import json
from pathlib import Path

import pytest

from src.research.protocol import candidate_id, describe_plan, trial_spec, validate_plan
from src.research.runner import (preflight, regenerate_report, run_study, schedule_next, session_lock,
                                 smoke_plan, validate_result)


@pytest.fixture
def plan(tmp_path):
    return smoke_plan(tmp_path / "study")


def record(spec, metric=3.0, status="completed"):
    return {**spec, "metric": metric, "status": status}


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(seeds=[0, 11]),
    lambda p: p.update(learning_rates=[.001, float("nan")]),
    lambda p: p.update(widths=[8, 10]),
    lambda p: p["model"].update(optimizer="muon"),
    lambda p: p["data"].update(val_fraction=.2),
    lambda p: p["limits"].update(max_trials=0),
])
def test_plan_rejects_ambiguous_or_invalid_protocol(plan, mutation):
    mutation(plan)
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_plan_budget_matches_scheduled_matrix(plan):
    description = describe_plan(plan)
    assert description["training_trials"] == 10
    assert description["test_evaluations_max"] == 8
    assert description["training_character_tokens_per_trial"] == 32


def test_selection_is_frozen_before_transfer_and_test(plan):
    records, selection, seen = [], {}, []
    while True:
        spec, selection, reason = schedule_next(plan, records, selection)
        if reason:
            assert reason == "completed"
            break
        # Challenger wins confirm but loses catastrophically at larger widths
        # and on test. Neither can revise the small-width locked selection.
        challenger = spec["learning_rate"] != plan["baseline_lr"]
        metric = (2.0 if challenger else 3.0) if spec["stage"] in ("screen", "confirm") else (100.0 if challenger else 1.0)
        if spec["stage"] in ("transfer", "test"):
            assert selection["winner_id"] == candidate_id(.003)
        if spec["stage"] == "test":
            assert "transfer" in seen
            assert "parent_trial_id" in spec
        records.append(record(spec, metric))
        seen.append(spec["stage"])
    assert len(records) == 18
    assert selection["winner_id"] == candidate_id(.003)


def test_confirmation_failure_cannot_reduce_seed_count(plan):
    records, selection = [], {}
    while True:
        spec, selection, reason = schedule_next(plan, records, selection)
        if reason:
            assert reason == "confirmation_incomplete"
            break
        status = "failed" if spec["stage"] == "confirm" and spec["seed"] == 22 else "completed"
        records.append(record(spec, 2.0, status))
    assert "winner_id" not in selection
    assert not any(r["stage"] in ("transfer", "test") for r in records)


def test_inconsistent_confirm_deltas_retain_baseline(plan):
    records, selection = [], {}
    while True:
        spec, selection, reason = schedule_next(plan, records, selection)
        if reason or spec["stage"] == "transfer":
            break
        metric = 3.0
        if spec["learning_rate"] != plan["baseline_lr"]:
            metric = 1.0 if spec["seed"] == 11 else 3.1
        records.append(record(spec, metric))
    assert selection["mean_delta"] < 0
    assert selection["winner_id"] == candidate_id(plan["baseline_lr"])


@pytest.fixture
def fake_worker(monkeypatch):
    calls = []

    def launch(request_path, output, timeout, disk_limit, lock_fd):
        calls.append(request_path)
        request = json.loads(request_path.read_text())
        cfg = request["config"]
        training = cfg["training"]
        loss = 4.0 - training["learning_rate"] * 100
        tokens = training["batch_size"] * cfg["model"]["block_size"]
        if request["stage"] == "test":
            result = {"device": cfg["training"]["device"], "split": "test",
                      "checkpoint_step": training["max_iters"], "ce_loss": loss,
                      "tokens_evaluated": training["eval_iters"] * tokens}
        else:
            checkpoint = request_path.parent / "checkpoints" / "final.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"unit-test fixture, not a real model")
            result = {"device": training["device"], "completed_steps": training["max_iters"],
                      "stop_reason": "max_iters", "final_checkpoint": str(checkpoint), "training_time_sec": .01,
                      "final_metric": {"step": training["max_iters"] - 1,
                                       "tokens_processed": tokens * training["max_iters"],
                                       "val_ce_loss": loss, "rss_mb": 1.0, "accel_mem_mb": None,
                                       "accel_mem_kind": None}}
        Path(request["result_path"]).write_text(json.dumps(result))
        return "completed", None

    monkeypatch.setattr("src.research.runner.launch", launch)
    return calls


def test_session_completes_and_resume_does_not_repeat_trials(plan, tmp_path, fake_worker):
    output = tmp_path / "study"
    first = run_study(plan, output)
    assert first["stop_reason"] == "completed"
    assert len(fake_worker) == 18
    assert (output / "report.md").is_file()
    second = run_study(plan, output)
    assert second["records"] == first["records"]
    assert len(fake_worker) == 18
    assert second["active_seconds"] >= first["active_seconds"]


def test_changed_dataset_rejects_resume(plan, tmp_path, fake_worker):
    output = tmp_path / "study"
    run_study(plan, output)
    with Path(plan["data"]["path"]).open("a") as corpus:
        corpus.write("changed")
    with pytest.raises(ValueError, match="Frozen plan"):
        run_study(plan, output)
    assert len(fake_worker) == 18


def test_trial_budget_stops_before_extra_job(plan, tmp_path, fake_worker):
    plan["limits"]["max_trials"] = 2
    state = run_study(plan, tmp_path / "study")
    assert state["stop_reason"] == "max_trials"
    assert len(fake_worker) == len(state["records"]) == 2
    assert "winner_id" not in state["selection"]


def test_failure_cap_stops_and_does_not_score_failure(plan, tmp_path, monkeypatch):
    monkeypatch.setattr("src.research.runner.launch", lambda *args: ("timeout", "timeout fixture"))
    state = run_study(plan, tmp_path / "study")
    assert state["stop_reason"] == "max_failures"
    assert len(state["records"]) == 1
    assert state["records"][0]["metric"] is None


def test_session_lock_excludes_concurrent_controller(tmp_path):
    with session_lock(tmp_path):
        with pytest.raises(ValueError, match="already running"):
            with session_lock(tmp_path):
                pass


def test_report_regeneration_obeys_session_lock(tmp_path):
    with session_lock(tmp_path):
        with pytest.raises(ValueError, match="already running"):
            regenerate_report(tmp_path)


def test_dataset_change_during_worker_invalidates_result(plan, tmp_path, fake_worker, monkeypatch):
    from src.research import runner
    original = runner.launch

    def changed_during_worker(*args):
        outcome = original(*args)
        with Path(plan["data"]["path"]).open("a") as corpus:
            corpus.write("changed during training")
        return outcome

    monkeypatch.setattr(runner, "launch", changed_during_worker)
    state = run_study(plan, tmp_path / "study")
    assert state["stop_reason"] == "evidence_changed"
    assert len(fake_worker) == 1
    assert state["records"][0]["status"] == "invalid"
    assert state["records"][0]["metric"] is None


def test_partial_training_result_cannot_be_ranked(plan, tmp_path):
    result = {"device": "cpu", "completed_steps": 1, "stop_reason": "time_budget",
              "final_metric": {"step": 0, "tokens_processed": 16}}
    with pytest.raises(ValueError, match="Incomplete fixed-token"):
        validate_result(result, {"stage": "screen"}, plan, tmp_path)


def test_preflight_uses_same_split_rounding_as_loader(plan, tmp_path):
    path = Path(plan["data"]["path"])
    path.write_text("x" * 109)
    plan["data"].update(train_fraction=.7, val_fraction=.2)
    plan["training"]["block_size"] = 11
    with pytest.raises(ValueError, match="Each split"):
        preflight(plan)


@pytest.mark.parametrize("cause,expected", [("time", "timeout"), ("disk", "budget_exhausted")])
def test_worker_limits_terminate_process_without_accepting_score(tmp_path, monkeypatch, cause, expected):
    from src.research import runner
    terminated = []
    class FakeProcess:
        def poll(self):
            return None
    process = FakeProcess()
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *args, **kwargs: process)
    monkeypatch.setattr(runner, 'terminate', lambda p: terminated.append(p))
    clock = iter([0.0, 2.0 if cause == 'time' else 0.0])
    monkeypatch.setattr(runner.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(runner, 'disk_mb', lambda _: 100.0)
    status, _ = runner.launch(tmp_path / 'request.json', tmp_path, timeout=1.0,
                              disk_limit=1.0, lock_fd=0)
    assert status == expected
    assert terminated == [process]


def test_expired_session_budget_launches_no_worker(plan, tmp_path, fake_worker):
    plan['limits']['session_seconds'] = 1e-12
    state = run_study(plan, tmp_path / 'study')
    assert state['stop_reason'] == 'session_time'
    assert fake_worker == []
