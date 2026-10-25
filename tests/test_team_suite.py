"""Protocol tests only: every execution substitutes a fake worker process."""
from __future__ import annotations

import copy
import importlib.metadata
import json
from pathlib import Path

import pytest

from src.research.team import suite


@pytest.fixture
def specification(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unit tests must never launch a training process")
    monkeypatch.setattr(suite.subprocess, "Popen", forbidden)
    monkeypatch.setattr(suite.platform, "platform", lambda: "mock-test-platform")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("A tiny deterministic corpus.\n" * 100)
    config = {"data": {"path": str(corpus), "train_fraction": .8, "val_fraction": .1},
              "model": {"dim": 16, "n_heads": 2, "n_layers": 1, "block_size": 8,
                        "dropout": 0., "norm_type": "rmsnorm", "rope": 1,
                        "attn_type": "mha", "mlp_type": "dense", "vocab_size": 1},
              "training": {"max_iters": 4, "batch_size": 2, "eval_every": 2,
                           "eval_iters": 3, "cpu_threads": 1, "device": "cpu",
                           "learning_rate": .001, "weight_decay": .1, "grad_clip": 1.}}
    spec = {"version": 1, "hypothesis": "A declared model change improves small and larger models.",
            "targets": [{"name": "small", "width": 16, "n_heads": 2},
                        {"name": "large", "width": 32, "n_heads": 4}],
            "seeds": [11, 22], "eval_seed": 100, "min_improvement": .01,
            "limits": {"max_trials": 16, "session_seconds": 1000, "trial_seconds": 20,
                       "max_disk_mb": 100, "max_failures": 4, "max_consecutive_failures": 3}}
    for arm in suite.ARMS:
        root = tmp_path / arm
        (root / "src" / "models").mkdir(parents=True)
        (root / "src" / "train").mkdir()
        (root / "src" / "models" / "gpt.py").write_text(f"# frozen {arm} model\n")
        (root / "src" / "train" / "train.py").write_text("# same trusted trainer\n")
        (root / "requirements.txt").write_text("torch\n")
        spec[arm] = {"root": str(root), "config": copy.deepcopy(config),
                     "source_sha256": suite.source_digest(root)}
    spec["candidate"]["config"]["training"]["learning_rate"] = .002
    return spec


def test_schedule_contract_validation(specification):
    spec = copy.deepcopy(specification)
    train = spec['candidate']['config']['training']
    train['lr_schedule'] = {'kind': 'cosine', 'total_steps': 4, 'min_ratio': .1}
    assert suite.validate_suite(spec)['candidate']['config']['training']['lr_schedule'] == train['lr_schedule']
    train['lr_schedule']['total_steps'] = 5
    with pytest.raises(ValueError, match='total_steps'): suite.validate_suite(spec)


def fake_launcher(monkeypatch, *, deltas=None, fail=None, partial=None, mutate=None):
    calls = []
    deltas = deltas or {16: -.1, 32: -.05}

    def launch(request_path, output, timeout, disk_limit, lock_fd, *, root):
        request = json.loads(request_path.read_text())
        directory = request_path.parent
        cfg = request["config"]
        arm = Path(root).name
        call = {"arm": arm, "stage": request["stage"], "seed": cfg["training"]["seed"],
                "width": cfg["model"]["dim"], "request": request, "timeout": timeout}
        calls.append(call)
        if fail and fail(call):
            return "timeout", "Mock timeout"
        metric = 2.0 + (deltas[call["width"]] if arm == "candidate" else 0)
        runtime = {"torch": importlib.metadata.version("torch"), "cpu_threads": request["cpu_threads"],
                   "requested_device": "cpu"}
        train = cfg["training"]
        if request["stage"] == "test":
            result = {"device": "cpu", "split": "test", "checkpoint_step": train["max_iters"],
                      "ce_loss": metric, "tokens_evaluated": train["eval_iters"] * train["batch_size"] * cfg["model"]["block_size"]}
        else:
            checkpoint = directory / "checkpoints" / "final.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"mock checkpoint; never loaded")
            result = {"device": "cpu", "completed_steps": train["max_iters"], "stop_reason": "max_iters",
                      "final_metric": {"step": train["max_iters"] - 1,
                                       "tokens_processed": train["max_iters"] * train["batch_size"] * cfg["model"]["block_size"],
                                       "val_ce_loss": metric, "rss_mb": 1., "accel_mem_mb": None, "accel_mem_kind": None},
                      "final_checkpoint": str(checkpoint), "training_time_sec": 1.}
            if partial and partial(call):
                result["completed_steps"] -= 1
        result["runtime"] = runtime
        Path(request["result_path"]).write_text(json.dumps(result))
        if mutate:
            mutate(call)
        return "completed", None

    monkeypatch.setattr(suite, "_launch", launch)
    return calls


def test_disabled_guard_precedes_any_side_effect(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "validate_suite", lambda *_: pytest.fail("Validation must not execute"))
    with pytest.raises(PermissionError, match="Training is disabled"):
        suite.run_suite({}, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def test_describe_is_pure_and_counts_all_jobs(specification):
    original = copy.deepcopy(specification)
    description = suite.describe_suite(specification)
    assert description["training_trials"] == 8
    assert description["test_evaluations_max"] == 8
    assert description["training_character_tokens_per_trial"] == 64
    assert description["training_enabled"] is False
    assert specification == original


@pytest.mark.parametrize("change", [
    lambda s: s["candidate"]["config"]["training"].update(max_iters=5),
    lambda s: s["candidate"]["config"]["training"].update(eval_iters=4),
    lambda s: s["candidate"]["config"]["training"].update(device="mps"),
    lambda s: s["candidate"]["config"]["model"].update(block_size=16),
    lambda s: s["candidate"]["config"]["data"].update(val_fraction=.2),
    lambda s: s["candidate"]["config"]["training"].update(max_train_seconds=10),
    lambda s: s["candidate"]["config"]["training"].update(optimizer="Muon"),
    lambda s: s["candidate"]["config"]["model"].update(causal=False),
    lambda s: s["targets"][1].update(n_heads=2),
    lambda s: s.update(seeds=[11, 11]),
    lambda s: s.update(min_improvement=float("nan")),
    lambda s: s.update(audit_test="false"),
    lambda s: s.update(data_sha256="not-a-sha256"),
])
def test_unfair_or_unsupported_comparisons_rejected(specification, change):
    change(specification)
    with pytest.raises(ValueError):
        suite.validate_suite(specification)


def test_full_fake_suite_freezes_selection_before_transfer_and_test(specification, tmp_path, monkeypatch):
    calls = fake_launcher(monkeypatch, deltas={16: -.1, 32: .2})
    report = suite.run_suite(specification, tmp_path / "result", allow_training=True)
    assert report["stop_reason"] == "completed"
    assert report["local_outcome"] == "supported"
    assert report["transfer_outcome"] == "contradicted"
    assert report["selection"]["winner"] == "candidate"
    assert len(report["performance"]) == 6
    assert all(row["complete"] for row in report["performance"])
    throughput = next(row for row in report["performance"] if row["id"] == "small:training_throughput_pct")
    assert throughput["mean_delta"] == 0
    assert throughput["baseline_mean"] == throughput["candidate_mean"] == 64
    assert [c["stage"] for c in calls] == ["local"] * 4 + ["transfer"] * 4 + ["test"] * 8
    assert [c["arm"] for c in calls[:4]] == ["baseline", "candidate", "candidate", "baseline"]
    assert all(c["request"]["config"]["training"]["strict_device"] for c in calls)
    assert all(c["request"]["config"]["training"]["eval_seed"] == 100 for c in calls)
    assert (tmp_path / "result" / "manifest.json").exists()


def test_baseline_retention_skips_candidate_holdout(specification, tmp_path, monkeypatch):
    calls = fake_launcher(monkeypatch, deltas={16: .1, 32: -.1})
    report = suite.run_suite(specification, tmp_path / "result", allow_training=True)
    assert report["local_outcome"] == "contradicted"
    assert report["transfer_outcome"] == "not_applicable"
    assert report["audit_outcome"] == "baseline_only"
    assert report["selection"]["winner"] == "baseline"
    assert len(calls) == 12
    assert all(c["arm"] == "baseline" for c in calls if c["stage"] == "test")


def test_failed_seed_is_recorded_and_never_replaced(specification, tmp_path, monkeypatch):
    calls = fake_launcher(monkeypatch, fail=lambda c: c["arm"] == "candidate" and c["seed"] == 11)
    report = suite.run_suite(specification, tmp_path / "result", allow_training=True)
    assert report["stop_reason"] == "local_incomplete"
    assert report["local_outcome"] == "incomplete"
    assert report["comparisons"][0]["missing_seeds"] == [11]
    assert len(calls) == 4
    resumed = suite.run_suite(specification, tmp_path / "result", allow_training=True)
    assert len(calls) == 4
    assert resumed["stop_reason"] == "local_incomplete"


def test_partial_token_budget_does_not_score(specification, tmp_path, monkeypatch):
    fake_launcher(monkeypatch, partial=lambda c: c["arm"] == "candidate")
    report = suite.run_suite(specification, tmp_path / "result", allow_training=True)
    assert report["local_outcome"] == "incomplete"
    candidate = [r for r in report["records"] if r["arm"] == "candidate"]
    assert all(r["status"] == "invalid" and r["metric"] is None for r in candidate)


def test_completed_resume_does_not_duplicate_workers(specification, tmp_path, monkeypatch):
    calls = fake_launcher(monkeypatch)
    first = suite.run_suite(specification, tmp_path / "result", allow_training=True)
    second = suite.run_suite(specification, tmp_path / "result", allow_training=True)
    assert len(calls) == 16
    assert second["selection"] == first["selection"]
    assert second["stop_reason"] == "completed"


def test_source_mutation_invalidates_evidence(specification, tmp_path, monkeypatch):
    def mutate(call):
        (Path(specification["candidate"]["root"]) / "src" / "models" / "gpt.py").write_text("# altered\n")
    calls = fake_launcher(monkeypatch, mutate=mutate)
    report = suite.run_suite(specification, tmp_path / "result", allow_training=True)
    assert len(calls) == 1
    assert report["stop_reason"] == "evidence_changed"
    assert report["local_outcome"] == "invalid"
    assert report["evidence_integrity"] is False
    assert all(not row["complete"] and row["mean_delta"] is None for row in report["performance"])
    assert report["records"][0]["metric"] is None


def test_protected_trainer_change_rejected_before_workers(specification, tmp_path, monkeypatch):
    root = Path(specification["candidate"]["root"])
    (root / "src" / "train" / "train.py").write_text("# changed evaluator\n")
    specification["candidate"]["source_sha256"] = suite.source_digest(root)
    calls = fake_launcher(monkeypatch)
    with pytest.raises(ValueError, match="protected"):
        suite.run_suite(specification, tmp_path / "result", allow_training=True)
    assert not calls


def test_trial_limit_bounds_training_and_audits(specification, tmp_path, monkeypatch):
    specification["limits"]["max_trials"] = 5
    calls = fake_launcher(monkeypatch)
    report = suite.run_suite(specification, tmp_path / "result", allow_training=True)
    assert len(calls) == 5
    assert report["stop_reason"] == "max_trials"
    assert report["transfer_outcome"] == "incomplete"
    assert not any(c["stage"] == "test" for c in calls)


def test_resume_rejects_changed_specification(specification, tmp_path, monkeypatch):
    calls = fake_launcher(monkeypatch)
    suite.run_suite(specification, tmp_path / "result", allow_training=True)
    specification["candidate"]["config"]["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="Frozen specification"):
        suite.run_suite(specification, tmp_path / "result", allow_training=True)
    assert len(calls) == 16


def test_bootstrap_is_syntactically_valid_but_not_executed():
    compile(suite.WORKER_BOOTSTRAP, "trusted-scoring-bootstrap", "exec")
    assert "super().forward(idx, None" in suite.WORKER_BOOTSTRAP
    assert "cross_entropy(logits.reshape" in suite.WORKER_BOOTSTRAP


def test_changed_completed_checkpoint_blocks_resume(specification, tmp_path, monkeypatch):
    calls = fake_launcher(monkeypatch)
    report = suite.run_suite(specification, tmp_path / "result", allow_training=True)
    Path(report["records"][0]["checkpoint"]).write_bytes(b"changed checkpoint")
    resumed = suite.run_suite(specification, tmp_path / "result", allow_training=True)
    assert resumed["stop_reason"] == "checkpoint_changed"
    assert resumed["local_outcome"] == "invalid"
    assert len(calls) == 16


def test_consecutive_failure_cap_does_not_replace_seeds(specification, tmp_path, monkeypatch):
    specification["limits"]["max_consecutive_failures"] = 2
    calls = fake_launcher(monkeypatch, fail=lambda _: True)
    report = suite.run_suite(specification, tmp_path / "result", allow_training=True)
    assert len(calls) == 2
    assert report["stop_reason"] == "max_consecutive_failures"
    assert report["local_outcome"] == "incomplete"


def test_disk_cap_prevents_worker_dispatch(specification, tmp_path, monkeypatch):
    calls = fake_launcher(monkeypatch)
    monkeypatch.setattr(suite, "disk_mb", lambda _: 101)
    report = suite.run_suite(specification, tmp_path / "result", allow_training=True)
    assert report["stop_reason"] == "disk_limit"
    assert not calls


def test_stale_running_job_is_charged_and_not_retried(specification, tmp_path, monkeypatch):
    specification["limits"]["max_trials"] = 1
    calls = fake_launcher(monkeypatch)
    suite.run_suite(specification, tmp_path / "result", allow_training=True)
    state_path = tmp_path / "result" / "state.json"
    state = json.loads(state_path.read_text())
    state["records"][0].update(status="running", metric=None)
    state["active_seconds"] = 20
    state_path.write_text(json.dumps(state))
    report = suite.run_suite(specification, tmp_path / "result", allow_training=True)
    assert len(calls) == 1
    assert report["records"][0]["status"] == "failed"
    assert report["active_seconds"] >= 20


def test_audit_can_be_deferred_until_final_campaign_iteration(specification, tmp_path, monkeypatch):
    specification["audit_test"] = False
    calls = fake_launcher(monkeypatch)
    description = suite.describe_suite(specification)
    assert description["test_evaluations_max"] == 0
    assert description["jobs_max"] == description["training_trials"] == 8
    report = suite.run_suite(specification, tmp_path / "result", allow_training=True)
    assert len(calls) == 8
    assert report["stop_reason"] == "completed"
    assert report["local_outcome"] == report["transfer_outcome"] == "supported"
    assert report["audit_outcome"] == "not_requested"
    assert report["holdout_consumed"] is False
    assert not any(c["stage"] == "test" for c in calls)
    assert suite.run_suite(specification, tmp_path / "result", allow_training=True)["audit_outcome"] == "not_requested"
    assert len(calls) == 8


def test_audit_default_and_consumption_are_explicit(specification, tmp_path, monkeypatch):
    assert suite.validate_suite(specification)["audit_test"] is True
    assert suite.validate_suite(specification)["data_sha256"] is None
    fake_launcher(monkeypatch)
    report = suite.run_suite(specification, tmp_path / "result", allow_training=True)
    assert report["holdout_consumed"] is True


def test_expected_corpus_hash_mismatch_blocks_all_workers(specification, tmp_path, monkeypatch):
    specification["data_sha256"] = "0" * 64
    calls = fake_launcher(monkeypatch)
    with pytest.raises(ValueError, match="Corpus changed: data_sha256"):
        suite.run_suite(specification, tmp_path / "result", allow_training=True)
    assert not calls
    assert not (tmp_path / "result").exists()


def test_expected_corpus_hash_is_preserved_in_contract(specification, tmp_path, monkeypatch):
    path = Path(specification["baseline"]["config"]["data"]["path"])
    specification["data_sha256"] = suite.file_digest(path)
    specification["audit_test"] = False
    calls = fake_launcher(monkeypatch)
    report = suite.run_suite(specification, tmp_path / "result", allow_training=True)
    assert len(calls) == 8
    assert report["contract"]["data"] == specification["data_sha256"]


def test_expected_corpus_hash_detects_worker_time_mutation(specification, tmp_path, monkeypatch):
    path = Path(specification["baseline"]["config"]["data"]["path"])
    specification["data_sha256"] = suite.file_digest(path)
    calls = fake_launcher(monkeypatch, mutate=lambda _: path.write_text("a changed corpus\n" * 100))
    report = suite.run_suite(specification, tmp_path / "result", allow_training=True)
    assert len(calls) == 1
    assert report["stop_reason"] == "evidence_changed"
    assert report["contract"]["data"] == specification["data_sha256"]
    assert report["records"][0]["metric"] is None
