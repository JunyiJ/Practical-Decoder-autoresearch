"""Control-flow tests use fabricated receipts; no training or agents execute."""
from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from src.research.team import coordinator
from src.research.team.knowledge import KnowledgeBank
from src.research.team.roles import validate_response
from src.research.team.runtime import AgentRunError


def forbidden(*args, **kwargs):
    raise AssertionError("A real agent, worker, or test process must not run in these tests")


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinator.subprocess, "Popen", forbidden)
    monkeypatch.setattr(coordinator, "run_call", forbidden)
    monkeypatch.setattr(coordinator, "run_suite", forbidden)
    monkeypatch.setattr(coordinator, "_local_tests", forbidden)
    monkeypatch.setattr(coordinator, "source_digest", lambda root: "a" * 64)
    monkeypatch.setattr(coordinator, "inspect_host", lambda device: {"device": device, "device_available": True})
    config = {"version": 1, "name": "fixture", "objective": "Check paired transfer.",
        "repository": str(tmp_path / "repo"), "base_ref": "HEAD", "base_plan": str(tmp_path / "plan.yaml"),
        "knowledge_bank": str(tmp_path / "knowledge"),
        "runtime": {"executable": "codex", "model": None, "web_search": True},
        "limits": {"max_iterations": 2, "max_agent_calls": 8, "agent_seconds": 10,
                   "total_agent_seconds": 60, "agent_output_mb": 2, "local_test_seconds": 10,
                   "total_training_seconds": 40}, "local_tests": ["tests/test_unit.py"]}
    plan = {"widths": [16, 32], "seeds": [1, 2], "baseline_lr": .001, "eval_seed": 123,
            "min_improvement": .005, "model": {"head_dim": 8}, "training": {"cpu_threads": 1},
            "limits": {"max_trials": 16, "session_seconds": 30, "trial_seconds": 5,
                       "max_disk_mb": 50, "max_failures": 2, "max_consecutive_failures": 2}}
    (tmp_path / "corpus.txt").write_text("Synthetic corpus fixture; no training reads this file.\n")
    base_config = {"data": {"path": str(tmp_path / "corpus.txt"), "train_fraction": .8, "val_fraction": .1},
                   "model": {"dim": 16, "n_heads": 2, "n_layers": 1, "block_size": 8, "dropout": 0.,
                             "vocab_size": 10, "rope": 1, "attn_type": "mha", "mlp_type": "dense"},
                   "training": {"max_iters": 2, "batch_size": 2, "eval_every": 2, "eval_iters": 1,
                                "device": "cpu", "cpu_threads": 1, "learning_rate": .001, "weight_decay": .1, "grad_clip": 1}}
    events = {"workspaces": [], "freezes": [], "local_tests": [], "roles": [], "suites": []}

    def create(repo, destination, branch, base_ref):
        destination.mkdir(parents=True, exist_ok=True)
        events["workspaces"].append(str(destination))
        return {"path": str(destination), "branch": branch, "base_commit": "b" * 40}

    def freeze(workspace, destination, base_commit):
        destination.mkdir(parents=True, exist_ok=True)
        worker = destination / "src/research/worker.py"
        worker.parent.mkdir(parents=True)
        worker.write_text("# Pure fixture: never executed.\n")
        patch = destination / "candidate.patch"
        patch.write_text("fixture patch\n")
        events["freezes"].append(str(destination))
        return {"path": str(destination), "base_commit": base_commit,
                "source_digest": "a" * 64, "patch_path": str(patch)}

    monkeypatch.setattr(coordinator, "create_workspace", create)
    monkeypatch.setattr(coordinator, "freeze_workspace", freeze)
    monkeypatch.setattr(coordinator, "inspect_workspace", lambda *args: {"changes": [], "branch": "codex/fixture/iteration-001"})
    monkeypatch.setattr(coordinator, "load_plan", lambda path: deepcopy(plan))
    monkeypatch.setattr(coordinator, "training_config", lambda *args: deepcopy(base_config))
    output = tmp_path / "campaign"
    state = coordinator.init_campaign(config, output)
    return {"output": output, "state": state, "config": config, "plan": plan,
            "base_config": base_config, "events": events, "monkeypatch": monkeypatch}


def save(campaign, state):
    coordinator.atomic_json(campaign["output"] / "campaign.json", state)


def read(campaign):
    return json.loads((campaign["output"] / "campaign.json").read_text())


def responses():
    return {
        "idea": {"title": "Lower learning rate", "hypothesis": "A lower LR might transfer.", "rationale": "A falsifiable comparison.",
                 "change_kind": "config", "based_on": [], "proposed_changes": ["Reduce LR."],
                 "transfer_prediction": "Compare widths 16 and 32.", "falsification": "Reject a paired regression.",
                 "prediction": {"mechanism": "Synthetic lower-LR hypothesis; no measurement.", "assumptions": ["Same corpus and token budget."],
                     "estimates": [{"target": f"width-{width}", "metric": metric, "expected": 0., "lower": -.1, "upper": .1,
                                    "confidence": "low", "rationale": "Synthetic test fixture, not a scientific forecast."}
                                   for width in (16, 32) for metric in ("validation_ce_delta", "training_throughput_pct", "peak_rss_pct")]},
                 "sources": [], "search_notes": "Synthetic unit-test response; no literature read."},
        "coder": {"summary": "Prepared lower LR.", "config_overrides": [{"path": "training.learning_rate", "value_json": "0.0003"}],
                  "test_files": ["tests/test_candidate.py"], "risks": []},
        "executor": {"ready": True, "reasons": ["Synthetic preflight only."]},
        "evaluator": {"summary": "Synthetic interpretation, not an actual experiment.", "validity_concerns": [],
                      "learnings": [{"finding": "Synthetic lesson; no real result.", "explanation": "Mock protocol evidence only.",
                          "limitation": "No training was executed.", "next_action": "Inspect the mocked comparison.",
                          "evidence_ids": ["width-16:validation_ce_delta"], "confidence": "low"}],
                      "next_ideas": ["Test weight decay next."], "literature_connections": []},
    }


def install_fake_handoffs(campaign, *, replies=None, report=None):
    replies = responses() if replies is None else replies
    report = {"local_outcome": "supported", "transfer_outcome": "supported"} if report is None else report
    events, monkeypatch = campaign["events"], campaign["monkeypatch"]

    def run_call(role, prompt, cwd, output, settings, **kwargs):
        events["roles"].append({"role": role, "prompt": prompt, "cwd": str(cwd), **kwargs})
        output.mkdir(parents=True)
        result = validate_response(role, replies[role])
        (output / "response.json").write_text(json.dumps(result))
        return {"response": result, "duration_seconds": .01, "usage": []}

    def local_tests(workspace, files, output, timeout):
        events["local_tests"].append({"files": files, "timeout": timeout})
        output.mkdir(parents=True)
        receipt = {"exit_code": 0, "duration_seconds": .01}
        coordinator.atomic_json(output / "tests.json", receipt)
        return receipt

    def run_suite(spec, output, **kwargs):
        events["suites"].append({"spec": deepcopy(spec), **kwargs})
        assert kwargs["allow_training"] is True
        return deepcopy(report)

    monkeypatch.setattr(coordinator, "run_call", run_call)
    monkeypatch.setattr(coordinator, "_local_tests", local_tests)
    monkeypatch.setattr(coordinator, "run_suite", run_suite)
    return replies


def advance_to(campaign, stage):
    for _ in range(6):
        state = read(campaign)
        if state["iterations"] and state["iterations"][-1]["stage"] == stage:
            return state
        state = coordinator.advance(campaign["output"], run_agents=True, allow_training=True)
        if state["status"] == "needs_attention":
            pytest.fail(state["iterations"][-1]["error"])
    pytest.fail(f"Fixture did not reach {stage}")


def test_init_and_default_advance_only_prepare_handoff(campaign):
    assert campaign["state"]["status"] == "prepared"
    bank = KnowledgeBank(campaign["config"]["knowledge_bank"])
    negative = bank.add({"kind": "observation", "title": "Prior numerical failure", "summary": "A previous intervention diverged.",
                         "outcome": "failed", "created_by": "fixture", "scope": {"width": 64}})
    request = coordinator.advance(campaign["output"])
    assert request["state"] == "prepared" and request["role"] == "idea"
    prompt = (Path(request["handoff"]) / "prompt.md").read_text()
    assert negative in prompt and "Prior numerical failure" in prompt
    state = read(campaign)
    assert state["agent_calls"] == [] and state["training_seconds"] == 0
    assert state["iterations"][0]["stage"] == "idea"
    assert len(campaign["events"]["workspaces"]) == 1
    assert not campaign["events"]["local_tests"]


def test_each_role_handoff_and_explicit_training_gate(campaign):
    install_fake_handoffs(campaign)
    output, events = campaign["output"], campaign["events"]
    for stage in ("coder", "executor", "train"):
        assert coordinator.advance(output, run_agents=True)["iterations"][0]["stage"] == stage
    assert [item["role"] for item in events["roles"]] == ["idea", "coder", "executor"]
    assert events["roles"][1]["cwd"] == read(campaign)["iterations"][0]["workspace"]["path"]
    state = coordinator.advance(output, run_agents=True)
    assert state["status"] == "awaiting_training_authorization"
    assert events["suites"] == []
    state = coordinator.advance(output, run_agents=True, allow_training=True)
    assert state["iterations"][0]["stage"] == "evaluator"
    assert len(events["suites"]) == 1
    assert state["iterations"][0]["suite"]["candidate"]["config"]["training"]["learning_rate"] == .0003
    assert events["local_tests"][0]["files"] == ["tests/test_unit.py", "tests/test_candidate.py"]
    state = coordinator.advance(output, run_agents=True)
    iteration = state["iterations"][0]
    assert iteration["stage"] == "completed"
    bank = KnowledgeBank(campaign["config"]["knowledge_bank"])
    evidence = bank.get(iteration["evaluation_id"])
    assert evidence["outcome"] == "supported"
    assert evidence["scope"]["generalizes_beyond_tested_models"] is False
    assert iteration["execution_id"] in evidence["related_ids"]
    request = coordinator.prepare_next(output)
    next_prompt = (Path(request["handoff"]) / "prompt.md").read_text()
    assert iteration["evaluation_id"] in next_prompt
    assert "Synthetic interpretation" in next_prompt


def test_prediction_is_persisted_before_coder_and_included_in_handoff(campaign):
    install_fake_handoffs(campaign)
    state = coordinator.advance(campaign["output"], run_agents=True)
    iteration = state["iterations"][0]
    assert iteration["stage"] == "coder"
    path = Path(iteration["prediction_artifact"]["path"])
    document = json.loads(path.read_text())
    assert document["prediction"] == iteration["idea"]["prediction"]
    bank = KnowledgeBank(campaign["config"]["knowledge_bank"])
    assert bank.get(iteration["prediction_id"])["scope"]["registered_before_execution"] is True
    assert bank.search(kind="trial") == []
    assert campaign["events"]["suites"] == []
    request = coordinator.prepare_next(campaign["output"])
    assert iteration["prediction_id"] in (Path(request["handoff"]) / "prompt.md").read_text()


def test_missing_target_forecast_stops_before_candidate_workspace(campaign):
    replies = responses()
    replies["idea"]["prediction"]["estimates"] = replies["idea"]["prediction"]["estimates"][:3]
    install_fake_handoffs(campaign, replies=replies)
    state = coordinator.advance(campaign["output"], run_agents=True)
    assert state["status"] == "needs_attention"
    assert "cover every target" in state["iterations"][0]["error"]
    assert len(campaign["events"]["workspaces"]) == 1


@pytest.mark.parametrize("tamper", ["prediction", "corpus"])
def test_changed_forecast_or_data_blocks_training(campaign, tamper):
    install_fake_handoffs(campaign)
    state = advance_to(campaign, "train")
    if tamper == "prediction":
        path = Path(state["iterations"][0]["prediction_artifact"]["path"])
        path.chmod(0o644)
        path.write_text("{}")
    else:
        Path(campaign["base_config"]["data"]["path"]).write_text("Changed scope after the prediction")
    state = coordinator.advance(campaign["output"], run_agents=True, allow_training=True)
    assert state["status"] == "needs_attention"
    assert campaign["events"]["suites"] == []


def test_complete_mock_iteration_persists_results_learnings_and_trajectory(campaign):
    records = [{"trial_id": f"{stage}_{width}_{seed}_{arm}", "target": f"width-{width}", "width": width,
                "stage": stage, "seed": seed, "arm": arm, "status": "completed", "metric": 2.0 if arm == "baseline" else 1.98,
                "tokens_processed": 32, "training_time_sec": 2.0 if arm == "baseline" else 1.0,
                "rss_mb": 100 if arm == "baseline" else 90, "duration_seconds": 3.0}
               for stage, width in (("local", 16), ("transfer", 32)) for seed in (1, 2) for arm in ("baseline", "candidate")]
    report = {"local_outcome": "supported", "transfer_outcome": "supported", "audit_outcome": "not_requested",
              "evidence_integrity": True, "holdout_consumed": False, "records": records, "active_seconds": 24}
    install_fake_handoffs(campaign, report=report)
    state = advance_to(campaign, "completed")
    iteration = state["iterations"][0]
    checks = {row["id"]: row for row in iteration["prediction_check"]["comparisons"]}
    assert checks["width-16:training_throughput_pct"]["observed_mean_delta"] == 100
    assert checks["width-16:training_throughput_pct"]["verdict"] == "above_range"
    bank = KnowledgeBank(campaign["config"]["knowledge_bank"])
    assert len(bank.search(kind="trial")) == 8
    assert bank.get(iteration["result_id"])["outcome"] == "supported"
    assert bank.get(iteration["learning_ids"][0])["outcome"] == "reported"
    assert iteration["prediction_id"] in bank.get(iteration["learning_ids"][0])["related_ids"]
    assert (campaign["output"] / iteration["id"] / "learnings.json").is_file()
    with (campaign["output"] / "trials.csv").open() as stream:
        trial_rows = list(csv.DictReader(stream))
    assert len(trial_rows) == 8 and all(row["status"] == "completed" for row in trial_rows)
    trajectory = json.loads((campaign["output"] / "trajectory.json").read_text())
    assert trajectory["iterations"][0]["learning_ids"] == iteration["learning_ids"]
    request = coordinator.prepare_next(campaign["output"])
    prompt = (Path(request["handoff"]) / "prompt.md").read_text()
    assert iteration["learning_ids"][0] in prompt
    assert "prediction_targets" in prompt


def test_invented_learning_evidence_cannot_enter_evaluation_records(campaign):
    replies = responses()
    replies["evaluator"]["learnings"][0]["evidence_ids"] = ["imaginary-trial"]
    install_fake_handoffs(campaign, replies=replies)
    advance_to(campaign, "evaluator")
    for _ in range(2):
        retry = coordinator.advance(campaign["output"], run_agents=True)
        assert retry["status"] == "ready_for_evaluator"
    state = coordinator.advance(campaign["output"], run_agents=True)
    assert state["status"] == "needs_attention"
    bank = KnowledgeBank(campaign["config"]["knowledge_bank"])
    assert bank.search(kind="evaluation") == []
    iteration = state["iterations"][0]
    assert bank.get(iteration["failure_learning_id"])["scope"]["not_a_negative_model_result"] is True


@pytest.mark.parametrize("deterministic,concerns,expected", [
    ("contradicted", [], "contradicted"), ("supported", ["Unequal budgets detected."], "inconclusive"),
    ("smoke_only", [], "inconclusive"), ("mixed", [], "mixed"),
])
def test_evaluator_narrative_cannot_upgrade_deterministic_report(campaign, deterministic, concerns, expected):
    replies = responses()
    replies["evaluator"].update(summary="The prose claims a universal win, which must not alter evidence.", validity_concerns=concerns)
    install_fake_handoffs(campaign, replies=replies, report={"local_outcome": "supported", "transfer_outcome": deterministic})
    state = advance_to(campaign, "completed")
    record = KnowledgeBank(campaign["config"]["knowledge_bank"]).get(state["iterations"][0]["evaluation_id"])
    assert record["outcome"] == expected
    assert record["scope"]["transfer_outcome"] == deterministic


@pytest.mark.parametrize("cap,expected", [("calls", "agent_call_budget_exhausted"), ("time", "agent_time_budget_exhausted")])
def test_agent_caps_prevent_any_call(campaign, cap, expected):
    state = read(campaign)
    if cap == "calls":
        state["config"]["limits"]["max_agent_calls"] = 1
        state["agent_calls"] = [{"status": "completed"}]
    else:
        state["agent_seconds"] = state["config"]["limits"]["total_agent_seconds"]
    save(campaign, state)
    assert coordinator.advance(campaign["output"], run_agents=True)["status"] == expected


def test_agent_call_timeout_is_clamped_to_remaining_campaign_budget(campaign):
    install_fake_handoffs(campaign)
    state = read(campaign)
    state["agent_seconds"] = 59
    save(campaign, state)
    state = coordinator.advance(campaign["output"], run_agents=True)
    assert campaign["events"]["roles"][0]["timeout"] == 1
    assert state["agent_calls"][0]["reserved_seconds"] == 1


def test_failed_agent_keeps_reserved_budget_and_records_operational_failure(campaign):
    def fail(*args, **kwargs):
        raise AgentRunError("Synthetic agent deadline exceeded")
    campaign["monkeypatch"].setattr(coordinator, "run_call", fail)
    state = coordinator.advance(campaign["output"], run_agents=True)
    assert state["status"] == "needs_attention"
    assert state["agent_seconds"] == state["config"]["limits"]["agent_seconds"]
    assert state["agent_calls"][0]["status"] == "failed"
    failure = KnowledgeBank(campaign["config"]["knowledge_bank"]).get(state["iterations"][0]["failure_id"])
    assert failure["outcome"] == "failed" and failure["details"]["not_a_negative_model_result"] is True


def test_interrupted_handoff_cannot_silently_repeat_call(campaign):
    state = read(campaign)
    state["agent_calls"] = [{"status": "running", "reserved_seconds": 10}]
    save(campaign, state)
    with pytest.raises(ValueError, match="Interrupted agent handoff"):
        coordinator.advance(campaign["output"], run_agents=True)


def test_unknown_knowledge_id_is_rejected_before_candidate_workspace(campaign):
    replies = responses()
    replies["idea"]["based_on"] = ["made-up-record"]
    install_fake_handoffs(campaign, replies=replies)
    state = coordinator.advance(campaign["output"], run_agents=True)
    assert state["status"] == "needs_attention"
    assert "does not exist" in state["iterations"][0]["error"]
    assert len(campaign["events"]["workspaces"]) == 1


def test_noop_candidate_fails_before_local_tests(campaign):
    replies = responses()
    replies["coder"]["config_overrides"] = []
    install_fake_handoffs(campaign, replies=replies)
    coordinator.advance(campaign["output"], run_agents=True)
    state = coordinator.advance(campaign["output"], run_agents=True)
    assert state["status"] == "needs_attention"
    assert "no code or configuration change" in state["iterations"][0]["error"]
    assert campaign["events"]["local_tests"] == []


def test_malformed_response_cannot_advance_phase(campaign):
    replies = responses()
    replies["idea"]["empirical_result"] = "supported"
    install_fake_handoffs(campaign, replies=replies)
    state = coordinator.advance(campaign["output"], run_agents=True)
    assert state["iterations"][0]["stage"] == "failed"
    assert "unexpected fields" in state["iterations"][0]["error"]


def test_preflight_rejection_never_reaches_suite(campaign):
    replies = responses()
    replies["executor"] = {"ready": False, "reasons": ["Missing prerequisites."]}
    install_fake_handoffs(campaign, replies=replies)
    advance_to(campaign, "executor")
    state = coordinator.advance(campaign["output"], run_agents=True, allow_training=True)
    assert state["iterations"][0]["stage"] == "failed"
    assert campaign["events"]["suites"] == []


@pytest.mark.parametrize("path", ["training.max_iters", "training.device", "data.path", "model.dim", "model.n_heads", "model.block_size", "model.vocab_size"])
def test_protected_overrides_cannot_change_protocol(path):
    with pytest.raises(ValueError, match="Protected|protocol"):
        coordinator.apply_overrides({}, [{"path": path, "value_json": "1"}])


def test_overrides_copy_input_and_reject_duplicates_nonfinite_and_scalar_traversal():
    original = {"model": {"dropout": .1}, "training": {"learning_rate": .001}}
    result = coordinator.apply_overrides(original, [{"path": "training.learning_rate", "value_json": "0.0003"}])
    assert original["training"]["learning_rate"] == .001 and result["training"]["learning_rate"] == .0003
    for values in ([{"path": "model.dropout", "value_json": "NaN"}],
                   [{"path": "model.dropout.child", "value_json": "1"}],
                   [{"path": "model.dropout", "value_json": "0"}] * 2):
        with pytest.raises(ValueError):
            coordinator.apply_overrides(original, values)


def test_config_uses_inherited_model_and_rejects_unknown_runtime_flags(campaign, tmp_path):
    path = tmp_path / "team.yaml"
    cfg = deepcopy(campaign["config"])
    path.write_text(json.dumps(cfg))
    loaded = coordinator.load_config(path)
    assert loaded["runtime"]["model"] is None
    assert loaded["runtime"]["web_search"] is True
    cfg["runtime"]["dangerously_bypass"] = True
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="runtime"):
        coordinator.load_config(path)


def test_training_budget_exhaustion_never_launches_suite(campaign):
    install_fake_handoffs(campaign)
    state = advance_to(campaign, "train")
    state["training_seconds"] = state["config"]["limits"]["total_training_seconds"]
    save(campaign, state)
    state = coordinator.advance(campaign["output"], run_agents=True, allow_training=True)
    assert state["status"] == "training_budget_exhausted" and campaign["events"]["suites"] == []


def test_controller_source_change_stops_before_agent(campaign):
    campaign["monkeypatch"].setattr(coordinator, "source_digest", lambda root: "c" * 64)
    with pytest.raises(ValueError, match="Controller source changed"):
        coordinator.advance(campaign["output"], run_agents=True)


def test_interrupted_training_cannot_restore_reserved_compute(campaign):
    install_fake_handoffs(campaign)
    state = advance_to(campaign, "train")
    state["iterations"][0]["training_started"] = True
    state["training_seconds"] = 30
    save(campaign, state)
    state = coordinator.advance(campaign["output"], run_agents=True, allow_training=True)
    assert state["status"] == "needs_attention"
    assert "Interrupted suite" in state["iterations"][0]["error"]
    assert state["training_seconds"] == 30
    assert campaign["events"]["suites"] == []


def test_run_campaign_defaults_to_preparation_and_stops_at_training_gate(campaign):
    request = coordinator.run_campaign(campaign["output"])
    assert request["state"] == "prepared" and request["role"] == "idea"
    install_fake_handoffs(campaign)
    state = coordinator.run_campaign(campaign["output"], run_agents=True)
    assert state["status"] == "awaiting_training_authorization"
    assert [event["role"] for event in campaign["events"]["roles"]] == ["idea", "coder", "executor"]
    assert campaign["events"]["suites"] == []


def test_holdout_audit_is_only_planned_at_last_iteration(campaign):
    install_fake_handoffs(campaign)
    state = advance_to(campaign, "executor")
    assert state["iterations"][0]["suite"]["audit_test"] is False
    state["config"]["limits"]["max_iterations"] = 1
    spec = coordinator._suite_spec(state, state["iterations"][0])
    assert spec["audit_test"] is True


def test_previous_holdout_consumption_blocks_new_audit(campaign):
    install_fake_handoffs(campaign)
    state = advance_to(campaign, "executor")
    state["config"]["limits"]["max_iterations"] = 1
    corpus = Path(state["baseline_config"]["data"]["path"])
    corpus.write_text("A fixture corpus for checking provenance only.")
    data_hash = coordinator.file_digest(corpus)
    KnowledgeBank(state["config"]["knowledge_bank"]).add({
        "kind": "observation", "title": "Prior holdout audit", "summary": "The corpus holdout was already examined.",
        "outcome": "inconclusive", "created_by": "fixture", "scope": {"data_sha256": data_hash, "holdout_consumed": True},
    })
    spec = coordinator._suite_spec(state, state["iterations"][0])
    assert spec["audit_test"] is False


def test_config_file_only_patch_without_effective_overrides_is_noop(campaign):
    replies = responses()
    replies["coder"]["config_overrides"] = []
    install_fake_handoffs(campaign, replies=replies)
    campaign["monkeypatch"].setattr(coordinator, "inspect_workspace", lambda *args: {
        "changes": [{"path": "src/config/experimental.yaml"}], "branch": "codex/fixture/iteration-001"})
    coordinator.advance(campaign["output"], run_agents=True)
    state = coordinator.advance(campaign["output"], run_agents=True)
    assert state["status"] == "needs_attention"
    assert campaign["events"]["local_tests"] == []


def test_handoff_exposes_all_historical_evidence_without_bank_writes(campaign):
    bank = KnowledgeBank(campaign["config"]["knowledge_bank"])
    oldest = bank.add({"kind": "observation", "title": "Old rejected intervention", "summary": "A numerical failure matters for this idea.",
                       "outcome": "failed", "created_by": "fixture", "details": {"lesson": "Avoid this initialization at large width."}})
    for index in range(35):
        bank.add({"kind": "idea", "title": f"Later idea {index}", "summary": "An unrelated recent record.",
                  "outcome": "untried", "created_by": "fixture"})
    request = coordinator.prepare_next(campaign["output"])
    prompt = (Path(request["handoff"]) / "prompt.md").read_text()
    encoded = prompt.split("<evidence_json>\n", 1)[1].split("\n</evidence_json>", 1)[0]
    context = json.loads(encoded)
    assert len(context["knowledge"]) == 30
    assert oldest not in {record["id"] for record in context["knowledge"]}
    assert "knowledge_bank" not in context
    snapshot = context["knowledge_snapshot"]
    path = Path(snapshot["path"])
    assert path.is_absolute() and snapshot["includes_all_records"] is True
    assert coordinator.file_digest(path) == snapshot["sha256"]
    assert path.stat().st_mode & 0o222 == 0
    exported = path.read_text()
    assert oldest in exported and "Avoid this initialization at large width." in exported
    assert exported.count('"created_by": "fixture"') == 36


def test_each_handoff_retains_its_snapshot_when_bank_grows(campaign):
    state = read(campaign)
    first = coordinator._context(state, {"id": "iteration-001", "stage": "idea"})["knowledge_snapshot"]
    before = Path(first["path"]).read_bytes()
    new_id = KnowledgeBank(campaign["config"]["knowledge_bank"]).add({
        "kind": "observation", "title": "New finding from another campaign", "summary": "Appended after the first handoff.",
        "outcome": "inconclusive", "created_by": "fixture"})
    second = coordinator._context(state, {"id": "iteration-001", "stage": "idea"})["knowledge_snapshot"]
    assert first["path"] != second["path"]
    assert Path(first["path"]).read_bytes() == before
    assert new_id not in before.decode()
    assert new_id in Path(second["path"]).read_text()
    assert coordinator._context(state, {"id": "iteration-001", "stage": "idea"})["knowledge_snapshot"] == second


def test_explicit_optimizer_baseline_is_frozen_without_changing_protocol(campaign, tmp_path):
    config = deepcopy(campaign['config'])
    config['baseline_options'] = {'optimizer': 'muon', 'decay_mode': 'independent', 'parameterization': 'mup'}
    state = coordinator.init_campaign(config, tmp_path / 'optimizer-campaign')
    baseline = state['baseline_config']
    assert baseline['training']['optimizer'] == 'muon'
    assert baseline['training']['decay_mode'] == 'independent'
    assert baseline['model']['parameterization'] == 'mup'
    assert baseline['model']['base_width'] == 64
    assert baseline['data'] == campaign['base_config']['data']
    assert baseline['training']['max_iters'] == campaign['base_config']['training']['max_iters']
    assert state['plan'] == campaign['plan']
    assert 'optimizer' not in campaign['state']['baseline_config']['training']
    assert 'parameterization' not in campaign['state']['baseline_config']['model']
    assert campaign['events']['suites'] == []
    saved = json.loads((tmp_path / 'optimizer-campaign/campaign.json').read_text())
    assert saved['baseline_config'] == baseline
    candidate = coordinator.apply_overrides(baseline, [{'path': 'training.weight_decay', 'value_json': '0'}])
    assert candidate['training']['optimizer'] == 'muon'
    assert candidate['model']['parameterization'] == 'mup'
    assert baseline['training']['weight_decay'] == .1


@pytest.mark.parametrize('options', [None, {}, {'optimizer': 'muon'},
    {'optimizer': 'sgd', 'decay_mode': 'independent', 'parameterization': 'mup'},
    {'optimizer': 'muon', 'decay_mode': 'bad', 'parameterization': 'mup'},
    {'optimizer': 'muon', 'decay_mode': 'independent', 'parameterization': 'bad'},
    {'optimizer': 'muon', 'decay_mode': 'independent', 'parameterization': 'mup', 'max_iters': 1}])
def test_invalid_baseline_options_rejected_before_workspaces(campaign, tmp_path, options):
    config = deepcopy(campaign['config']); config['baseline_options'] = options
    path = tmp_path / 'options.yaml'; path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='baseline_options'):
        coordinator.load_config(path)
    count = len(campaign['events']['workspaces'])
    with pytest.raises(ValueError, match='baseline_options'):
        coordinator.init_campaign(config, tmp_path / 'invalid-options')
    assert len(campaign['events']['workspaces']) == count


def test_independent_baseline_rejects_invalid_decay_before_workspaces(campaign, tmp_path):
    config = deepcopy(campaign['config'])
    config['baseline_options'] = {'optimizer': 'muon', 'decay_mode': 'independent', 'parameterization': 'mup'}
    campaign['base_config']['training']['weight_decay'] = 1
    count = len(campaign['events']['workspaces'])
    with pytest.raises(ValueError, match='Independent baseline decay'):
        coordinator.init_campaign(config, tmp_path / 'invalid-decay')
    assert len(campaign['events']['workspaces']) == count
