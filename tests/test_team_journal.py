"""Evidence provenance tests use temporary files and fabricated measurements."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.research.team import journal
from src.research.team.knowledge import KnowledgeBank
from src.research.team.predictions import METRICS, compare_prediction


def entry(kind, **updates):
    return {"kind": kind, "title": "Synthetic research record", "summary": "Fixture only, no experiment executed.",
            "outcome": "untried", "created_by": "unit-test", **updates}


@pytest.fixture
def research(tmp_path):
    bank = KnowledgeBank(tmp_path / "knowledge")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Synthetic corpus bytes for provenance checks only.\n")
    state = {"key": "campaign-test", "baseline": {"base_commit": "a" * 40},
             "baseline_config": {"data": {"path": str(corpus)}, "model": {"dim": 64}, "training": {"learning_rate": .001}},
             "plan": {"widths": [64, 128], "seeds": [1, 2], "model": {"head_dim": 16}}}
    prediction = {"mechanism": "Test a speculative optimization mechanism.", "assumptions": ["Identical token budgets."],
        "estimates": [{"target": f"width-{width}", "metric": metric, "expected": 0, "lower": -1, "upper": 1,
                       "confidence": "low", "rationale": "Synthetic expected range."}
                      for width in state["plan"]["widths"] for metric in METRICS]}
    idea_id = bank.add(entry("idea"))
    iteration = {"id": "iteration-001", "stage": "idea", "idea_id": idea_id,
                 "idea": {"title": "Synthetic model intervention", "prediction": prediction}}
    directory = tmp_path / "iteration-001"
    directory.mkdir()
    return bank, state, iteration, directory


def populate_results(research, transfer_outcome="mixed", missing_resources=False):
    bank, state, iteration, directory = research
    journal.register_prediction(bank, state, iteration, directory)
    iteration["candidate_config"] = {"model": {"dim": 64}, "training": {"learning_rate": .0003}}
    iteration["execution_id"] = bank.add(entry("execution", outcome="inconclusive", related_ids=[iteration["prediction_id"]]))
    report = {"local_outcome": "supported", "transfer_outcome": transfer_outcome, "audit_outcome": "unavailable",
              "contract": {"data": journal.file_digest(Path(state["baseline_config"]["data"]["path"]))}, "holdout_consumed": False, "stop_reason": "completed",
              "evidence_integrity": {"status": "checked"},
              "records": [{"trial_id": "baseline-64-seed-1", "stage": "confirm", "arm": "baseline", "target": "width-64", "seed": 1, "status": "completed"},
                          {"trial_id": "candidate-128-seed-2", "stage": "transfer", "arm": "candidate", "target": "width-128", "seed": 2, "status": "timeout"}],
              "performance": [{"id": f"{row['target']}:{row['metric']}", "mean_delta": .1, "complete": True, "missing_seeds": []}
                              for row in iteration["idea"]["prediction"]["estimates"]]}
    if missing_resources:
        for row in report["performance"]:
            if not row["id"].endswith("validation_ce_delta"):
                row.update(mean_delta=None, complete=False, missing_seeds=[2])
    iteration["report"] = report
    iteration["prediction_check"] = compare_prediction(iteration["idea"]["prediction"], report["performance"])
    journal.atomic_json(directory / "evidence.json", report)
    journal.atomic_json(directory / "prediction-check.json", iteration["prediction_check"])
    journal.record_results(bank, state, iteration, directory)
    return research


def lesson_result():
    return {"learnings": [{"finding": "The fixture has mixed transfer evidence.",
            "explanation": "An interpretation to test, not causal proof.",
            "limitation": "Only the registered tiny-model scope is represented.",
            "next_action": "Design a new paired test of the explanation.",
            "evidence_ids": ["width-64:validation_ce_delta", "candidate-128-seed-2"], "confidence": "low"}]}


def test_prediction_is_frozen_before_execution_and_linked_to_idea(research):
    bank, state, iteration, directory = research
    with pytest.raises(ValueError, match="registered pre-execution prediction"):
        journal.verify_prediction(state, iteration)
    journal.register_prediction(bank, state, iteration, directory)
    path = directory / "prediction.json"
    saved = json.loads(path.read_text())
    assert saved["registered_at"].endswith("Z")
    assert path.stat().st_mode & 0o222 == 0
    assert journal.verify_prediction(state, iteration) == iteration["idea"]["prediction"]
    record = bank.get(iteration["prediction_id"])
    assert record["kind"] == "prediction" and record["outcome"] == "untried"
    assert record["scope"]["registered_before_execution"] is True
    assert record["scope"]["data_sha256"] == journal.file_digest(Path(state["baseline_config"]["data"]["path"]))
    assert saved["plan_sha256"] == journal.digest(state["plan"])
    assert record["related_ids"] == [iteration["idea_id"]]
    assert record["artifacts"] == [iteration["prediction_artifact"]]
    assert not (directory / "evidence.json").exists()
    with pytest.raises(ValueError, match="already registered"):
        journal.register_prediction(bank, state, iteration, directory)
    assert bank.get(iteration["prediction_id"])["details"] == saved


@pytest.mark.parametrize("already_run", [{"training_started": True}, {"report": {"transfer_outcome": "supported"}}, {"stage": "coder"}])
def test_post_execution_prediction_registration_is_rejected(research, already_run):
    bank, state, iteration, directory = research
    iteration.update(already_run)
    with pytest.raises(ValueError, match="before|execution|started"):
        journal.register_prediction(bank, state, iteration, directory)
    assert bank.search(kind="prediction") == []
    assert not (directory / "prediction.json").exists()


def test_prediction_artifact_tampering_blocks_verification(research):
    bank, state, iteration, directory = research
    journal.register_prediction(bank, state, iteration, directory)
    path = directory / "prediction.json"
    path.chmod(0o644)
    path.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        journal.verify_prediction(state, iteration)


@pytest.mark.parametrize("change", ["idea", "baseline", "targets", "seeds", "data", "plan"])
def test_prediction_rejects_later_changes_to_registered_experiment(research, change):
    bank, state, iteration, directory = research
    journal.register_prediction(bank, state, iteration, directory)
    if change == "idea":
        iteration["idea"]["prediction"]["mechanism"] = "A different explanation after seeing results."
    elif change == "baseline":
        state["baseline_config"]["training"]["learning_rate"] = .01
    elif change == "targets":
        state["plan"]["model"]["head_dim"] = 8
    elif change == "seeds":
        state["plan"]["seeds"] = [3, 4]
    elif change == "data":
        Path(state["baseline_config"]["data"]["path"]).write_text("Different corpus at the same path.\n")
    else:
        state["plan"]["min_improvement"] = .5
    with pytest.raises(ValueError, match="no longer matches"):
        journal.verify_prediction(state, iteration)


def test_prediction_symlink_substitution_is_rejected_even_with_same_contents(research, tmp_path):
    bank, state, iteration, directory = research
    journal.register_prediction(bank, state, iteration, directory)
    path = directory / "prediction.json"
    copied = tmp_path / "copied.json"
    copied.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(copied)
    with pytest.raises(ValueError, match="changed"):
        journal.verify_prediction(state, iteration)


def test_trials_and_result_preserve_failures_scope_and_prediction_lineage(research):
    bank, state, iteration, directory = populate_results(research)
    trial_records = [bank.get(identity) for identity in iteration["trial_ids"]]
    assert [record["outcome"] for record in trial_records] == ["reported", "failed"]
    assert trial_records[1]["details"]["status"] == "timeout"
    for trial in trial_records:
        assert trial["related_ids"] == sorted([iteration["execution_id"], iteration["prediction_id"]])
        assert trial["scope"]["generalizes_beyond_tested_models"] is False
        assert trial["scope"]["data_sha256"] == journal.file_digest(Path(state["baseline_config"]["data"]["path"]))
    result = bank.get(iteration["result_id"])
    assert result["kind"] == "result" and result["outcome"] == "mixed"
    assert set(result["related_ids"]) == {iteration["execution_id"], iteration["prediction_id"], *iteration["trial_ids"]}
    assert result["details"]["prediction_check"] == iteration["prediction_check"]
    assert result["details"]["performance"] == iteration["report"]["performance"]
    before = deepcopy(iteration["trial_ids"]), iteration["result_id"]
    journal.record_results(bank, state, iteration, directory)
    assert (iteration["trial_ids"], iteration["result_id"]) == before


def test_unknown_empirical_result_outcome_is_inconclusive(research):
    bank, state, iteration, directory = populate_results(research, transfer_outcome="smoke_only")
    assert bank.get(iteration["result_id"])["outcome"] == "inconclusive"


def test_learnings_link_prediction_result_evaluation_and_exact_evidence(research):
    bank, state, iteration, directory = populate_results(research)
    iteration["evaluation_id"] = bank.add(entry("evaluation", outcome="inconclusive", related_ids=[iteration["result_id"]]))
    result = lesson_result()
    response = directory / "evaluator-response.json"
    journal.atomic_json(response, result)
    journal.record_learnings(bank, state, iteration, result, directory, response, "inconclusive")
    learning = bank.get(iteration["learning_ids"][0])
    assert learning["kind"] == "learning" and learning["outcome"] == "inconclusive"
    assert learning["related_ids"] == sorted([iteration["prediction_id"], iteration["result_id"], iteration["evaluation_id"]])
    assert learning["details"]["evidence_ids"] == result["learnings"][0]["evidence_ids"]
    assert learning["details"]["interpretation_not_proof_of_mechanism"] is True
    assert learning["scope"]["generalizes_beyond_tested_models"] is False
    for ref in learning["artifacts"]:
        assert journal.file_digest(Path(ref["path"])) == ref["sha256"]
    assert json.loads((directory / "learnings.json").read_text())["prediction_id"] == iteration["prediction_id"]
    assert (directory / "learnings.json").stat().st_mode & 0o222 == 0
    before = (directory / "learnings.json").read_bytes()
    with pytest.raises(ValueError, match="already recorded learnings"):
        journal.record_learnings(bank, state, iteration, result, directory, response, "inconclusive")
    assert (directory / "learnings.json").read_bytes() == before
    assert len(bank.search(kind="learning")) == 1


def test_supported_quality_does_not_upgrade_lesson_or_missing_resource_measurements(research):
    bank, state, iteration, directory = populate_results(research, transfer_outcome="supported", missing_resources=True)
    iteration["evaluation_id"] = bank.add(entry("evaluation", outcome="reported", related_ids=[iteration["result_id"]]))
    result = lesson_result()
    lesson = result["learnings"][0]
    lesson.update(finding="Resource improvement remains unknown.",
                  explanation="Quality support does not establish the missing throughput effect.",
                  evidence_ids=["width-128:training_throughput_pct"])
    response = directory / "evaluator-response.json"
    journal.atomic_json(response, result)
    journal.record_learnings(bank, state, iteration, result, directory, response, "supported")
    record = bank.get(iteration["learning_ids"][0])
    assert record["outcome"] == "reported"
    assert record["scope"]["transfer_outcome"] == "supported"
    document = json.loads((directory / "learnings.json").read_text())
    assert document["experiment_outcome"] == "supported" and document["outcome"] == "reported"
    check = next(row for row in document["prediction_check"]["comparisons"] if row["id"] == "width-128:training_throughput_pct")
    assert check["verdict"] == "unavailable" and check["observed_mean_delta"] is None
    assert record["details"]["interpretation_not_proof_of_mechanism"] is True


@pytest.mark.parametrize("references", [["invented-metric"], [], ["width-64:validation_ce_delta", "invented-trial"]])
def test_invented_or_empty_evidence_rejected_before_writing_learning(research, references):
    bank, state, iteration, directory = populate_results(research)
    result = lesson_result()
    result["learnings"][0]["evidence_ids"] = references
    response = directory / "evaluator-response.json"
    journal.atomic_json(response, result)
    with pytest.raises(ValueError, match="missing performance or trial evidence"):
        journal.record_learnings(bank, state, iteration, result, directory, response, "inconclusive")
    assert bank.search(kind="learning") == []
    assert not (directory / "learnings.json").exists()


def test_operational_failure_learning_does_not_claim_model_regression(research):
    bank, state, iteration, directory = research
    error_path = directory / "coder-failure.json"
    journal.atomic_json(error_path, {"phase": "coder", "error": "A local test failed."})
    failure_id = bank.add(entry("observation", outcome="failed", artifacts=[journal.artifact(error_path)]))
    iteration.update(failed_phase="coder", error="A local test failed.", failure_id=failure_id)
    journal.record_failure_learning(bank, state, iteration, error_path)
    learning = bank.get(iteration["failure_learning_id"])
    assert "learning_ids" not in iteration
    assert learning["outcome"] == "inconclusive"
    assert learning["scope"]["not_a_negative_model_result"] is True
    assert learning["related_ids"] == [failure_id]
    assert learning["details"]["evidence_ids"] == [failure_id]
    assert "does not establish a model-quality or efficiency effect" in learning["details"]["explanation"]
    assert bank.search(kind="result") == [] and bank.search(kind="trial") == []


def test_partial_learning_import_preserves_ids_and_operational_failure_is_separate(research, monkeypatch):
    bank, state, iteration, directory = populate_results(research)
    iteration["evaluation_id"] = bank.add(entry("evaluation", outcome="reported", related_ids=[iteration["result_id"]]))
    result = lesson_result()
    result["learnings"].append({**deepcopy(result["learnings"][0]), "finding": "A second scoped lesson."})
    response = directory / "evaluator-response.json"
    journal.atomic_json(response, result)
    original_add = bank.add
    attempted = []

    def fail_second_learning(value):
        if value["kind"] == "learning":
            attempted.append(value)
            if len(attempted) == 2:
                raise RuntimeError("Synthetic interruption during the second learning import")
        return original_add(value)

    monkeypatch.setattr(bank, "add", fail_second_learning)
    with pytest.raises(RuntimeError, match="Synthetic interruption"):
        journal.record_learnings(bank, state, iteration, result, directory, response, "mixed")
    accepted = list(iteration["learning_ids"])
    assert len(accepted) == 1
    assert bank.get(accepted[0])["summary"] == result["learnings"][0]["finding"]
    monkeypatch.setattr(bank, "add", original_add)
    error_path = directory / "evaluator-failure.json"
    journal.atomic_json(error_path, {"error": "Synthetic interruption"})
    iteration.update(failed_phase="evaluator", error="Synthetic interruption",
        failure_id=bank.add(entry("observation", outcome="failed", artifacts=[journal.artifact(error_path)])))
    journal.record_failure_learning(bank, state, iteration, error_path)
    assert iteration["learning_ids"] == accepted
    assert iteration["failure_learning_id"] not in accepted
    assert bank.get(iteration["failure_learning_id"])["outcome"] == "inconclusive"
