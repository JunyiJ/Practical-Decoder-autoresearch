"""Regression coverage for real knowledge-bank citations; no agents or training."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.research.team import coordinator, journal
from src.research.team.knowledge import KnowledgeBank
from src.research.team.recovery import recover_evaluation
from test_team_coordinator import campaign, install_fake_handoffs, advance_to, read


def test_current_bank_ids_are_valid_but_unrelated_ids_are_not(campaign):
    replies = install_fake_handoffs(campaign, report={"local_outcome": "supported", "transfer_outcome": "supported",
        "records": [{"trial_id": "local_width-16_s1_baseline", "stage": "local", "arm": "baseline",
                     "target": "width-16", "seed": 1, "status": "failed", "error": "fixture"}]})
    state = advance_to(campaign, "evaluator")
    iteration = state["iterations"][0]
    result = replies["evaluator"]
    result["learnings"][0]["evidence_ids"] = [iteration["result_id"], iteration["trial_ids"][0]]
    journal.validate_learning_evidence(iteration, result)
    context = coordinator._context(state, iteration)
    assert iteration["result_id"] in context["evidence_catalog"]["result_record_ids"]
    assert iteration["trial_ids"][0] in context["evidence_catalog"]["trial_record_ids"]
    for invalid in (iteration["idea_id"], iteration["prediction_id"], "kb_" + "0" * 64):
        bad = deepcopy(result)
        bad["learnings"][0]["evidence_ids"].append(invalid)
        with pytest.raises(journal.LearningEvidenceError, match=invalid):
            journal.validate_learning_evidence(iteration, bad)
    state = coordinator.advance(campaign["output"], run_agents=True)
    assert state["iterations"][0]["stage"] == "completed"
    assert len(campaign["events"]["suites"]) == 1


def test_corrective_retry_does_not_repeat_training_or_erase_rejection(campaign):
    replies = install_fake_handoffs(campaign)
    advance_to(campaign, "evaluator")
    replies["evaluator"]["learnings"][0]["evidence_ids"] = ["invented-id"]
    state = coordinator.advance(campaign["output"], run_agents=True)
    iteration = state["iterations"][0]
    assert state["status"] == "ready_for_evaluator"
    rejection = iteration["evaluation_rejections"][0]
    original_bytes = Path(rejection["path"]).read_bytes()
    assert "invented-id" in iteration["evaluation_feedback"]
    assert KnowledgeBank(state["config"]["knowledge_bank"]).search(kind="evaluation") == []
    spent = state["agent_seconds"]
    replies["evaluator"]["learnings"][0]["evidence_ids"] = [iteration["result_id"]]
    state = coordinator.advance(campaign["output"], run_agents=True)
    assert state["iterations"][0]["stage"] == "completed"
    assert state["agent_seconds"] > spent
    assert len(state["agent_calls"]) == 5
    assert len(campaign["events"]["suites"]) == 1
    assert Path(rejection["path"]).read_bytes() == original_bytes
    assert "invented-id" in campaign["events"]["roles"][-1]["prompt"]


@pytest.fixture
def rejected(campaign):
    replies = install_fake_handoffs(campaign, report={"local_outcome": "supported", "transfer_outcome": "supported", "records": []})
    state = advance_to(campaign, "evaluator")
    replies["evaluator"]["learnings"][0]["evidence_ids"] = [state["iterations"][0]["result_id"]]
    def old_validator(*args):
        raise ValueError("Learning cites missing performance or trial evidence IDs")
    campaign["monkeypatch"].setattr(coordinator, "validate_learning_evidence", old_validator)
    state = coordinator.advance(campaign["output"], run_agents=True)
    call_dir = Path(state["agent_calls"][-1]["path"])
    response = call_dir / "response.json"
    journal.atomic_json(call_dir / "receipt.json", {"role": "evaluator", "state": "completed",
        "response": json.loads(response.read_text()), "response_sha256": journal.file_digest(response)})
    return campaign


def test_recover_saved_response_is_idempotent_and_preserves_originals(rejected):
    output = rejected["output"]
    originals = {str(p): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    result = recover_evaluation(output)
    assert result["status"] == "evaluation_recovered"
    assert len(result["learning_ids"]) == 1
    assert all(Path(path).read_bytes() == content for path, content in originals.items())
    assert read(rejected)["status"] == "needs_attention"
    assert len(rejected["events"]["suites"]) == 1
    bank = KnowledgeBank(rejected["config"]["knowledge_bank"])
    learning = bank.get(result["learning_ids"][0])
    assert learning["details"]["evidence_ids"] == [result["result_id"]]
    assert result["original_failure_id"] in bank.get(result["evaluation_id"])["related_ids"]
    count = len(bank.search(limit=10000))
    assert recover_evaluation(output) == result
    assert len(bank.search(limit=10000)) == count


@pytest.mark.parametrize("tamper", ["response", "evidence", "prediction-check", "trial-inventory", "failure-phase"])
def test_recovery_rejects_changed_or_ineligible_evidence(rejected, tamper):
    output = rejected["output"]
    state = read(rejected)
    iteration = state["iterations"][0]
    if tamper == "response":
        Path(state["agent_calls"][-1]["path"], "response.json").write_text("{}")
    elif tamper in {"evidence", "prediction-check"}:
        (output / iteration["id"] / (tamper + ".json")).write_text("{}")
    else:
        if tamper == "trial-inventory":
            iteration["trial_ids"] = ["kb_" + "0" * 64]
        else:
            iteration["failed_phase"] = "train"
        journal.atomic_json(output / "campaign.json", state)
    with pytest.raises(ValueError):
        recover_evaluation(output)
    assert not (output / iteration["id"] / "recovered-evaluation").exists()
