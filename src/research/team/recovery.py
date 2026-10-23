"""Recover a citation-rejected evaluation without rewriting a frozen campaign.

No agent or training process is launched. The original response, failed state,
budgets, predictions and numerical results remain unchanged. Recovery appends
interpretations to the shared knowledge bank for use by a new campaign.
"""
from copy import deepcopy
import json
from pathlib import Path

from ..runner import atomic_json, session_lock
from .journal import (artifact, record_learnings, validate_learning_evidence,
                      verify_prediction, utc_now)
from .knowledge import KnowledgeBank
from .roles import validate_response


def _verify_artifact(reference):
    path = Path(reference["path"])
    if path.is_symlink() or artifact(path) != reference:
        raise ValueError(f"Recovery artifact changed: {path}")


def recover_evaluation(output: Path) -> dict:
    output = output.resolve()
    with session_lock(output):
        state = json.loads((output / "campaign.json").read_text())
        if not state.get("iterations"):
            raise ValueError("No failed evaluation to recover")
        original = state["iterations"][-1]
        if (state["status"] != "needs_attention" or original.get("stage") != "failed"
                or original.get("failed_phase") != "evaluator"
                or not original.get("error", "").startswith("Learning cites missing performance or trial evidence IDs")):
            raise ValueError("Only a citation-rejected evaluator response is eligible for recovery")
        if any(call["status"] == "running" for call in state["agent_calls"]):
            raise ValueError("An interrupted agent must be inspected before recovery")
        directory = output / original["id"]
        evidence = directory / "evidence.json"
        bank = KnowledgeBank(state["config"]["knowledge_bank"])
        verify_prediction(state, original)
        if json.loads(evidence.read_text()) != original["report"]:
            raise ValueError("Saved report differs from campaign evidence")
        result_record = bank.get(original["result_id"])
        if (result_record["kind"] != "result"
                or result_record["scope"].get("campaign") != state["key"]
                or result_record["scope"].get("iteration") != original["id"]
                or result_record["details"]["performance"] != original["report"]["performance"]
                or artifact(evidence) not in result_record["artifacts"]):
            raise ValueError("Result record does not bind this iteration's evidence")
        for reference in result_record["artifacts"]:
            _verify_artifact(reference)
        rows = original["report"]["records"]
        if len(rows) != len(original["trial_ids"]):
            raise ValueError("Trial record inventory is incomplete")
        for identity, row in zip(original["trial_ids"], rows):
            record = bank.get(identity)
            if (record["kind"] != "trial" or record["details"] != row
                    or record["scope"].get("campaign") != state["key"]
                    or record["scope"].get("iteration") != original["id"]
                    or artifact(evidence) not in record["artifacts"]):
                raise ValueError("Trial record does not match this iteration")
        calls = [call for call in state["agent_calls"]
                 if call["role"] == "evaluator" and call["iteration"] == original["id"]]
        if not calls or calls[-1]["status"] != "completed":
            raise ValueError("No completed evaluator response to recover")
        call_dir = Path(calls[-1]["path"])
        response_path = call_dir / "response.json"
        receipt = json.loads((call_dir / "receipt.json").read_text())
        if receipt.get("role") != "evaluator" or receipt.get("state") != "completed":
            raise ValueError("Missing completed evaluator receipt")
        _verify_artifact({"path": str(response_path.resolve()), "sha256": receipt["response_sha256"]})
        response = validate_response("evaluator", json.loads(response_path.read_text()))
        if response != receipt["response"]:
            raise ValueError("Evaluator receipt and response differ")
        validate_learning_evidence(original, response)
        recovered = directory / "recovered-evaluation"
        if (recovered / "recovery.json").exists():
            result = json.loads((recovered / "recovery.json").read_text())
            for reference in result["artifacts"]:
                _verify_artifact(reference)
            return result
        recovered.mkdir(exist_ok=False)  # Do not silently overwrite a partial recovery.
        iteration = deepcopy(original)
        outcome = iteration["report"]["transfer_outcome"]
        if outcome not in {"supported", "contradicted", "mixed", "inconclusive", "failed"} or response["validity_concerns"]:
            outcome = "inconclusive"
        iteration["evaluation_id"] = bank.add({
            "kind": "evaluation", "title": iteration["idea"]["title"], "summary": response["summary"],
            "outcome": outcome, "created_by": "evaluation-recovery",
            "scope": {"campaign": state["key"], "iteration": iteration["id"],
                      "generalizes_beyond_tested_models": False, "recovered_after_citation_rejection": True},
            "related_ids": [iteration["result_id"], iteration["prediction_id"], iteration["failure_id"]],
            "artifacts": [artifact(evidence), artifact(response_path), artifact(directory / "evaluator-failure.json")],
            "details": response})
        record_learnings(bank, state, iteration, response, recovered, response_path, outcome, evidence_path=evidence)
        result = {"status": "evaluation_recovered", "created_at": utc_now(),
                  "campaign": str(output), "iteration": iteration["id"],
                  "original_failure_id": original["failure_id"], "result_id": iteration["result_id"],
                  "evaluation_id": iteration["evaluation_id"], "learning_ids": iteration["learning_ids"],
                  "artifacts": [artifact(evidence), artifact(response_path), artifact(recovered / "learnings.json")],
                  "next_step": "Start a new campaign using the same knowledge bank; the original failed state is preserved."}
        atomic_json(recovered / "recovery.json", result)
        return result
