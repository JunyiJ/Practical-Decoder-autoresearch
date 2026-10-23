"""Persist the prediction -> trial -> result -> learning chain in the shared bank."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..protocol import digest, file_digest
from ..runner import atomic_json
from .predictions import METRICS, validate_prediction


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_digest(path)}


def _targets(state):
    plan = state["plan"]
    return [{"name": f"width-{width}", "width": width, "n_heads": width // plan["model"]["head_dim"]}
            for width in plan["widths"]]


def _prediction_contract(state, iteration):
    return {"version": 1, "campaign": state["key"], "iteration": iteration["id"],
            "idea_id": iteration["idea_id"], "baseline_commit": state["baseline"]["base_commit"],
            "baseline_config_sha256": digest(state["baseline_config"]),
            "data_sha256": file_digest(Path(state["baseline_config"]["data"]["path"])),
            "plan_sha256": digest(state["plan"]),
            "targets": _targets(state), "seeds": state["plan"]["seeds"],
            "prediction": validate_prediction(iteration["idea"]["prediction"], _targets(state)),
            "metrics": METRICS}


def register_prediction(bank, state, iteration, directory: Path):
    """Save before coding/dispatch, so observed outcomes cannot rewrite forecasts."""
    if iteration.get("training_started") or "report" in iteration or iteration.get("stage") not in {None, "idea"}:
        raise ValueError("A prediction must be registered before coding or execution begins")
    document = {**_prediction_contract(state, iteration), "registered_at": utc_now()}
    path = directory / "prediction.json"
    if path.exists():
        raise ValueError("Refusing to replace an already registered prediction")
    atomic_json(path, document)
    path.chmod(0o444)
    iteration["prediction_artifact"] = artifact(path)
    iteration["prediction_id"] = bank.add({"kind": "prediction", "title": iteration["idea"]["title"],
        "summary": document["prediction"]["mechanism"], "outcome": "untried", "created_by": "idea-agent",
        "scope": {"campaign": state["key"], "iteration": iteration["id"], "targets": document["targets"],
                  "seeds": document["seeds"], "data_sha256": document["data_sha256"], "registered_before_execution": True},
        "related_ids": [iteration["idea_id"]], "artifacts": [iteration["prediction_artifact"]], "details": document})


def verify_prediction(state, iteration) -> dict:
    ref = iteration.get("prediction_artifact")
    if not ref or not iteration.get("prediction_id"):
        raise ValueError("A registered pre-execution prediction is required; start a new iteration")
    path = Path(ref["path"])
    if path.is_symlink() or file_digest(path) != ref["sha256"]:
        raise ValueError("Registered prediction changed; execution and interpretation are blocked")
    document = json.loads(path.read_text())
    contract = _prediction_contract(state, iteration)
    if any(document.get(key) != value for key, value in contract.items()):
        raise ValueError("Prediction no longer matches the idea or registered baseline/targets")
    return document["prediction"]


def evidence_scope(state, iteration) -> dict:
    report = iteration.get("report", {})
    suite = iteration.get("suite", {})
    return {"campaign": state["key"], "iteration": iteration["id"],
            "targets": suite.get("targets", _targets(state)), "seeds": state["plan"]["seeds"],
            "data_sha256": report.get("contract", {}).get("data"),
            "contract": report.get("contract", {}),
            "baseline_config_sha256": digest(state["baseline_config"]),
            "candidate_config_sha256": digest(iteration["candidate_config"]) if "candidate_config" in iteration else None,
            "local_outcome": report.get("local_outcome"), "transfer_outcome": report.get("transfer_outcome"),
            "holdout_consumed": report.get("holdout_consumed", False),
            "generalizes_beyond_tested_models": False}


def record_results(bank, state, iteration, directory: Path):
    report = iteration["report"]
    evidence = artifact(directory / "evidence.json")
    scope = evidence_scope(state, iteration)
    trials = []
    for row in report.get("records", []):
        trials.append(bank.add({"kind": "trial", "title": f"{iteration['id']}: {row['trial_id']}",
            "summary": f"{row['stage']} {row['arm']} at {row['target']}, seed {row['seed']}: {row['status']}.",
            "outcome": "reported" if row["status"] == "completed" else "failed",
            "created_by": "bounded-executor", "scope": {**scope, "trial_id": row["trial_id"],
                "target": row["target"], "seed": row["seed"], "arm": row["arm"], "stage": row["stage"]},
            "related_ids": [iteration["execution_id"], iteration["prediction_id"]],
            "artifacts": [evidence], "details": row}))
    iteration["trial_ids"] = trials
    outcome = report["transfer_outcome"]
    if outcome not in {"supported", "contradicted", "mixed", "failed"}:
        outcome = "inconclusive"
    iteration["result_id"] = bank.add({"kind": "result", "title": iteration["idea"]["title"],
        "summary": f"Local outcome: {report['local_outcome']}; transfer: {report['transfer_outcome']}; audit: {report.get('audit_outcome', 'unavailable')}.",
        "outcome": outcome, "created_by": "deterministic-evaluator", "scope": scope,
        "related_ids": [iteration["execution_id"], iteration["prediction_id"], *trials],
        "artifacts": [evidence, artifact(directory / "prediction-check.json")],
        "details": {"performance": report.get("performance", []), "prediction_check": iteration["prediction_check"],
                    "stop_reason": report.get("stop_reason"), "evidence_integrity": report.get("evidence_integrity")}})


class LearningEvidenceError(ValueError):
    """A completed evaluator response used citations outside this iteration."""


def learning_evidence_catalog(iteration):
    """Include both local identifiers and the records produced by record_results.

    Only this iteration's recorded result/trial IDs are allowed. Other bank
    entries (including literature, predictions and older results) are not aliases.
    """
    report = iteration.get("report", {})
    return {"metric_ids": [row["id"] for row in report.get("performance", [])],
            "trial_ids": [row["trial_id"] for row in report.get("records", [])],
            "result_record_ids": [iteration["result_id"]] if iteration.get("result_id") else [],
            "trial_record_ids": list(iteration.get("trial_ids", []))}


def validate_learning_evidence(iteration, result):
    available = {identity for values in learning_evidence_catalog(iteration).values() for identity in values}
    for lesson in result["learnings"]:
        if not lesson["evidence_ids"] or set(lesson["evidence_ids"]) - available:
            invalid = sorted(set(lesson["evidence_ids"]) - available)
            raise LearningEvidenceError(f"Learning cites missing performance or trial evidence IDs: {invalid}. "
                                        f"Choose at least one ID from the current evidence_catalog: {sorted(available)}")


def record_learnings(bank, state, iteration, result, directory: Path, response_path: Path, outcome: str,
                    *, evidence_path: Path | None = None):
    validate_learning_evidence(iteration, result)
    # A supported intervention does not establish every explanation, resource
    # claim, or proposed mechanism in an agent's narrative.
    learning_status = "inconclusive" if outcome == "inconclusive" else "reported"
    document = {"version": 1, "created_at": utc_now(), "prediction_id": iteration["prediction_id"],
                "result_id": iteration["result_id"], "outcome": learning_status, "experiment_outcome": outcome,
                "scope": evidence_scope(state, iteration),
                "prediction_check": iteration["prediction_check"], "learnings": result["learnings"]}
    path = directory / "learnings.json"
    if path.exists():
        raise ValueError("Refusing to replace already recorded learnings")
    atomic_json(path, document)
    path.chmod(0o444)
    ids = iteration.setdefault("learning_ids", [])
    for lesson in result["learnings"]:
        ids.append(bank.add({"kind": "learning", "title": iteration["idea"]["title"],
            "summary": lesson["finding"], "outcome": learning_status, "created_by": "evaluator-agent",
            "scope": document["scope"],
            "related_ids": [iteration["prediction_id"], iteration["result_id"], iteration["evaluation_id"]],
            "artifacts": [artifact(path), artifact(response_path), artifact(evidence_path or directory / "evidence.json")],
            "details": {**lesson, "interpretation_not_proof_of_mechanism": True}}))


def record_failure_learning(bank, state, iteration, error_path: Path):
    """Retain an operational lesson; a failed implementation is no model result."""
    lesson = {"finding": f"{iteration['failed_phase']} failed: {iteration['error']}",
              "explanation": "This interruption does not establish a model-quality or efficiency effect.",
              "limitation": "No new empirical claim follows from the failure itself.",
              "next_action": f"Inspect the {iteration['failed_phase']} failure artifact before a new attempt.",
              "evidence_ids": [iteration["failure_id"]], "confidence": "high"}
    identity = bank.add({"kind": "learning", "title": f"Operational lesson: {iteration['id']}",
        "summary": lesson["finding"], "outcome": "inconclusive", "created_by": "coordinator",
        "scope": {"campaign": state["key"], "iteration": iteration["id"], "not_a_negative_model_result": True},
        "related_ids": [iteration["failure_id"]], "artifacts": [artifact(error_path)], "details": lesson})
    iteration["failure_learning_id"] = identity
    iteration["failure_learning"] = lesson
