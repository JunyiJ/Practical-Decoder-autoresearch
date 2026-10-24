"""Persistent role handoffs; only the explicit training phase may train models."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

from ..protocol import ROOT, digest, file_digest, load_plan, training_config, trial_spec
from ..runner import atomic_json, session_lock, terminate
from .knowledge import KnowledgeBank
from .history import export_history
from .journal import (register_prediction, verify_prediction, record_results, record_learnings,
                      record_failure_learning, validate_learning_evidence, utc_now,
                      learning_evidence_catalog, LearningEvidenceError)
from .performance import summarize_performance
from .predictions import METRICS, compare_prediction
from .roles import build_prompt
from .runtime import prepare_call, run_call
from .control import run_control
from .host import inspect_host
from .evolution import validate_settings, plan_generation, validate_child, compare_parents
from .suite import describe_suite, run_suite, validate_suite
from .workspace import create_workspace, freeze_workspace, inspect_workspace, source_digest


# This catches accidental training in local correctness tests. It is not a
# hostile-code sandbox: arbitrary Python still requires a trusted code agent.
TEST_BOOTSTRAP = """
import sys
import torch
import pytest
from src.train import train
from src.research import worker, runner
from src.research.team import suite
def no_training(*args, **kwargs):
    raise RuntimeError('Training is disabled during code-agent correctness tests')
train.run_training = no_training
worker.run_training = no_training
worker.main = no_training
runner.run_study = no_training
suite.run_suite = no_training
for cls in vars(torch.optim).values():
    if isinstance(cls, type) and issubclass(cls, torch.optim.Optimizer):
        cls.step = no_training
raise SystemExit(pytest.main(['-q', *sys.argv[1:]]))
"""


def load_config(path: Path) -> dict:
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    required = {"version", "name", "objective", "repository", "base_ref", "base_plan", "knowledge_bank", "runtime", "limits", "local_tests"}
    if not isinstance(cfg, dict) or set(cfg) - {"continuous", "evolution", "baseline_options"} != required or cfg["version"] != 1:
        raise ValueError("Invalid team config; use research/team_mac.yaml as the schema")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", str(cfg["name"])) or not str(cfg["objective"]).strip():
        raise ValueError("Set a short name and a concrete objective")
    for key in ("repository", "base_plan", "knowledge_bank"):
        cfg[key] = str((ROOT / Path(cfg[key]).expanduser()).resolve())
    if set(cfg["runtime"]) != {"executable", "model", "web_search"}:
        raise ValueError("runtime supports executable, optional model, and web_search")
    if not isinstance(cfg["runtime"]["web_search"], bool) or not isinstance(cfg["runtime"]["executable"], str):
        raise ValueError("Invalid runtime settings")
    if cfg["runtime"]["model"] is not None and not isinstance(cfg["runtime"]["model"], str):
        raise ValueError("runtime.model must be a string or null")
    limit_keys = {"max_iterations", "max_agent_calls", "agent_seconds", "total_agent_seconds", "agent_output_mb", "local_test_seconds", "total_training_seconds"}
    if set(cfg["limits"]) != limit_keys:
        raise ValueError("Invalid campaign limits")
    unlimited = {"max_iterations", "max_agent_calls", "total_agent_seconds", "total_training_seconds"}
    if "continuous" in cfg:
        settings = cfg["continuous"]
        if not isinstance(settings, dict) or set(settings) != {"target_hours", "min_free_disk_mb"}:
            raise ValueError("continuous requires target_hours and min_free_disk_mb")
        for key, value in settings.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"continuous.{key} must be finite and positive")
        if any(cfg["limits"][key] is not None for key in unlimited):
            raise ValueError("Continuous campaigns require null aggregate limits; per-call and per-suite limits remain finite")
    if "evolution" in cfg:
        validate_settings(cfg["evolution"])
    if "baseline_options" in cfg:
        validate_baseline_options(cfg["baseline_options"])
    for key, value in cfg["limits"].items():
        if "continuous" in cfg and key in unlimited and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"limits.{key} must be finite and positive")
        if key.startswith("max_") and not isinstance(value, int):
            raise ValueError(f"limits.{key} must be an integer")
    if not isinstance(cfg["local_tests"], list) or not cfg["local_tests"]:
        raise ValueError("Specify baseline local_tests")
    return cfg


def validate_baseline_options(options):
    """Reviewed optimizer families only; data, targets and budgets stay protected."""
    choices = {"optimizer": {"adamw", "muon"},
               "decay_mode": {"conventional", "independent"},
               "parameterization": {"sp", "mup"}}
    if not isinstance(options, dict) or set(options) != set(choices):
        raise ValueError("baseline_options requires optimizer, decay_mode and parameterization")
    for key, allowed in choices.items():
        if not isinstance(options[key], str) or options[key] not in allowed:
            raise ValueError(f"Invalid baseline_options.{key}")


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_digest(path)}


def _knowledge_snapshot(bank: KnowledgeBank, output: Path) -> dict:
    """Publish complete evidence for read-only agents without changing the bank.

    Each handoff keeps its own content-addressed snapshot as the shared bank
    grows. Read-only CLI agents can search this plain file without changing
    stored records or requesting write permissions.
    """
    content = bank.export_markdown()
    checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
    directory = output.resolve() / "knowledge-snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{checksum}.md"
    if path.exists():
        if path.is_symlink() or file_digest(path) != checksum:
            raise ValueError("Knowledge snapshot was changed outside the controller")
    else:
        temporary = directory / f".{checksum}.{os.getpid()}.{time.time_ns()}.tmp"
        try:
            temporary.write_text(content, encoding="utf-8")
            temporary.chmod(0o444)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return {"path": str(path), "sha256": checksum, "format": "markdown",
            "includes_all_records": True}


def _save(output: Path, state: dict):
    bank = KnowledgeBank(state["config"]["knowledge_bank"])
    state["knowledge_snapshot"] = _knowledge_snapshot(bank, output)
    state["reports"] = export_history(output, state)
    atomic_json(output / "campaign.json", state)
    (output / "knowledge.md").write_text(Path(state["knowledge_snapshot"]["path"]).read_text(encoding="utf-8"), encoding="utf-8")


def init_campaign(config: dict, output: Path) -> dict:
    """Set up committed baseline snapshots, but never call Codex or train."""
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty campaign directory")
    repo = Path(config["repository"])
    plan = load_plan(config["base_plan"])
    cfg = training_config(plan, trial_spec("confirm", plan["widths"][0], plan["seeds"][0], plan["baseline_lr"]), output / "unused")
    cfg["training"]["cpu_threads"] = plan["training"]["cpu_threads"]
    if "baseline_options" in config:
        options = config["baseline_options"]
        validate_baseline_options(options)
        if options["decay_mode"] == "independent" and cfg["training"]["weight_decay"] >= 1:
            raise ValueError("Independent baseline decay must be < 1 per step")
        cfg["training"].update(optimizer=options["optimizer"], decay_mode=options["decay_mode"])
        cfg["model"].update(parameterization=options["parameterization"], base_width=64)
    # The clean-root guard in create_workspace prevents accidentally using an
    # old commit while the controller implementation is still uncommitted.
    key = config["name"] + "-" + digest(str(output))[:8]
    workspaces = repo.parent / ".research-worktrees" / key
    base = create_workspace(repo, workspaces / "baseline", f"codex/{key}/baseline", config["base_ref"])
    output.mkdir(parents=True, exist_ok=True)
    frozen = freeze_workspace(Path(base["path"]), output / "baseline", base["base_commit"])
    if not (Path(frozen["path"]) / "src/research/worker.py").is_file():
        raise ValueError("Committed baseline lacks the research worker; commit the reviewed harness first")
    state = {"version": 1, "created_at": utc_now(), "config": config, "plan": plan,
             "controller_sha256": source_digest(repo), "key": key,
             "workspaces": str(workspaces), "baseline": frozen, "baseline_config": cfg,
             "iterations": [], "agent_calls": [], "agent_seconds": 0.0,
             "training_seconds": 0.0, "status": "prepared"}
    _save(output, state)
    return state


def apply_overrides(config: dict, overrides: list[dict]) -> dict:
    cfg = copy.deepcopy(config)
    seen = set()
    for override in overrides:
        path = override["path"]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+", path) or path in seen:
            raise ValueError(f"Invalid or duplicate config path: {path}")
        seen.add(path)
        if not (path.startswith("model.") or path in {"training.learning_rate", "training.weight_decay", "training.grad_clip", "training.optimizer", "training.decay_mode"}):
            raise ValueError(f"Protected experimental setting: {path}")
        if path in {"model.dim", "model.n_heads", "model.block_size", "model.vocab_size", "model.base_width"}:
            raise ValueError(f"Target/protocol controls {path}; the candidate cannot change it")
        value = json.loads(override["value_json"])
        digest(value)  # Reject NaN/Infinity accepted by Python's JSON decoder.
        node = cfg
        parts = path.split(".")
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            if not isinstance(node[part], dict):
                raise ValueError(f"Config path traverses a scalar: {path}")
            node = node[part]
        node[parts[-1]] = value
    return cfg


def _current(state: dict) -> dict | None:
    if state["iterations"] and state["iterations"][-1]["stage"] != "completed":
        return state["iterations"][-1]
    maximum = state["config"]["limits"]["max_iterations"]
    if maximum is not None and len(state["iterations"]) >= maximum:
        state["status"] = "completed"
        return None
    iteration = {"id": f"iteration-{len(state['iterations']) + 1:03d}", "stage": "idea", "created_at": utc_now()}
    if "evolution" in state["config"]:
        iteration["evolution"] = plan_generation(state)
    state["iterations"].append(iteration)
    return iteration


def _context(state: dict, iteration: dict) -> dict:
    bank = KnowledgeBank(state["config"]["knowledge_bank"])
    # Refresh before every handoff: another campaign may have appended evidence
    # since this campaign was last saved. The immutable export remains readable
    # for the whole agent invocation, including from a read-only sandbox.
    snapshot = _knowledge_snapshot(bank, Path(state["baseline"]["path"]).parent)
    entries = bank.search(limit=30)
    # Keep complete provenance on disk; compact retrieval avoids unbounded prompts.
    compact = [{k: entry[k] for k in ("id", "kind", "title", "summary", "outcome", "scope", "sources", "related_ids")}
               for entry in entries]
    return {"objective": state["config"]["objective"], "iteration": iteration,
            "knowledge": compact, "knowledge_snapshot": snapshot,
            "knowledge_excerpt_limit": 30,
            "baseline_config": state["baseline_config"], "transfer_plan": state["plan"],
            "prediction_metrics": METRICS,
            "prediction_targets": [{"name": f"width-{width}", "width": width} for width in state["plan"]["widths"]],
            "exploration_reports": state.get("reports", {}),
            "limits": state["config"]["limits"], "local_tests": state["config"]["local_tests"],
            "environment": "16 GB Mac mini target; no assumption that this host is that machine",
            "host_preflight": state.get("host_preflight"),
            "evidence_catalog": learning_evidence_catalog(iteration),
            "training_policy": "Codex roles never train; only the bounded Python executor may train when explicitly enabled."}


def _role_cwd(state: dict, iteration: dict) -> Path:
    return Path(iteration["workspace"]["path"]) if iteration["stage"] == "coder" else Path(state["baseline"]["path"])


def prepare_next(output: Path) -> dict:
    """Export a role prompt/schema with no model, tests, or training execution."""
    output = output.resolve()
    with session_lock(output):
        state = json.loads((output / "campaign.json").read_text())
        iteration = _current(state)
        if iteration is None or iteration["stage"] in {"train", "failed"}:
            _save(output, state)
            return {"status": state["status"], "stage": iteration["stage"] if iteration else "completed"}
        role = iteration["stage"]
        directory = output / "handoffs" / f"{iteration['id']}-{role}-{time.time_ns()}"
        request = prepare_call(role, build_prompt(role, _context(state, iteration)), _role_cwd(state, iteration), directory, state["config"]["runtime"])
        _save(output, state)
        return {**request, "handoff": str(directory)}


def _local_tests(workspace: Path, files: list[str], output: Path, timeout: float) -> dict:
    paths = []
    for name in dict.fromkeys(files):
        if not isinstance(name, str):
            raise ValueError("Local test paths must be strings")
        path = workspace / name
        if (not name.startswith("tests/") or not name.endswith(".py")
                or path.is_symlink() or not path.resolve().is_relative_to((workspace / "tests").resolve()) or not path.is_file()):
            raise ValueError(f"Invalid local test path: {name}")
        paths.append(name)
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-c", TEST_BOOTSTRAP, *paths]
    started = time.monotonic()
    with (output / "tests.log").open("w") as log:
        process = subprocess.Popen(command, cwd=workspace, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        try:
            process.wait(timeout=timeout)
        except BaseException:
            terminate(process)
            raise
    receipt = {"command": command, "exit_code": process.returncode,
               "duration_seconds": time.monotonic() - started, "log": artifact(output / "tests.log")}
    atomic_json(output / "tests.json", receipt)
    if process.returncode:
        raise ValueError(f"Candidate local tests failed; see {output / 'tests.log'}")
    return receipt


def _suite_spec(state: dict, iteration: dict) -> dict:
    plan = state["plan"]
    data_path = Path(state["baseline_config"]["data"]["path"])
    data_hash = file_digest(data_path) if data_path.is_file() else None
    bank = KnowledgeBank(state["config"]["knowledge_bank"])
    consumed = data_hash and any(entry["scope"].get("holdout_consumed")
                                 for entry in bank.search(query=data_hash, limit=1000))
    return validate_suite({"version": 1, "hypothesis": iteration["idea"]["hypothesis"],
        "data_sha256": data_hash,
        "audit_test": len(state["iterations"]) == state["config"]["limits"]["max_iterations"] and not bool(consumed),
        "baseline": {"root": state["baseline"]["path"], "source_sha256": state["baseline"]["source_digest"], "config": state["baseline_config"]},
        "candidate": {"root": iteration["snapshot"]["path"], "source_sha256": iteration["snapshot"]["source_digest"], "config": iteration["candidate_config"]},
        "targets": [{"name": f"width-{width}", "width": width, "n_heads": width // plan["model"]["head_dim"]} for width in plan["widths"]],
        "seeds": plan["seeds"], "eval_seed": plan["eval_seed"], "min_improvement": plan["min_improvement"], "limits": copy.deepcopy(plan["limits"])})


def reserve_audit(bank: KnowledgeBank, state: dict, iteration: dict, spec: dict, directory: Path):
    """Conservatively consume a heldout before launch, even if later interrupted."""
    actual = file_digest(Path(spec["baseline"]["config"]["data"]["path"]))
    if spec.get("data_sha256") is not None and spec["data_sha256"] != actual:
        raise ValueError("Dataset changed after the candidate protocol was frozen")
    spec["data_sha256"] = actual
    if not spec["audit_test"]:
        return
    # Shared across campaigns using this knowledge directory. Query+reservation is one
    # critical section, so concurrent campaigns cannot both claim fresh audit.
    with (bank.path / ".audit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if any(row["scope"].get("holdout_consumed") for row in bank.search(actual, kind="observation", limit=10000)):
            spec["audit_test"] = False
            iteration["audit_skipped_reason"] = "Dataset holdout was previously used or reserved"
            return
        path = directory / "audit-reservation.json"
        atomic_json(path, {"dataset_sha256": actual, "campaign": state["key"], "iteration": iteration["id"],
                           "policy": "Reserved before dispatch; conservatively consumed even if interrupted"})
        bank.add({"kind": "observation", "title": "Held-out test audit reserved",
            "summary": "An audited suite reserved this dataset. Treat the holdout as consumed even if execution is interrupted before an audit result is recorded.",
            "outcome": "reported", "scope": {"dataset_sha256": actual, "holdout_consumed": True},
            "artifacts": [artifact(path)], "created_by": "coordinator",
            "details": {"campaign": state["key"], "iteration": iteration["id"]}})


def _record_idea(bank: KnowledgeBank, state: dict, iteration: dict, result: dict, response_path: Path):
    for reference in result["based_on"]:
        bank.get(reference)  # Reject invented evidence IDs.
    literature_ids = []
    for source in result["sources"]:
        literature_ids.append(bank.add({"kind": "literature", "title": source["title"], "summary": source["claim"],
            "outcome": "reported", "scope": {"replicated_locally": False},
            "sources": [{key: source[key] for key in ("title", "url", "locator") if source[key]}],
            "artifacts": [artifact(response_path)], "created_by": "idea-agent",
            "details": {"verification": "Agent-reported reading; retained with its Codex event log, not independently replicated."}}))
    iteration["idea_id"] = bank.add({"kind": "idea", "title": result["title"], "summary": result["hypothesis"],
        "outcome": "untried", "scope": {"campaign": state["key"], "iteration": iteration["id"]},
        "related_ids": list(dict.fromkeys(result["based_on"] + literature_ids)),
        "artifacts": [artifact(response_path)], "created_by": "idea-agent", "details": result})


def advance(output: Path, *, run_agents=False, allow_training=False) -> dict:
    if not run_agents:
        return prepare_next(output)
    output = output.resolve()
    with session_lock(output) as lock_fd:
        state = json.loads((output / "campaign.json").read_text())
        if (output / "STOP").exists():
            state["status"] = "paused"
            _save(output, state)
            return state
        if source_digest(Path(state["config"]["repository"])) != state["controller_sha256"]:
            raise ValueError("Controller source changed; start a new campaign from the reviewed revision")
        if any(call["status"] == "running" for call in state["agent_calls"]):
            raise ValueError("Interrupted agent handoff requires inspection; its time reservation remains charged")
        continuous = state["config"].get("continuous")
        if continuous and shutil.disk_usage(output).free < continuous["min_free_disk_mb"] * 1024**2:
            state["status"] = "insufficient_disk_space"
            _save(output, state)
            return state
        if "run_started_at" not in state:
            state["run_started_at"] = utc_now()
        iteration = _current(state)
        if iteration is None or iteration["stage"] == "failed":
            _save(output, state)
            return state
        bank = KnowledgeBank(state["config"]["knowledge_bank"])
        limits = state["config"]["limits"]
        directory = output / iteration["id"]
        directory.mkdir(parents=True, exist_ok=True)
        phase = iteration["stage"]
        try:
            if not state.get("host_preflight") or phase in {"executor", "train"}:
                state["host_preflight"] = {**inspect_host(state["baseline_config"]["training"]["device"]),
                                           "checked_at": utc_now()}
            if phase != "idea":
                verify_prediction(state, iteration)
            if phase == "train":
                if not allow_training:
                    state["status"] = "awaiting_training_authorization"
                    _save(output, state)
                    return state
                if iteration.get("training_started"):
                    raise ValueError("Interrupted suite requires inspection before restarting; its reserved training time remains charged")
                total = limits["total_training_seconds"]
                remaining = (iteration["suite"]["limits"]["session_seconds"] if total is None
                             else total - state["training_seconds"])
                if remaining <= 0:
                    state["status"] = "training_budget_exhausted"
                    _save(output, state)
                    return state
                spec = copy.deepcopy(iteration["suite"])
                spec["limits"]["session_seconds"] = min(spec["limits"]["session_seconds"], remaining)
                reserve_audit(bank, state, iteration, spec, directory)
                iteration["suite"] = spec
                # Reserve the complete suite allowance before launch. A crash
                # must never silently restore consumed campaign compute.
                allowance = spec["limits"]["session_seconds"]
                previous = state["training_seconds"]
                state["training_seconds"] += allowance
                iteration["training_started"] = True
                _save(output, state)
                started = time.monotonic()
                report = run_suite(spec, directory / "suite", allow_training=True)
                state["training_seconds"] = previous + time.monotonic() - started
                report["performance"] = summarize_performance(spec, report.get("records", []),
                                                             integrity=report.get("evidence_integrity", True))
                if "evolution" in iteration:
                    report["evolution"] = compare_parents(state, iteration, report)
                iteration["report"] = report
                atomic_json(directory / "evidence.json", report)
                prediction = verify_prediction(state, iteration)
                iteration["prediction_check"] = compare_prediction(prediction, report["performance"])
                atomic_json(directory / "prediction-check.json", iteration["prediction_check"])
                iteration["execution_id"] = bank.add({"kind": "execution", "title": iteration["idea"]["title"],
                    "summary": "Paired small-model and transfer execution recorded; see deterministic evidence.",
                    "outcome": "inconclusive", "scope": {"campaign": state["key"], "iteration": iteration["id"]},
                    "related_ids": [iteration["implementation_id"]], "artifacts": [artifact(directory / "evidence.json")],
                    "created_by": "bounded-executor", "details": {"local_outcome": report["local_outcome"], "transfer_outcome": report["transfer_outcome"]}})
                record_results(bank, state, iteration, directory)
                if any(row["stage"] == "test" for row in report.get("records", [])):
                    bank.add({"kind": "observation", "title": "Held-out test data consumed", "summary": "This dataset's test audit has been exposed; future adaptive searches cannot call it a fresh holdout.",
                        "outcome": "reported", "scope": {"dataset_sha256": report["contract"]["data"], "holdout_consumed": True},
                        "related_ids": [iteration["execution_id"]], "artifacts": [artifact(directory / "evidence.json")], "created_by": "coordinator"})
                iteration["stage"] = "evaluator"
                state["status"] = "ready_for_evaluator"
            else:
                if limits["max_agent_calls"] is not None and len(state["agent_calls"]) >= limits["max_agent_calls"]:
                    state["status"] = "agent_call_budget_exhausted"
                    _save(output, state)
                    return state
                total = limits["total_agent_seconds"]
                remaining = limits["agent_seconds"] if total is None else total - state["agent_seconds"]
                if remaining <= 0:
                    state["status"] = "agent_time_budget_exhausted"
                    _save(output, state)
                    return state
                call_dir = directory / f"{phase}-{len(state['agent_calls']) + 1:03d}"
                allowance = min(limits["agent_seconds"], remaining)
                call = {"role": phase, "iteration": iteration["id"], "path": str(call_dir), "status": "running", "reserved_seconds": allowance,
                        "started_at": utc_now()}
                state["agent_calls"].append(call)
                previous = state["agent_seconds"]
                state["agent_seconds"] += allowance
                _save(output, state)
                started = time.monotonic()
                receipt = run_call(phase, build_prompt(phase, _context(state, iteration), training_authorized=allow_training), _role_cwd(state, iteration), call_dir,
                                   state["config"]["runtime"], allow_agents=True, timeout=allowance,
                                   max_output_mb=limits["agent_output_mb"], lock_fd=lock_fd)
                if source_digest(Path(state["config"]["repository"])) != state["controller_sha256"]:
                    raise ValueError("Controller source changed during the agent call; response not accepted")
                state["agent_seconds"] = previous + time.monotonic() - started
                call.update(status="completed", duration_seconds=receipt["duration_seconds"], usage=receipt["usage"], completed_at=utc_now())
                result = receipt["response"]
                if phase == "idea":
                    if "evolution" in iteration and result["change_kind"] != "config":
                        raise ValueError("Evolution requires a configuration proposal")
                    iteration["idea"] = result
                    _record_idea(bank, state, iteration, result, call_dir / "response.json")
                    register_prediction(bank, state, iteration, directory)
                    iteration["workspace"] = create_workspace(Path(state["config"]["repository"]), Path(state["workspaces"]) / iteration["id"],
                        f"codex/{state['key']}/{iteration['id']}", state["baseline"]["base_commit"])
                    iteration["stage"] = "coder"
                elif phase == "coder":
                    workspace = Path(iteration["workspace"]["path"])
                    inspected = inspect_workspace(workspace, state["baseline"]["base_commit"])
                    iteration["candidate_config"] = apply_overrides(state["baseline_config"], result["config_overrides"])
                    validate_child(state, iteration, inspected["changes"])
                    model_changed = any(change["path"].startswith(("src/models/", "src/attention/", "src/moe/")) for change in inspected["changes"])
                    if not model_changed and iteration["candidate_config"] == state["baseline_config"]:
                        raise ValueError("Candidate makes no code or configuration change")
                    iteration["local_tests"] = _local_tests(workspace, state["config"]["local_tests"] + result["test_files"], directory / "local-tests", limits["local_test_seconds"])
                    iteration["snapshot"] = freeze_workspace(workspace, directory / "candidate", state["baseline"]["base_commit"])
                    iteration["implementation"] = result
                    iteration["suite"] = _suite_spec(state, iteration)
                    atomic_json(directory / "suite-spec.json", iteration["suite"])
                    iteration["implementation_id"] = bank.add({"kind": "implementation", "title": iteration["idea"]["title"],
                        "summary": result["summary"], "outcome": "untried", "scope": {"campaign": state["key"], "iteration": iteration["id"]},
                        "related_ids": [iteration["idea_id"], iteration["prediction_id"]], "artifacts": [iteration["prediction_artifact"], artifact(directory / "suite-spec.json"), artifact(directory / "local-tests/tests.json"), artifact(Path(iteration["snapshot"]["patch_path"]))],
                        "created_by": "code-agent", "details": {"branch": inspected["branch"], "config_overrides": result["config_overrides"], "risks": result["risks"]}})
                    iteration["stage"] = "executor"
                elif phase == "executor":
                    iteration["preflight"] = result
                    if not result["ready"]:
                        raise ValueError("Executor preflight rejected the candidate: " + "; ".join(result["reasons"]))
                    iteration["stage"] = "train"
                elif phase == "evaluator":
                    report = iteration["report"]
                    validate_learning_evidence(iteration, result)
                    # The narrative cannot upgrade the deterministic conclusion.
                    outcome = report["transfer_outcome"]
                    if outcome not in {"supported", "contradicted", "mixed", "inconclusive", "failed"}:
                        outcome = "inconclusive"
                    if result["validity_concerns"]:
                        outcome = "inconclusive"
                    iteration["evaluation_id"] = bank.add({"kind": "evaluation", "title": iteration["idea"]["title"],
                        "summary": result["summary"], "outcome": outcome,
                        "scope": {"campaign": state["key"], "iteration": iteration["id"], "targets": iteration["suite"]["targets"],
                                  "local_outcome": report["local_outcome"], "transfer_outcome": report["transfer_outcome"], "generalizes_beyond_tested_models": False},
                        "related_ids": [iteration["execution_id"], iteration["idea_id"], iteration["prediction_id"], iteration["result_id"]],
                        "artifacts": [artifact(directory / "evidence.json"), artifact(call_dir / "response.json")],
                        "created_by": "evaluator-agent", "details": result})
                    iteration["evaluation"] = result
                    record_learnings(bank, state, iteration, result, directory, call_dir / "response.json", outcome)
                    iteration["stage"] = "completed"
                    iteration["completed_at"] = utc_now()
                state["status"] = "ready_for_" + iteration["stage"]
        except (Exception, KeyboardInterrupt) as exc:
            if phase == "evaluator" and isinstance(exc, LearningEvidenceError):
                # Repair citation-only mistakes without repeating the training suite.
                # Every attempt consumes the ordinary agent call/time budgets.
                rejected = iteration.setdefault("evaluation_rejections", [])
                rejection_path = call_dir / "evidence-rejection.json"
                atomic_json(rejection_path, {"error": str(exc), "response": artifact(call_dir / "response.json"),
                                            "evidence_catalog": learning_evidence_catalog(iteration)})
                call["evaluation_status"] = "rejected_evidence"
                rejected.append(artifact(rejection_path))
                if len(rejected) <= 2:
                    iteration["evaluation_feedback"] = str(exc)
                    state["status"] = "ready_for_evaluator"
                    _save(output, state)
                    return state
            if state["agent_calls"] and state["agent_calls"][-1]["status"] == "running":
                state["agent_calls"][-1].update(status="failed", error=str(exc))
            iteration.update(stage="failed", failed_phase=phase, error=str(exc) or "Interrupted")
            state["status"] = "needs_attention"
            error_path = directory / f"{phase}-failure.json"
            atomic_json(error_path, {"phase": phase, "error": iteration["error"]})
            iteration["failure_id"] = bank.add({"kind": "observation", "title": f"{iteration['id']} {phase} failed", "summary": iteration["error"],
                "outcome": "failed", "scope": {"campaign": state["key"], "phase": phase}, "related_ids": [iteration["idea_id"]] if "idea_id" in iteration else [],
                "artifacts": [artifact(error_path)], "created_by": "coordinator", "details": {"not_a_negative_model_result": True}})
            record_failure_learning(bank, state, iteration, error_path)
        _save(output, state)
        return state


def run_campaign(output: Path, *, run_agents=False, allow_training=False, resume=False) -> dict:
    if not run_agents:
        if resume:
            raise ValueError("Resuming execution requires --run-agents")
        return prepare_next(output)
    with run_control(output, resume=resume):
        while True:
            state = advance(output, run_agents=True, allow_training=allow_training)
            if state["status"] in {"completed", "needs_attention", "awaiting_training_authorization", "training_budget_exhausted", "agent_call_budget_exhausted", "agent_time_budget_exhausted", "paused", "insufficient_disk_space"}:
                return state
