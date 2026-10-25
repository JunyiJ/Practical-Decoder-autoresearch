"""Frozen, paired baseline/candidate experiments for the MLE agent team.

The controller is deterministic. It never proposes edits, changes seeds after
failure, or selects on larger-width/test scores. Training requires an explicit
``allow_training=True`` at the process boundary.
"""
from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev

from ..protocol import digest, file_digest
from ..runner import atomic_json, disk_mb, session_lock, terminate, validate_result
from .performance import summarize_performance


ARMS = ("baseline", "candidate")
TUNABLE_TRAINING = {"learning_rate", "weight_decay", "grad_clip", "optimizer", "decay_mode"}
RUNTIME_TRAINING = {"seed", "eval_seed", "strict_device", "checkpoint_dir", "checkpoint_every",
                    "metrics_file", "summary_file", "keep_best_checkpoint", "resume_from", "max_train_seconds"}
SHARED_TRAINING = {"max_iters", "batch_size", "eval_every", "eval_iters", "device", "cpu_threads"}
LIMIT_KEYS = {"max_trials", "session_seconds", "trial_seconds", "max_disk_mb", "max_failures",
              "max_consecutive_failures"}

# Candidates may change the model's training objective. Their reported evaluation
# loss is never authoritative: recompute CE from their logits in trusted code.
WORKER_BOOTSTRAP = """
import torch
from src.train import train as trainer
OriginalGPT = trainer.GPT
class ScoredGPT(OriginalGPT):
    def forward(self, idx, targets=None, *args, **kwargs):
        if targets is not None and kwargs.get('include_aux_loss', True) is False:
            result = super().forward(idx, None, *args, **kwargs)
            logits = result[0]
            if logits.ndim != 3 or logits.shape[:-1] != targets.shape or logits.shape[-1] != self.cfg.vocab_size:
                raise ValueError('Evaluation logits do not match target shape and vocabulary')
            if not getattr(self, '_team_causality_checked', False):
                cut = idx.shape[1] // 2
                perturbed = idx.clone()
                perturbed[:, cut:] = (perturbed[:, cut:] + 1) % self.cfg.vocab_size
                alternate = super().forward(perturbed, None, *args, **kwargs)[0]
                if not torch.allclose(logits[:, :cut], alternate[:, :cut], rtol=1e-5, atol=1e-6):
                    raise ValueError('Causality check failed: future tokens changed prefix logits')
                self._team_causality_checked = True
            ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            if kwargs.get('return_loss_breakdown', False):
                return logits, ce, {'ce_loss': ce, 'total_loss': ce, 'aux_loss': None}
            return logits, ce
        return super().forward(idx, targets, *args, **kwargs)
trainer.GPT = ScoredGPT
from src.research.worker import main
main()
"""


def source_digest(root: Path | str) -> str:
    root = Path(root).resolve()
    paths = sorted((root / "src").rglob("*.py")) + [root / "requirements.txt"]
    if not paths or not (root / "src").is_dir():
        raise ValueError(f"Missing source tree: {root}")
    if any(path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root)
           for path in paths):
        raise ValueError("Frozen sources cannot contain symlinks")
    return digest({str(path.relative_to(root)): file_digest(path) for path in paths})


def _number(value, label, *, integer=False, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or (value < 0 if zero else value <= 0) or (integer and not isinstance(value, int))):
        raise ValueError(f"{label} must be a finite {'nonnegative' if zero else 'positive'} {'integer' if integer else 'number'}")


def _keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{label} must have exactly these keys: {sorted(keys)}")


def validate_suite(spec: dict) -> dict:
    spec = copy.deepcopy(spec)
    if isinstance(spec, dict):
        spec.setdefault("audit_test", True)
        spec.setdefault("data_sha256", None)
    _keys(spec, {"version", "hypothesis", "baseline", "candidate", "targets", "seeds", "eval_seed",
                 "min_improvement", "limits", "audit_test", "data_sha256"}, "suite")
    if type(spec["version"]) is not int or spec["version"] != 1:
        raise ValueError("Only suite version 1 is supported")
    if not isinstance(spec["hypothesis"], str) or not spec["hypothesis"].strip():
        raise ValueError("A nonempty hypothesis is required")
    if not isinstance(spec["audit_test"], bool):
        raise ValueError("audit_test must be a boolean")
    if spec["data_sha256"] is not None and (
            not isinstance(spec["data_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", spec["data_sha256"])):
        raise ValueError("data_sha256 must be a lowercase SHA-256 hex digest or null")
    seeds = spec["seeds"]
    if not isinstance(seeds, list) or len(seeds) < 2:
        raise ValueError("At least two paired seeds are required")
    for seed in seeds + [spec["eval_seed"]]:
        _number(seed, "seed", integer=True, zero=True)
        if seed >= 2**63 - 2:
            raise ValueError("Seed exceeds supported range")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Paired seeds must be distinct")
    _number(spec["min_improvement"], "min_improvement", zero=True)
    _keys(spec["limits"], LIMIT_KEYS, "limits")
    for key, value in spec["limits"].items():
        _number(value, key, integer=key in {"max_trials", "max_failures", "max_consecutive_failures"})
    targets = spec["targets"]
    if not isinstance(targets, list) or len(targets) < 2:
        raise ValueError("Declare a small model and at least one larger-width target")
    for target in targets:
        _keys(target, {"name", "width", "n_heads"}, "target")
        if not isinstance(target["name"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", target["name"]):
            raise ValueError("Target names must contain only letters, numbers, hyphens or underscores")
        for key in ("width", "n_heads"):
            _number(target[key], key, integer=True)
        if target["width"] % target["n_heads"]:
            raise ValueError("Every width must be divisible by its number of heads")
    widths = [t["width"] for t in targets]
    if (len({t["name"] for t in targets}) != len(targets) or len(set(widths)) != len(widths)
            or widths != sorted(widths)):
        raise ValueError("Target names must be unique and widths strictly increasing")
    if len({t["width"] // t["n_heads"] for t in targets}) != 1:
        raise ValueError("Keep head dimension fixed across width-transfer targets")
    for arm in ARMS:
        value = spec[arm]
        _keys(value, {"root", "config", "source_sha256"}, arm)
        if not isinstance(value["root"], str) or not Path(value["root"]).is_absolute():
            raise ValueError("Frozen repository roots must be absolute paths")
        value["root"] = str(Path(value["root"]).resolve())
        if not isinstance(value["source_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["source_sha256"]):
            raise ValueError("Each arm requires a frozen source_sha256")
        cfg = value["config"]
        _keys(cfg, {"data", "model", "training"}, f"{arm}.config")
        _keys(cfg["data"], {"path", "train_fraction", "val_fraction"}, f"{arm}.data")
        if not isinstance(cfg["data"]["path"], str) or not cfg["data"]["path"]:
            raise ValueError("A corpus path is required")
        path = Path(cfg["data"]["path"]).expanduser()
        cfg["data"]["path"] = str((Path(value["root"]) / path).resolve() if not path.is_absolute() else path.resolve())
        for key in ("train_fraction", "val_fraction"):
            _number(cfg["data"][key], key)
        if cfg["data"]["train_fraction"] + cfg["data"]["val_fraction"] >= 1:
            raise ValueError("Reserve a nonempty held-out test split")
        model, train = cfg["model"], cfg["training"]
        if not isinstance(model, dict) or not isinstance(train, dict):
            raise ValueError("Model and training configs must be mappings")
        for key in ("n_layers", "block_size"):
            _number(model.get(key), f"model.{key}", integer=True)
        if model["block_size"] < 2:
            raise ValueError("Context length must be at least 2 to check autoregressive causality")
        if model.get("causal", True) is not True:
            raise ValueError("Autoregressive experiments require causal attention")
        _number(model.get("dropout", 0), "dropout", zero=True)
        if model.get("dropout", 0) >= 1:
            raise ValueError("dropout must be < 1")
        if model.get("rope", 0) == 1 and any(t["width"] // t["n_heads"] % 2 for t in targets):
            raise ValueError("RoPE requires an even head dimension")
        unknown = set(train) - SHARED_TRAINING - TUNABLE_TRAINING - RUNTIME_TRAINING - {'batch_layout'}
        if unknown:
            raise ValueError(f"Unsupported training fields: {sorted(unknown)}")
        for key in SHARED_TRAINING - {"device"}:
            _number(train.get(key), f"training.{key}", integer=True)
        if train.get('batch_layout', 'independent') not in {'independent', 'paired_blocks512', 'paired_batch512'}:
            raise ValueError('Unsupported training.batch_layout')
        if train.get('batch_layout') == 'paired_blocks512' and (
                model['block_size'] not in {128,256,512} or train['batch_size']*model['block_size'] != 1024):
            raise ValueError('paired_blocks512 requires context128/256/512 and1024 targets/update')
        if not isinstance(train.get("device"), str) or not re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", train["device"]):
            raise ValueError("Device must be cpu, mps, cuda or cuda:N")
        if train.get('batch_layout') == 'paired_batch512' and (
                model['block_size'] != 128 or train['batch_size'] not in {8,16,32}):
            raise ValueError('paired_batch512 requires context128 and batch8/16/32')
        train.setdefault("weight_decay", 0.0)
        train.setdefault("grad_clip", 0.0)
        for key, allowed, default in [('optimizer', {'adamw', 'muon'}, 'adamw'),
                                      ('decay_mode', {'conventional', 'independent'}, 'conventional')]:
            if train.get(key, default) not in allowed:
                raise ValueError(f'Unsupported training.{key}')
        if train.get('decay_mode') == 'independent' and train['weight_decay'] >= 1:
            raise ValueError('Independent per-step decay must be < 1')
        for key in TUNABLE_TRAINING - {'optimizer', 'decay_mode'}:
            _number(train.get(key), f"training.{key}", zero=key != "learning_rate")
        if train.get("resume_from") is not None or train.get("max_train_seconds") is not None:
            raise ValueError("Trials must start fresh and use the full fixed-token budget")
        # Runtime bookkeeping is controlled by the executor, never by a proposal.
        for key in RUNTIME_TRAINING:
            train.pop(key, None)
        model["dim"], model["n_heads"], model["vocab_size"] = targets[0]["width"], targets[0]["n_heads"], 1
    base, candidate = (spec[arm]["config"] for arm in ARMS)
    if base["data"] != candidate["data"] or base["model"]["block_size"] != candidate["model"]["block_size"]:
        raise ValueError("Both arms must use identical data, splits, tokenizer and context length")
    if any(base["training"][key] != candidate["training"][key] for key in SHARED_TRAINING):
        raise ValueError("Both arms must use identical token budgets, evaluation settings, threads and device")
    if base['training'].get('batch_layout', 'independent') != candidate['training'].get('batch_layout', 'independent'):
        raise ValueError('Both arms must use identical batch layout')
    digest(spec)  # Reject non-JSON or NaN model config values before recording.
    return spec


def describe_suite(spec: dict) -> dict:
    spec = validate_suite(spec)
    cfg = spec["baseline"]["config"]
    train = cfg["training"]
    count = 2 * len(spec["seeds"]) * len(spec["targets"])
    tests = count if spec["audit_test"] else 0
    return {"version": 1, "hypothesis": spec["hypothesis"], "targets": spec["targets"],
            "seeds": spec["seeds"], "training_trials": count, "test_evaluations_max": tests,
            "jobs_max": count + tests, "audit_test": spec["audit_test"], "limits": spec["limits"],
            "training_character_tokens_per_trial": train["max_iters"] * train["batch_size"] * cfg["model"]["block_size"],
            "objective": "final-budget validation CE in nats/character; equal tokens, not equal compute",
            "selection_rule": "Candidate supported only if mean paired delta < -min_improvement and every paired delta < 0",
            "training_enabled": False}


def _fingerprint(spec):
    sources = {arm: source_digest(spec[arm]["root"]) for arm in ARMS}
    if any(sources[arm] != spec[arm]["source_sha256"] for arm in ARMS):
        raise ValueError("Frozen source digest does not match the supplied snapshot")
    # Model implementation can vary; trainer/data/worker/evaluator cannot.
    roots = [Path(spec[arm]["root"]) for arm in ARMS]
    mutable = ("src/models/", "src/attention/", "src/moe/")
    protected = [{str(p.relative_to(root)): file_digest(p) for p in (root / "src").rglob("*.py")
                  if not str(p.relative_to(root)).startswith(mutable)} for root in roots]
    if protected[0] != protected[1]:
        raise ValueError("Candidate changed protected trainer, data, evaluation or controller sources")
    if file_digest(roots[0] / "requirements.txt") != file_digest(roots[1] / "requirements.txt"):
        raise ValueError("Both arms must use the same dependencies")
    packages = {name: importlib.metadata.version(name) for name in
                ("torch", "numpy", "omegaconf", "hydra-core", "tiktoken")}
    data_hash = file_digest(Path(spec["baseline"]["config"]["data"]["path"]))
    if spec["data_sha256"] is not None and spec["data_sha256"] != data_hash:
        raise ValueError("Corpus changed: data_sha256 does not match the frozen expected dataset digest")
    return {"spec": digest(spec), "source": sources,
            "data": data_hash,
            "evaluation_bootstrap": digest(WORKER_BOOTSTRAP),
            "controller_source": file_digest(Path(__file__)),
            "runtime": {"python": sys.version, "executable": sys.executable, "packages": packages,
                        "platform": platform.platform(), "host": platform.node(),
                        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0")}}


def _preflight(spec):
    cfg = spec["baseline"]["config"]
    data = cfg["data"]
    text = Path(data["path"]).read_text(encoding="utf-8")
    length = len(text)
    if len(set(text)) < 2:
        raise ValueError("The character corpus must contain at least two distinct symbols")
    a, b = int(length * data["train_fraction"]), int(length * (data["train_fraction"] + data["val_fraction"]))
    if min(a, b - a, length - b) <= cfg["model"]["block_size"]:
        raise ValueError("Each data split must be longer than the context length")
    if cfg["training"]["device"] == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("Disable MPS CPU fallback for strict-device experiments")


def _trial(stage, target, seed, arm):
    return {"trial_id": f"{stage}_{target['name']}_s{seed}_{arm}", "stage": stage,
            "target": target["name"], "width": target["width"], "n_heads": target["n_heads"],
            "seed": seed, "arm": arm}


def _pairs(stage, targets, seeds, arms=ARMS):
    return [_trial(stage, target, seed, arm) for target in targets for i, seed in enumerate(seeds)
            for arm in (arms if i % 2 == 0 else tuple(reversed(arms)))]


def _comparison(spec, records, target, stage):
    pairs, missing = [], []
    for seed in spec["seeds"]:
        rows = [[r for r in records if r["target"] == target["name"] and r["stage"] == stage
                 and r["seed"] == seed and r["arm"] == arm and r["status"] == "completed"
                 and isinstance(r.get("metric"), (int, float)) and math.isfinite(r["metric"])] for arm in ARMS]
        if any(len(group) != 1 for group in rows):
            missing.append(seed)
        else:
            a, b = rows[0][0]["metric"], rows[1][0]["metric"]
            pairs.append({"seed": seed, "baseline": a, "candidate": b, "delta": b - a})
    deltas = [p["delta"] for p in pairs]
    delta = mean(deltas) if deltas else None
    outcome = ("incomplete" if missing else "supported" if delta < -spec["min_improvement"] and all(d < 0 for d in deltas)
               else "contradicted" if delta >= 0 else "mixed")
    return {"target": target["name"], "width": target["width"], "stage": stage,
            "outcome": outcome, "complete": not missing, "pairs": pairs, "missing_seeds": missing,
            "mean_delta": delta, "sample_std_delta": stdev(deltas) if len(deltas) > 1 else None}


def _schedule(spec, records, selection):
    by_id = {r["trial_id"]: r for r in records}
    selection = dict(selection)
    small = spec["targets"][0]
    local = _pairs("local", [small], spec["seeds"])
    for job in local:
        if job["trial_id"] not in by_id:
            return job, selection, None
    comparison = _comparison(spec, records, small, "local")
    if not comparison["complete"]:
        return None, selection, "local_incomplete"
    if not selection:
        selection = {"winner": "candidate" if comparison["outcome"] == "supported" else "baseline",
                     "local_outcome": comparison["outcome"], "paired_deltas": [p["delta"] for p in comparison["pairs"]],
                     "mean_delta": comparison["mean_delta"], "selected_by": "small_width_validation_only"}
    transfer = _pairs("transfer", spec["targets"][1:], spec["seeds"])
    for job in transfer:
        if job["trial_id"] not in by_id:
            return job, selection, None
    if any(by_id[j["trial_id"]]["status"] != "completed" for j in transfer):
        return None, selection, "transfer_incomplete"
    if not spec["audit_test"]:
        return None, selection, "completed"
    arms = ARMS if selection["winner"] == "candidate" else ("baseline",)
    tests = _pairs("test", spec["targets"], spec["seeds"], arms)
    for job in tests:
        if job["trial_id"] not in by_id:
            stage = "local" if job["target"] == small["name"] else "transfer"
            job["parent_trial_id"] = _trial(stage, next(t for t in spec["targets"] if t["name"] == job["target"]), job["seed"], job["arm"])["trial_id"]
            return job, selection, None
    return None, selection, "completed" if all(by_id[j["trial_id"]]["status"] == "completed" for j in tests) else "test_incomplete"


def _report(state):
    spec, records = state["spec"], state["records"]
    comparisons = [_comparison(spec, records, target, "local" if index == 0 else "transfer")
                   for index, target in enumerate(spec["targets"])]
    larger = comparisons[1:]
    transfer = ("incomplete" if not all(c["complete"] for c in larger) else
                "not_applicable" if state["selection"].get("winner") != "candidate" else
                "supported" if all(c["outcome"] == "supported" for c in larger) else
                "contradicted" if all(c["outcome"] == "contradicted" for c in larger) else "mixed")
    audits = [_comparison(spec, records, target, "test") for target in spec["targets"]] if spec["audit_test"] else []
    audit = ("not_requested" if not spec["audit_test"] else
             "baseline_only" if state["selection"].get("winner") == "baseline" and state.get("stop_reason") == "completed"
             else "incomplete" if not all(a["complete"] for a in audits) else
             "supported" if all(a["outcome"] == "supported" for a in audits) else
             "contradicted" if all(a["outcome"] == "contradicted" for a in audits) else "mixed")
    integrity_ok = state.get("stop_reason") not in {"evidence_changed", "checkpoint_changed"}
    return {"version": 1, "hypothesis": spec["hypothesis"], "evidence_integrity": integrity_ok,
            "holdout_consumed": any(r["stage"] == "test" for r in records),
            "local_outcome": comparisons[0]["outcome"] if integrity_ok else "invalid",
            "transfer_outcome": transfer if integrity_ok else "invalid",
            "audit_outcome": audit if integrity_ok or not spec["audit_test"] else "invalid", "comparisons": comparisons,
            "audits": audits, "selection": state["selection"], "records": records,
            "performance": summarize_performance(spec, records, integrity=integrity_ok),
            "stop_reason": state.get("stop_reason"), "active_seconds": state["active_seconds"],
            "contract": state["contract"], "protocol": describe_suite(spec),
            "caveats": ["Negative paired CE deltas favor the candidate. This is a declared heuristic, not a significance test.",
                        "Width transfer is limited to these model families, data, token budget, seeds and optimizer; it does not establish production-scale transfer.",
                        "Equal character-token budgets do not imply equal compute. Timing and memory are recorded per trial.",
                        "Test scores audit the frozen choice; using them to propose a next iteration consumes the holdout.",
                        "No paired score from a failed, missing or incomplete trial is treated as evidence."]}


def _launch(request_path, output, timeout, disk_limit, lock_fd, *, root, bootstrap=WORKER_BOOTSTRAP):
    started = time.monotonic()
    # Inherit an advisory lock so a surviving child prevents a competing resume.
    with (request_path.parent / "worker.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-c", bootstrap, "--request", str(request_path)],
                                   cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, pass_fds=(lock_fd,))
        try:
            next_disk = started
            while process.poll() is None:
                now = time.monotonic()
                if now - started >= timeout:
                    terminate(process)
                    return "timeout", "Whole-process wall-time limit reached"
                if now >= next_disk:
                    if disk_mb(output) >= disk_limit:
                        terminate(process)
                        return "budget_exhausted", "Sampled output disk limit reached"
                    next_disk = now + 1
                time.sleep(.1)
        except BaseException:
            terminate(process)
            raise
    if time.monotonic() - started > timeout:
        return "timeout", "Worker exceeded wall-time allowance"
    if disk_mb(output) >= disk_limit:
        return "budget_exhausted", "Sampled output disk limit reached"
    return ("failed", f"Worker exited {process.returncode}; see worker.log") if process.returncode else ("completed", None)


def _request(spec, job, directory):
    cfg = copy.deepcopy(spec[job["arm"]]["config"])
    cfg["model"].update(dim=job["width"], n_heads=job["n_heads"])
    threads = cfg["training"].pop("cpu_threads")
    cfg["training"].update(seed=job["seed"], eval_seed=spec["eval_seed"], strict_device=True,
                           checkpoint_dir=str(directory / "checkpoints"), checkpoint_every=0,
                           keep_best_checkpoint=False, resume_from=None, max_train_seconds=None,
                           metrics_file=str(directory / "metrics.jsonl"), summary_file=str(directory / "training_summary.json"))
    return {"stage": job["stage"], "config": cfg, "cpu_threads": threads,
            "result_path": str(directory / "result.json")}


def _verify_checkpoints(records, output):
    for record in records:
        if record["status"] != "completed" or record["stage"] == "test":
            continue
        path = Path(record["checkpoint"]).resolve()
        directory = (output / "trials" / record["trial_id"]).resolve()
        if not path.is_relative_to(directory) or file_digest(path) != record["checkpoint_sha256"]:
            raise ValueError(f"Recorded final checkpoint changed: {record['trial_id']}")


def run_suite(spec: dict, output: Path, *, allow_training=False) -> dict:
    if allow_training is not True:
        raise PermissionError("Training is disabled. Explicit allow_training=True is required to execute the suite.")
    spec = validate_suite(spec)
    _preflight(spec)
    contract = _fingerprint(spec)
    output = Path(output).expanduser().resolve()
    if any(output == Path(spec[a]["root"]) or output.is_relative_to(Path(spec[a]["root"]) / "src") for a in ARMS):
        raise ValueError("Place suite outputs outside frozen source directories")
    output.mkdir(parents=True, exist_ok=True)
    with session_lock(output) as lock_fd:
        path = output / "state.json"
        if path.exists():
            state = json.loads(path.read_text())
            if state["contract"] != contract or state["spec"] != spec:
                raise ValueError("Frozen specification, sources, corpus or runtime changed; use a new output directory")
            for row in state["records"]:
                if row["status"] == "running":
                    row.update(status="failed", error="Interrupted controller; no automatic retry", metric=None)
        else:
            if any(p.name != ".lock" for p in output.iterdir()):
                raise ValueError("A new suite output directory must be empty")
            state = {"version": 1, "spec": spec, "contract": contract, "records": [], "selection": {},
                     "active_seconds": 0.0, "stop_reason": None}
            atomic_json(output / "manifest.json", {"contract": contract, "protocol": describe_suite(spec)})
        prior, start = state["active_seconds"], time.monotonic()

        def save(reserve=0.0):
            state["active_seconds"] = prior + time.monotonic() - start + reserve
            atomic_json(path, state)
            atomic_json(output / "report.json", _report(state))

        state["stop_reason"] = None
        try:
            while True:
                try:
                    if _fingerprint(spec) != contract:
                        raise ValueError("Fingerprint changed")
                except (ValueError, OSError) as exc:
                    state.update(stop_reason="evidence_changed", error=str(exc))
                    break
                try:
                    _verify_checkpoints(state["records"], output)
                except (ValueError, OSError, KeyError) as exc:
                    state.update(stop_reason="checkpoint_changed", error=str(exc))
                    break
                job, state["selection"], reason = _schedule(spec, state["records"], state["selection"])
                if reason:
                    state["stop_reason"] = reason
                    break
                limits = spec["limits"]
                failed = sum(r["status"] != "completed" for r in state["records"])
                consecutive = 0
                for record in reversed(state["records"]):
                    if record["status"] == "completed":
                        break
                    consecutive += 1
                remaining = limits["session_seconds"] - prior - (time.monotonic() - start)
                reason = ("max_trials" if len(state["records"]) >= limits["max_trials"] else
                          "max_failures" if failed >= limits["max_failures"] else
                          "max_consecutive_failures" if consecutive >= limits["max_consecutive_failures"] else
                          "session_time" if remaining <= 0 else
                          "disk_limit" if disk_mb(output) >= limits["max_disk_mb"] else None)
                if reason:
                    state["stop_reason"] = reason
                    break
                directory = output / "trials" / job["trial_id"]
                request = _request(spec, job, directory)
                if job["stage"] == "test":
                    parent = next(r for r in state["records"] if r["trial_id"] == job["parent_trial_id"])
                    parent_req = json.loads((output / "trials" / parent["trial_id"] / "request.json").read_text())
                    if file_digest(Path(parent["checkpoint"])) != parent["checkpoint_sha256"]:
                        state["stop_reason"] = "checkpoint_changed"
                        break
                    request.update(config=parent_req["config"], checkpoint=parent["checkpoint"],
                                   checkpoint_sha256=parent["checkpoint_sha256"])
                directory.mkdir(parents=True, exist_ok=False)
                atomic_json(directory / "request.json", request)
                row = {**job, "status": "running", "metric": None, "tokens_processed": 0,
                       "duration_seconds": 0.0, "error": None}
                state["records"].append(row)
                allowance = min(remaining, limits["trial_seconds"])
                save(reserve=allowance)
                trial_start = time.monotonic()
                try:
                    status, error = _launch(directory / "request.json", output, allowance,
                                            limits["max_disk_mb"], lock_fd, root=spec[job["arm"]]["root"])
                    row.update(status=status, error=error)
                    if _fingerprint(spec) != contract:
                        raise ValueError("Evidence changed during worker execution")
                    if status == "completed":
                        result = json.loads((directory / "result.json").read_text())
                        cfg = request["config"]
                        plan = {"device": cfg["training"]["device"], "training": {
                            **cfg["training"], "block_size": cfg["model"]["block_size"]}}
                        row.update(validate_result(result, request, plan, directory))
                        if result.get("runtime") != {"torch": contract["runtime"]["packages"]["torch"],
                                                     "cpu_threads": request["cpu_threads"], "requested_device": plan["device"]}:
                            raise ValueError("Worker runtime differs from the frozen contract")
                except KeyboardInterrupt:
                    row.update(status="failed", error="Interrupted; no automatic retry", metric=None)
                    state["stop_reason"] = "interrupted"
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    row.update(status="invalid", error=str(exc), metric=None)
                    try:
                        if _fingerprint(spec) != contract:
                            state["stop_reason"] = "evidence_changed"
                    except (ValueError, OSError):
                        state["stop_reason"] = "evidence_changed"
                finally:
                    row["duration_seconds"] = time.monotonic() - trial_start
                save()
                if state["stop_reason"]:
                    break
                if row["status"] == "budget_exhausted":
                    state["stop_reason"] = "disk_limit"
                    break
        finally:
            save()
        return _report(state)
