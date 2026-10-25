"""A local controller, not an LLM: the coding agent proposes; this runs evidence."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.metadata
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

from .protocol import (ROOT, candidate_id, describe_plan, digest, file_digest,
                       load_plan, training_config, trial_spec, validate_plan)
from .report import build_report, render_markdown


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_fingerprint() -> dict:
    paths = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "requirements.txt"]
    return {str(path.relative_to(ROOT)): file_digest(path) for path in paths}


def evidence_contract(plan: dict) -> dict:
    packages = {name: importlib.metadata.version(name)
                for name in ("torch", "numpy", "omegaconf", "hydra-core", "tiktoken")}
    return {"plan": digest(plan), "data": file_digest(Path(plan["data"]["path"])),
            "source": source_fingerprint(), "runtime": {
                "python": sys.version, "executable": sys.executable, "packages": packages,
                "platform": platform.platform(), "machine": platform.machine(), "host": platform.node(),
                "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0")}}


def preflight(plan: dict):
    path = Path(plan["data"]["path"])
    if not path.is_file():
        raise ValueError(f"Dataset missing: {path}. Supply a local UTF-8 text corpus before running.")
    length = len(path.read_text(encoding="utf-8"))
    train_end = int(length * plan["data"]["train_fraction"])
    val_end = int(length * (plan["data"]["train_fraction"] + plan["data"]["val_fraction"]))
    if min(train_end, val_end - train_end, length - val_end) <= plan["training"]["block_size"]:
        raise ValueError("Each split must contain more characters than block_size")
    if plan["device"] == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("Unset PYTORCH_ENABLE_MPS_FALLBACK for a strict-device research study")


def _complete(records):
    return all(r["status"] == "completed" and math.isfinite(r["metric"]) for r in records)


def schedule_next(plan: dict, records: list, selection: dict):
    """Selection may read screen/confirm validation only, never transfer or test."""
    by_id = {r["trial_id"]: r for r in records}
    selection = dict(selection)
    small = plan["widths"][0]
    baseline = plan["baseline_lr"]
    lrs = [baseline] + [lr for lr in plan["learning_rates"] if lr != baseline]
    screen = [trial_spec("screen", small, plan["screen_seed"], lr) for lr in lrs]
    for spec in screen:
        if spec["trial_id"] not in by_id:
            return spec, selection, None
    if not _complete([by_id[screen[0]["trial_id"]]]):
        return None, selection, "screen_baseline_failed"
    if "challenger_id" not in selection:
        valid = [by_id[s["trial_id"]] for s in screen[1:]
                 if _complete([by_id[s["trial_id"]]])]
        if not valid:
            return None, selection, "no_valid_challenger"
        challenger = min(valid, key=lambda r: (r["metric"], r["candidate"]))
        selection.update(baseline_id=candidate_id(baseline), challenger_id=challenger["candidate"],
                         challenger_lr=challenger["learning_rate"])
    challenger_lr = selection["challenger_lr"]

    def pairs(stage, widths, rates):
        # Alternate arm order to reduce systematic ordering effects.
        return [trial_spec(stage, w, seed, lr) for w in widths
                for i, seed in enumerate(plan["seeds"])
                for lr in (rates if i % 2 == 0 else list(reversed(rates)))]

    confirmation = pairs("confirm", [small], [baseline, challenger_lr])
    for spec in confirmation:
        if spec["trial_id"] not in by_id:
            return spec, selection, None
    if not _complete([by_id[s["trial_id"]] for s in confirmation]):
        return None, selection, "confirmation_incomplete"
    if "winner_id" not in selection:
        deltas = [by_id[trial_spec("confirm", small, seed, challenger_lr)["trial_id"]]["metric"] -
                  by_id[trial_spec("confirm", small, seed, baseline)["trial_id"]]["metric"]
                  for seed in plan["seeds"]]
        mean_delta = math.fsum(deltas) / len(deltas)
        winner_lr = challenger_lr if mean_delta < -plan["min_improvement"] and all(d < 0 for d in deltas) else baseline
        selection.update(winner_id=candidate_id(winner_lr), winner_lr=winner_lr,
                         selected_by="confirm_mean_val_loss", mean_delta=mean_delta,
                         paired_deltas=deltas,
                         rule="mean delta < -min_improvement AND every paired delta < 0; otherwise baseline")

    transfer = pairs("transfer", plan["widths"][1:], [baseline, challenger_lr])
    for spec in transfer:
        if spec["trial_id"] not in by_id:
            return spec, selection, None
    if not _complete([by_id[s["trial_id"]] for s in transfer]):
        return None, selection, "transfer_incomplete"

    # Holdout is opened only after all transfer validation trials, and only for
    # the small-width locked winner and baseline. It cannot change selection.
    test_lrs = list(dict.fromkeys([baseline, selection["winner_lr"]]))
    test = pairs("test", plan["widths"], test_lrs)
    for spec in test:
        if spec["trial_id"] not in by_id:
            parent_stage = "confirm" if spec["width"] == small else "transfer"
            spec["parent_trial_id"] = trial_spec(parent_stage, spec["width"], spec["seed"], spec["learning_rate"])["trial_id"]
            return spec, selection, None
    reason = "completed" if _complete([by_id[s["trial_id"]] for s in test]) else "test_incomplete"
    return None, selection, reason


def validate_result(result: dict, request: dict, plan: dict, directory: Path) -> dict:
    expected_device = plan["device"]
    if result.get("device") != expected_device:
        raise ValueError("Worker device differs from frozen plan")
    if request["stage"] == "test":
        if result.get("split") != "test" or result.get("checkpoint_step") != plan["training"]["max_iters"]:
            raise ValueError("Holdout did not evaluate the final fixed-budget checkpoint")
        metric = result["ce_loss"]
        tokens = result["tokens_evaluated"]
        expected = plan["training"]["eval_iters"] * plan["training"]["batch_size"] * plan["training"]["block_size"]
        if tokens != expected:
            raise ValueError("Incomplete held-out evaluation")
        extra = {}
    else:
        steps = plan["training"]["max_iters"]
        final = result["final_metric"]
        tokens = final["tokens_processed"]
        if (result.get("completed_steps") != steps or result.get("stop_reason") != "max_iters"
                or final["step"] != steps - 1
                or tokens != steps * plan["training"]["batch_size"] * plan["training"]["block_size"]):
            raise ValueError("Incomplete fixed-token trial; partial/best-checkpoint scores cannot rank")
        metric = final["val_ce_loss"]
        if request['config']['training'].get('lr_schedule') is not None:
            from omegaconf import OmegaConf
            from ..train.lr_schedule import schedule_contract, multiplier
            training = request['config']['training']
            schedule = schedule_contract(OmegaConf.create(training))
            metrics = result.get('metrics')
            if not isinstance(metrics, list) or not metrics or metrics[-1] != final:
                raise ValueError('Scheduled trial requires its complete metric records')
            expected_steps=sorted(set(range(0,steps,training['eval_every']))|{steps-1})
            if [row.get('step') for row in metrics]!=expected_steps:
                raise ValueError('Scheduled trial has missing or extra metric updates')
            previous = -1
            for row in metrics:
                step = row.get('step')
                if type(step) is not int or not previous < step < steps:
                    raise ValueError('Invalid scheduled metric update index')
                previous = step
                ratio = multiplier(schedule, step)
                values = {'lr_multiplier': ratio,
                          'base_learning_rate': training['learning_rate'],
                          'scheduled_learning_rate': training['learning_rate'] * ratio}
                for key, expected_value in values.items():
                    actual = row.get(key)
                    if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                            or not math.isfinite(actual)
                            or not math.isclose(actual, expected_value, rel_tol=0, abs_tol=1e-12)):
                        raise ValueError('Scheduled learning rate differs from frozen request')
        if request['config']['training'].get('weight_averaging') is not None:
            from omegaconf import OmegaConf
            from ..train.weight_averaging import averaging_contract
            training = request['config']['training']
            averaging = averaging_contract(OmegaConf.create(training))
            def count_at(completed):
                return 0 if completed < averaging['start_step'] else 1 + (completed-averaging['start_step']) // averaging['interval']
            expected = dict(spec=averaging, completed_steps=steps, snapshots=count_at(steps))
            if result.get('weight_averaging') != expected:
                raise ValueError('Averaging result metadata differs from frozen request')
            metrics = result.get('metrics')
            expected_steps=sorted(set(range(0,steps,training['eval_every']))|{steps-1})
            if not isinstance(metrics,list) or [row.get('step') for row in metrics]!=expected_steps:
                raise ValueError('Incomplete averaging metric history')
            for row in metrics:
                actual = row.get('averaging_snapshots')
                if type(actual) is not int or actual != count_at(row['step']+1):
                    raise ValueError('Averaging snapshot metric differs from frozen update index')
        checkpoint = Path(result["final_checkpoint"]).resolve()
        if not checkpoint.is_relative_to(directory.resolve()) or not checkpoint.is_file():
            raise ValueError("Missing or out-of-trial final checkpoint")
        extra = {"checkpoint": str(checkpoint), "checkpoint_sha256": file_digest(checkpoint),
                 "training_time_sec": result["training_time_sec"], "rss_mb": final["rss_mb"],
                 "accel_mem_mb": final["accel_mem_mb"], "accel_mem_kind": final["accel_mem_kind"]}
    if isinstance(metric, bool) or not isinstance(metric, (float, int)) or not math.isfinite(metric):
        raise ValueError("Nonfinite or invalid evaluation score")
    return {"metric": float(metric), "tokens_processed": tokens, **extra}


@contextlib.contextmanager
def session_lock(output: Path):
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("This study is already running (or its worker is still exiting)") from exc
        try:
            yield lock.fileno()
        finally:
            # Children inherit this descriptor, retaining the lock if the
            # controller is unexpectedly killed. Normal exit reaps the child.
            fcntl.flock(lock, fcntl.LOCK_UN)


def disk_mb(output: Path) -> float:
    return sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) / 1024**2


def terminate(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        process.wait()


def launch(request_path: Path, output: Path, timeout: float, disk_limit: float, lock_fd: int):
    started = time.monotonic()
    with (request_path.parent / "worker.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "src.research.worker", "--request", str(request_path)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, pass_fds=(lock_fd,))
        try:
            next_disk_check = started
            while process.poll() is None:
                now = time.monotonic()
                if now - started >= timeout:
                    terminate(process)
                    return "timeout", "Whole-process wall-time limit reached"
                if now >= next_disk_check:
                    if disk_mb(output) >= disk_limit:
                        terminate(process)
                        return "budget_exhausted", "Sampled output disk limit reached"
                    next_disk_check = now + 1.0
                time.sleep(0.1)
        except BaseException:
            terminate(process)
            raise
    if time.monotonic() - started > timeout:
        return "timeout", "Worker finished beyond its wall-time allowance"
    if disk_mb(output) >= disk_limit:
        return "budget_exhausted", "Sampled output disk limit reached"
    if process.returncode:
        return "failed", f"Worker exited {process.returncode}; see worker.log"
    return "completed", None


def write_reports(output: Path, state: dict):
    report = build_report(state["plan"], state["records"], state["selection"])
    report["session"] = {k: state.get(k) for k in ("stop_reason", "active_seconds", "contract")}
    atomic_json(output / "report.json", report)
    (output / "report.md").write_text(render_markdown(report) +
        f"\nSession: {state.get('stop_reason') or 'running'}; active seconds: {state['active_seconds']:.2f}.\n")


def regenerate_report(output: Path):
    with session_lock(output):
        state = json.loads((output / "state.json").read_text())
        write_reports(output, state)


def _git_metadata():
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"), "status_at_start": git("status", "--short")}


def run_study(plan: dict, output: Path) -> dict:
    plan = validate_plan(plan)
    preflight(plan)
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with session_lock(output) as lock_fd:
        state_path = output / "state.json"
        contract = evidence_contract(plan)
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["contract"] != contract or state["plan"] != plan:
                raise ValueError("Frozen plan, source, data or runtime changed. Start a new output directory.")
            # A crashed controller charged the full reserved timeout. Do not
            # pretend abandoned jobs succeeded or retry them without accounting.
            for record in state["records"]:
                if record["status"] == "running":
                    record.update(status="failed", error="Interrupted previous controller; no automatic retry")
        else:
            if any(p.name != ".lock" for p in output.iterdir()):
                raise ValueError("New output directory must be empty")
            state = {"version": 1, "plan": plan, "contract": contract, "git": _git_metadata(),
                     "records": [], "selection": {}, "active_seconds": 0.0, "stop_reason": None}
            atomic_json(output / "plan.json", plan)
            atomic_json(output / "manifest.json", {"contract": contract, "git": state["git"], "protocol": describe_plan(plan)})
        previous_seconds = state["active_seconds"]
        start = time.monotonic()

        def save(reserve=0.0):
            state["active_seconds"] = previous_seconds + time.monotonic() - start + reserve
            atomic_json(state_path, state)
            write_reports(output, state)

        state["stop_reason"] = None
        try:
            while True:
                if evidence_contract(plan) != state["contract"]:
                    state["stop_reason"] = "evidence_changed"
                    break
                spec, selection, reason = schedule_next(plan, state["records"], state["selection"])
                state["selection"] = selection
                if reason:
                    state["stop_reason"] = reason
                    break
                limits = plan["limits"]
                failed = sum(r["status"] != "completed" for r in state["records"])
                consecutive = 0
                for record in reversed(state["records"]):
                    if record["status"] == "completed":
                        break
                    consecutive += 1
                elapsed = previous_seconds + time.monotonic() - start
                remaining = limits["session_seconds"] - elapsed
                stop = ("max_trials" if len(state["records"]) >= limits["max_trials"] else
                        "max_failures" if failed >= limits["max_failures"] else
                        "max_consecutive_failures" if consecutive >= limits["max_consecutive_failures"] else
                        "session_time" if remaining <= 0 else
                        "disk_limit" if disk_mb(output) >= limits["max_disk_mb"] else None)
                if stop:
                    state["stop_reason"] = stop
                    break
                directory = output / "trials" / spec["trial_id"]
                directory.mkdir(parents=True, exist_ok=False)
                cfg = training_config(plan, spec, directory)
                request = {"stage": spec["stage"], "config": cfg, "cpu_threads": plan["training"]["cpu_threads"],
                           "result_path": str(directory / "result.json")}
                if spec["stage"] == "test":
                    parent = next(r for r in state["records"] if r["trial_id"] == spec["parent_trial_id"])
                    # Reuse the saved training config exactly; only the output
                    # for this evaluation is new. The trainer verifies contract.
                    parent_request = json.loads((output / "trials" / parent["trial_id"] / "request.json").read_text())
                    request.update(config=parent_request["config"], checkpoint=parent["checkpoint"],
                                   checkpoint_sha256=parent["checkpoint_sha256"])
                atomic_json(directory / "request.json", request)
                record = {**spec, "status": "running", "metric": None, "tokens_processed": 0,
                          "duration_seconds": 0.0, "error": None}
                state["records"].append(record)
                allowance = min(limits["trial_seconds"], max(0, limits["session_seconds"] - previous_seconds - (time.monotonic() - start)))
                save(reserve=allowance)
                print(f"[{len(state['records'])}/{limits['max_trials']}] {spec['stage']} width={spec['width']} seed={spec['seed']} lr={spec['learning_rate']}", flush=True)
                trial_start = time.monotonic()
                try:
                    status, error = launch(directory / "request.json", output, allowance, limits["max_disk_mb"], lock_fd)
                    record.update(status=status, error=error)
                    if evidence_contract(plan) != state["contract"]:
                        record.update(status="invalid", error="Source, data or runtime changed during trial")
                        state["stop_reason"] = "evidence_changed"
                    elif status == "completed":
                        try:
                            result = json.loads((directory / "result.json").read_text())
                            record.update(validate_result(result, request, plan, directory))
                        except (ValueError, KeyError, TypeError, OSError) as exc:
                            record.update(status="invalid", error=str(exc))
                except KeyboardInterrupt:
                    record.update(status="failed", error="Interrupted by user; no automatic retry")
                    state["stop_reason"] = "interrupted"
                    break
                except OSError as exc:
                    record.update(status="failed", error=str(exc))
                finally:
                    record["duration_seconds"] = time.monotonic() - trial_start
                save()
                print(f"  {record['status']}: {record['metric'] if record['metric'] is not None else record['error']}", flush=True)
                if state["stop_reason"] == "evidence_changed":
                    break
                if record["status"] == "budget_exhausted":
                    state["stop_reason"] = "disk_limit"
                    break
        finally:
            save()
        return state


def smoke_plan(output: Path) -> dict:
    """Synthetic data is deliberately separate from a real research conclusion."""
    corpus = output.parent / (output.name + "-synthetic.txt")
    content = "a quick brown fox jumps over the lazy dog.\n" * 100
    if not corpus.exists():
        corpus.parent.mkdir(parents=True, exist_ok=True)
        corpus.write_text(content)
    elif corpus.read_text() != content:
        raise ValueError("Existing smoke corpus differs; use a new output path")
    plan = load_plan(ROOT / "research/mac_lr_transfer.yaml")
    plan.update(name="synthetic-smoke", purpose="smoke", hypothesis="Exercise the protocol; no scientific conclusion.",
                device="cpu", widths=[8, 16], learning_rates=[0.001, 0.003], baseline_lr=0.001,
                seeds=[11, 22], min_improvement=0.0)
    plan["data"]["path"] = str(corpus.resolve())
    plan["model"].update(head_dim=4, n_layers=1)
    plan["training"].update(max_iters=2, batch_size=2, block_size=8, eval_every=2, eval_iters=2, cpu_threads=1)
    plan["limits"].update(max_trials=18, session_seconds=180, trial_seconds=20,
                          max_disk_mb=128, max_failures=1, max_consecutive_failures=1)
    return validate_plan(plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", type=Path, default=ROOT / "research/mac_lr_transfer.yaml")
        if name == "run":
            sub.add_argument("--output", type=Path, required=True)
    for name in ("report", "smoke"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            print(json.dumps(describe_plan(load_plan(args.config)), indent=2))
        elif args.command == "report":
            regenerate_report(args.output)
            print(args.output / "report.md")
        else:
            plan = smoke_plan(args.output.resolve()) if args.command == "smoke" else load_plan(args.config)
            state = run_study(plan, args.output)
            print(f"Study {state['stop_reason']}; report: {args.output.resolve() / 'report.md'}")
            if state["stop_reason"] != "completed":
                return 2
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
