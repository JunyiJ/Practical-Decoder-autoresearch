"""The deliberately narrow v1 experiment contract: dense AdamW LR transfer."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def candidate_id(lr: float) -> str:
    # Hex represents the actual float, without collisions from rounded labels.
    return "lr_" + float(lr).hex().replace(".", "_").replace("+", "p").replace("-", "m")


def _keys(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    if set(value) != set(allowed):
        raise ValueError(f"{label}: missing {set(allowed) - set(value)}, unknown {set(value) - set(allowed)}")


def _positive(value, label, integer=False, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{label} must be finite and {'nonnegative' if zero else 'positive'}")
    if integer and (not isinstance(value, int)):
        raise ValueError(f"{label} must be an integer")


def validate_plan(plan: dict) -> dict:
    plan = copy.deepcopy(plan)
    _keys(plan, {"version", "purpose", "name", "hypothesis", "device", "data", "model", "training",
                 "widths", "learning_rates", "baseline_lr", "seeds", "screen_seed",
                 "eval_seed", "min_improvement", "limits"}, "plan")
    if plan["version"] != 1:
        raise ValueError("Only protocol version 1 is supported")
    if plan["purpose"] not in ("research", "smoke"):
        raise ValueError("purpose must be research or smoke")
    if not isinstance(plan["name"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", plan["name"]):
        raise ValueError("name must contain only letters, numbers, underscores and hyphens")
    if not isinstance(plan["hypothesis"], str) or not plan["hypothesis"].strip():
        raise ValueError("Write a hypothesis before starting a study")
    if not isinstance(plan["device"], str) or not re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", plan["device"]):
        raise ValueError("device must be cpu, mps, cuda or cuda:N")
    _keys(plan["data"], {"path", "train_fraction", "val_fraction"}, "data")
    for key in ("train_fraction", "val_fraction"):
        _positive(plan["data"][key], f"data.{key}")
    if sum(plan["data"][k] for k in ("train_fraction", "val_fraction")) >= 1:
        raise ValueError("Reserve a nonempty test fraction; train_fraction + val_fraction < 1")
    path = Path(plan["data"]["path"]).expanduser()
    plan["data"]["path"] = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
    _keys(plan["model"], {"n_layers", "head_dim", "dropout", "norm_type"}, "model")
    for key in ("n_layers", "head_dim"):
        _positive(plan["model"][key], f"model.{key}", integer=True)
    _positive(plan["model"]["dropout"], "model.dropout", zero=True)
    if plan["model"]["dropout"] >= 1 or plan["model"]["head_dim"] % 2:
        raise ValueError("dropout must be < 1 and head_dim must be even for RoPE")
    if plan["model"]["norm_type"] not in ("rmsnorm", "layernorm"):
        raise ValueError("Unsupported norm_type")
    _keys(plan["training"], {"max_iters", "batch_size", "block_size", "weight_decay", "grad_clip",
                            "eval_every", "eval_iters", "cpu_threads"}, "training")
    for key, value in plan["training"].items():
        _positive(value, f"training.{key}", integer=key not in ("weight_decay", "grad_clip"),
                  zero=key == "weight_decay")
    for field, minimum in (("widths", 2), ("learning_rates", 2), ("seeds", 2)):
        values = plan[field]
        if not isinstance(values, list) or len(values) < minimum or len(set(values)) != len(values):
            raise ValueError(f"{field} must contain at least {minimum} distinct values")
        for value in values:
            _positive(value, field, integer=field != "learning_rates", zero=field == "seeds")
    if plan["widths"] != sorted(plan["widths"]) or any(w % plan["model"]["head_dim"] for w in plan["widths"]):
        raise ValueError("widths must increase and be divisible by head_dim")
    _positive(plan["baseline_lr"], "baseline_lr")
    if plan["baseline_lr"] not in plan["learning_rates"]:
        raise ValueError("baseline_lr must be in learning_rates")
    for key in ("screen_seed", "eval_seed"):
        _positive(plan[key], key, integer=True, zero=True)
    if plan["screen_seed"] in plan["seeds"]:
        raise ValueError("Confirmation seeds must be held out from screening")
    if max(plan["seeds"] + [plan["screen_seed"], plan["eval_seed"]]) >= 2**63 - 2:
        raise ValueError("Seeds must be below 2**63 - 2")
    _positive(plan["min_improvement"], "min_improvement", zero=True)
    _keys(plan["limits"], {"max_trials", "session_seconds", "trial_seconds", "max_disk_mb",
                          "max_failures", "max_consecutive_failures"}, "limits")
    for key, value in plan["limits"].items():
        _positive(value, f"limits.{key}", integer=key in ("max_trials", "max_failures", "max_consecutive_failures"))
    return plan


def load_plan(path: str | Path) -> dict:
    return validate_plan(OmegaConf.to_container(OmegaConf.load(path), resolve=True))


def describe_plan(plan: dict) -> dict:
    seeds, widths = len(plan["seeds"]), len(plan["widths"])
    training_trials = len(plan["learning_rates"]) + 2 * seeds * widths
    return {"name": plan["name"], "purpose": plan["purpose"], "hypothesis": plan["hypothesis"],
            "widths": plan["widths"], "heads": [w // plan["model"]["head_dim"] for w in plan["widths"]],
            "training_trials": training_trials, "test_evaluations_max": 2 * seeds * widths,
            "jobs_max": training_trials + 2 * seeds * widths,
            "training_character_tokens_per_trial": plan["training"]["max_iters"] * plan["training"]["batch_size"] * plan["training"]["block_size"],
            "objective": "final validation CE in nats/character at equal training tokens",
            "parameterization": "standard PyTorch initialization; NOT muP",
            "optimizer": "AdamW, constant LR, default betas/eps; decay on all parameters with gradients",
            "device": plan["device"], "limits": plan["limits"]}


def trial_spec(stage: str, width: int, seed: int, lr: float) -> dict:
    candidate = candidate_id(lr)
    return {"trial_id": f"{stage}_w{width}_s{seed}_{candidate}", "stage": stage,
            "width": width, "seed": seed, "learning_rate": lr, "candidate": candidate}


def training_config(plan: dict, trial: dict, directory: Path) -> dict:
    """Exactly one architecture family; research changes only LR and width."""
    train = plan["training"]
    return {"data": plan["data"], "model": {
        "dim": trial["width"], "n_layers": plan["model"]["n_layers"],
        "n_heads": trial["width"] // plan["model"]["head_dim"], "vocab_size": 1,
        "block_size": train["block_size"], "dropout": plan["model"]["dropout"],
        "norm_type": plan["model"]["norm_type"], "rope": 1, "attn_type": "mha", "mlp_type": "dense", "ffn": "gelu",
    }, "training": {
        **{k: v for k, v in train.items() if k not in ("block_size", "cpu_threads")},
        "device": plan["device"], "strict_device": True, "seed": trial["seed"],
        "eval_seed": plan["eval_seed"], "learning_rate": trial["learning_rate"],
        "max_train_seconds": None, "resume_from": None, "keep_best_checkpoint": False,
        "checkpoint_dir": str(directory / "checkpoints"), "checkpoint_every": 0,
        "metrics_file": str(directory / "metrics.jsonl"), "summary_file": str(directory / "training_summary.json"),
    }}
