"""One disposable process per training run or held-out evaluation."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .protocol import file_digest
from ..train.train import run_training, evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    torch.set_num_threads(request["cpu_threads"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = OmegaConf.create(request["config"])
    if request["stage"] == "test":
        checkpoint = Path(request["checkpoint"])
        if file_digest(checkpoint) != request["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed since its training trial")
        result = evaluate_checkpoint(cfg, checkpoint, split="test")
    else:
        result = run_training(cfg)
    result["runtime"] = {"torch": str(torch.__version__), "cpu_threads": torch.get_num_threads(),
                         "requested_device": str(cfg.training.device)}
    target = Path(request["result_path"])
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)


if __name__ == "__main__":
    main()
