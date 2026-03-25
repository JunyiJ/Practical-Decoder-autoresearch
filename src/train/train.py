from pathlib import Path
import json
import math
import logging
import shutil
import subprocess
import sys
import time
import resource

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import torch
from datetime import datetime
from ..models.gpt import GPT
from ..data.loaders import TinyDataLoader, TextDataset
from ..utils.checkpoint import save_checkpoint
from ..moe.moe_block import MoEBlock

log = logging.getLogger(__name__)

def _get_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        rss_bytes = rss
    else:
        rss_bytes = rss * 1024
    return rss_bytes / (1024 * 1024)


def _get_accel_mem_mb(device: str) -> float | None:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    if device == "mps" and hasattr(torch, "mps"):
        current = getattr(torch.mps, "current_allocated_memory", None)
        if callable(current):
            return current() / (1024 * 1024)
    return None


def _get_cuda_device_index(device: str) -> int | None:
    if not device.startswith("cuda"):
        return None
    if ":" in device:
        _, _, suffix = device.partition(":")
        try:
            return int(suffix)
        except ValueError:
            return None
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return 0


def _get_cuda_usage(device: str) -> dict | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    if shutil.which("nvidia-smi") is None:
        return None
    device_index = _get_cuda_device_index(device)
    if device_index is None:
        return None
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpu_index = int(parts[0])
            utilization = float(parts[1])
            memory_used = float(parts[2])
            memory_total = float(parts[3])
        except ValueError:
            continue
        if gpu_index == device_index:
            return {
                "gpu_index": gpu_index,
                "utilization_pct": utilization,
                "memory_used_mb": memory_used,
                "memory_total_mb": memory_total,
            }
    return None


def _collect_moe_hists(model: GPT) -> list[tuple[int, torch.Tensor]]:
    hists = []
    for layer_idx, block in enumerate(model.transformer.h):
        mlp = getattr(block, "mlp", None)
        if isinstance(mlp, MoEBlock) and mlp.last_expert_hist is not None:
            hists.append((layer_idx, mlp.last_expert_hist.detach().to("cpu")))
    return hists


def _summarize_moe_usage(model: GPT) -> dict | None:
    train_hists = _collect_moe_hists(model)
    if not train_hists:
        return None

    stacked = torch.stack([hist for _, hist in train_hists], dim=0)
    aggregate_total = stacked.sum().item()
    if aggregate_total > 0:
        aggregate_usage = (stacked.sum(dim=0).float() / aggregate_total).tolist()
    else:
        aggregate_usage = []

    per_layer = []
    for layer_idx, hist in train_hists:
        layer_total = hist.sum().item()
        if layer_total > 0:
            layer_usage = (hist.float() / layer_total).tolist()
        else:
            layer_usage = []
        per_layer.append(
            {
                "layer_idx": layer_idx,
                "usage": [float(x) for x in layer_usage],
            }
        )

    return {
        "aggregate": [float(x) for x in aggregate_usage],
        "per_layer": per_layer,
    }


def _append_metric(metric_path: Path, metric: dict) -> None:
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    with metric_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(metric, sort_keys=True) + "\n")


def run_training(cfg: DictConfig) -> dict:
    eval_every = int(getattr(cfg.training, "eval_every", 400))
    eval_iters = int(getattr(cfg.training, "eval_iters", 100))
    if eval_every < 1:
        raise ValueError("cfg.training.eval_every must be >= 1")
    if eval_iters < 1:
        raise ValueError("cfg.training.eval_iters must be >= 1")
    metrics_file = getattr(cfg.training, "metrics_file", None)
    summary_file = getattr(cfg.training, "summary_file", None)
    metrics_path = Path(to_absolute_path(metrics_file)) if metrics_file else None
    summary_path = Path(to_absolute_path(summary_file)) if summary_file else None

    device = cfg.training.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    loader = TinyDataLoader(
        data_path=to_absolute_path(cfg.data.path),
        batch_size = cfg.training.batch_size,
        block_size = cfg.model.block_size,
        device = device
    )
    # The tiktokenizer doesn't work well for mini-shakspear dataset because the training
    # data is too limited to learn a big vocab as tiktokenizer
    # loader = TextDataset(
    #     data_path=to_absolute_path(cfg.data.path),
    #     batch_size = cfg.training.batch_size,
    #     block_size = cfg.model.block_size,
    #     device = device
    # )
    cfg.model.vocab_size = loader.vocab_size
    model = GPT(cfg.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    checkpoint_dir = to_absolute_path(cfg.training.checkpoint_dir)
    checkpoint_every = cfg.training.checkpoint_every
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    log.info("Starting training: %s + %s on %s", cfg.model.attn_type, cfg.model.mlp_type, device)

    start_time = time.perf_counter()
    metrics = []
    final_checkpoint_path = None
    for iter in range(cfg.training.max_iters):
        model.train()
        iter_start = time.perf_counter()
        xb, yb = loader.get_batch('train')
        logits, loss, train_loss_breakdown = model(xb, yb, return_loss_breakdown=True)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        iter_time = time.perf_counter() - iter_start

        if iter % eval_every == 0 or iter == cfg.training.max_iters - 1:
            elapsed = time.perf_counter() - start_time
            rss_mb = _get_rss_mb()
            accel_mb = _get_accel_mem_mb(device)
            cuda_usage = _get_cuda_usage(device)
            model.eval()
            eval_losses = 0.0
            eval_aux_losses = 0.0
            saw_eval_aux_loss = False
            with torch.no_grad():
                for _ in range(eval_iters):
                    x_eval, y_eval = loader.get_batch('val')
                    _, eval_loss, eval_loss_breakdown = model(
                        x_eval,
                        y_eval,
                        include_aux_loss=False,
                        return_loss_breakdown=True,
                    )
                    eval_losses += eval_loss.item()
                    if eval_loss_breakdown["aux_loss"] is not None:
                        saw_eval_aux_loss = True
                        eval_aux_losses += eval_loss_breakdown["aux_loss"].item()
            avg_eval_loss = eval_losses / eval_iters
            avg_eval_aux_loss = (eval_aux_losses / eval_iters) if saw_eval_aux_loss else None
            ppl = math.exp(avg_eval_loss)
            moe_usage = _summarize_moe_usage(model)
            train_ce_loss = train_loss_breakdown["ce_loss"]
            train_aux_loss = train_loss_breakdown["aux_loss"]

            if train_aux_loss is None and accel_mb is None:
                log.info(
                    "Step %d: Loss %.4f | val %.4f | ppl %.2f | iter %.3fs | elapsed %.1fs | rss %.1f MB",
                    iter,
                    loss.item(),
                    avg_eval_loss,
                    ppl,
                    iter_time,
                    elapsed,
                    rss_mb,
                )
            elif train_aux_loss is None:
                log.info(
                    "Step %d: Loss %.4f | val %.4f | ppl %.2f | iter %.3fs | elapsed %.1fs | rss %.1f MB | accel %.1f MB",
                    iter,
                    loss.item(),
                    avg_eval_loss,
                    ppl,
                    iter_time,
                    elapsed,
                    rss_mb,
                    accel_mb,
                )
            elif accel_mb is None:
                log.info(
                    "Step %d: Loss %.4f | ce %.4f | aux %.4f | val_ce %.4f | val_aux %.4f | ppl %.2f | iter %.3fs | elapsed %.1fs | rss %.1f MB",
                    iter,
                    loss.item(),
                    train_ce_loss.item(),
                    train_aux_loss.item(),
                    avg_eval_loss,
                    0.0 if avg_eval_aux_loss is None else avg_eval_aux_loss,
                    ppl,
                    iter_time,
                    elapsed,
                    rss_mb,
                )
            else:
                log.info(
                    "Step %d: Loss %.4f | ce %.4f | aux %.4f | val_ce %.4f | val_aux %.4f | ppl %.2f | iter %.3fs | elapsed %.1fs | rss %.1f MB | accel %.1f MB",
                    iter,
                    loss.item(),
                    train_ce_loss.item(),
                    train_aux_loss.item(),
                    avg_eval_loss,
                    0.0 if avg_eval_aux_loss is None else avg_eval_aux_loss,
                    ppl,
                    iter_time,
                    elapsed,
                    rss_mb,
                    accel_mb,
                )
            if cuda_usage is not None:
                log.info(
                    "CUDA usage (gpu %d): util %.1f%% | used %.1f / %.1f MB",
                    cuda_usage["gpu_index"],
                    cuda_usage["utilization_pct"],
                    cuda_usage["memory_used_mb"],
                    cuda_usage["memory_total_mb"],
                )
            if moe_usage is not None:
                usage_str = ", ".join(f"{u:.3f}" for u in moe_usage["aggregate"]) or "n/a"
                log.info("MoE expert usage (train batch, top-k assignments): %s", usage_str)
                for layer_usage in moe_usage["per_layer"]:
                    layer_usage_str = ", ".join(f"{u:.3f}" for u in layer_usage["usage"]) or "n/a"
                    log.info("MoE expert usage (layer %d): %s", layer_usage["layer_idx"], layer_usage_str)

            metric = {
                "step": iter,
                "train_loss": float(loss.item()),
                "train_total_loss": float(loss.item()),
                "train_ce_loss": float(train_ce_loss.item()),
                "train_aux_loss": None if train_aux_loss is None else float(train_aux_loss.item()),
                "val_loss": float(avg_eval_loss),
                "val_ce_loss": float(avg_eval_loss),
                "val_aux_loss": None if avg_eval_aux_loss is None else float(avg_eval_aux_loss),
                "ppl": float(ppl),
                "iter_time_sec": float(iter_time),
                "elapsed_time_sec": float(elapsed),
                "rss_mb": float(rss_mb),
                "accel_mem_mb": None if accel_mb is None else float(accel_mb),
                "device": device,
                "cuda": cuda_usage,
                "moe_usage": moe_usage,
            }
            metrics.append(metric)
            if metrics_path is not None:
                _append_metric(metrics_path, metric)
        if checkpoint_dir and checkpoint_every > 0 and (iter + 1) % checkpoint_every == 0:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_path = Path(checkpoint_dir) / f"step_{iter + 1:06d}_{stamp}.pt"
            save_checkpoint(
                ckpt_path,
                model,
                optimizer=optimizer,
                step=iter + 1,
                cfg=cfg,
            )
            final_checkpoint_path = str(ckpt_path)
    if checkpoint_dir:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_path = Path(checkpoint_dir) / f"final_{stamp}.pt"
        save_checkpoint(
            ckpt_path,
            model,
            optimizer=optimizer,
            step=cfg.training.max_iters,
            cfg=cfg,
        )
        final_checkpoint_path = str(ckpt_path)

    best_metric = min(metrics, key=lambda item: item["val_loss"]) if metrics else None
    result = {
        "device": device,
        "metrics": metrics,
        "best_metric": best_metric,
        "final_metric": metrics[-1] if metrics else None,
        "final_checkpoint": final_checkpoint_path,
        "max_iters": int(cfg.training.max_iters),
        "eval_every": eval_every,
        "eval_iters": eval_iters,
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
    return result


@hydra.main(version_base=None, config_path="../config", config_name="mac_tinyshakespeare")
def train(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    train()
