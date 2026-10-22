from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: Optional[int] = None,
    cfg: Optional[Any] = None,
    training_state: Optional[dict] = None,
) -> None:
    ckpt_path = Path(path)
    ckpt = {"model_state": model.state_dict(), "model_math_version": 2}
    if training_state is not None:
        ckpt["training_state"] = training_state
    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()
    if step is not None:
        ckpt["step"] = step
    if cfg is not None:
        try:
            from omegaconf import OmegaConf

            ckpt["cfg"] = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            ckpt["cfg"] = cfg

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    torch.save(ckpt, temporary)
    temporary.replace(ckpt_path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str | torch.device] = None,
) -> dict:
    ckpt = read_checkpoint(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


def read_checkpoint(path, map_location='cpu'):
    """Reject pre-fix math rather than silently interpreting old weights differently."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if ckpt.get('model_math_version') != 2:
        raise ValueError(
            'Legacy/incompatible checkpoint: RoPE and MLA math changed in version 2. '
            'Use the original code revision for legacy inference; start a fresh run '
            'for corrected training. Automatic conversion is not lossless.'
        )
    return ckpt


def capture_training_state(loader, device, **progress):
    state = dict(progress)
    state['torch_rng'] = torch.get_rng_state()
    state['loader_rng'] = {k: g.get_state() for k, g in loader.generators.items()}
    if torch.device(device).type == 'cuda':
        state['cuda_rng'] = torch.cuda.get_rng_state_all()
    elif torch.device(device).type == 'mps':
        state['mps_rng'] = torch.mps.get_rng_state()
    return state


def restore_training_state(state, loader, device):
    torch.set_rng_state(state['torch_rng'].cpu())
    for key, value in state['loader_rng'].items():
        loader.generators[key].set_state(value.cpu())
    if torch.device(device).type == 'cuda':
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
    elif torch.device(device).type == 'mps':
        torch.mps.set_rng_state(state['mps_rng'].cpu())
