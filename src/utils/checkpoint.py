from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import warnings


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: Optional[int] = None,
    cfg: Optional[Any] = None,
    training_state: Optional[dict] = None,
) -> None:
    ckpt_path = Path(path)
    ckpt = {"model_state": model.state_dict(), "model_math_version": 2, "parameter_layout_version": 2}
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
    load_model_weights(model, ckpt, for_resume=optimizer is not None)
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


def load_model_weights(model, checkpoint, *, for_resume=False):
    """Load strictly; discard only the historical unused RoPE position table.

    This in-memory inference adapter never rewrites the checkpoint. Its active
    weights and forward math are unchanged. Legacy optimizer parameter groups
    include the removed table, so training continuation requires original source.
    """
    from ..models.embedding_scale import validate_input_scale_load
    validate_input_scale_load(model, checkpoint)
    from ..models.readout_norm import validate_readout_norm_load
    validate_readout_norm_load(model, checkpoint)
    if for_resume and 'averaged_weights' in checkpoint:
        raise ValueError('Averaged inference weights cannot resume the online optimizer; use the raw parent checkpoint')
    state = checkpoint['model_state']
    key = 'transformer.wpe.weight'
    if getattr(model, 'enable_rope', False) and key in state:
        if for_resume:
            raise ValueError('Legacy RoPE checkpoint includes an unused position table; resume with its original source. New training must use a new run.')
        saved = checkpoint.get('cfg', {}).get('model', {})
        if saved.get('rope') != 1 or tuple(state[key].shape) != (model.cfg.block_size, model.cfg.dim):
            raise ValueError('Unexpected legacy RoPE position-table contract')
        state = dict(state)
        del state[key]
        warnings.warn('Loading historical RoPE weights without the unused position table; checkpoint is unchanged.', UserWarning, stacklevel=2)
    return model.load_state_dict(state, strict=True)


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
