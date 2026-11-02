"""Narrow instrumented trainer; frozen incumbent helpers and model math are reused."""
import json, logging, math, time
from pathlib import Path
import torch
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from research.readout_centering_model_v1.runtime.models.gpt import GPT
from research.readout_centering_model_v1.runtime.train.train import (
    _resolve_device, _make_loader, _training_contract as inherited_contract,
    _contracts_match, _evaluate, _get_rss_mb, _get_accel_mem_mb,
    _get_cuda_usage, _summarize_moe_usage, _append_metric,
)
from research.readout_centering_model_v1.runtime.train.lr_schedule import LearningRateSchedule
from research.readout_centering_model_v1.runtime.train.weight_averaging import TailWeightAverages
from research.readout_centering_model_v1.runtime.utils.device import synchronize_device
from research.readout_centering_model_v1.runtime.utils.checkpoint import (
    save_checkpoint, read_checkpoint, capture_training_state, restore_training_state,
)
from .integration import identity, build_optimizer, load_model_weights
from research.gradient_clip_model_v1.clipping import clip_gradients
log = logging.getLogger(__name__)


def _training_contract(cfg, device, loader):
    return dict(inherited_contract(cfg, device, loader), readout_centering=identity(cfg))


def validate_history(rows, completed_steps, cfg):
    threshold = identity(cfg)['grad_clip']
    if type(rows) is not list or len(rows) != completed_steps or not 0 <= completed_steps <= cfg.training.max_iters:
        raise ValueError('Complete bounded per-update clipping history required')
    for step, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {'step','threshold','pre_norm','post_norm','coefficient','gradient_tensors'}:
            raise ValueError('Exact clipping observation fields required')
        if type(row['step']) is not int or type(row['gradient_tensors']) is not int or row['step'] != step or row['threshold'] != threshold or row['gradient_tensors'] != 88:
            raise ValueError('Clipping history identity or sequence differs')
        if not all(type(row[k]) is float and math.isfinite(row[k]) for k in ('pre_norm','post_norm','coefficient')):
            raise ValueError('Finite clipping measurements required')
        pre, post, coef = (row[k] for k in ('pre_norm','post_norm','coefficient'))
        if pre < 0 or post < 0 or not 0 <= coef <= 1 or post > threshold + 1e-5:
            raise ValueError('Invalid clipping observation range')
        expected = min(1., threshold / (pre + 1e-6))
        if abs(coef-expected) > 2e-5 or abs(post-pre*coef) > max(1e-5,post*2e-5):
            raise ValueError('Clipping observation arithmetic differs')
    return [dict(row) for row in rows]

