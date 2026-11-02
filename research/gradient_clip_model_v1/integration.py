"""Threshold is training/checkpoint identity, not a change to model or cache math."""
from collections.abc import Mapping
from omegaconf import OmegaConf
from research.branch_scale_model_v1.integration import (
    identity as inherited_identity, build_optimizer as inherited_optimizer,
    load_model_weights as inherited_load,
)


def identity(cfg):
    result = inherited_identity(cfg)
    if result['branch_gain']['kind'] != 'unit':
        raise ValueError('Clipping comparison requires incumbent unit residual gains')
    threshold = cfg.training.grad_clip
    if type(threshold) is not float or threshold not in (1.0, 0.5):
        raise ValueError('Explicit float global clipping threshold1.0 or0.5 required')
    return dict(result, policy='fixed_lr_global_gradient_clip_v1',
                grad_clip=threshold, norm_type=2.0, epsilon=1e-6,
                error_if_nonfinite=True, order='backward_clip_schedule_decay_children_ema')


def build_optimizer(model, cfg):
    identity(cfg)
    return inherited_optimizer(model, cfg)


def load_model_weights(model, payload, cfg, *, for_resume=False):
    expected = identity(cfg)
    if not isinstance(payload, Mapping) or 'cfg' not in payload:
        raise ValueError('Saved full clipping configuration required')
    if identity(OmegaConf.create(payload['cfg'])) != expected:
        raise ValueError('Cross-threshold clipping checkpoint identity differs')
    return inherited_load(model, payload, cfg, for_resume=for_resume)
