"""Full training/checkpoint identity for the fixed AdamW beta2 comparison."""
from collections.abc import Mapping
from omegaconf import OmegaConf
from research.readout_centering_model_v1.integration import identity as inherited_identity
from .runtime.train.optimizers import beta2_contract, build_optimizer as construct
from .runtime.utils.checkpoint import load_model_weights as inherited_load

def identity(cfg):
    parent = inherited_identity(cfg)
    if parent['readout']['kind'] != 'layer_affine':
        raise ValueError('Incumbent final LayerNorm required')
    return dict(parent, policy='fixed_lr_adamw_beta2_v1', adamw=beta2_contract(cfg.training))

def build_optimizer(model, cfg):
    expected=identity(cfg)
    result=construct(model,cfg.training)
    if result.decay != expected['decay'] or any(g['weight_decay'] != 0. for g in result.param_groups):
        raise ValueError('Exact independent decay and zero child decay required')
    return result

def load_model_weights(model, payload, cfg, *, for_resume=False):
    expected=identity(cfg)
    if not isinstance(payload, Mapping) or 'cfg' not in payload or identity(OmegaConf.create(payload['cfg'])) != expected:
        raise ValueError('Foreign AdamW beta2 training identity')
    # The namespace-local resume preflight independently constructs the declared
    # optimizer and validates every saved group/rate/state before loading weights.
    return inherited_load(model,payload,for_resume=for_resume)
