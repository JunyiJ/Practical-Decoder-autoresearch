"""Explicit final-normalization semantics for a prospective MHA quality round.

This is a separate unlaunched prototype. Matching shapes do not imply matching
normalization, and declared width factors do not establish optimal LR transfer.
"""
from collections.abc import Mapping
import math

POLICY = 'mha4_readout_norm'
EPSILON = 1e-5
KINDS = ('layer_affine', 'layer_no_bias', 'rms')
FIELDS = ('readout_norm_kind', 'readout_norm_width_policy', 'readout_norm_epsilon')


def _get(cfg, key, default=None):
    return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)


def _has(cfg, key):
    return key in cfg if isinstance(cfg, Mapping) else hasattr(cfg, key)


def readout_norm_contract(cfg):
    if not any(_has(cfg, key) for key in FIELDS):
        return None
    kind, epsilon = _get(cfg, FIELDS[0]), _get(cfg, FIELDS[2])
    if (kind not in KINDS or _get(cfg, FIELDS[1]) != POLICY
            or type(epsilon) not in (int, float) or not math.isfinite(epsilon)
            or epsilon != EPSILON):
        raise ValueError('Final normalization requires the explicit reviewed kind, policy and epsilon')
    expected = dict(n_layers=4, mlp_type='dense', ffn='swiglu', ffn_bias=False,
        ffn_width_policy='swiglu_21_8', depth_width_policy='depth4_unscaled',
        attn_type='mha', attention_width_policy='mha_head32',
        parameterization='mup', base_width=64, norm_type='rmsnorm',
        rope=1, dropout=0., attn_dropout=0., block_size=128,
        input_width_policy='mha4_input_scale')
    if any(_get(cfg, key) != value for key, value in expected.items()):
        raise ValueError('Final normalization differs from its reviewed backbone')
    scale = _get(cfg, 'input_embedding_scale')
    if (type(scale) not in (int, float) or not math.isfinite(scale)
            or scale not in (1., .5, .25)):
        raise ValueError('Final normalization requires an explicit inherited input scale')
    if (any(type(_get(cfg, key)) is not int for key in ('n_layers', 'dim', 'n_heads', 'base_width'))
            or _get(cfg, 'dim') not in (64, 128, 256)
            or _get(cfg, 'dim') != 32*_get(cfg, 'n_heads')
            or _get(cfg, 'ffn_bias') is not False):
        raise ValueError('Final normalization shape/head/layer contract changed')
    forbidden = ('hidden_dim', 'n_kv_heads', 'n_groups', 'moe', 'dimckv', 'dimcq',
        'head_dim_r', 'attn_bias', 'causal', 'residual_scale', 'residual_scaling',
        'position_encoding', 'tie_weights', 'tie_embeddings', 'embedding_init_std',
        'embedding_norm', 'qk_norm', 'qk_norm_width_policy', 'attention_output_gate',
        'output_gate_width_policy', 'ffn_hidden_norm', 'ffn_norm_width_policy',
        'final_norm', 'final_norm_type', 'norm_eps', 'layer_norm_epsilon')
    if any(_has(cfg, key) for key in forbidden):
        raise ValueError('Final normalization rejects unreviewed combined overrides')
    return dict(readout_norm_kind=kind, readout_norm_width_policy=POLICY,
                readout_norm_epsilon=float(epsilon))


def readout_norm_signature(cfg):
    contract = readout_norm_contract(cfg)
    return ('layer_affine', EPSILON) if contract is None else (
        contract['readout_norm_kind'], contract['readout_norm_epsilon'])


def build_readout_norm(cfg):
    from torch import nn
    contract = readout_norm_contract(cfg)
    if contract is None:
        return nn.LayerNorm(cfg.dim)
    kind = contract['readout_norm_kind']
    if kind == 'rms':
        return nn.RMSNorm(cfg.dim, eps=EPSILON, elementwise_affine=True)
    return nn.LayerNorm(cfg.dim, eps=EPSILON, elementwise_affine=True,
                        bias=kind == 'layer_affine')


def validate_readout_norm_load(model, checkpoint):
    current_cfg = getattr(model, 'cfg', {})
    saved_cfg = _get(checkpoint.get('cfg', {}), 'model', {})
    if readout_norm_contract(current_cfg) != readout_norm_contract(saved_cfg):
        raise ValueError('Checkpoint final-normalization semantics differ from model')
    if getattr(model, 'readout_norm_signature', ('layer_affine', EPSILON)) != readout_norm_signature(current_cfg):
        raise ValueError('Constructed final normalization differs from mutable model config')
