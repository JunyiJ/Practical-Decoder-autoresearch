"""Explicit input-coordinate intervention; no change to legacy model math.

The scalar is width independent. This preserves parameter shapes and the named
optimizer exponents, not a claim of optimal learning-rate transfer.
"""
from collections.abc import Mapping
import math


POLICY = 'mha4_input_scale'


def _get(cfg, key, default=None):
    return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)


def _has(cfg, key):
    return key in cfg if isinstance(cfg, Mapping) else hasattr(cfg, key)


def input_scale_contract(cfg):
    fields = ('input_embedding_scale', 'input_width_policy')
    if not any(_has(cfg, k) for k in fields):
        return None
    scale = _get(cfg, 'input_embedding_scale')
    if (type(scale) not in (int, float) or not math.isfinite(scale)
            or scale not in (1., .5, .25) or _get(cfg, 'input_width_policy') != POLICY):
        raise ValueError('Input embedding scale requires the explicit reviewed policy and scalar')
    expected = dict(n_layers=4, mlp_type='dense', ffn='swiglu', ffn_bias=False,
        ffn_width_policy='swiglu_21_8', depth_width_policy='depth4_unscaled',
        attn_type='mha', attention_width_policy='mha_head32',
        parameterization='mup', base_width=64, norm_type='rmsnorm',
        rope=1, dropout=0., attn_dropout=0., block_size=128)
    if any(_get(cfg, k) != v for k, v in expected.items()):
        raise ValueError('Input embedding scale recipe differs from its explicit contract')
    if (any(type(_get(cfg, k)) is not int for k in ('n_layers','dim','n_heads','base_width'))
            or _get(cfg,'dim') not in (64,128,256)
            or _get(cfg,'dim') != 32*_get(cfg,'n_heads') or _get(cfg,'ffn_bias') is not False):
        raise ValueError('Input embedding scale shape/head/layer contract changed')
    forbidden = ('hidden_dim','n_kv_heads','n_groups','moe','dimckv','dimcq',
        'head_dim_r','attn_bias','causal','residual_scale','residual_scaling',
        'position_encoding','tie_weights','tie_embeddings','embedding_init_std',
        'embedding_norm','qk_norm','qk_norm_width_policy','attention_output_gate',
        'output_gate_width_policy','ffn_hidden_norm','ffn_norm_width_policy')
    if any(_has(cfg, k) for k in forbidden):
        raise ValueError('Input embedding scale rejects unreviewed combined overrides')
    if any(_has(cfg, k) for k in ('readout_norm_kind', 'readout_norm_width_policy', 'readout_norm_epsilon')):
        from .readout_norm import readout_norm_contract
        readout_norm_contract(cfg)
    return dict(input_width_policy=POLICY, input_embedding_scale=float(scale))


def resolve_input_scale(cfg):
    spec = input_scale_contract(cfg)
    return 1. if spec is None else spec['input_embedding_scale']


def validate_input_scale_load(model, checkpoint):
    """Same-shaped tensors must not silently acquire a different input scalar."""
    current_cfg = getattr(model, 'cfg', {})
    saved_cfg = checkpoint.get('cfg', {})
    saved_cfg = _get(saved_cfg, 'model', {})
    current = input_scale_contract(current_cfg)
    saved = input_scale_contract(saved_cfg)
    if current != saved:
        raise ValueError('Checkpoint input embedding scale contract differs from model')
    expected_scale = 1. if current is None else current['input_embedding_scale']
    if getattr(model, 'input_embedding_scale', 1.) != expected_scale:
        raise ValueError('Constructed input embedding scale differs from mutable model config')
