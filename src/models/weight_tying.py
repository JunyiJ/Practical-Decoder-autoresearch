"""Prospective scaled table sharing; metadata helpers never import torch.

The matched control changes readout coordinates as well as initialization.
Sharing then couples the two gradient roles and their adaptive state. These
contracts do not establish optimal muP transfer or local quality improvement.
"""
from collections.abc import Mapping
import math

POLICY = 'mha4_scaled_table_sharing'
KINDS = ('legacy_untied', 'matched_untied', 'scaled_tied')
FIELDS = ('weight_tying_kind', 'weight_tying_width_policy')


def _get(cfg, key, default=None):
    return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)


def _has(cfg, key):
    return key in cfg if isinstance(cfg, Mapping) else hasattr(cfg, key)


def weight_tying_contract(cfg):
    if not any(_has(cfg, k) for k in FIELDS): return None
    if (_get(cfg, FIELDS[0]) not in KINDS or _get(cfg, FIELDS[1]) != POLICY):
        raise ValueError('Table sharing requires its explicit kind and policy')
    from .output_bias import output_bias_contract
    if output_bias_contract(cfg) is None:
        raise ValueError('Table sharing requires explicit inherited bias/prior and normalization')
    if any(_has(cfg, k) for k in ('readout_scale', 'weight_tying_scale', 'shared_table_scale',
                                 'output_init_std', 'readout_init_std')):
        raise ValueError('Table sharing rejects independent scale or initialization overrides')
    # Inherited readout contract already binds dense MHA4/head32/Swi21D/8,
    # RMS blocks/RoPE/unscaled/base64/width64,128,256 and rejects old tie flags.
    return dict(weight_tying_kind=_get(cfg, FIELDS[0]), weight_tying_width_policy=POLICY)


def weight_tying_signature(cfg):
    c = weight_tying_contract(cfg)
    return None if c is None else (c['weight_tying_kind'], POLICY, 64)


def readout_scale(cfg):
    c = weight_tying_contract(cfg)
    return 1. if c is None or c['weight_tying_kind'] == 'legacy_untied' else 1. / math.sqrt(3 * 64)


def validate_weight_tying_model(model):
    cfg = getattr(model, 'cfg', {})
    signature = weight_tying_signature(cfg)
    if getattr(model, 'weight_tying_signature', None) != signature:
        raise ValueError('Constructed table-sharing semantics differ from mutable config')
    if getattr(model, 'table_readout_scale', 1.) != readout_scale(cfg):
        raise ValueError('Constructed table readout scale differs from fixed contract')
    if signature is None: return
    kind = signature[0]
    table = model.transformer.wte.weight
    head = getattr(model, 'lm_head', None)
    if kind == 'scaled_tied':
        if head is not None:
            raise ValueError('Shared table must have no separately registered readout')
    elif head is None or head.weight is table:
        raise ValueError('Untied table requires a distinct readout parameter')
    # remove_duplicate=False exposes accidental aliases instead of hiding them.
    pairs = list(model.named_parameters(remove_duplicate=False))
    names = [name for name, _ in pairs]
    ids = [id(p) for _, p in pairs]
    if len(ids) != len(set(ids)) or len(names) != len(set(names)):
        raise ValueError('Table sharing requires unique canonical parameter entries')
    if names.count('transformer.wte.weight') != 1:
        raise ValueError('Missing canonical embedding table')
    if ('lm_head.weight' in names) != (kind != 'scaled_tied'):
        raise ValueError('Readout parameter registration differs from table policy')


def validate_weight_tying_load(model, checkpoint):
    current = getattr(model, 'cfg', {})
    saved = _get(checkpoint.get('cfg', {}), 'model', {})
    if weight_tying_contract(current) != weight_tying_contract(saved):
        raise ValueError('Checkpoint table-sharing semantics differ')
    validate_weight_tying_model(model)
    c = weight_tying_contract(current)
    if c is None: return
    keys = checkpoint.get('model_state', {})
    if ('transformer.wte.weight' not in keys or
            ('lm_head.weight' in keys) != (c['weight_tying_kind'] != 'scaled_tied')):
        raise ValueError('Checkpoint table layout is not canonical')
