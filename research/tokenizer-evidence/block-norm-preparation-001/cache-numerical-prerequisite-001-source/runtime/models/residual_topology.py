"""Separate prospective serial/parallel block contract; no tensor imports.

Parallel describes the dependency graph, not concurrent kernels. Both existing
normalizers remain separate. No parameter or empirical muP benefit is implied.
"""
from collections.abc import Mapping

POLICY = 'mha4_separate_norm_residual_topology'
KINDS = ('serial', 'parallel_separate_norms')
FIELDS = ('residual_topology_kind', 'residual_topology_width_policy')


def _get(cfg, key, default=None):
    return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)


def _has(cfg, key):
    return key in cfg if isinstance(cfg, Mapping) else hasattr(cfg, key)


def residual_topology_contract(cfg):
    if not any(_has(cfg, k) for k in FIELDS):
        return None
    if (_get(cfg, FIELDS[0]) not in KINDS or _get(cfg, FIELDS[1]) != POLICY):
        raise ValueError('Residual topology requires explicit reviewed kind and policy')
    from .weight_tying import weight_tying_contract
    if weight_tying_contract(cfg) is None:
        raise ValueError('Residual topology requires explicit final inherited table and readout contracts')
    if any(_has(cfg, k) for k in ('parallel_residual', 'parallel_attn', 'shared_block_norm',
                                 'residual_branch_scale', 'ffn_input_scale')):
        raise ValueError('Residual topology rejects unreviewed branch overrides')
    return dict(residual_topology_kind=_get(cfg, FIELDS[0]), residual_topology_width_policy=POLICY)


def residual_topology_signature(cfg):
    c = residual_topology_contract(cfg)
    return None if c is None else (c['residual_topology_kind'], POLICY)


def resolve_residual_topology(cfg):
    c = residual_topology_contract(cfg)
    return 'serial' if c is None else c['residual_topology_kind']


def validate_residual_topology_model(model):
    cfg = getattr(model, 'cfg', {})
    signature = residual_topology_signature(cfg)
    if getattr(model, 'residual_topology_signature', None) != signature:
        raise ValueError('Constructed residual topology differs from mutable config')
    blocks = getattr(getattr(model, 'transformer', None), 'h', ())
    if signature is not None and len(blocks) != 4:
        raise ValueError('Residual topology requires four reviewed blocks')
    kind = resolve_residual_topology(cfg)
    for block in blocks:
        if getattr(block, 'residual_topology_kind', 'serial') != kind:
            raise ValueError('Block topology differs from model contract')
        if signature is not None and block.ln1 is block.ln2:
            raise ValueError('Parallel topology retains separate block normalizers')


def validate_residual_topology_load(model, checkpoint):
    current = getattr(model, 'cfg', {})
    saved = _get(checkpoint.get('cfg', {}), 'model', {})
    if residual_topology_contract(current) != residual_topology_contract(saved):
        raise ValueError('Checkpoint residual topology differs despite matching tensor shapes')
    validate_residual_topology_model(model)
