"""Narrow width256 tokenizer MLA gate comparison; historical policies unchanged."""
from collections.abc import Mapping

POLICY = 'tokenizer_mla_gate_v1'
KINDS = ('silu', 'gelu_exact')
FIELDS = ('gated_activation_kind', 'gated_activation_width_policy')


def get(cfg, key, default=None):
    return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)


def activation_contract(cfg):
    if get(cfg, 'tokenizer_architecture_policy') != POLICY:
        return None
    kind = get(cfg, FIELDS[0])
    if type(kind) is not str or kind not in KINDS or get(cfg, FIELDS[1]) != POLICY:
        raise ValueError('Tokenizer MLA gate requires explicit silu/gelu_exact kind and policy')
    from .tokenizer_architecture import MLA_POLICY, architecture_contract
    base = dict(cfg) if isinstance(cfg, Mapping) else vars(cfg).copy()
    for key in FIELDS:
        del base[key]
    base['tokenizer_architecture_policy'] = MLA_POLICY
    contract = architecture_contract(base)
    if contract['width'] != 256:
        raise ValueError('Tokenizer MLA gate comparison is restricted to width256')
    return dict(contract, policy=POLICY, gate_kind=kind, gate_policy=POLICY)
