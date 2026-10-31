"""Exact GPT-2 GQA table-sharing adaptation; separate from character policies."""
from collections.abc import Mapping

POLICY = 'tokenizer_gqa_scaled_table_sharing_v1'
FIELDS = ('weight_tying_kind', 'weight_tying_width_policy')
KINDS = ('legacy_untied', 'matched_untied', 'scaled_tied')


def get(cfg, key, default=None):
    return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)


def table_contract(cfg):
    if get(cfg, 'tokenizer_architecture_policy') != POLICY:
        return None
    if get(cfg, FIELDS[0]) not in KINDS or get(cfg, FIELDS[1]) != POLICY:
        raise ValueError('Tokenizer table sharing requires its exact kind and policy')
    # Validate all inherited keys through the existing strict GQA policy, with
    # only the two declared table fields removed; arbitrary overrides still fail.
    from .tokenizer_architecture import POLICY as BASE_POLICY, architecture_contract
    base = dict(cfg) if isinstance(cfg, Mapping) else vars(cfg).copy()
    for key in FIELDS:
        del base[key]
    base['tokenizer_architecture_policy'] = BASE_POLICY
    contract = architecture_contract(base)
    if contract['family'] != 'gqa':
        raise ValueError('Tokenizer table sharing is restricted to the GQA backbone')
    return dict(contract, policy=POLICY, table_kind=get(cfg, FIELDS[0]),
                table_policy=POLICY, table_base_width=64)
