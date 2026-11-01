"""Explicit product normalization for the isolated tokenizer MLA policy."""
from research.mla_score_v1.runtime.models.ffn_hidden_norm import (
    resolve_ffn_hidden_norm, make_ffn_hidden_norm, validate_ffn_hidden_norm_contract as legacy_validate)

def validate_ffn_hidden_norm_contract(cfg):
    from .block_norm import FFN_NORM_POLICY, contract, get
    if get(cfg, 'tokenizer_architecture_policy') != FFN_NORM_POLICY:
        return legacy_validate(cfg)
    value = contract(cfg)
    return 2688 if value['ffn_product_normalization']['kind'] == 'rms' else 0
