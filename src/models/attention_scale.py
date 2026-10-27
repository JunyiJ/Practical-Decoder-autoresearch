"""Fixed post-RoPE attention-logit multiplier contract; no tensor imports."""
from collections.abc import Mapping

POLICY = 'mha4_fixed_attention_logit_scale'
SCALES = (1., .5, 2.)
FIELDS = ('attention_logit_scale', 'attention_logit_scale_width_policy')

def _get(cfg, key, default=None):
    return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)

def _has(cfg, key):
    return key in cfg if isinstance(cfg, Mapping) else hasattr(cfg, key)

def attention_scale_contract(cfg):
    if not any(_has(cfg, k) for k in FIELDS):
        return None
    value = _get(cfg, FIELDS[0])
    from .tokenizer_architecture import SCORE_POLICY, architecture_contract
    if _get(cfg, FIELDS[1]) == SCORE_POLICY:
        if architecture_contract(cfg) is None:
            raise ValueError('Tokenizer score scaling requires its explicit backbone')
        return dict(attention_logit_scale=float(value), attention_logit_scale_width_policy=SCORE_POLICY)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in SCALES or _get(cfg, FIELDS[1]) != POLICY:
        raise ValueError('Attention scale requires explicit reviewed scalar and policy')
    from .residual_topology import residual_topology_contract
    if residual_topology_contract(cfg) is None:
        raise ValueError('Attention scale requires explicit final inherited residual/table/readout contracts')
    if any(_has(cfg, k) for k in ('attention_temperature', 'attn_scale', 'softmax_scale', 'qk_scale', 'logit_softcap', 'learned_temperature')):
        raise ValueError('Attention scale rejects unreviewed score overrides')
    return dict(attention_logit_scale=float(value), attention_logit_scale_width_policy=POLICY)

def attention_scale_signature(cfg):
    c = attention_scale_contract(cfg)
    return None if c is None else (c['attention_logit_scale'], c['attention_logit_scale_width_policy'])

def resolve_attention_scale(cfg):
    c = attention_scale_contract(cfg)
    return 1. if c is None else c['attention_logit_scale']

def validate_attention_scale_model(model):
    cfg = getattr(model, 'cfg', {})
    signature = attention_scale_signature(cfg)
    if getattr(model, 'attention_scale_signature', None) != signature:
        raise ValueError('Constructed attention scale differs from mutable config')
    blocks = getattr(getattr(model, 'transformer', None), 'h', ())
    if signature is not None and len(blocks) != 4:
        raise ValueError('Attention scale requires four reviewed blocks')
    value = resolve_attention_scale(cfg)
    for block in blocks:
        if getattr(block.attn, 'attention_logit_scale', 1.) != value:
            raise ValueError('Constructed attention scalar differs from model contract')

def validate_attention_scale_load(model, checkpoint):
    current = getattr(model, 'cfg', {})
    saved = _get(checkpoint.get('cfg', {}), 'model', {})
    if attention_scale_contract(current) != attention_scale_contract(saved):
        raise ValueError('Checkpoint attention scale differs despite matching tensor shapes')
    validate_attention_scale_model(model)
