"""Explicit complete-branch channel gains; contracts import no tensor runtime.

The inherited depth4_unscaled key describes the existing depth/width geometry.
Only this separately named policy authorizes post-projection branch gains; old
contracts and generic branch-scale overrides remain unchanged.
"""
from collections.abc import Mapping
POLICY = 'mha4_channel_residual_gain'
KINDS = ('unit', 'static_tenth', 'learned_tenth')
FIELDS = ('residual_gain_kind', 'residual_gain_width_policy')
PARAMETERS = ('attention_residual_gain', 'ffn_residual_gain')
INITIAL_GAIN = 0.1

def _get(c,k,d=None): return c.get(k,d) if isinstance(c,Mapping) else getattr(c,k,d)
def _has(c,k): return k in c if isinstance(c,Mapping) else hasattr(c,k)
def residual_gain_contract(cfg):
    if not any(_has(cfg,k) for k in FIELDS): return None
    kind = _get(cfg,FIELDS[0])
    if type(kind) is not str or kind not in KINDS or _get(cfg,FIELDS[1]) != POLICY:
        raise ValueError('Residual gain requires explicit reviewed kind and policy')
    from .gated_activation import gated_activation_contract
    if gated_activation_contract(cfg) is None:
        raise ValueError('Residual gain requires explicit final inherited activation/backbone')
    if any(_has(cfg,k) for k in ('residual_gain_init','residual_gain_scale','residual_gain_shared',
            'residual_gain_activation','layer_scale','layerscale','rezero','residual_branch_scale',
            'attention_residual_scale','ffn_residual_scale')):
        raise ValueError('Residual gain rejects unreviewed initialization/sharing/branch overrides')
    return dict(residual_gain_kind=kind,residual_gain_width_policy=POLICY)
def residual_gain_signature(cfg):
    c=residual_gain_contract(cfg)
    return None if c is None else (c['residual_gain_kind'],POLICY)
def resolve_residual_gain(cfg):
    c=residual_gain_contract(cfg)
    return 'unit' if c is None else c['residual_gain_kind']
def install_residual_gains(block,cfg):
    block.residual_gain_kind=resolve_residual_gain(cfg)
    if block.residual_gain_kind == 'learned_tenth':
        import torch
        from torch import nn
        for name in PARAMETERS:
            block.register_parameter(name,nn.Parameter(torch.full((cfg.dim,),INITIAL_GAIN)))
def apply_residual_gain(block,value,branch):
    if branch not in ('attention','ffn'): raise ValueError('Unknown residual branch')
    kind=block.residual_gain_kind
    if kind == 'unit': return value
    if kind == 'static_tenth': return value * INITIAL_GAIN
    if kind == 'learned_tenth': return value * getattr(block,branch+'_residual_gain')
    raise ValueError('Unknown constructed residual gain kind')
def validate_residual_gain_model(model):
    cfg=getattr(model,'cfg',{})
    signature=residual_gain_signature(cfg)
    if getattr(model,'residual_gain_signature',None) != signature:
        raise ValueError('Constructed residual gain differs from mutable config')
    blocks=getattr(getattr(model,'transformer',None),'h',())
    if signature is not None and len(blocks) != 4:
        raise ValueError('Residual gain requires four reviewed blocks')
    kind=resolve_residual_gain(cfg)
    seen=set()
    for block in blocks:
        if getattr(block,'residual_gain_kind','unit') != kind:
            raise ValueError('Block residual gain differs from model contract')
        for name in PARAMETERS:
            p=getattr(block,name,None)
            if kind == 'learned_tenth':
                if (p is None or tuple(getattr(p,'shape',())) != (_get(cfg,'dim'),)
                        or getattr(p,'requires_grad',False) is not True
                        or getattr(block,'_parameters',{}).get(name) is not p
                        or name in getattr(block,'_buffers',{}) or id(p) in seen):
                    raise ValueError('Residual gains must be distinct learned channel vectors')
                seen.add(id(p))
            elif p is not None or name in getattr(block,'_parameters',{}) or name in getattr(block,'_buffers',{}):
                raise ValueError('Unit/static residual gains must add no tensor state')
def validate_residual_gain_load(model,checkpoint):
    saved=_get(checkpoint.get('cfg',{}),'model',{})
    if residual_gain_contract(getattr(model,'cfg',{})) != residual_gain_contract(saved):
        raise ValueError('Checkpoint residual gain differs despite compatible tensor shapes')
    validate_residual_gain_model(model)
