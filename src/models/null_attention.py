"""Named fixed-null normalization; metadata imports no tensor runtime.

Normalize FINAL masked attention scores. Null is always score zero and value
zero, never a parameter, persistent activation, rotary position or cached key.
"""
from collections.abc import Mapping
POLICY = 'mha4_fixed_null_attention'
KINDS = ('standard', 'prefix_control', 'null_zero')
FIELDS = ('null_attention_kind', 'null_attention_policy')
CONVENTION = 'final_masked_scores_fixed_zero_null_visible_real_prefix_v1'
FORBIDDEN = ('null_attention_score', 'null_attention_value', 'null_attention_learned',
             'null_attention_temperature', 'null_attention_detach', 'null_attention_clip',
             'null_attention_width_scale', 'null_attention_cache', 'null_attention_count')
STATE_NAMES = ('null_key', 'null_value', 'null_logit', 'null_attention_state')
def _get(c,k,d=None): return c.get(k,d) if isinstance(c,Mapping) else getattr(c,k,d)
def _has(c,k): return k in c if isinstance(c,Mapping) else hasattr(c,k)
def null_attention_contract(cfg):
    if not any(_has(cfg,k) for k in FIELDS): return None
    kind,policy=(_get(cfg,k) for k in FIELDS)
    if type(kind) is not str or kind not in KINDS or type(policy) is not str or policy!=POLICY:
        raise ValueError('Null attention requires an explicit reviewed kind and policy')
    from .logit_z import logit_z_contract
    if logit_z_contract(cfg) is None:
        raise ValueError('Null attention requires final inherited objective and backbone')
    if _get(cfg,'causal',True) is not True:
        raise ValueError('Null attention requires causal real-key visibility')
    if any(_has(cfg,k) for k in FORBIDDEN):
        raise ValueError('Unreviewed null attention override')
    return dict(null_attention_kind=kind,null_attention_policy=POLICY)
def null_attention_signature(cfg):
    c=null_attention_contract(cfg)
    return None if c is None else (c[FIELDS[0]],POLICY,CONVENTION)
def resolve_null_attention(cfg):
    c=null_attention_contract(cfg)
    return 'standard' if c is None else c[FIELDS[0]]
def normalize_attention(scores, visible, kind):
    if type(kind) is not str or kind not in KINDS:
        raise ValueError('Invalid null attention normalization kind')
    import torch
    # Exact historical arithmetic for standard; no count/null operations.
    if kind=='standard': return torch.softmax(scores,dim=-1)
    if (scores.dtype!=torch.float32 or scores.ndim!=4 or visible.dtype!=torch.bool
            or visible.ndim!=2 or tuple(visible.shape)!=tuple(scores.shape[-2:])
            or visible.device!=scores.device):
        raise ValueError('Expected FP32 masked attention and matching bool visibility')
    counts=visible.sum(dim=-1,keepdim=True)
    if not (counts>0).all().item():
        raise ValueError('Every query requires a visible real key')
    if kind=='prefix_control':
        mass=counts.to(dtype=scores.dtype)
        return torch.softmax(scores,dim=-1)*(mass/(mass+1))
    # scores already include content scaling, position bias and causal -inf.
    # Softmax stabilizes real AND zero-null columns together, including extremes.
    null=scores.new_zeros((*scores.shape[:-1],1))
    return torch.softmax(torch.cat((scores,null),dim=-1),dim=-1)[...,:-1]
def validate_null_attention_model(model):
    signature=null_attention_signature(getattr(model,'cfg',{}))
    stored=getattr(model,'null_attention_signature',None)
    if type(stored) is not type(signature) or stored!=signature:
        raise ValueError('Constructed/current null attention differs or was deleted')
    blocks=list(getattr(getattr(model,'transformer',None),'h',()))
    if signature is not None and len(blocks)!=4:
        raise ValueError('Null attention requires four inherited fused MHA blocks')
    for block in blocks:
        attn=getattr(block,'attn',None)
        if signature is not None:
            if (getattr(attn,'null_attention_signature',None)!=signature
                    or type(getattr(attn,'null_attention_signature',None)) is not tuple
                    or getattr(attn,'null_attention_kind',None)!=signature[0]
                    or getattr(attn,'causal',None) is not True):
                raise ValueError('Constructed attention normalization differs')
        elif getattr(attn,'null_attention_signature',None) is not None:
            raise ValueError('Unexpected explicit attention normalization')
        for obj in (attn,block):
            for name in STATE_NAMES:
                if hasattr(obj,name): raise ValueError('Null K/V or activation state is forbidden')
    for obj in [model]+[getattr(b,'attn',None) for b in blocks]:
        for name in STATE_NAMES:
            if hasattr(obj,name): raise ValueError('Null state is forbidden')
        for name in FIELDS+('null_attention_signature',):
            if name in getattr(obj,'_parameters',{}) or name in getattr(obj,'_buffers',{}):
                raise ValueError('Null attention adds no tensor state')
    return signature
def validate_null_attention_load(model,checkpoint):
    saved=_get(checkpoint.get('cfg',{}),'model',{})
    if null_attention_contract(getattr(model,'cfg',{}))!=null_attention_contract(saved):
        raise ValueError('Checkpoint null attention differs despite compatible weights')
    validate_null_attention_model(model)
    if any(any(part in STATE_NAMES+FIELDS+('null_attention_signature',) for part in key.split('.'))
           for key in checkpoint.get('model_state',{})):
        raise ValueError('Checkpoint contains forbidden null tensor state')
