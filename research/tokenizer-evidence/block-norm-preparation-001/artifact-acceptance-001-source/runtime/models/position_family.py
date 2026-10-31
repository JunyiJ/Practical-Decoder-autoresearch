"""Named positional family, separate from historical rope=0 learned tables.

The inherited rope=1 key is a legacy structural container. Only this explicit
policy overrides its positional meaning; no inherited geometry guard is relaxed.
Contracts and slope calculations import no tensor runtime.
"""
from collections.abc import Mapping
POLICY = 'mha4_position_family'
KINDS = ('rope', 'nope', 'alibi')
FIELDS = ('position_family_kind', 'position_family_width_policy')
SLOPE_FORMULA = 'power2_head_order_exp_minus8'
def _get(c,k,d=None): return c.get(k,d) if isinstance(c,Mapping) else getattr(c,k,d)
def _has(c,k): return k in c if isinstance(c,Mapping) else hasattr(c,k)
def slopes_for_heads(heads):
    if type(heads) is not int or heads not in (2,4,8):
        raise ValueError('Position family requires 2,4,8 heads')
    return tuple(2.**(-8.*(h+1)/heads) for h in range(heads))
def position_family_contract(cfg):
    if not any(_has(cfg,k) for k in FIELDS): return None
    kind = _get(cfg,FIELDS[0])
    if type(kind) is not str or kind not in KINDS or _get(cfg,FIELDS[1]) != POLICY:
        raise ValueError('Position family requires explicit reviewed kind and policy')
    from .value_residual import value_residual_contract
    if value_residual_contract(cfg) is None:
        raise ValueError('Position family requires explicit final inherited value/backbone')
    if type(_get(cfg,'rope')) is not int or _get(cfg,'rope') != 1:
        raise ValueError('Named position policy preserves legacy rope=1 container')
    d,h = _get(cfg,'dim'),_get(cfg,'n_heads')
    if type(d) is not int or d not in (64,128,256) or type(h) is not int or h != d//32:
        raise ValueError('Position family requires reviewed width and head32')
    if any(_has(cfg,k) for k in ('position_encoding','alibi_slopes','alibi_scale',
        'alibi_slope_formula','position_offset','position_theta','position_learned',
        'position_interpolation','position_family_slopes','enable_rope')):
        raise ValueError('Position family rejects unreviewed positional overrides')
    return dict(position_family_kind=kind,position_family_width_policy=POLICY)
def position_family_signature(cfg):
    c=position_family_contract(cfg)
    return None if c is None else (c[FIELDS[0]],POLICY,SLOPE_FORMULA,_get(cfg,'n_heads'))
def resolve_position_family(cfg):
    c=position_family_contract(cfg)
    return None if c is None else c[FIELDS[0]]
def distance_bias_scalar(heads,head,query,key):
    slopes=slopes_for_heads(heads)
    if type(head) is not int or not 0<=head<heads:
        raise ValueError('Invalid head index')
    if any(type(x) is not int or x<0 for x in (query,key)):
        raise ValueError('Positions must be nonnegative integers')
    return -slopes[head]*(query-key)
def add_alibi_bias(logits,slopes,start_pos,*,key_start_pos=0):
    # Fixed slopes have no parameter/buffer/optimizer/averaging state. Tensor
    # materialization consumes no RNG; score and distance terms stay separate.
    import torch
    if any(type(x) is not int or x<0 for x in (start_pos,key_start_pos)):
        raise ValueError('Position offset must be a nonnegative integer')
    heads,queries,keys=logits.shape[-3:]
    if tuple(slopes) != slopes_for_heads(heads) or start_pos<key_start_pos or start_pos+queries>key_start_pos+keys:
        raise ValueError('ALiBi slope or cached-prefix geometry mismatch')
    q=torch.arange(start_pos,start_pos+queries,device=logits.device,dtype=logits.dtype)
    k=torch.arange(key_start_pos,key_start_pos+keys,device=logits.device,dtype=logits.dtype)
    m=torch.tensor(slopes,device=logits.device,dtype=logits.dtype).view(1,heads,1,1)
    return logits - m*(q[:,None]-k[None,:])
def validate_position_family_model(model):
    cfg=getattr(model,'cfg',{})
    signature=position_family_signature(cfg)
    if getattr(model,'position_family_signature',None) != signature:
        raise ValueError('Constructed position family differs from mutable config')
    if signature is None: return
    kind=signature[0];rope=kind=='rope'
    transformer=getattr(model,'transformer',None)
    blocks=getattr(transformer,'h',())
    if len(blocks)!=4 or hasattr(transformer,'wpe'):
        raise ValueError('Position family requires four blocks and no position table')
    if getattr(model,'position_encoding',None)!=kind or getattr(model,'enable_rope',None) is not rope:
        raise ValueError('Constructed model positional mode mismatch')
    for block in blocks:
        attn=getattr(block,'attn',None)
        if getattr(attn,'position_family_kind',None)!=kind:
            raise ValueError('Constructed attention positional mode mismatch')
        expected=slopes_for_heads(_get(cfg,'n_heads')) if kind=='alibi' else ()
        if getattr(attn,'position_slopes',None)!=expected:
            raise ValueError('Constructed ALiBi slopes mismatch')
        if getattr(block,'enable_rope',None) is not rope:
            raise ValueError('Constructed block rotation mismatch')
        for name in ('freq_cos','freq_sin'):
            freq=getattr(block,name,None)
            if rope:
                if freq is None or tuple(getattr(freq,'shape',())) != (128,16):
                    raise ValueError('RoPE frequencies absent or malformed')
            elif freq is not None:
                raise ValueError('Unrotated family must have no rotary frequencies')
        if any(n in getattr(attn,'_parameters',{}) or n in getattr(attn,'_buffers',{})
                for n in ('position_slopes','alibi_slopes','position_state')):
            raise ValueError('Position family must have no learned or cached position state')
def validate_position_family_load(model,checkpoint):
    saved=_get(checkpoint.get('cfg',{}),'model',{})
    if position_family_contract(getattr(model,'cfg',{})) != position_family_contract(saved):
        raise ValueError('Checkpoint positional family differs despite compatible weights')
    validate_position_family_model(model)
    if position_family_contract(saved) is not None and 'transformer.wpe.weight' in checkpoint.get('model_state',{}):
        raise ValueError('Named positional family rejects a learned position table')
