"""Explicit call-local, attached first-layer value reuse; no persistent activation state."""
from collections.abc import Mapping
POLICY = 'mha4_first_value_residual'
KINDS = ('local', 'local_half', 'first_blend')
FIELDS = ('value_residual_kind', 'value_residual_width_policy')
def _get(c,k,d=None): return c.get(k,d) if isinstance(c,Mapping) else getattr(c,k,d)
def _has(c,k): return k in c if isinstance(c,Mapping) else hasattr(c,k)
def value_residual_contract(cfg):
    if not any(_has(cfg,k) for k in FIELDS): return None
    kind=_get(cfg,FIELDS[0]);policy=_get(cfg,FIELDS[1])
    if not isinstance(kind,str) or kind not in KINDS or policy != POLICY:
        raise ValueError('Value reuse requires an explicit supported kind and named policy')
    from .residual_gain import residual_gain_contract
    if residual_gain_contract(cfg) is None:
        raise ValueError('Value reuse requires the explicit inherited residual-gain contract')
    for k in ('value_residual_coefficient','value_residual_detach','value_residual_shared_cache',
              'value_residual_recursive','value_residual_learned','value_residual_first_layer'):
        if _has(cfg,k): raise ValueError('Unsupported value-residual override: '+k)
    return dict(value_residual_kind=kind,value_residual_width_policy=POLICY)
def value_residual_signature(cfg):
    c=value_residual_contract(cfg)
    return None if c is None else (c[FIELDS[0]],POLICY)
def resolve_value_residual(cfg):
    c=value_residual_contract(cfg)
    return 'local' if c is None else c[FIELDS[0]]
def mix_values(local, first, kind, layer):
    if kind not in KINDS or isinstance(layer,bool) or not isinstance(layer,int) or layer not in range(4):
        raise ValueError('Invalid value-reuse kind or layer index')
    if layer == 0:
        if first is not None: raise ValueError('First layer must establish its own raw values')
        return local
    if kind == 'local': return local
    if kind == 'local_half': return .5 * local
    if first is None: raise ValueError('First-blend requires attached values from this call first-layer values')
    if (first.shape != local.shape or first.device != local.device or first.dtype != local.dtype):
        raise ValueError('First-layer value chunk must match local batch/head/token/channel geometry and dtype/device')
    # No detach, clone, cache access, or model-side activation storage.
    return .5 * local + .5 * first

def validate_value_residual_model(model):
    cfg=getattr(model,'cfg',{})
    signature=value_residual_signature(cfg)
    if signature != getattr(model,'value_residual_signature',None):
        raise ValueError('Constructed/current value-reuse contracts differ')
    blocks=list(getattr(getattr(model,'transformer',None),'h',[]))
    if signature is not None and len(blocks)!=4: raise ValueError('Value reuse requires four blocks')
    kind=resolve_value_residual(cfg)
    for i,b in enumerate(blocks):
        a=b.attn
        if getattr(a,'value_residual_kind','local')!=kind:
            raise ValueError('Constructed attention value-reuse kind differs')
        layer=getattr(a,'value_residual_layer',None)
        if signature is not None and (type(layer) is not int or layer!=i):
            raise ValueError('Constructed attention layer identity differs')
        # The first-layer value is always a forward-local tensor, never module state.
        for n in ('first_layer_value','value_residual_state'):
            if hasattr(a,n) or hasattr(b,n) or hasattr(model,n):
                raise ValueError('Persistent first-layer value state is forbidden')
def validate_value_residual_load(model,checkpoint):
    saved=value_residual_contract(checkpoint.get('cfg',{}).get('model',{}))
    current=value_residual_contract(getattr(model,'cfg',{}))
    if saved!=current: raise ValueError('Checkpoint/current value-reuse contracts differ')
    validate_value_residual_model(model)
