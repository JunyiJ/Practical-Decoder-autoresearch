"""Same-shape gated FFN activation semantics; metadata only.

Legacy ffn='swiglu' names the existing three-projection structural container.
Under this named policy, gated_activation_kind is the authoritative nonlinearity.
No implicit reinterpretation of historical checkpoints is allowed.
"""
from collections.abc import Mapping
POLICY = 'mha4_gated_activation_21_8'
KINDS = ('silu', 'gelu_exact', 'relu')
FIELDS = ('gated_activation_kind', 'gated_activation_width_policy')
def _get(c,k,d=None):return c.get(k,d) if isinstance(c,Mapping) else getattr(c,k,d)
def _has(c,k):return k in c if isinstance(c,Mapping) else hasattr(c,k)
def gated_activation_contract(cfg):
    if not any(_has(cfg,k) for k in FIELDS):return None
    from .tokenizer_activation import POLICY as TOKENIZER_POLICY, activation_contract
    if (_get(cfg, 'tokenizer_architecture_policy') == TOKENIZER_POLICY
            or _get(cfg, FIELDS[1]) == TOKENIZER_POLICY):
        contract = activation_contract(cfg)
        if contract is None:
            raise ValueError('Tokenizer gate requires its explicit MLA backbone')
        return dict(gated_activation_kind=contract['gate_kind'],
                    gated_activation_width_policy=TOKENIZER_POLICY)
    kind=_get(cfg,FIELDS[0])
    if type(kind) is not str or kind not in KINDS or _get(cfg,FIELDS[1])!=POLICY:
        raise ValueError('Gated activation requires explicit reviewed kind and policy')
    from .attention_scale import attention_scale_contract
    if attention_scale_contract(cfg) is None:
        raise ValueError('Gated activation requires explicit final inherited attention/backbone contracts')
    if any(_has(cfg,k) for k in ('activation','gate_activation','swiglu_beta','gelu_approximate',
                                'gate_multiplier','ffn_activation_scale','ffn_hidden_norm_policy')):
        raise ValueError('Gated activation rejects unreviewed nonlinearity overrides')
    return dict(gated_activation_kind=kind,gated_activation_width_policy=POLICY)
def gated_activation_signature(cfg):
    c=gated_activation_contract(cfg)
    return None if c is None else (c['gated_activation_kind'],c['gated_activation_width_policy'])
def resolve_gated_activation(cfg):
    c=gated_activation_contract(cfg)
    return 'silu' if c is None else c['gated_activation_kind']
def validate_gated_activation_model(model):
    cfg=getattr(model,'cfg',{})
    sig=gated_activation_signature(cfg)
    if getattr(model,'gated_activation_signature',None)!=sig:
        raise ValueError('Constructed gate activation differs from mutable config')
    blocks=getattr(getattr(model,'transformer',None),'h',())
    if sig is not None and len(blocks)!=4:
        raise ValueError('Gated activation requires four reviewed blocks')
    kind=resolve_gated_activation(cfg)
    for b in blocks:
        mlp=getattr(b,'mlp',None)
        if getattr(mlp,'gate_activation','silu')!=kind:
            raise ValueError('Block gate activation differs from model contract')
        if sig is not None:
            if (getattr(mlp,'hidden_dim',None)!=21*_get(cfg,'dim')//8
                    or getattr(mlp,'hidden_norm',None) is not None):
                raise ValueError('Gated activation requires unchanged hidden shape and no product normalization')
            for name in ('linear_gate','linear_up','linear_down'):
                if not hasattr(mlp,name) or getattr(getattr(mlp,name),'bias',False) is not None:
                    raise ValueError('Gated activation requires three bias-free projections')
def validate_gated_activation_load(model,checkpoint):
    saved=_get(checkpoint.get('cfg',{}),'model',{})
    if gated_activation_contract(getattr(model,'cfg',{}))!=gated_activation_contract(saved):
        raise ValueError('Checkpoint gate activation differs despite identical tensor shapes')
    validate_gated_activation_model(model)
