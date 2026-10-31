"""Explicit training-only output log-normalizer objective; evaluation is pure CE.

Metadata contracts import no tensor runtime. No parameter/buffer/RNG is added.
The coefficient applies to FINAL returned FP32 logits, after all readout scales
and output bias. This is unrelated to router auxiliary losses.
"""
from collections.abc import Mapping
import math

POLICY = 'mha4_final_logit_z'
COEFFICIENTS = {'none': 0., 'z1e4': .0001, 'z1e3': .001}
FIELDS = ('logit_z_kind', 'logit_z_coefficient', 'logit_z_objective_policy')
CONVENTION = 'final_fp32_logsumexp_squared_valid_target_mean_v1'

def _get(c,k,d=None): return c.get(k,d) if isinstance(c,Mapping) else getattr(c,k,d)
def _has(c,k): return k in c if isinstance(c,Mapping) else hasattr(c,k)

def logit_z_contract(cfg):
    if not any(_has(cfg,k) for k in FIELDS): return None
    kind, coefficient, policy = (_get(cfg,k) for k in FIELDS)
    if type(kind) is not str or kind not in COEFFICIENTS or policy != POLICY:
        raise ValueError('Logit-z requires explicit reviewed kind and objective policy')
    if (type(coefficient) not in (int,float) or not math.isfinite(coefficient)
            or coefficient != COEFFICIENTS[kind]):
        raise ValueError('Logit-z coefficient must exactly match the reviewed family')
    from .position_family import position_family_contract
    if position_family_contract(cfg) is None:
        raise ValueError('Logit-z requires the final explicit inherited position/backbone')
    for key in ('logit_z_center', 'logit_z_temperature', 'logit_z_clip',
                'logit_z_detach', 'logit_z_width_scale', 'label_smoothing',
                'logit_z_ignore_index', 'logit_z_log_vocab_subtract'):
        if _has(cfg,key): raise ValueError('Unreviewed logit-z override: '+key)
    return dict(logit_z_kind=kind,logit_z_coefficient=float(coefficient),
                logit_z_objective_policy=POLICY)

def logit_z_signature(cfg):
    c=logit_z_contract(cfg)
    return None if c is None else (c[FIELDS[0]],c[FIELDS[1]],POLICY,CONVENTION)

def validate_logit_z_model(model):
    signature=logit_z_signature(getattr(model,'cfg',{}))
    stored=getattr(model,'logit_z_signature',None)
    if (type(stored) is not type(signature) or stored != signature
            or (signature is not None and any(type(a) is not type(b) for a,b in zip(stored,signature)))):
        raise ValueError('Constructed/current logit-z objectives differ or were deleted')
    # The signature is immutable metadata, never learned/averaged tensor state.
    for key in FIELDS+('logit_z_signature',):
        if key in getattr(model,'_parameters',{}) or key in getattr(model,'_buffers',{}):
            raise ValueError('Logit-z adds no parameter or buffer state')
    return signature

def validate_logit_z_load(model,checkpoint):
    saved=_get(checkpoint.get('cfg',{}),'model',{})
    if logit_z_contract(getattr(model,'cfg',{})) != logit_z_contract(saved):
        raise ValueError('Checkpoint logit-z objective differs despite compatible weights')
    validate_logit_z_model(model)

def objective_requested(model, enabled, targets):
    if type(enabled) is not bool: raise ValueError('Training objective flag must be boolean')
    signature=validate_logit_z_model(model)
    if enabled and (signature is None or targets is None):
        raise ValueError('Training objective requires explicit contract and targets')
    return signature if enabled else None

def training_logit_z(logits, targets, ce_loss, signature):
    """Return loss/metrics; zero arm returns the original CE object unchanged.

    CE's historical ignore_index=-100 supplies exactly the valid-token mask.
    Empty valid batches fail for the nonzero objective, rather than produce NaN.
    For zero, logZ diagnostics are deliberately null: computing them would add
    arithmetic to the promised historical zero path. No tensor is detached here.
    """
    if (not isinstance(signature,tuple) or len(signature)!=4
            or type(signature[0]) is not str or type(signature[1]) is not float
            or signature[0] not in COEFFICIENTS
            or signature != (signature[0],COEFFICIENTS[signature[0]],POLICY,CONVENTION)):
        raise ValueError('Invalid constructed logit-z objective signature')
    coefficient=signature[1]
    if coefficient == 0.:
        return ce_loss, dict(logit_z_coefficient=0., logit_z_mean=None,
                            logit_z_squared_mean=None, logit_z_penalty=None)
    import torch
    if logits.dtype != torch.float32 or targets.shape != logits.shape[:-1]:
        raise ValueError('Logit-z requires final FP32 logits and matching targets')
    flat=logits.reshape(-1,logits.shape[-1])
    valid=targets.reshape(-1).ne(-100)
    if not valid.any().item(): raise ValueError('Logit-z requires at least one valid target')
    log_z=torch.logsumexp(flat[valid],dim=-1)
    squared=log_z.square().mean()
    penalty=coefficient*squared
    return ce_loss+penalty, dict(logit_z_coefficient=coefficient,
        logit_z_mean=log_z.mean(),logit_z_squared_mean=squared,logit_z_penalty=penalty)
