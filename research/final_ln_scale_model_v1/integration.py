"""Narrow training identity around the unchanged adopted grouped MLA model."""
from collections.abc import Mapping
from research.final_ln_scale_model_v1.runtime.train.lr_schedule import schedule_contract
from research.final_ln_scale_model_v1.runtime.models.block_norm import contract, READOUT_POLICY
from research.final_ln_scale_model_v1.runtime.train.optimizers import build_optimizer as inherited_optimizer
from research.final_ln_scale_model_v1.runtime.utils.checkpoint import load_model_weights as inherited_load

def identity(cfg):
    model = contract(cfg.model)
    if model is None or model['policy'] != READOUT_POLICY or model['normalization']['kind'] != 'rms_incumbent':
        raise ValueError('Adopted grouped MLA with block RMSNorm required')
    if model['branch_gain']['kind'] != 'unit':
        raise ValueError('Unit gains required')
    t = cfg.training
    if type(t.grad_clip) is not float or t.grad_clip != 1.:
        raise ValueError('Incumbent clipping1.0 required')
    if type(t.weight_decay) is not float or t.weight_decay != .00006:
        raise ValueError('Exactly independent decay0.00006 required')
    if t.optimizer != 'muon' or t.decay_mode != 'independent' or t.learning_rate != .00125:
        raise ValueError('Fixed peak LR0.00125 and inherited Muon/AdamW required')
    if t.lr_schedule.kind != 'linear_cooldown':
        raise ValueError('No warmup; inherited linear cooldown required')
    if (getattr(t, 'adamw_beta2_policy', None) != 'mla4_adamw_beta2_fixed_v1'
            or type(getattr(t, 'adamw_beta2', None)) is not float or t.adamw_beta2 != .999):
        raise ValueError('Incumbent AdamW beta2 .999 required')
    if getattr(t,'adamw_epsilon_policy',None) != 'mla4_adamw_epsilon_fixed_v1' or type(getattr(t,'adamw_epsilon',None)) is not float or t.adamw_epsilon!=1e-6:raise ValueError('Adopted explicit AdamW epsilon1e-6 required')
    schedule = schedule_contract(t)
    if schedule['min_ratio'] != .1:
        raise ValueError('Cooldown floor0.1 required')
    if (schedule['total_steps'], schedule['start_step']) not in ((4,2),(20,13),(1200,800)):
        raise ValueError('Exact inherited cooldown horizon/onset required')
    return dict(adamw=dict(beta1=.9,beta2=.999,epsilon=1e-6), readout=model['final_normalization'], branch_gain=model['branch_gain'], grad_clip=1., policy='fixed_lr_final_ln_scale_v1', schedule=schedule, decay=float(t.weight_decay),
                peak_lr=.00125, optimizer='muon', decay_mode='independent')

def build_optimizer(model, cfg):
    expected = identity(cfg)
    result = inherited_optimizer(model, cfg.training)
    if result.decay != expected['decay']:
        raise ValueError('Actual independent decay differs')
    if any(g['weight_decay'] != 0. for g in result.param_groups):
        raise ValueError('Child optimizers must not apply another decay')
    return result

def load_model_weights(model, payload, cfg, *, for_resume=False):
    from omegaconf import OmegaConf
    expected = identity(cfg)
    if not isinstance(payload, Mapping) or 'cfg' not in payload:
        raise ValueError('Saved full configuration required')
    saved = identity(OmegaConf.create(payload['cfg']))
    if saved != expected:
        raise ValueError('Cross-arm final readout checkpoint identity differs')
    if for_resume and payload.get('optimizer_state', {}).get('independent_decay') != expected['decay']:
        raise ValueError('Optimizer decay differs from saved training configuration')
    return inherited_load(model, payload, for_resume=for_resume)
