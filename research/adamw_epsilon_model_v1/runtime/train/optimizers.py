"""Fixed epsilon only; inherited grouping, beta2 and shrinkage stay unchanged."""
import torch
from research.final_ln_bias_model_v1.runtime.train.optimizers import (
    build_optimizer as inherited_build, optimizer_options as inherited_options,
    Beta2Optimizer,
)
POLICY = 'mla4_adamw_epsilon_fixed_v1'

def epsilon_contract(training):
    epsilon = getattr(training, 'adamw_epsilon', None)
    if (getattr(training, 'adamw_epsilon_policy', None) != POLICY
            or type(epsilon) is not float or epsilon not in (1e-8, 1e-6)):
        raise ValueError('Explicit AdamW epsilon1e-8/1e-6 policy required')
    inherited_options(training)  # Enforce incumbent beta2 and inherited decay mode.
    return dict(policy=POLICY, beta1=.9, beta2=.999, epsilon=epsilon)

def optimizer_options(training):
    return dict(inherited_options(training), adamw_epsilon=epsilon_contract(training))

class EpsilonOptimizer(Beta2Optimizer):
    # Inherited load validates ALL children and groups, including eps, before
    # loading any child. Scheduled LR is separately checked by resume preflight.
    pass

def build_optimizer(model, training):
    spec = epsilon_contract(training)
    if getattr(model.cfg, 'final_ln_bias_kind', None) != 'learned':
        raise ValueError('Incumbent learned final LayerNorm bias required')
    result = inherited_build(model, training)
    adam = result.optimizers[1]
    if type(adam) is not torch.optim.AdamW or len(adam.param_groups) != 44:
        raise ValueError('Exact44 incumbent AdamW groups required')
    if any(g['betas'] != (.9,.999) or g['eps'] != 1e-8 or g['weight_decay'] != 0. for g in adam.param_groups):
        raise ValueError('Inherited AdamW defaults differ')
    adam.defaults['eps'] = spec['epsilon']
    for group in adam.param_groups: group['eps'] = spec['epsilon']
    return EpsilonOptimizer(result.optimizers, result.decay)
