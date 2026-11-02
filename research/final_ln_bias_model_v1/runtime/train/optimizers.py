"""Explicit second-moment policy; unchanged grouping and independent shrinkage."""
import torch
from ._grouping import (
    build_optimizer as inherited_build, optimizer_options as inherited_options,
    HybridOptimizer,
)
POLICY = 'mla4_adamw_beta2_fixed_v1'

def beta2_contract(training):
    beta2 = getattr(training, 'adamw_beta2', None)
    if (getattr(training, 'adamw_beta2_policy', None) != POLICY
            or type(beta2) is not float or beta2 != .999):
        raise ValueError('Incumbent AdamW beta2 .999 policy required')
    return dict(policy=POLICY, beta1=.9, beta2=beta2, epsilon=1e-8)

def optimizer_options(training):
    return dict(inherited_options(training), adamw_beta2=beta2_contract(training))

class Beta2Optimizer(HybridOptimizer):
    def load_state_dict(self, state):
        # Validate every child/group BEFORE any PyTorch child mutates state.
        if (type(state) is not dict or state.get('version') != 1
                or state.get('independent_decay') != self.decay
                or type(state.get('optimizers')) is not list
                or len(state['optimizers']) != len(self.optimizers)):
            raise ValueError('Optimizer composition/decay differs')
        for child, saved in zip(self.optimizers, state['optimizers']):
            expected = child.state_dict()['param_groups']
            if type(saved) is not dict or type(saved.get('param_groups')) is not list or len(saved['param_groups']) != len(expected):
                raise ValueError('Optimizer group inventory differs')
            for actual, wanted in zip(saved['param_groups'], expected):
                # Scheduled LR is checked by the checkpoint resume preflight.
                if not isinstance(actual, dict) or {k:v for k,v in actual.items() if k != 'lr'} != {k:v for k,v in wanted.items() if k != 'lr'}:
                    raise ValueError('Foreign optimizer beta2 or group identity')
        return super().load_state_dict(state)

def build_optimizer(model, training):
    spec = beta2_contract(training)
    if getattr(model.cfg, 'readout_norm_kind', None) != 'layer_affine':
        raise ValueError('Incumbent final LayerNorm required')
    result = inherited_build(model, training)
    if len(result.optimizers) != 2 or type(result.optimizers[0]) is not torch.optim.Muon or type(result.optimizers[1]) is not torch.optim.AdamW:
        raise ValueError('Exact inherited Muon/AdamW composition required')
    adam = result.optimizers[1]
    from ..models.block_norm import contract
    expected = 44 if contract(model.cfg)['final_normalization']['bias'] else 43
    if len(adam.param_groups) != expected or any(g['betas'] != (.9,.999) or g['eps'] != 1e-8 or g['weight_decay'] != 0 for g in adam.param_groups):
        raise ValueError('Inherited AdamW defaults/group inventory differ')
    # No tensor, RNG, grouping, LR or Muon changes; set the declared beta2 before step0.
    adam.defaults['betas'] = (.9, spec['beta2'])
    for group in adam.param_groups: group['betas'] = (.9, spec['beta2'])
    return Beta2Optimizer(result.optimizers, result.decay)
