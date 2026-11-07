import copy
import weakref
import torch
from research.final_ln_scale_model_v1.runtime.train.optimizers import EpsilonOptimizer
from .contract import require


class Optimizer(EpsilonOptimizer):
    def __init__(self, children, model):
        super().__init__(children, .00006)
        self.architecture = copy.deepcopy(model.signature)
        self.owner = weakref.ref(model)
        self.completed_steps = 0
        self.gradient_steps = {g['parameter_names'][0]: 0 for g in self.param_groups}

    def step(self):
        active = [g['parameter_names'][0] for g in self.param_groups if g['params'][0].grad is not None]
        super().step()
        self.completed_steps += 1
        for name in active:
            self.gradient_steps[name] += 1
        owner = self.owner()
        if owner is not None:
            owner.weight_epoch += 1

    def state_dict(self):
        return dict(super().state_dict(), readout_mixture_signature=copy.deepcopy(self.architecture),
                    completed_steps=self.completed_steps, gradient_steps=copy.deepcopy(self.gradient_steps))

    def load_state_dict(self, state):
        require(type(state) is dict and set(state) == {'version', 'independent_decay', 'optimizers',
                'readout_mixture_signature', 'completed_steps', 'gradient_steps'}
                and state['readout_mixture_signature'] == self.architecture, 'Own optimizer signature required')
        step, counts = state['completed_steps'], state['gradient_steps']
        require(type(step) is int and step >= 0 and type(counts) is dict and counts.keys() == self.gradient_steps.keys()
                and all(type(x) is int and 0 <= x <= step for x in counts.values()), 'Perparameter exposure differs')
        super().load_state_dict(state)
        self.completed_steps, self.gradient_steps = step, copy.deepcopy(counts)
        owner = self.owner()
        if owner is not None:
            owner.weight_epoch += 1


def build_optimizer(model, training):
    model.validate()
    t = training
    require(t.optimizer == 'muon' and t.decay_mode == 'independent' and type(t.weight_decay) is float
            and t.weight_decay == .00006, 'Incumbent optimizer/decay required')
    require(type(t.learning_rate) is float and t.learning_rate in (.000625, .00125, .0025)
            and t.adamw_beta2 == .999 and t.adamw_epsilon == 1e-6
            and t.adamw_beta2_policy == 'mla4_adamw_beta2_fixed_v1'
            and t.adamw_epsilon_policy == 'mla4_adamw_epsilon_fixed_v1', 'Finite LR/adopted moment policies required')
    muon, adam = [], []
    for name, p in model.named_parameters():
        hidden = (name.startswith('backbone.transformer.h.') and p.ndim == 2) or name.startswith('context_maps.')
        scale = .5 if hidden else 1.
        group = dict(params=[p], lr=t.learning_rate*scale, weight_decay=0., parameter_names=[name], width_lr_scale=scale)
        (muon if hidden else adam).append(group)
    require(len(muon) == model.signature['muon_groups'] and len(adam) == model.signature['adam_groups'], 'Own complete grouping required')
    children = [torch.optim.Muon(muon, lr=t.learning_rate, weight_decay=0., adjust_lr_fn='match_rms_adamw'),
                torch.optim.AdamW(adam, lr=t.learning_rate, weight_decay=0., betas=(.9, .999), eps=1e-6)]
    return Optimizer(children, model)
