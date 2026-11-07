"""Own future-token optimizer grouping/state; generated runtime still unqualified."""
import copy
import math
import weakref
import torch
from research.final_ln_scale_model_v1.runtime.train.optimizers import EpsilonOptimizer
from .contract import require


def hidden_matrix(name, parameter):
    return (name.startswith('native.transformer.h.') or name.startswith('auxiliary.block.')) and parameter.ndim == 2


class Optimizer(EpsilonOptimizer):
    def __init__(self, children, model):
        super().__init__(children, .00006)
        self.architecture, self.owner = copy.deepcopy(model.signature), weakref.ref(model)
        self.completed_steps = 0
        self.gradient_steps = {g['parameter_names'][0]: 0 for g in self.param_groups}
        self._parameters = {n: p for n, p in model.named_parameters()}
        self._group_specs = [{k: copy.deepcopy(v) for k, v in g.items() if k not in ('params', 'lr')}
                             for g in self.param_groups]
        self._maximum_lrs = [g['lr'] for g in self.param_groups]
        self.validate()

    def validate(self):
        model = self.owner()
        require(model is not None and model.signature == self.architecture, 'Live own optimizer owner required')
        model.validate()
        require(self.decay == .00006 and len(self.optimizers) == 2
                and type(self.optimizers[0]) is torch.optim.Muon and type(self.optimizers[1]) is torch.optim.AdamW,
                'Own exact optimizer composition required')
        groups = [g for opt in self.optimizers for g in opt.param_groups]
        require(len(groups) == len(self._group_specs) and len(self.param_groups) == len(groups)
                and all(a is b for a, b in zip(groups, self.param_groups)), 'Own complete group inventory required')
        named = dict(model.named_parameters())
        require(named.keys() == self._parameters.keys() and all(p is self._parameters[n] for n, p in named.items()),
                'Replaced/foreign optimizer parameters')
        for group, spec, maximum in zip(groups, self._group_specs, self._maximum_lrs):
            require({k: v for k, v in group.items() if k not in ('params', 'lr')} == spec
                    and type(group['lr']) is float and math.isfinite(group['lr']) and 0 <= group['lr'] <= maximum
                    and len(group['params']) == 1 and group['params'][0] is named[spec['parameter_names'][0]],
                    'Own optimizer group policy/identity/LR differs')
        require(type(self.completed_steps) is int and self.completed_steps >= 0
                and type(self.gradient_steps) is dict and self.gradient_steps.keys() == self._parameters.keys()
                and all(type(v) is int and v == self.completed_steps for v in self.gradient_steps.values()),
                'Own complete perparameter exposure required')

    def step(self):
        self.validate()
        # Preflight all gradients before independent decay or either child mutates anything.
        for group in self.param_groups:
            p = group['params'][0]
            require(p.grad is not None and not p.grad.is_sparse and p.grad.shape == p.shape
                    and p.grad.dtype == p.dtype and p.grad.device == p.device and bool(torch.isfinite(p.grad).all()),
                    'Own finite full gradient exposure required')
        super().step()
        self.completed_steps += 1
        self.gradient_steps = {n: self.completed_steps for n in self.gradient_steps}
        self.owner().weight_epoch += 1

    def _preflight(self, state):
        require(type(state) is dict and set(state) == {'version', 'independent_decay', 'optimizers',
                'future_token_signature', 'completed_steps', 'gradient_steps'}
                and state['version'] == 1 and type(state['version']) is int
                and type(state['independent_decay']) is float and state['independent_decay'] == self.decay
                and state['future_token_signature'] == self.architecture, 'Own future-token optimizer signature required')
        step, counts = state['completed_steps'], state['gradient_steps']
        require(type(step) is int and step >= 0 and type(counts) is dict and counts.keys() == self.gradient_steps.keys()
                and all(type(v) is int and v == step for v in counts.values()), 'Own optimizer step/exposure differs')
        require(type(state['optimizers']) is list and len(state['optimizers']) == 2, 'Own child optimizer states required')
        offset = 0
        for index, (child, saved) in enumerate(zip(self.optimizers, state['optimizers'])):
            wanted = child.state_dict()['param_groups']
            require(type(saved) is dict and set(saved) == {'state', 'param_groups'} and type(saved['state']) is dict
                    and all(type(k) is int for k in saved['state'])
                    and type(saved['param_groups']) is list and len(saved['param_groups']) == len(wanted),
                    'Own child state/group inventory differs')
            expected_ids = []
            for local, (actual, expected, live) in enumerate(zip(saved['param_groups'], wanted, child.param_groups)):
                require(type(actual) is dict and {k: v for k, v in actual.items() if k != 'lr'} ==
                        {k: v for k, v in expected.items() if k != 'lr'}
                        and type(actual.get('lr')) is float and math.isfinite(actual['lr'])
                        and 0 <= actual['lr'] <= self._maximum_lrs[offset+local], 'Own saved child group/LR differs')
                ids = actual['params']
                require(type(ids) is list and len(ids) == 1 and type(ids[0]) is int, 'Own one-parameter state IDs required')
                pid, parameter = ids[0], live['params'][0]
                expected_ids.append(pid)
                if step == 0:
                    continue
                require(pid in saved['state'] and type(saved['state'][pid]) is dict, 'Missing complete optimizer moments')
                moments = saved['state'][pid]
                keys = {'momentum_buffer'} if index == 0 else {'step', 'exp_avg', 'exp_avg_sq'}
                require(set(moments) == keys, 'Own exact optimizer moment inventory required')
                for key in keys-{'step'}:
                    value = moments[key]
                    require(type(value) is torch.Tensor and value.shape == parameter.shape and value.dtype == parameter.dtype
                            and value.device == parameter.device and not value.requires_grad and bool(torch.isfinite(value).all()),
                            'Own same-backend finite moment tensor required')
                    if key == 'exp_avg_sq':
                        require(bool((value >= 0).all()), 'Negative second moments forbidden')
                if index == 1:
                    value = moments['step']
                    require(type(value) is torch.Tensor and value.shape == () and value.dtype == torch.float32
                            and value.device.type == 'cpu' and not value.requires_grad and float(value) == step,
                            'Own exact CPU Adam step required')
            require(len(expected_ids) == len(set(expected_ids)) and set(saved['state']) == (set(expected_ids) if step else set()),
                    'Foreign/extra optimizer moment IDs')
            offset += len(wanted)

    def state_dict(self):
        self.validate()
        state = dict(super().state_dict(), future_token_signature=copy.deepcopy(self.architecture),
                     completed_steps=self.completed_steps, gradient_steps=copy.deepcopy(self.gradient_steps))
        self._preflight(state)
        return state

    def load_state_dict(self, state):
        self.validate()
        self._preflight(state)
        # Preflight and clone BEFORE either child mutation; future resource gates must budget this clone.
        owned = copy.deepcopy(state)
        super().load_state_dict(owned)
        self.completed_steps, self.gradient_steps = owned['completed_steps'], copy.deepcopy(owned['gradient_steps'])
        self.owner().weight_epoch += 1


def build_optimizer(model, training):
    model.validate()
    t = training
    require(t.optimizer == 'muon' and t.decay_mode == 'independent' and type(t.weight_decay) is float
            and t.weight_decay == .00006, 'Native fixed optimizer/independent decay required')
    require(type(t.learning_rate) is float and t.learning_rate in (.000625, .00125, .0025)
            and type(t.adamw_beta2) is float and t.adamw_beta2 == .999
            and type(t.adamw_epsilon) is float and t.adamw_epsilon == 1e-6
            and t.adamw_beta2_policy == 'mla4_adamw_beta2_fixed_v1'
            and t.adamw_epsilon_policy == 'mla4_adamw_epsilon_fixed_v1', 'Native finite LR and adopted moments required')
    muon, adam = [], []
    for name, parameter in model.named_parameters():
        hidden = hidden_matrix(name, parameter)
        scale = .5 if hidden else 1.
        group = dict(params=[parameter], lr=t.learning_rate*scale, weight_decay=0.,
                     parameter_names=[name], width_lr_scale=scale)
        (muon if hidden else adam).append(group)
    require(len(muon) == model.signature['muon_groups'] and len(adam) == model.signature['adam_groups'],
            'Own exact complete Muon/AdamW grouping required')
    children = [torch.optim.Muon(muon, lr=t.learning_rate, weight_decay=0., adjust_lr_fn='match_rms_adamw'),
                torch.optim.AdamW(adam, lr=t.learning_rate, weight_decay=0., betas=(.9, .999), eps=1e-6)]
    return Optimizer(children, model)
