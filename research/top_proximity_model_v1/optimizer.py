"""Own finite-gradient optimizer composition; SOURCE ONLY, persistence unsupported."""
import math
import torch
from .model import TopModel
from . import contract as C


class TopOptimizer:
    def __init__(self, model, learning_rate):
        C.require(type(model) is TopModel, 'Own exact TOP model required')
        model.validate()
        C.require(type(learning_rate) is float and math.isfinite(learning_rate) and 0. < learning_rate <= 1.,
                  'Positive bounded Python-float base LR required')
        self.model, self.identity = model, model.identity
        self._items = model.ordered_parameters()
        C.require(len({id(p) for _, p in self._items}) == len(self._items), 'Duplicate optimizer parameter objects')
        matrices, adam = [], []
        for name, parameter in self._items:
            hidden = name.startswith('transformer.h.') and parameter.ndim == 2
            scale = .5 if hidden else 1.
            group = dict(params=[parameter], lr=learning_rate * scale, weight_decay=0.,
                         parameter_names=[name], width_lr_scale=scale)
            (matrices if hidden else adam).append(group)
        C.require(len(matrices) == 44 and len(adam) == (44 if model.spec.variant == 'native' else 45),
                  'Exact native/augmented grouping required')
        self.optimizers = (
            torch.optim.Muon(matrices, lr=learning_rate, weight_decay=0., adjust_lr_fn='match_rms_adamw'),
            torch.optim.AdamW(adam, lr=learning_rate, betas=(.9, .999), eps=1e-6, weight_decay=0.),
        )
        self.param_groups = [group for optimizer in self.optimizers for group in optimizer.param_groups]
        self._policies = tuple({key: value for key, value in group.items()
                                if key not in ('params', 'lr', 'parameter_names', 'width_lr_scale')}
                               for group in self.param_groups)
        self._bindings = tuple((group, tuple(group['parameter_names']), group['params'][0], group['width_lr_scale'])
                               for group in self.param_groups)
        self.validate(require_gradients=False)

    def validate(self, *, require_gradients):
        self.model.validate()
        C.require(self.model.identity == self.identity, 'Foreign or changed TOP optimizer identity')
        items = self.model.ordered_parameters()
        C.require(len(items) == len(self._items) and all(a == b and p is q for (a, p), (b, q) in zip(items, self._items)),
                  'Optimizer model parameter coverage/order changed')
        groups = [g for optimizer in self.optimizers for g in optimizer.param_groups]
        C.require(len(groups) == len(self._bindings) and len(self.param_groups) == len(groups)
                  and all(a is b for a, b in zip(groups, self.param_groups)), 'Optimizer group ownership differs')
        rates = []
        for actual, (original, names, parameter, scale), policy in zip(groups, self._bindings, self._policies):
            C.require(actual is original and len(actual['params']) == 1 and actual['params'][0] is parameter
                      and actual['parameter_names'] == list(names) and actual['width_lr_scale'] == scale
                      and actual['weight_decay'] == 0., 'Optimizer grouping/decay differs')
            options = {key: value for key, value in actual.items()
                       if key not in ('params', 'lr', 'parameter_names', 'width_lr_scale')}
            C.require(C.same_typed(options, policy), 'Optimizer child options changed')
            rate = actual['lr']
            C.require(type(rate) is float and math.isfinite(rate) and 0. <= rate <= scale,
                      'Finite bounded scheduled group LR required')
            rates.append(rate / scale)
            if require_gradients:
                grad = parameter.grad
                C.require(type(grad) is torch.Tensor and grad.layout == torch.strided and grad.shape == parameter.shape
                          and grad.dtype == parameter.dtype and grad.device == parameter.device
                          and bool(torch.isfinite(grad).all()), 'All finite dense same-device gradients required before mutation')
        C.require(all(rate == rates[0] for rate in rates), 'Uniform native width-scaled LR schedule required')
        C.require(type(self.optimizers[0]) is torch.optim.Muon and type(self.optimizers[1]) is torch.optim.AdamW,
                  'Exact Muon/AdamW children required')
        for group in self.optimizers[0].param_groups:
            C.require(group['adjust_lr_fn'] == 'match_rms_adamw', 'Muon RMS-matching policy differs')
        for group in self.optimizers[1].param_groups:
            C.require(group['betas'] == (.9, .999) and group['eps'] == 1e-6, 'AdamW policy differs')

    def zero_grad(self, set_to_none=True):
        C.require(type(set_to_none) is bool, 'Exact boolean zero-grad policy required')
        self.validate(require_gradients=False)
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        # ALL gradients preflight before decay or either child can mutate anything.
        self.validate(require_gradients=True)
        with torch.no_grad():
            for _, parameter in self._items:
                parameter.mul_(1. - .00006)
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self, *args, **kwargs):
        raise ValueError('TOP model-math optimizer persistence is unqualified')

    def load_state_dict(self, *args, **kwargs):
        raise ValueError('TOP model-math optimizer loader is unqualified')


def build_optimizer(model, learning_rate):
    return TopOptimizer(model, learning_rate)
