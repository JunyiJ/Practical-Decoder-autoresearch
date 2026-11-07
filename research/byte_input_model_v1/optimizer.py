"""Own canonical optimizer math SOURCE; no state or worker admission."""
import math
import torch
from . import contract as C
from .model import ByteInputModel


def is_muon(name, parameter):
    return parameter.ndim == 2 and (name.startswith('transformer.h.') or
           name.startswith('composer.filters.') or name in (
               'composer.highway_weight', 'composer.transform_weight',
               'composer.projection_weight'))


class ByteInputOptimizer:
    def __init__(self, model, learning_rate):
        C.require(type(model) is ByteInputModel, 'Own model required')
        model.validate()
        C.require(type(learning_rate) is float and math.isfinite(learning_rate) and
                  learning_rate > 0., 'Positive finite base LR')
        self.model = model
        self._items = model.ordered_parameters()
        muon, adam = [], []
        self._scales = []
        for name, parameter in self._items:
            use_muon = is_muon(name, parameter)
            scale = .5 if use_muon else 1.
            group = dict(params=[parameter], lr=learning_rate*scale, weight_decay=0.)
            (muon if use_muon else adam).append(group)
        _, _, muon_count, adam_count = C.counts(model.spec.variant)
        C.require(len(muon) == muon_count and len(adam) == adam_count, 'Full canonical optimizer grouping')
        self._optimizers = (
            torch.optim.Muon(muon, adjust_lr_fn='match_rms_adamw'),
            torch.optim.AdamW(adam, betas=(.9, .999), eps=1e-6))
        self.param_groups = [group for optimizer in self._optimizers for group in optimizer.param_groups]
        by_id = {id(parameter): name for name, parameter in self._items}
        self._bindings = tuple((group, group['params'][0],
                               .5 if is_muon(by_id[id(group['params'][0])], group['params'][0]) else 1.)
                              for group in self.param_groups)
        self._options = tuple({key: value for key, value in group.items() if key not in ('params', 'lr')}
                              for group in self.param_groups)
        self.validate()

    def validate(self, require_gradients=False):
        device = self.model.validate()
        items = self.model.ordered_parameters()
        C.require(len(items) == len(self._items) and all(name == old and parameter is original
                  for (name, parameter), (old, original) in zip(items, self._items)), 'Own canonical parameters')
        actual = [group for optimizer in self._optimizers for group in optimizer.param_groups]
        C.require(len(actual) == len(self._bindings) == len(self.param_groups) and
                  all(a is b for a, b in zip(actual, self.param_groups)), 'Exact child group ownership')
        base_lr = None
        for index, (group, parameter, scale) in enumerate(self._bindings):
            C.require(actual[index] is group and len(group['params']) == 1 and group['params'][0] is parameter,
                      'Canonical single-parameter groups')
            C.require(C.same_typed({key: value for key, value in group.items() if key not in ('params', 'lr')},
                                  self._options[index]), 'Unchanged optimizer policy')
            lr = group['lr']
            C.require(type(lr) is float and math.isfinite(lr) and lr >= 0., 'Finite scheduled LR')
            if base_lr is None:
                base_lr = lr / scale
            C.require(lr / scale == base_lr, 'Shared scheduled base LR')
            if require_gradients:
                gradient = parameter.grad
                C.require(type(gradient) is torch.Tensor and gradient.shape == parameter.shape and
                          gradient.device == device and gradient.dtype == parameter.dtype and
                          gradient.layout == torch.strided and bool(torch.isfinite(gradient).all()),
                          'All finite dense gradients before any mutation')
        C.require(len({id(parameter) for _, parameter, _ in self._bindings}) == len(items),
                  'Full unique canonical exposure')

    def zero_grad(self, set_to_none=True):
        C.require(type(set_to_none) is bool, 'Explicit gradient clearing mode')
        self.validate()
        for optimizer in self._optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.validate(require_gradients=True)
        with torch.no_grad():
            for _, parameter in self._items:
                parameter.mul_(1. - .00006)
        for optimizer in self._optimizers:
            optimizer.step()

    def state_dict(self, *args, **kwargs):
        raise ValueError('Byte-input optimizer persistence is unqualified')

    def load_state_dict(self, *args, **kwargs):
        raise ValueError('Byte-input optimizer loading is unqualified')


def build_optimizer(model, learning_rate):
    return ByteInputOptimizer(model, learning_rate)
