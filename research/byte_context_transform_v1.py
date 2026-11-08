"""Isolated byte-incumbent context transform; no frozen namespace edits."""
import torch
from torch import nn
from torch.nn import functional as F
from research.byte_input_model_v6 import ByteInputModel
from research.byte_input_model_v6.optimizer import is_muon
from research.byte_input_model_v3 import Spec

ARMS = ('byte', 'linear', 'tanh')


def transform(hidden, weight, arm):
    if arm == 'byte':
        assert weight is None
        return hidden
    assert arm in ('linear', 'tanh')
    result = F.linear(hidden, weight)
    return result if arm == 'linear' else torch.tanh(result)


class Model(nn.Module):
    def __init__(self, arm, composer_seed, projection_seed, codebook):
        super().__init__()
        assert arm in ARMS
        self.arm = arm
        self.base = ByteInputModel(Spec('raw_bytes', composer_seed, codebook.identity), codebook)
        self.codebook_identity = codebook.identity
        self.context_weight = None
        if arm != 'byte':
            generator = torch.Generator(device='cpu').manual_seed(projection_seed)
            self.context_weight = nn.Parameter(torch.empty((256, 256)).uniform_(-1/16, 1/16, generator=generator))
        self.identity = ('byte_context_transform_v1', arm, composer_seed, projection_seed, codebook.identity)
        self.validate()

    def ordered_parameters(self):
        return self.base.ordered_parameters() + (() if self.context_weight is None else (('context_weight', self.context_weight),))

    def validate(self):
        device = self.base.validate()
        items = self.ordered_parameters()
        assert self.arm in ARMS and ((self.context_weight is None) == (self.arm == 'byte'))
        assert len(items) == (105 if self.arm == 'byte' else 106)
        assert sum(p.numel() for _, p in items) == (16351859 if self.arm == 'byte' else 16417395)
        assert {id(p) for _, p in items} == {id(p) for p in self.parameters()}
        assert len({p.untyped_storage().data_ptr() for _, p in items}) == len(items)
        if self.context_weight is not None:
            p = self.context_weight
            assert p.dtype == torch.float32 and p.device == device and tuple(p.shape) == (256, 256)
            assert bool(torch.isfinite(p).all())
        return items

    def forward(self, ids, targets=None):
        device = self.base.validate()
        self.base._ids(ids, device)
        if targets is not None: self.base._ids(targets, device, tuple(ids.shape))
        hidden = self.base._composed_hidden(ids.contiguous())
        logits = self.base.readout(transform(hidden, self.context_weight, self.arm)) / 4.
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, 50257), targets.contiguous().reshape(-1))
        return logits, loss


class Optimizer:
    def __init__(self, model, learning_rate=.0025):
        self.items = model.validate()
        muon, adam = [], []
        for name, p in self.items:
            use_muon = is_muon(name, p) or name == 'context_weight'
            (muon if use_muon else adam).append(dict(params=[p], lr=learning_rate*(.5 if use_muon else 1.), weight_decay=0.))
        assert len(muon) == (53 if model.arm == 'byte' else 54) and len(adam) == 52
        self.children = (torch.optim.Muon(muon, adjust_lr_fn='match_rms_adamw'),
                         torch.optim.AdamW(adam, betas=(.9, .999), eps=1e-6))
        self.param_groups = [g for child in self.children for g in child.param_groups]
        assert len({id(g['params'][0]) for g in self.param_groups}) == len(self.items)
        self.steps = 0

    def zero_grad(self):
        for child in self.children: child.zero_grad(set_to_none=True)

    def step(self):
        assert all(p.grad is not None for _, p in self.items)
        assert bool(torch.stack([torch.isfinite(p.grad).all() for _, p in self.items]).all())
        with torch.no_grad():
            for _, p in self.items: p.mul_(1. - .00006)
        for child in self.children: child.step()
        self.steps += 1


def construct(arm, roles, projection_seed, codebook, device):
    mps = torch.mps.get_rng_state().clone() if torch.backends.mps.is_available() else None
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(roles['native_initializer'])
            return Model(arm, roles['composer_initializer'], projection_seed, codebook).to(device)
    finally:
        if mps is not None: torch.mps.set_rng_state(mps)


def weights(model):
    return {n:p.detach().cpu().clone() for n,p in model.ordered_parameters()}


@torch.no_grad()
def load_weights(model, values):
    items = model.validate()
    assert list(values) == [n for n,p in items]
    for name, p in items:
        v = values[name]
        assert v.shape == p.shape and v.dtype == p.dtype and bool(torch.isfinite(v).all())
    for name, p in items: p.copy_(values[name])
    model.validate()
