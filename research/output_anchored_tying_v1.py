"""Isolated output-coordinate tying experiment; never patches frozen models."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from research.final_ln_scale_model_v1.runtime.models.gpt import GPT
from research.byte_input_model_v3.contract import native_config

ARMS = ('native', 'matched', 'tied')
K = 1 / math.sqrt(192)


class Model(nn.Module):
    def __init__(self, arm):
        super().__init__()
        assert arm in ARMS
        self.arm = arm
        original = GPT(native_config())
        assert sum(p.numel() for p in original.parameters()) == 28492416
        assert original.width_mult == 4 and original.input_embedding_scale == 1
        self.transformer = original.transformer
        self.lm_head = original.lm_head if arm != 'tied' else None
        if arm != 'native':
            with torch.no_grad():
                self.transformer.wte.weight.copy_(original.lm_head.weight)
        self.validate()

    def validate(self):
        parameters = tuple(self.named_parameters())
        assert len(parameters) == (87 if self.arm == 'tied' else 88)
        assert sum(p.numel() for _, p in parameters) == (15626624 if self.arm == 'tied' else 28492416)
        assert len({p.untyped_storage().data_ptr() for _, p in parameters}) == len(parameters)
        assert (self.lm_head is None) == (self.arm == 'tied')
        assert all(p.dtype == torch.float32 for _, p in parameters)
        assert bool(torch.stack([torch.isfinite(p).all() for _, p in parameters]).all())
        return parameters

    def forward(self, ids, targets=None):
        assert ids.dtype == torch.int64 and ids.ndim == 2 and 0 < ids.shape[1] <= 128
        x = self.transformer.wte(ids)
        if self.arm != 'native':
            x = x / K
        for block in self.transformer.h:
            x, auxiliary = block(x, None, 0)
            assert auxiliary is None
        x = self.transformer.ln_f(x)
        weight = self.transformer.wte.weight if self.arm == 'tied' else self.lm_head.weight
        logits = F.linear(x, weight) / 4.
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, 50257), targets.reshape(-1))
        return logits, loss


class Optimizer:
    def __init__(self, model, lr=.00125):
        self.model = model
        self.items = model.validate()
        muon, adam = [], []
        for name, p in self.items:
            is_muon = p.ndim == 2 and name.startswith('transformer.h.')
            (muon if is_muon else adam).append(dict(params=[p], lr=lr*(.5 if is_muon else 1.), weight_decay=0.))
        assert len(muon) == 44 and len(adam) == (43 if model.arm == 'tied' else 44)
        self.children = (torch.optim.Muon(muon, adjust_lr_fn='match_rms_adamw'),
                         torch.optim.AdamW(adam, betas=(.9, .999), eps=1e-6))
        self.param_groups = [g for child in self.children for g in child.param_groups]
        assert len({id(g['params'][0]) for g in self.param_groups}) == len(self.items)
        self.steps = 0

    def zero_grad(self):
        for child in self.children:
            child.zero_grad(set_to_none=True)

    def step(self):
        assert all(p.grad is not None for _, p in self.items)
        assert bool(torch.stack([torch.isfinite(p.grad).all() for _, p in self.items]).all())
        with torch.no_grad():
            for _, p in self.items:
                p.mul_(1. - .00006)
        for child in self.children:
            child.step()
        self.steps += 1


def construct(arm, seed, device):
    # Initialization is CPU-only and deliberately identical before intervention.
    mps = torch.mps.get_rng_state().clone() if torch.backends.mps.is_available() else None
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return Model(arm).to(device)
    finally:
        if mps is not None:
            torch.mps.set_rng_state(mps)
