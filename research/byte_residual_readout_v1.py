"""Zero-gated residual readout; isolated from closed transform studies."""
import torch
from torch import nn
from torch.nn import functional as F
from research.byte_input_model_v6 import ByteInputModel
from research.byte_input_model_v6.optimizer import is_muon
from research.byte_input_model_v3 import Spec
from research.byte_context_transform_v1 import Optimizer as PriorOptimizer

ARMS = ('byte', 'residual_linear', 'residual_tanh')


def transform(hidden, weight, gate, arm):
    if arm == 'byte':
        assert weight is None and gate is None
        return hidden
    assert arm in ARMS[1:]
    branch = F.linear(hidden, weight)
    if arm == 'residual_tanh':
        branch = branch.tanh()
    return hidden + gate * branch


class Model(nn.Module):
    def __init__(self, arm, composer_seed, projection_seed, codebook):
        super().__init__()
        assert arm in ARMS
        self.arm = arm
        self.base = ByteInputModel(Spec('raw_bytes', composer_seed, codebook.identity), codebook)
        self.codebook_identity = codebook.identity
        self.context_weight = self.context_gate = None
        if arm != 'byte':
            generator = torch.Generator(device='cpu').manual_seed(projection_seed)
            self.context_weight = nn.Parameter(torch.empty(256, 256).uniform_(-1/16, 1/16, generator=generator))
            self.context_gate = nn.Parameter(torch.zeros(()))
        self.identity = ('byte_residual_readout_v1', arm, composer_seed, projection_seed, codebook.identity)
        self.validate()

    def ordered_parameters(self):
        extra = () if self.arm == 'byte' else (('context_weight', self.context_weight), ('context_gate', self.context_gate))
        return self.base.ordered_parameters() + extra

    def validate(self):
        device = self.base.validate()
        assert self.arm in ARMS
        assert (self.context_weight is None) == (self.context_gate is None) == (self.arm == 'byte')
        items = self.ordered_parameters()
        assert len(items) == (105 if self.arm == 'byte' else 107)
        assert sum(p.numel() for _, p in items) == (16351859 if self.arm == 'byte' else 16417396)
        assert {id(p) for _, p in items} == {id(p) for p in self.parameters()}
        assert len({p.untyped_storage().data_ptr() for _, p in items}) == len(items)
        for value, shape in ((self.context_weight, (256, 256)), (self.context_gate, ())):
            if value is not None:
                assert value.dtype == torch.float32 and value.device == device and tuple(value.shape) == shape
                assert bool(torch.isfinite(value).all())
        return items

    def forward(self, ids, targets=None):
        device = self.base.validate()
        self.base._ids(ids, device)
        if targets is not None:
            self.base._ids(targets, device, tuple(ids.shape))
        hidden = self.base._composed_hidden(ids.contiguous())
        hidden = transform(hidden, self.context_weight, self.context_gate, self.arm)
        logits = self.base.readout(hidden) / 4.
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, 50257), targets.contiguous().reshape(-1))
        return logits, loss


class Optimizer(PriorOptimizer):
    def __init__(self, model, learning_rate=.0025):
        self.items = model.validate()
        muon, adam = [], []
        for name, p in self.items:
            use_muon = is_muon(name, p) or name == 'context_weight'
            (muon if use_muon else adam).append(dict(params=[p], lr=learning_rate*(.5 if use_muon else 1.), weight_decay=0.))
        assert len(muon) == (53 if model.arm == 'byte' else 54)
        assert len(adam) == (52 if model.arm == 'byte' else 53)
        self.children = (torch.optim.Muon(muon, adjust_lr_fn='match_rms_adamw'),
                         torch.optim.AdamW(adam, betas=(.9, .999), eps=1e-6))
        self.param_groups = [g for child in self.children for g in child.param_groups]
        assert len({id(g['params'][0]) for g in self.param_groups}) == len(self.items)
        self.steps = 0


def construct(arm, roles, projection_seed, codebook, device):
    mps = torch.mps.get_rng_state().clone() if torch.backends.mps.is_available() else None
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(roles['native_initializer'])
            return Model(arm, roles['composer_initializer'], projection_seed, codebook).to(device)
    finally:
        if mps is not None:
            torch.mps.set_rng_state(mps)
