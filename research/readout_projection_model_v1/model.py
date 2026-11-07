"""Actual native child and one private projection; no mixture or router."""
import copy
import hashlib
import weakref
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from research.body_allocation_model_v1.model import GPT as Native
from research.block_norm_v1.runtime.models.rope import precompute_freqs_cis
from .contract import signature, require, projection_seed, POLICY

KINDS = {'native': 'native_logits', 'linear_projection1': 'linear_projection_logits',
         'tanh_projection1': 'tanh_projection_logits'}


@dataclass(frozen=True)
class Distribution:
    policy: str
    model_fingerprint: str
    kind: str
    values: torch.Tensor

    def log_probs(self):
        require(self.policy == POLICY and self.kind in KINDS.values(), 'Own tagged single-projection logits required')
        return F.log_softmax(self.values, dim=-1)


class Projection(nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(256, 256))
        with torch.no_grad():
            self.weight.uniform_(-1/16, 1/16, generator=generator)


class Cache:
    def __init__(self, model, batch):
        self.owner = weakref.ref(model)
        self.signature = copy.deepcopy(model.signature)
        self.epoch = model.weight_epoch
        self.layers = model.backbone.caches(batch)

    def reset(self):
        model = self.owner()
        require(model is not None and model.signature == self.signature, 'Live own cache owner required')
        for layer in self.layers:
            layer.reset()
        self.epoch = model.weight_epoch


class GPT(nn.Module):
    def __init__(self, cfg, *, initializer_seed):
        super().__init__()
        require(torch.get_default_dtype() == torch.float32 and torch.get_default_device() == torch.device('cpu'),
                'Genuine native CPU FP32 construction required')
        self.cfg = copy.deepcopy(cfg)
        self.signature = signature(cfg)
        self.initializer_seed = initializer_seed
        self.private_projection_seed = projection_seed(initializer_seed)
        self._initialization_identity = (initializer_seed, self.private_projection_seed)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(initializer_seed)
            self.backbone = Native(copy.deepcopy(self.cfg['backbone_template']))
            self.native_cpu_draw_digest = hashlib.sha256(bytes(torch.get_rng_state().tolist())).hexdigest()
        # Register the projection AFTER the native child, preserving common88 parameter order.
        self.add_module('projection', None)
        if self.signature['arm'] != 'native':
            gen = torch.Generator(device='cpu').manual_seed(self.private_projection_seed)
            self.projection = Projection(gen)
        self.width_mult = 4.
        self.weight_epoch = 0
        self.validate()

    def validate(self):
        require(signature(self.cfg) == self.signature, 'Mutable own projection configuration differs')
        require((self.initializer_seed, self.private_projection_seed) == self._initialization_identity
                and self.private_projection_seed == projection_seed(self.initializer_seed), 'Initializer identity differs')
        require(type(self.backbone) is Native and self.backbone.cfg == self.cfg['backbone_template'], 'Genuine native child required')
        self.backbone.validate()
        p = list(self.parameters())
        require(len(p) == self.signature['muon_groups']+self.signature['adam_groups']
                and sum(t.numel() for t in p) == self.signature['parameters'], 'Exact single-projection inventory required')
        device = p[0].device
        require(device.type in ('cpu', 'mps') and all(t.dtype == torch.float32 and t.device == device and t.requires_grad for t in p),
                'CPU/MPS FP32 single-device learned tensors required')
        require(len({id(t) for t in p}) == len(p) and len({t.untyped_storage().data_ptr() for t in p}) == len(p), 'Aliased parameters forbidden')
        require(self.width_mult == 4. and type(self.weight_epoch) is int and self.weight_epoch >= 0, 'Readout/cache epoch differs')
        if self.signature['arm'] == 'native':
            require(self.projection is None, 'Native adds no learned state')
        else:
            require(type(self.projection) is Projection and tuple(self.projection.weight.shape) == (256, 256),
                    'One exact biasfree projection required')

    def caches(self, batch):
        self.validate()
        require(type(batch) is int and 0 < batch <= 8, 'Bounded exact cache batch required')
        return Cache(self, batch)

    def validate_cache(self, cache, batch, start):
        require(type(cache) is Cache and cache.owner() is self and cache.signature == self.signature
                and cache.epoch == self.weight_epoch, 'Foreign/stale cache rejected')
        self.backbone.validate_caches(cache.layers, batch, start)

    def tokens(self, idx, start=0, cache=None):
        self.validate()
        require(type(idx) is torch.Tensor and idx.dtype == torch.int64 and idx.ndim == 2
                and 0 < idx.shape[0] <= 8 and 0 < idx.shape[1] <= 128
                and idx.device == self.backbone.lm_head.weight.device, 'Bounded device-local native tokens required')
        require(not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('mps'), 'Autocast forbidden')
        require(type(start) is int and 0 <= start and start+idx.shape[1] <= 128
                and bool(((idx >= 0) & (idx < 50257)).all()), 'Native IDs/positions differ')
        if cache is None:
            require(start == 0, 'Offset requires caches')
        else:
            require(not torch.is_grad_enabled(), 'Caches are inference only')
            self.validate_cache(cache, idx.shape[0], start)

    def _features(self, idx, cache=None, start=0):
        self.tokens(idx, start, cache)
        x = self.backbone.transformer.wte(idx)
        for i, block in enumerate(self.backbone.transformer.h):
            x = block(x, None if cache is None else cache.layers[i], start)
        return self.backbone.transformer.ln_f(x)

    def _logits(self, z):
        if self.projection is not None:
            z = F.linear(z, self.projection.weight)
            if self.signature['arm'] == 'tanh_projection1':
                z = torch.tanh(z)
        return self.backbone.lm_head(z)/4.

    def forward(self, idx, *, cache=None, start_pos=0):
        return Distribution(POLICY, self.signature['model_fingerprint'], KINDS[self.signature['arm']],
                            self._logits(self._features(idx, cache, start_pos)))

    def loss(self, idx, targets, *, reduction='mean'):
        self.tokens(idx)
        require(type(targets) is torch.Tensor and targets.shape == idx.shape and targets.dtype == torch.int64
                and targets.device == idx.device and bool(((targets >= 0) & (targets < 50257)).all()), 'Exact native targets required')
        require(type(reduction) is str and reduction in ('none', 'mean'), 'Explicit loss reduction required')
        return F.cross_entropy(self._logits(self._features(idx)).reshape(-1, 50257), targets.reshape(-1), reduction=reduction)


def load_weights(model, payload):
    model.validate()
    require(type(payload) is dict and set(payload) == {'kind', 'architecture', 'model_state'}
            and payload['kind'] == 'native_readout_projection_weights_v1'
            and payload['architecture'] == model.signature, 'Exact own projection weights required')
    current, saved = model.state_dict(), payload['model_state']
    require(type(saved) in (dict, type(current)) and saved.keys() == current.keys(), 'Exact persisted state inventory required')
    cos, sin = precompute_freqs_cis(32, 128)
    for name, tensor in saved.items():
        require(type(tensor) is torch.Tensor and tensor.device.type in ('cpu', 'mps')
                and tensor.shape == current[name].shape and tensor.dtype == current[name].dtype
                and bool(torch.isfinite(tensor).all()), 'Finite exact weight tensor required: '+name)
        if name.endswith(('freq_cos', 'freq_sin')):
            require(torch.equal(tensor.cpu(), cos if name.endswith('freq_cos') else sin), 'Persisted RoPE differs')
    # Native causal masks are reconstructible and nonpersistent, never indexed in saved state.
    model.load_state_dict(saved)
    model.weight_epoch += 1
