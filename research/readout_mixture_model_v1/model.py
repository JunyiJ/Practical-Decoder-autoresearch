import copy
import hashlib
import weakref
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from research.body_allocation_model_v1.model import GPT as Native
from research.block_norm_v1.runtime.models.rope import precompute_freqs_cis
from .contract import signature, require, context_seed, POLICY


@dataclass(frozen=True)
class Distribution:
    policy: str
    model_fingerprint: str
    kind: str
    values: torch.Tensor

    def log_probs(self):
        require(self.policy == POLICY and self.kind in ('native_logits', 'context_mix_logits', 'probability_mix_log_probs'),
                'Own tagged distribution required')
        return self.values if self.kind == 'probability_mix_log_probs' else F.log_softmax(self.values, dim=-1)


class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 256))


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
        self.private_context_seed = context_seed(initializer_seed)
        self._initialization_identity = (initializer_seed, self.private_context_seed)
        # Native child has its TRUE unchanged configuration, never an arm masquerade.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(initializer_seed)
            self.backbone = Native(copy.deepcopy(self.cfg['backbone_template']))
            state = torch.get_rng_state()
            self.native_cpu_draw_digest = hashlib.sha256(bytes(state.tolist())).hexdigest()
        self.context_maps = nn.ParameterList()
        self.add_module('router', None)
        if self.signature['arm'] != 'native':
            gen = torch.Generator(device='cpu').manual_seed(self.private_context_seed)
            for _ in range(2):
                p = nn.Parameter(torch.empty(256, 256))
                with torch.no_grad():
                    p.uniform_(-1/16, 1/16, generator=gen)
                self.context_maps.append(p)
            self.router = Router()
        self.width_mult = 4.
        self.weight_epoch = 0
        self.validate()

    def validate(self):
        require(signature(self.cfg) == self.signature, 'Mutable own model policy differs')
        require((self.initializer_seed, self.private_context_seed) == self._initialization_identity
                and self.private_context_seed == context_seed(self.initializer_seed), 'Initializer identity differs')
        require(type(self.backbone) is Native and self.backbone.cfg == self.cfg['backbone_template'], 'Genuine native child required')
        self.backbone.validate()
        p = list(self.parameters())
        require(len(p) == self.signature['muon_groups']+self.signature['adam_groups']
                and sum(t.numel() for t in p) == self.signature['parameters'], 'Own exact parameter inventory differs')
        device = p[0].device
        require(all(t.dtype == torch.float32 and t.device == device and t.requires_grad for t in p), 'FP32 single-device learned tensors required')
        require(len({id(t) for t in p}) == len(p) and len({t.untyped_storage().data_ptr() for t in p}) == len(p), 'Aliased parameters forbidden')
        require(self.width_mult == 4. and type(self.weight_epoch) is int and self.weight_epoch >= 0, 'Readout/cache state differs')
        if self.signature['arm'] == 'native':
            require(len(self.context_maps) == 0 and self.router is None, 'Native adds no learned state')
        else:
            require(len(self.context_maps) == 2 and all(t.shape == (256, 256) for t in self.context_maps)
                    and type(self.router) is Router and self.router.weight.shape == (2, 256), 'Own contexts/router differ')

    def caches(self, batch):
        self.validate()
        require(type(batch) is int and 0 < batch <= 8, 'Bounded exact cache batch required')
        return Cache(self, batch)

    def validate_cache(self, cache, batch, start):
        require(type(cache) is Cache and cache.owner() is self and cache.signature == self.signature
                and cache.epoch == self.weight_epoch, 'Foreign/stale own cache required to reject')
        self.backbone.validate_caches(cache.layers, batch, start)

    def tokens(self, idx, start=0, cache=None):
        self.validate()
        require(type(idx) is torch.Tensor and idx.dtype == torch.int64 and idx.ndim == 2
                and 0 < idx.shape[0] <= 8 and 0 < idx.shape[1] <= 128
                and idx.device == self.backbone.lm_head.weight.device, 'Bounded device-local native tokens required')
        require(not torch.is_autocast_enabled(idx.device.type), 'Autocast forbidden')
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

    def _contexts(self, z):
        return [torch.tanh(F.linear(z, p)) for p in self.context_maps], F.log_softmax(F.linear(z, self.router.weight), dim=-1)

    def forward(self, idx, *, cache=None, start_pos=0):
        z = self._features(idx, cache, start_pos)
        arm = self.signature['arm']
        if arm == 'native':
            kind, value = 'native_logits', self.backbone.lm_head(z)/4
        else:
            h, lp = self._contexts(z)
            if arm == 'context_mix2':
                value = self.backbone.lm_head(lp[..., 0:1].exp()*h[0]+lp[..., 1:2].exp()*h[1])/4
                kind = 'context_mix_logits'
            else:
                parts = [lp[..., k:k+1]+F.log_softmax(self.backbone.lm_head(h[k])/4, dim=-1) for k in range(2)]
                value, kind = torch.logaddexp(parts[0], parts[1]), 'probability_mix_log_probs'
        return Distribution(POLICY, self.signature['model_fingerprint'], kind, value)

    def loss(self, idx, targets, *, reduction='mean'):
        self.tokens(idx)
        require(type(targets) is torch.Tensor and targets.shape == idx.shape and targets.dtype == torch.int64
                and targets.device == idx.device and bool(((targets >= 0) & (targets < 50257)).all()), 'Exact native targets required')
        require(type(reduction) is str and reduction in ('none', 'mean'), 'Explicit loss reduction required')
        z = self._features(idx)
        y = targets.reshape(-1)
        if self.signature['arm'] == 'native':
            return F.cross_entropy((self.backbone.lm_head(z)/4).reshape(-1, 50257), y, reduction=reduction)
        h, lp = self._contexts(z)
        if self.signature['arm'] == 'context_mix2':
            bar = lp[..., 0:1].exp()*h[0]+lp[..., 1:2].exp()*h[1]
            return F.cross_entropy((self.backbone.lm_head(bar)/4).reshape(-1, 50257), y, reduction=reduction)
        terms = torch.stack([lp[..., k].reshape(-1)-F.cross_entropy(
            (self.backbone.lm_head(h[k])/4).reshape(-1, 50257), y, reduction='none') for k in range(2)], dim=-1)
        result = -torch.logsumexp(terms, dim=-1)
        return result if reduction == 'none' else result.mean()


def load_weights(model, payload):
    model.validate()
    require(type(payload) is dict and set(payload) == {'kind', 'architecture', 'model_state'}
            and payload['kind'] == 'native_readout_mixture_weights_v1'
            and payload['architecture'] == model.signature, 'Exact own readout weights required')
    current, saved = model.state_dict(), payload['model_state']
    require(type(saved) in (dict, type(current)) and saved.keys() == current.keys(), 'Exact persisted state inventory required')
    cos, sin = precompute_freqs_cis(32, 128)
    for name, tensor in saved.items():
        require(type(tensor) is torch.Tensor and tensor.device.type in ('cpu', 'mps')
                and tensor.shape == current[name].shape and tensor.dtype == current[name].dtype
                and bool(torch.isfinite(tensor).all()), 'Finite exact weight tensor required: '+name)
        if name.endswith(('freq_cos', 'freq_sin')):
            require(torch.equal(tensor.cpu(), cos if name.endswith('freq_cos') else sin), 'Persisted RoPE differs')
    # Only persisted buffers are indexed; native causal_mask is reconstructible/nonpersistent.
    model.load_state_dict(saved)
    model.weight_epoch += 1
