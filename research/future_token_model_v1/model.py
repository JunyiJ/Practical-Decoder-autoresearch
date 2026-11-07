"""Native child plus training-only copied head; SOURCE only until new proof gates."""
import copy
import hashlib
import weakref
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from research.body_allocation_model_v1.model import GPT as Native, Block, Attention, Cache as NativeCache
from research.ffn_structure_v1.runtime.models.mlp import SwiGLU
from research.final_ln_scale_v1.component import FinalLayerNorm
from research.block_norm_v1.runtime.models.rope import precompute_freqs_cis
from .contract import POLICY, signature, parameter_shapes, plain, require


@dataclass(frozen=True)
class Distribution:
    policy: str
    model_fingerprint: str
    kind: str
    values: torch.Tensor

    def log_probs(self):
        require(self.policy == POLICY and self.kind == 'future_token_primary_native_logits'
                and type(self.model_fingerprint) is str and len(self.model_fingerprint) == 64
                and type(self.values) is torch.Tensor and self.values.ndim == 3
                and self.values.shape[-1] == 50257 and self.values.dtype == torch.float32
                and self.values.device.type in ('cpu', 'mps'), 'Own primary distribution required')
        require(not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('mps'),
                'Autocast forbidden')
        return F.log_softmax(self.values, dim=-1)


class Cache:
    def __init__(self, model, batch):
        self.owner = weakref.ref(model)
        self.signature = copy.deepcopy(model.signature)
        self.batch, self.epoch = batch, model.weight_epoch
        self.layers = model.native.caches(batch)

    def reset(self):
        model = self.owner()
        require(model is not None, 'Live own cache owner required')
        model.validate()
        model.validate_cache(self, self.batch, None, allow_stale=True)
        for layer in self.layers:
            layer.reset()
        self.epoch = model.weight_epoch


class GPT(nn.Module):
    def __init__(self, cfg, *, initializer_seed):
        super().__init__()
        self.cfg, self.signature = plain(cfg), signature(cfg)
        require(type(initializer_seed) is int and 0 <= initializer_seed < 2**63,
                'Explicit newly audited initializer required')
        require(torch.get_default_dtype() == torch.float32 and torch.get_default_device() == torch.device('cpu'),
                'Genuine CPU FP32 native construction required')
        self.initializer_seed = initializer_seed
        self._initializer_identity = initializer_seed
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(initializer_seed)
            self.native = Native(copy.deepcopy(self.cfg['native_template']))
            self.native_cpu_draw_digest = hashlib.sha256(bytes(torch.get_rng_state().tolist())).hexdigest()
        if self.signature['arm'] != 'native_next':
            self.auxiliary = nn.ModuleDict(dict(block=copy.deepcopy(self.native.transformer.h[3]),
                                              norm=copy.deepcopy(self.native.transformer.ln_f)))
        self.weight_epoch = 0
        self.validate()

    def validate(self):
        s = self.signature
        require(signature(self.cfg) == s and self.initializer_seed == self._initializer_identity
                and type(self.weight_epoch) is int and self.weight_epoch >= 0, 'Mutable own model identity differs')
        require(type(self.native) is Native and plain(self.native.cfg) == self.cfg['native_template'],
                'Genuine unmodified native child required')
        self.native.validate()
        expected = parameter_shapes(s['arm'])
        named = list(self.named_parameters(remove_duplicate=False))
        require([name for name, _ in named] == list(expected), 'Exact ordered own parameter inventory required')
        parameters = [p for _, p in named]
        device = parameters[0].device
        require(device.type in ('cpu', 'mps') and all(type(p) is nn.Parameter and p.requires_grad
                and p.device == device and p.dtype == torch.float32 and tuple(p.shape) == expected[name]
                for name, p in named), 'Exact learned single-device FP32 parameters required')
        require(len(named) == s['parameter_tensors'] and sum(p.numel() for p in parameters) == s['parameters'],
                'Actual model size differs')
        require(len({id(p) for p in parameters}) == len(parameters)
                and len({p.untyped_storage().data_ptr() for p in parameters}) == len(parameters),
                'Aliased learned parameters forbidden; readout registered only once')
        if s['arm'] == 'native_next':
            require(not hasattr(self, 'auxiliary'), 'Native forbids auxiliary modules')
        else:
            require(type(self.auxiliary) is nn.ModuleDict and list(self.auxiliary) == ['block', 'norm'],
                    'Exact training-only auxiliary modules required')
            b, f = self.auxiliary.block, self.auxiliary.norm
            require(type(b) is Block and b.layer_index == 3 and type(b.attn) is Attention
                    and type(b.mlp) is SwiGLU, 'Genuine copied native final block required')
            for n in (b.ln1, b.ln2):
                require(type(n) is nn.RMSNorm and n.normalized_shape == (256,) and n.eps == 2**-23,
                        'Auxiliary block norm differs')
            a = b.attn
            require(a.dim == 256 and a.dimc_q == a.dimc_kv == 64 and a.n_heads == 8
                    and a.head_dim == a.head_dim_r == 32 and a.attention_logit_scale == .5 and a.causal
                    and a.max_seq_len == 128 and a.attn_dropout.p == a.out_dropout.p == 0.,
                    'Auxiliary causal MLA semantics differ')
            for name, shape in dict(dkv=(64, 256), uk=(256, 64), uv=(256, 64), dq=(64, 256),
                                    uq=(256, 64), qr=(256, 64), kr=(32, 256), out=(256, 256)).items():
                m = getattr(a, name)
                require(type(m) is nn.Linear and tuple(m.weight.shape) == shape and m.bias is not None,
                        'Auxiliary attention projection differs')
            require(b.mlp.dim == 256 and b.mlp.hidden_dim == 672 and b.mlp.gate_activation == 'silu'
                    and b.mlp.hidden_norm is None and b.mlp.dropout.p == 0., 'Auxiliary SwiGLU differs')
            for name in ('linear_gate', 'linear_up', 'linear_down'):
                m = getattr(b.mlp, name)
                require(type(m) is nn.Linear and m.bias is None, 'Auxiliary FFN matrix kind differs')
            require(type(f) is FinalLayerNorm and f.width == 256 and f.kind == 'learned'
                    and f.eps == 1e-5, 'Auxiliary final norm differs')
        blocks = [('native.transformer.h.'+str(i), b) for i, b in enumerate(self.native.transformer.h)]
        if s['arm'] != 'native_next':
            blocks.append(('auxiliary.block', self.auxiliary.block))
        expected_buffers = set()
        expected_mask = torch.ones((128, 128), dtype=torch.bool, device=device).tril()
        for prefix, b in blocks:
            for name in ('freq_cos', 'freq_sin'):
                value = getattr(b, name)
                require(type(value) is torch.Tensor and value.shape == (128, 16)
                        and value.dtype == torch.float32 and value.device == device and not value.requires_grad,
                        'Exact own RoPE buffer shape/dtype/device required')
                expected_buffers.add(prefix+'.'+name)
            a = b.attn; mask = a.causal_mask
            require(type(mask) is torch.Tensor and mask.shape == (128, 128) and mask.dtype == torch.bool
                    and mask.device == device and not mask.requires_grad
                    and 'causal_mask' in a._non_persistent_buffers_set and torch.equal(mask, expected_mask),
                    'Exact own nonpersistent causal mask required')
            expected_buffers.add(prefix+'.attn.causal_mask')
        buffers = list(self.named_buffers(remove_duplicate=False))
        require({name for name, _ in buffers} == expected_buffers and len(buffers) == len(expected_buffers),
                'Extra/missing native or auxiliary buffers')
        storage = [v.untyped_storage().data_ptr() for _, v in buffers]
        require(len(set(storage)) == len(storage) and not set(storage).intersection(
                p.untyped_storage().data_ptr() for p in parameters), 'Buffer/parameter alias forbidden')

    def tokens(self, idx, start=0, cache=None):
        self.validate()
        require(type(idx) is torch.Tensor and idx.layout == torch.strided and idx.dtype == torch.int64
                and idx.ndim == 2 and 0 < idx.shape[0] <= 8 and 0 < idx.shape[1] <= 128
                and idx.numel() <= 1024 and idx.device == self.native.lm_head.weight.device,
                'Own bounded device-local int64 IDs required')
        require(not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('mps'), 'Autocast forbidden')
        require(type(start) is int and 0 <= start and start+idx.shape[1] <= 128
                and bool(((idx >= 0) & (idx < 50257)).all()), 'Own token range/position differs')
        if cache is None:
            require(start == 0, 'Offset requires own cache')
        else:
            require(not torch.is_grad_enabled(), 'Cache is inference-only')
            self.validate_cache(cache, idx.shape[0], start)

    def caches(self, batch):
        self.validate()
        require(type(batch) is int and 0 < batch <= 8, 'Own bounded cache batch required')
        return Cache(self, batch)

    def validate_cache(self, cache, batch, start, *, allow_stale=False):
        require(type(cache) is Cache and cache.owner() is self and cache.signature == self.signature
                and type(cache.batch) is int and cache.batch == batch
                and type(cache.epoch) is int and (allow_stale or cache.epoch == self.weight_epoch)
                and type(cache.layers) is list and len(cache.layers) == 4, 'Foreign/stale own cache')
        require(all(type(c) is NativeCache for c in cache.layers), 'Own native cache layers required')
        position = cache.layers[0].position if start is None else start
        require(type(position) is int and 0 <= position <= 128, 'Own native cache position required')
        self.native.validate_caches(cache.layers, batch, position)
        forbidden = {v.untyped_storage().data_ptr() for v in (*self.parameters(), *self.buffers())}
        require(not any(v.untyped_storage().data_ptr() in forbidden for c in cache.layers
                        for v in (c.c_cache, c.r_cache)), 'Cache aliases owned model storage')

    def forward(self, idx, *, cache=None, start_pos=0):
        self.tokens(idx, start_pos, cache)
        logits, _ = self.native(idx, cache=None if cache is None else cache.layers, start_pos=start_pos)
        return Distribution(POLICY, self.signature['model_fingerprint'], 'future_token_primary_native_logits', logits)

    def loss(self, idx, targets, *, reduction='mean'):
        self.tokens(idx)
        require(type(targets) is torch.Tensor and targets.shape == idx.shape and targets.dtype == torch.int64
                and targets.device == idx.device and bool(((targets >= 0) & (targets < 50257)).all()),
                'Own exact primary targets required')
        require(type(reduction) is str and reduction in ('none', 'mean'), 'Own primary CE reduction required')
        logits, _ = self.native(idx)
        return F.cross_entropy(logits.reshape(-1, 50257), targets.reshape(-1), reduction=reduction)


def load_weights(model, payload):
    model.validate()
    require(type(payload) is dict and set(payload) == {'kind', 'architecture', 'model_state'}
            and payload['kind'] == 'future_token_native_weights_v1'
            and payload['architecture'] == model.signature, 'Own future-token weight identity required')
    saved, current = payload['model_state'], model.state_dict()
    require(type(saved) in (dict, type(current)) and list(saved) == list(current), 'Exact ordered own state required')
    cos, sin = precompute_freqs_cis(32, 128)
    for name, value in saved.items():
        require(type(value) is torch.Tensor and value.device.type in ('cpu', 'mps')
                and value.shape == current[name].shape and value.dtype == current[name].dtype
                and bool(torch.isfinite(value).all()), 'Own finite state tensor differs: '+name)
        if name.endswith(('freq_cos', 'freq_sin')):
            require(torch.equal(value.cpu(), cos if name.endswith('freq_cos') else sin), 'Own persisted RoPE differs')
    model.load_state_dict(saved)
    model.weight_epoch += 1
