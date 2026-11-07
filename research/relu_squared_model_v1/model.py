"""Own actual FFN model source; no generated/real-model runtime admission yet."""
import copy
import hashlib
import math
import weakref
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from research.body_allocation_model_v1.model import GPT as Native, Embedding, Attention
from research.ffn_structure_v1.runtime.models.mlp import SwiGLU
from research.final_ln_scale_v1.component import FinalLayerNorm
from research.block_norm_v1.runtime.models.rope import precompute_freqs_cis
from .contract import POLICY, signature, parameter_shapes, ungated_seed, require, plain


class PrivateLinear(nn.Module):
    def __init__(self, inputs, outputs, generator):
        super().__init__()
        self.in_features, self.out_features = inputs, outputs
        self.weight = nn.Parameter(torch.empty(outputs, inputs, dtype=torch.float32, device='cpu'))
        self.register_parameter('bias', None)
        with torch.no_grad():
            self.weight.uniform_(-1/math.sqrt(inputs), 1/math.sqrt(inputs), generator=generator)

    def forward(self, x):
        return F.linear(x, self.weight)


class UngatedFFN(nn.Module):
    def __init__(self, arm, generator):
        super().__init__()
        require(arm in ('ungated_silu', 'ungated_relu_squared'), 'Own ungated equation required')
        self.arm, self.dim, self.hidden_dim = arm, 256, 1008
        self.linear_up = PrivateLinear(256, 1008, generator)
        self.linear_down = PrivateLinear(1008, 256, generator)

    def forward(self, x):
        u = self.linear_up(x)
        hidden = F.silu(u) if self.arm == 'ungated_silu' else F.relu(u).square()
        return self.linear_down(hidden)


class Block(nn.Module):
    def __init__(self, original, arm, generator):
        super().__init__()
        # Registration order matches native; only the FFN module is replaced.
        self.layer_index = original.layer_index
        self.ln1, self.ln2 = original.ln1, original.ln2
        self.attn = original.attn
        self.mlp = original.mlp if arm == 'native_swiglu' else UngatedFFN(arm, generator)
        self.register_buffer('freq_cos', original.freq_cos)
        self.register_buffer('freq_sin', original.freq_sin)

    def forward(self, x, cache, start):
        y, _ = self.attn(self.ln1(x), cache=cache, start_pos=start,
                         freq_cos=self.freq_cos, freq_sin=self.freq_sin)
        x = x+y
        return x+self.mlp(self.ln2(x))


class LayerCache(nn.Module):
    def __init__(self, spec, index, batch, device):
        super().__init__()
        self.signature, self.layer_index, self.position = copy.deepcopy(spec), index, 0
        self.register_buffer('c_cache', torch.zeros(batch, 128, 64, device=device, dtype=torch.float32))
        self.register_buffer('r_cache', torch.zeros(batch, 128, 32, device=device, dtype=torch.float32))

    def update(self, c, r, start):
        require(type(c) is torch.Tensor and type(r) is torch.Tensor
                and type(start) is int and start == self.position and c.ndim == r.ndim == 3
                and c.shape[:2] == r.shape[:2] and c.shape[0] == self.c_cache.shape[0]
                and c.shape[1] > 0 and c.shape[2] == 64 and r.shape[2] == 32
                and c.dtype == r.dtype == torch.float32 and c.device == r.device == self.c_cache.device
                and not torch.is_grad_enabled(), 'Own inference cache update differs')
        end = start+c.shape[1]
        require(0 <= start < end <= 128, 'Own cache position overflow')
        self.c_cache[:, start:end, :] = c
        self.r_cache[:, start:end, :] = r
        self.position = end
        return self.c_cache[:, :end, :], self.r_cache[:, :end, :]


class Cache:
    def __init__(self, model, batch):
        self.owner, self.signature, self.epoch = weakref.ref(model), copy.deepcopy(model.signature), model.weight_epoch
        self.batch = batch
        self.layers = [LayerCache(model.signature, i, batch, model.lm_head.weight.device) for i in range(4)]

    def reset(self):
        model = self.owner()
        require(model is not None, 'Live own cache owner required')
        model.validate()
        model.validate_cache(self, self.batch, None, allow_stale=True)
        # All layers were preflighted before any mutation.
        for layer in self.layers:
            layer.c_cache.zero_(); layer.r_cache.zero_(); layer.position = 0
        self.epoch = model.weight_epoch


@dataclass(frozen=True)
class Distribution:
    policy: str
    model_fingerprint: str
    kind: str
    values: torch.Tensor

    def log_probs(self):
        require(self.policy == POLICY and self.kind == 'relu_squared_family_native_logits',
                'Own FFN tagged logits required')
        require(type(self.values) is torch.Tensor and self.values.dtype == torch.float32
                and self.values.device.type in ('cpu', 'mps') and self.values.ndim == 3
                and 0 < self.values.shape[0] <= 8 and 0 < self.values.shape[1] <= 128
                and self.values.shape[2] == 50257, 'Own bounded logits shape/dtype required')
        return F.log_softmax(self.values, dim=-1)


class GPT(nn.Module):
    def __init__(self, cfg, *, initializer_seed):
        super().__init__()
        require(torch.get_default_dtype() == torch.float32 and torch.get_default_device() == torch.device('cpu'),
                'Genuine native CPU FP32 construction required')
        self.cfg, self.signature = plain(cfg), signature(cfg)
        self.initializer_seed, self.private_ungated_seed = initializer_seed, ungated_seed(initializer_seed)
        self._initialization_identity = (self.initializer_seed, self.private_ungated_seed)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(initializer_seed)
            original = Native(copy.deepcopy(self.cfg['native_template']))
            self.native_cpu_draw_digest = hashlib.sha256(bytes(torch.get_rng_state().tolist())).hexdigest()
        arm = self.signature['arm']
        generator = None if arm == 'native_swiglu' else torch.Generator(device='cpu').manual_seed(self.private_ungated_seed)
        self.transformer = nn.ModuleDict(dict(wte=original.transformer.wte,
            h=nn.ModuleList([Block(b, arm, generator) for b in original.transformer.h]),
            ln_f=original.transformer.ln_f))
        self.lm_head = original.lm_head
        # Original is not a child; its discarded FFNs never enter own registration/state/optimizer.
        del original
        self.width_mult, self.weight_epoch = 4., 0
        self.validate()

    def validate(self):
        s = self.signature
        require(signature(self.cfg) == s, 'Mutable own FFN architecture differs')
        require((self.initializer_seed, self.private_ungated_seed) == self._initialization_identity
                and self.private_ungated_seed == ungated_seed(self.initializer_seed), 'Initializer identity differs')
        require(self.width_mult == 4. and type(self.weight_epoch) is int and self.weight_epoch >= 0,
                'Native readout scale/cache epoch differs')
        expected = parameter_shapes(s['arm'])
        named = list(self.named_parameters(remove_duplicate=False))
        require([n for n, _ in named] == list(expected), 'Exact ordered own parameter inventory required')
        parameters = [p for _, p in named]
        device = parameters[0].device
        require(device.type in ('cpu', 'mps') and all(type(p) is nn.Parameter and p.dtype == torch.float32
                and p.device == device and p.requires_grad and tuple(p.shape) == expected[n]
                for n, p in named), 'Actual own shaped FP32 learned tensors required')
        require(sum(p.numel() for p in parameters) == s['parameters'] and len(parameters) == s['muon_groups']+44,
                'Actual own parameter/group counts differ')
        require(len({id(p) for p in parameters}) == len(parameters)
                and len({p.untyped_storage().data_ptr() for p in parameters}) == len(parameters), 'Aliased own parameters')
        require(set(self._modules) == {'transformer', 'lm_head'} and type(self.transformer) is nn.ModuleDict
                and list(self.transformer.keys()) == ['wte', 'h', 'ln_f']
                and type(self.transformer.wte) is Embedding and type(self.transformer.h) is nn.ModuleList
                and len(self.transformer.h) == 4 and type(self.lm_head) is nn.Linear and self.lm_head.bias is None,
                'Own tree/embedding/readout layout differs')
        final = self.transformer.ln_f
        require(type(final) is FinalLayerNorm and final.width == 256 and final.kind == 'learned' and final.eps == 1e-5,
                'Native final affine LayerNorm required')
        expected_buffers = []
        for i, block in enumerate(self.transformer.h):
            require(type(block) is Block and type(block.layer_index) is int and block.layer_index == i
                    and set(block._modules) == {'ln1', 'ln2', 'attn', 'mlp'}, 'Own block tree differs')
            for norm in (block.ln1, block.ln2):
                require(type(norm) is nn.RMSNorm and norm.normalized_shape == (256,) and norm.eps == 2**-23
                        and norm.elementwise_affine, 'Native block RMSNorm required')
            a = block.attn
            require(type(a) is Attention and a.dim == 256 and a.dimc_q == a.dimc_kv == 64 and a.n_heads == 8
                    and a.head_dim == a.head_dim_r == 32 and a.attention_logit_scale == .5 and a.causal
                    and a.max_seq_len == 128 and a.attn_dropout.p == a.out_dropout.p == 0., 'Native MLA differs')
            for name, shape in dict(dkv=(64, 256), uk=(256, 64), uv=(256, 64), dq=(64, 256),
                                    uq=(256, 64), qr=(256, 64), kr=(32, 256), out=(256, 256)).items():
                layer = getattr(a, name)
                require(type(layer) is nn.Linear and layer.in_features == shape[1] and layer.out_features == shape[0]
                        and layer.bias is not None, 'Native attention projection differs')
            m = block.mlp
            if s['arm'] == 'native_swiglu':
                require(type(m) is SwiGLU and m.dim == 256 and m.hidden_dim == 672 and m.gate_activation == 'silu'
                        and m.hidden_norm is None and type(m.dropout) is nn.Dropout and m.dropout.p == 0.
                        and set(m._modules) == {'linear_gate', 'linear_up', 'linear_down', 'dropout'}, 'Native SwiGLU differs')
                layers = ((m.linear_gate, 256, 672), (m.linear_up, 256, 672), (m.linear_down, 672, 256))
                require(all(type(layer) is nn.Linear for layer, _, _ in layers), 'Genuine native FFN matrices required')
            else:
                require(type(m) is UngatedFFN and m.arm == s['arm'] and m.dim == 256 and m.hidden_dim == 1008
                        and set(m._modules) == {'linear_up', 'linear_down'}, 'Own ungated FFN differs')
                layers = ((m.linear_up, 256, 1008), (m.linear_down, 1008, 256))
                require(all(type(layer) is PrivateLinear for layer, _, _ in layers), 'Own private FFN matrices required')
            require(all(layer.in_features == inputs and layer.out_features == outputs and layer.bias is None
                        for layer, inputs, outputs in layers), 'Exact biasfree FFN shapes required')
            for name in ('freq_cos', 'freq_sin'):
                value = getattr(block, name)
                require(value.shape == (128, 16) and value.dtype == torch.float32 and value.device == device
                        and not value.requires_grad and name not in block._non_persistent_buffers_set,
                        'Native persistent RoPE buffer geometry differs')
                expected_buffers.append(f'transformer.h.{i}.'+name)
            mask = a.causal_mask
            require(type(mask) is torch.Tensor and mask.shape == (128, 128) and mask.dtype == torch.bool
                    and mask.device == device and not mask.requires_grad and 'causal_mask' in a._non_persistent_buffers_set,
                    'Native nonpersistent causal mask differs')
            expected_buffers.append(f'transformer.h.{i}.attn.causal_mask')
        buffers = list(self.named_buffers(remove_duplicate=False))
        require(set(n for n, _ in buffers) == set(expected_buffers) and len(buffers) == 12, 'Extra/missing own buffers')
        storages = [v.untyped_storage().data_ptr() for _, v in buffers]
        require(len(set(storages)) == len(storages) and not set(storages).intersection(
                p.untyped_storage().data_ptr() for p in parameters), 'Buffer/parameter alias forbidden')

    def caches(self, batch):
        self.validate()
        require(type(batch) is int and 0 < batch <= 8, 'Own bounded cache batch required')
        return Cache(self, batch)

    def validate_cache(self, cache, batch, start, *, allow_stale=False):
        require(type(cache) is Cache and cache.owner() is self and cache.signature == self.signature
                and type(cache.batch) is int and cache.batch == batch
                and type(cache.epoch) is int and (allow_stale or cache.epoch == self.weight_epoch)
                and type(cache.layers) is list and len(cache.layers) == 4, 'Foreign/stale own cache')
        tensors = []
        for i, layer in enumerate(cache.layers):
            require(type(layer) is LayerCache and layer.signature == self.signature and type(layer.layer_index) is int
                    and layer.layer_index == i and type(layer.position) is int and 0 <= layer.position <= 128
                    and (start is None or layer.position == start), 'Own cache position/layer differs')
            for name, shape in (('c_cache', (batch, 128, 64)), ('r_cache', (batch, 128, 32))):
                v = getattr(layer, name)
                require(type(v) is torch.Tensor and v.shape == shape and v.dtype == torch.float32
                        and v.device == self.lm_head.weight.device and v.is_contiguous() and not v.requires_grad,
                        'Own cache storage differs')
                tensors.append(v)
        require(len({id(v) for v in cache.layers}) == 4 and len({v.untyped_storage().data_ptr() for v in tensors}) == 8,
                'Aliased cache layers/storage')
        forbidden = {v.untyped_storage().data_ptr() for v in (*self.parameters(), *self.buffers())}
        require(not any(v.untyped_storage().data_ptr() in forbidden for v in tensors), 'Cache aliases model storage')

    def tokens(self, idx, start=0, cache=None):
        self.validate()
        require(type(idx) is torch.Tensor and idx.dtype == torch.int64 and idx.ndim == 2
                and 0 < idx.shape[0] <= 8 and 0 < idx.shape[1] <= 128 and idx.numel() <= 1024
                and idx.device == self.lm_head.weight.device, 'Own bounded native IDs required')
        require(not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('mps'), 'Autocast forbidden')
        require(type(start) is int and 0 <= start and start+idx.shape[1] <= 128
                and bool(((idx >= 0) & (idx < 50257)).all()), 'Native token range/position differs')
        if cache is None:
            require(start == 0, 'Own offset requires cache')
        else:
            require(not torch.is_grad_enabled(), 'Own cache is inference-only')
            self.validate_cache(cache, idx.shape[0], start)

    def _logits(self, idx, cache=None, start=0):
        x = self.transformer.wte(idx)
        for i, block in enumerate(self.transformer.h):
            x = block(x, None if cache is None else cache.layers[i], start)
        return self.lm_head(self.transformer.ln_f(x))/4.

    def forward(self, idx, *, cache=None, start_pos=0):
        self.tokens(idx, start_pos, cache)
        return Distribution(POLICY, self.signature['model_fingerprint'], 'relu_squared_family_native_logits',
                            self._logits(idx, cache, start_pos))

    def loss(self, idx, targets, *, reduction='mean'):
        self.tokens(idx)
        require(type(targets) is torch.Tensor and targets.shape == idx.shape and targets.dtype == torch.int64
                and targets.device == idx.device and bool(((targets >= 0) & (targets < 50257)).all()), 'Own exact targets required')
        require(type(reduction) is str and reduction in ('none', 'mean'), 'Own loss reduction required')
        return F.cross_entropy(self._logits(idx).reshape(-1, 50257), targets.reshape(-1), reduction=reduction)


def load_weights(model, payload):
    model.validate()
    require(type(payload) is dict and set(payload) == {'kind', 'architecture', 'model_state'}
            and payload['kind'] == 'native_matched_matrix_ffn_weights_v1'
            and payload['architecture'] == model.signature, 'Own FFN weight identity required')
    saved, current = payload['model_state'], model.state_dict()
    require(type(saved) in (dict, type(current)) and list(saved) == list(current), 'Exact ordered persisted FFN state required')
    cos, sin = precompute_freqs_cis(32, 128)
    for name, value in saved.items():
        require(type(value) is torch.Tensor and value.device.type in ('cpu', 'mps') and value.shape == current[name].shape
                and value.dtype == current[name].dtype and bool(torch.isfinite(value).all()), 'Own finite state tensor differs: '+name)
        if name.endswith(('freq_cos', 'freq_sin')):
            require(torch.equal(value.cpu(), cos if name.endswith('freq_cos') else sin), 'Persisted native RoPE differs')
    model.load_state_dict(saved)
    model.weight_epoch += 1
