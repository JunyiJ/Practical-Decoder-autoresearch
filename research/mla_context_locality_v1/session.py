"""256-token inference with a trained-range 128-key window at each layer.

Own policy and caches; the immutable global-context model is never patched.
Cross-layer information may travel farther than one attention window.
"""
import copy, math
import torch
from research.body_allocation_model_v1.contract import require
from research.mla_context_extension_v1.model import GPT, Cache
from research.block_norm_v1.runtime.models.rope import apply_rotary_emb

POLICY = 'incumbent_mla_local128_context256_fixed_weight_diagnostic_v1'

class Session:
    def __init__(self, model):
        require(type(model) is GPT, 'Reviewed inference-only extension required')
        model.validate(); require(model.checkpoint_sha is not None, 'Verified H200 required')
        self.model = model; self._model_id = id(model)
        self._signature = dict(model.cache_signature(), locality_policy=POLICY, attention_window=128)
        pos = torch.arange(256, device=model.lm_head.weight.device)
        self.mask = (pos[:, None] >= pos[None, :]) & (pos[:, None] - pos[None, :] < 128)
        self._mask_stamp = self.mask_stamp()

    def mask_stamp(self):
        t = self.mask
        return (id(t), t.untyped_storage().data_ptr(), t._version, t.dtype, t.device, tuple(t.shape))

    def validate(self):
        require(id(self.model) == self._model_id, 'Session model changed')
        self.model.validate()
        require(dict(self.model.cache_signature(), locality_policy=POLICY, attention_window=128) == self._signature,
                'Session/checkpoint policy changed')
        require(self.mask_stamp() == self._mask_stamp, 'Locality mask mutated or moved')

    def signature(self):
        self.validate(); return copy.deepcopy(self._signature)

    def caches(self, batch):
        spec = self.signature()
        return [Cache(spec, i, batch, self.model.lm_head.weight.device) for i in range(4)]

    def validate_caches(self, caches, batch, start):
        require(type(caches) in (list, tuple) and len(caches) == 4, 'Four locality caches required')
        tensors = []
        for i, c in enumerate(caches):
            require(type(c) is Cache and c.signature == self._signature and c.layer_index == i
                    and type(c.position) is int and c.position == start, 'Foreign locality/checkpoint/position rejected')
            require(tuple(c.c_cache.shape) == (batch, 256, 64) and tuple(c.r_cache.shape) == (batch, 256, 32),
                    'Locality cache geometry differs')
            for t in (c.c_cache, c.r_cache):
                require(t.dtype == torch.float32 and t.device == self.mask.device and t.is_contiguous()
                        and not t.requires_grad, 'Locality cache dtype/device/layout differs')
                tensors.append(t)
        ptrs = [t.untyped_storage().data_ptr() for t in tensors]
        forbidden = {t.untyped_storage().data_ptr() for t in (*self.model.parameters(), *self.model.buffers(), self.mask)}
        require(len(set(ptrs)) == 8 and not set(ptrs).intersection(forbidden), 'Locality cache aliases forbidden')

    def attention(self, a, x, cos, sin, cache, start):
        b, n, d = x.shape
        cq = a.dq(x); ckv = a.dkv(x)
        rq = apply_rotary_emb(a.qr(cq).view(b, n, 8, 32).transpose(1, 2), cos[start:start+n], sin[start:start+n])
        rk = apply_rotary_emb(a.kr(x).unsqueeze(1), cos[start:start+n], sin[start:start+n]).squeeze(1)
        if cache is not None: ckv, rk = cache.update(ckv, rk, start)
        total = ckv.shape[1]
        qb = a.uq(cq).view(b, n, 8, 32).transpose(1, 2)
        kb = a.uk(ckv).view(b, total, 8, 32).transpose(1, 2)
        v = a.uv(ckv).view(b, total, 8, 32).transpose(1, 2)
        q = torch.cat((qb, rq), -1); k = torch.cat((kb, rk.unsqueeze(1).expand(-1, 8, -1, -1)), -1)
        score = ((q @ k.transpose(-2, -1)) / math.sqrt(64)) * .5
        score = score.masked_fill(~self.mask[start:start+n, :total], float('-inf'))
        y = torch.softmax(score, -1) @ v
        return a.out(y.transpose(1, 2).contiguous().view(b, n, d))

    def __call__(self, idx, *, cache=None, start_pos=0):
        self.validate(); m = self.model
        require(not torch.is_grad_enabled() and not torch.is_autocast_enabled(self.mask.device.type),
                'FP32 no-grad inference only')
        require(idx.dtype == torch.int64 and idx.ndim == 2 and idx.shape[0] in (1, 8)
                and 0 < idx.shape[1] <= 256 and idx.device == self.mask.device
                and type(start_pos) is int and 0 <= start_pos and start_pos + idx.shape[1] <= 256,
                'Bounded locality token layout required')
        require(not bool((idx < 0).any()) and not bool((idx >= 50257).any()), 'Token outside vocabulary')
        if cache is None: require(start_pos == 0, 'Offset requires locality caches')
        else: self.validate_caches(cache, idx.shape[0], start_pos)
        x = m.transformer.wte(idx)
        for i, block in enumerate(m.transformer.h):
            x = x + self.attention(block.attn, block.ln1(x), block.freq_cos, block.freq_sin,
                                   None if cache is None else cache[i], start_pos)
            x = x + block.mlp(block.ln2(x))
        return m.lm_head(m.transformer.ln_f(x)) / m.width_mult
