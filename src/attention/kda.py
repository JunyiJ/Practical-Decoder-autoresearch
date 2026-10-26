"""Sequential reference KDA with Kimi K3's bounded decay and full-rank gate.

Equations (1), (2), (5), (6): https://arxiv.org/html/2607.24653v2#S2.SS1.SSS1
This is a correctness-oriented implementation, not a fused/chunk-parallel kernel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KDA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim
        self.n_heads = getattr(cfg, "n_heads", 1)
        if self.n_heads < 1 or self.dim < 1 or self.dim % self.n_heads != 0:
            raise ValueError("dim and n_heads must be positive, with dim divisible by n_heads")
        self.head_dim = self.dim // self.n_heads
        self.dim_alpha = getattr(cfg, "dim_alpha", max(1, self.head_dim // 4))
        if self.dim_alpha < 1:
            raise ValueError("dim_alpha must be positive")
        if not getattr(cfg, "causal", True):
            raise ValueError("KDA only supports causal attention")
        attn_bias = getattr(cfg, "attn_bias", True)
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=attn_bias)
        self.conv_size = 4
        self.q_conv = nn.Conv1d(self.dim, self.dim, self.conv_size, groups=self.dim, bias=False)
        self.k_conv = nn.Conv1d(self.dim, self.dim, self.conv_size, groups=self.dim, bias=False)
        self.v_conv = nn.Conv1d(self.dim, self.dim, self.conv_size, groups=self.dim, bias=False)
        self.beta_proj = nn.Linear(self.dim, self.n_heads, bias=attn_bias)
        self.z1 = nn.Linear(self.dim, self.dim_alpha, bias=False)
        self.z2 = nn.Linear(self.dim_alpha, self.dim, bias=True)
        self.g_min = -5.0
        self.A = nn.Parameter(torch.zeros(self.n_heads))
        self.head_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.output_gate = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_dropout = nn.Dropout(getattr(cfg, "dropout", 0.0))

    def forward(self, x, cache=None, start_pos=0, freq_cos=None, freq_sin=None):
        b, s, d = x.shape
        if s == 0 or d != self.dim:
            raise ValueError("KDA requires a nonempty sequence with the configured hidden dimension")
        if freq_cos is not None or freq_sin is not None:
            raise ValueError("KDA uses causal convolution and recurrence, not RoPE")
        if cache is None:
            if start_pos != 0:
                raise ValueError("Nonzero start_pos requires a KDA cache")
        else:
            if self.training or torch.is_grad_enabled():
                raise ValueError("KDA cache requires eval mode with gradients disabled")
            if getattr(cache, "attn_type", None) != "kda":
                raise ValueError("KDA requires a KDA cache")
            if start_pos != cache.position:
                raise ValueError("KDA cache requires contiguous start_pos; reset before reusing")
            if start_pos + s > cache.max_seq_len:
                raise ValueError("Cache size exceeded; increase block_size or reset cache.")
            if cache.state.shape != (b, self.n_heads, self.head_dim, self.head_dim):
                raise ValueError("KDA cache batch/head shape does not match input")

        features = []
        histories = []
        for name, projected in zip(("q", "k", "v"), self.qkv(x).chunk(3, dim=-1)):
            projected = projected.transpose(1, 2)
            if cache is None:
                padded = F.pad(projected, (self.conv_size - 1, 0))
            else:
                history = getattr(cache, name + "_history").to(projected.dtype)
                padded = torch.cat((history, projected), dim=-1)
            histories.append(padded[..., -(self.conv_size - 1):])
            convolved = F.silu(getattr(self, name + "_conv")(padded))
            features.append(convolved.transpose(1, 2).reshape(b, s, self.n_heads, self.head_dim))
        output_dtype = features[2].dtype
        z = self.z2(self.z1(x)).reshape(b, s, self.n_heads, self.head_dim)
        beta_logits = self.beta_proj(x)

        # Keep normalization, gates, and recurrent accumulation in FP32 under
        # mixed precision. Preserve FP64 for numerical reference/gradient checks.
        state_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            q, k, v = (feature.to(state_dtype) for feature in features)
            q = F.normalize(q, p=2, dim=-1, eps=1e-6)
            k = F.normalize(k, p=2, dim=-1, eps=1e-6)
            g = self.g_min * torch.sigmoid(self.A.to(state_dtype).exp()[None, None, :, None] * z.to(state_dtype))
            alpha = g.exp()
            beta = torch.sigmoid(beta_logits.to(state_dtype))
            state = (torch.zeros(b, self.n_heads, self.head_dim, self.head_dim,
                                 device=x.device, dtype=state_dtype)
                     if cache is None else cache.state.to(state_dtype))
            outputs = []
            for t in range(s):
                decayed = alpha[:, t].unsqueeze(-1) * state
                predicted_v = (decayed.transpose(-1, -2) @ k[:, t].unsqueeze(-1)).squeeze(-1)
                error = v[:, t] - predicted_v
                state = decayed + beta[:, t, :, None, None] * k[:, t].unsqueeze(-1) * error.unsqueeze(-2)
                outputs.append((state.transpose(-1, -2) @ q[:, t].unsqueeze(-1)).squeeze(-1))
            o = torch.stack(outputs, dim=1)
            o = F.rms_norm(o, (self.head_dim,), self.head_norm.weight.to(state_dtype), eps=1e-6)

        if cache is not None:
            cache.state.copy_(state)
            for name, history in zip(("q", "k", "v"), histories):
                getattr(cache, name + "_history").copy_(history)
            cache.position += s
        o = o.to(output_dtype).reshape(b, s, self.dim)
        gate = torch.sigmoid(self.output_gate(x))
        return self.out_dropout(self.out_proj(gate * o)), None
