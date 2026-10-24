import math

import torch
import torch.nn as nn

from ..models.rope import apply_rotary_emb, resolve_position_encoding


def latent_dimensions(cfg):
    """One contract for attention projections and their inference cache."""
    policy = getattr(cfg, 'attention_width_policy', None)
    if policy is not None:
        if (policy != 'mla_quarter_head32' or cfg.dim not in {64,128,256}
                or cfg.n_heads != cfg.dim // 32
                or any(hasattr(cfg, k) for k in ['dimckv','dimcq','head_dim_r','n_kv_heads','n_groups'])):
            raise ValueError('mla_quarter_head32 requires widths64/128/256, head32 and no latent/rotary/KV overrides')
        return cfg.dim // 4, cfg.dim // 4
    return getattr(cfg, 'dimcq', cfg.dim // 10), getattr(cfg, 'dimckv', cfg.dim // 10)


# https://arxiv.org/abs/2405.04434
class MLA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim
        self.dimc_q, self.dimc_kv = latent_dimensions(cfg)
        self.n_heads = getattr(cfg, "n_heads", 1)
        if self.dim % self.n_heads != 0:
            raise ValueError(f"cfg.dim must be divisible by cfg.n_heads (got {self.dim}, {self.n_heads})")
        self.head_dim = self.dim // self.n_heads
        self.head_dim_r = getattr(cfg, "head_dim_r", self.head_dim)

        attn_bias = getattr(cfg, "attn_bias", True)
        drop = getattr(cfg, "dropout", 0.0)
        attn_drop = getattr(cfg, "attn_dropout", drop)
        self.causal = getattr(cfg, "causal", True)

        self.dkv = nn.Linear(self.dim, self.dimc_kv, bias=attn_bias)
        self.uk = nn.Linear(self.dimc_kv, self.head_dim * self.n_heads, bias=attn_bias)
        self.uv = nn.Linear(self.dimc_kv, self.head_dim * self.n_heads, bias=attn_bias)
        self.dq = nn.Linear(self.dim, self.dimc_q, bias=attn_bias)
        self.uq = nn.Linear(self.dimc_q, self.head_dim * self.n_heads, bias=attn_bias)
        self.qr = nn.Linear(self.dimc_q, self.head_dim_r * self.n_heads, bias=attn_bias)
        self.kr = nn.Linear(self.dim, self.head_dim_r, bias=attn_bias)

        self.out = nn.Linear(self.dim, self.dim, bias=attn_bias)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.out_dropout = nn.Dropout(drop)

        self.max_seq_len = getattr(cfg, "block_size", None)
        if self.causal and self.max_seq_len is not None:
            mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool))
            self.register_buffer("causal_mask", mask, persistent=False)
        else:
            self.register_buffer("causal_mask", None, persistent=False)
    
    def _get_causal_mask(self, s, device):
        if self.causal_mask is None or self.causal_mask.size(0) < s:
            self.causal_mask = torch.tril(torch.ones(s, s, device=device, dtype=torch.bool))
        return self.causal_mask[:s, :s]

    def forward(self, x, cache=None, start_pos=0, freq_cos=None, freq_sin=None):
        b, s, d = x.size()
        c_q = self.dq(x)  # [b, s, dimc_q]
        c_kv = self.dkv(x)  # [b, s, dimc_kv]
        # [b, s, head_dim_r * n_heads] => [b, n_heads, s, head_dim_r]
        c_q_r = self.qr(c_q).view(b, s, self.n_heads, -1).transpose(1, 2)
        # [b, s, head_dim_r] => [b, 1, s, head_dim_r]
        k_r = self.kr(x).unsqueeze(1)
        if freq_cos is not None and freq_sin is not None:
            cos = freq_cos[start_pos:start_pos + s].to(device=c_q.device, dtype=c_q.dtype)
            sin = freq_sin[start_pos:start_pos + s].to(device=c_q.device, dtype=c_q.dtype)
            r_q = apply_rotary_emb(c_q_r, cos, sin)
            r_k = apply_rotary_emb(k_r, cos, sin)
        else:
            r_q = c_q_r
            r_k = k_r
        # Cache the shared rotary key and compressed KV, not expanded heads.
        r_k = r_k.squeeze(1)
        if cache is not None:
            c_kv, r_k = cache.update(c_kv, r_k, start_pos)
        total_len = c_kv.size(1)
        q_base = self.uq(c_q).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k_base = self.uk(c_kv).view(b, total_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.uv(c_kv).view(b, total_len, self.n_heads, self.head_dim).transpose(1, 2)
        # Concatenate within each head; concatenating flattened projections first
        # would mix content and rotary features across different heads.
        q = torch.cat([q_base, r_q], dim=-1)
        k_rotary = r_k.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        k = torch.cat([k_base, k_rotary], dim=-1)

        # [b, n_heads, s, s]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim + self.head_dim_r)
        if self.causal:
            total_len = k.size(-2)
            mask = self._get_causal_mask(total_len, x.device)
            mask = mask[start_pos:start_pos + s, :total_len]
            att = att.masked_fill(~mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        # [b, n_heads, s, head_dim]
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, s, d)
        y = self.out_dropout(self.out(y))
        return y, None


class GatedMLA(nn.Module):
    """MLA with optional decoupled RoPE and a channel-wise sigmoid gate."""

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim
        self.dimc_q, self.dimc_kv = latent_dimensions(cfg)
        self.n_heads = getattr(cfg, "n_heads", 1)
        if self.dim % self.n_heads != 0:
            raise ValueError(f"cfg.dim must be divisible by cfg.n_heads (got {self.dim}, {self.n_heads})")
        self.head_dim = self.dim // self.n_heads
        self.head_dim_r = getattr(cfg, "head_dim_r", self.head_dim)
        self.enable_rope = resolve_position_encoding(cfg, attn_type="gated_mla") == "rope"

        attn_bias = getattr(cfg, "attn_bias", True)
        drop = getattr(cfg, "dropout", 0.0)
        attn_drop = getattr(cfg, "attn_dropout", drop)
        self.causal = getattr(cfg, "causal", True)

        self.dkv = nn.Linear(self.dim, self.dimc_kv, bias=attn_bias)
        self.uk = nn.Linear(self.dimc_kv, self.head_dim * self.n_heads, bias=attn_bias)
        self.uv = nn.Linear(self.dimc_kv, self.head_dim * self.n_heads, bias=attn_bias)
        self.dq = nn.Linear(self.dim, self.dimc_q, bias=attn_bias)
        self.uq = nn.Linear(self.dimc_q, self.head_dim * self.n_heads, bias=attn_bias)
        self.qr = nn.Linear(self.dimc_q, self.head_dim_r * self.n_heads, bias=attn_bias)
        self.kr = nn.Linear(self.dim, self.head_dim_r, bias=attn_bias)
        self.gate = nn.Linear(self.dim, self.dim, bias=attn_bias)

        self.out = nn.Linear(self.dim, self.dim, bias=attn_bias)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.out_dropout = nn.Dropout(drop)

        self.max_seq_len = getattr(cfg, "block_size", None)
        if self.causal and self.max_seq_len is not None:
            mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool))
            self.register_buffer("causal_mask", mask, persistent=False)
        else:
            self.register_buffer("causal_mask", None, persistent=False)

    def _get_causal_mask(self, s, device):
        if self.causal_mask is None or self.causal_mask.size(0) < s:
            self.causal_mask = torch.tril(torch.ones(s, s, device=device, dtype=torch.bool))
        return self.causal_mask[:s, :s]

    def forward(self, x, cache=None, start_pos=0, freq_cos=None, freq_sin=None):
        b, s, d = x.size()
        c_q = self.dq(x)  # [b, s, dimc_q]
        c_kv = self.dkv(x)  # [b, s, dimc_kv]
        # [b, s, head_dim_r * n_heads] => [b, n_heads, s, head_dim_r]
        c_q_r = self.qr(c_q).view(b, s, self.n_heads, -1).transpose(1, 2)
        # [b, s, head_dim_r] => [b, 1, s, head_dim_r]
        k_r = self.kr(x).unsqueeze(1)
        if self.enable_rope and freq_cos is not None and freq_sin is not None:
            cos = freq_cos[start_pos:start_pos + s].to(device=c_q.device, dtype=c_q.dtype)
            sin = freq_sin[start_pos:start_pos + s].to(device=c_q.device, dtype=c_q.dtype)
            r_q = apply_rotary_emb(c_q_r, cos, sin)
            r_k = apply_rotary_emb(k_r, cos, sin)
        else:
            r_q = c_q_r
            r_k = k_r
        # Cache the shared rotary key and compressed KV, not expanded heads.
        r_k = r_k.squeeze(1)
        if cache is not None:
            c_kv, r_k = cache.update(c_kv, r_k, start_pos)
        total_len = c_kv.size(1)
        q_base = self.uq(c_q).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k_base = self.uk(c_kv).view(b, total_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.uv(c_kv).view(b, total_len, self.n_heads, self.head_dim).transpose(1, 2)
        # Concatenate within each head; concatenating flattened projections first
        # would mix content and rotary features across different heads.
        q = torch.cat([q_base, r_q], dim=-1)
        k_rotary = r_k.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        k = torch.cat([k_base, k_rotary], dim=-1)

        # [b, n_heads, s, s]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim + self.head_dim_r)
        if self.causal:
            total_len = k.size(-2)
            mask = self._get_causal_mask(total_len, x.device)
            mask = mask[start_pos:start_pos + s, :total_len]
            att = att.masked_fill(~mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        # [b, n_heads, s, head_dim]
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, s, d)
        gate = torch.sigmoid(self.gate(x))
        y = self.out_dropout(self.out(gate * y))
        return y, None
