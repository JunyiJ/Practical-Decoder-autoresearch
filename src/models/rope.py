import torch


def resolve_position_encoding(cfg, *, attn_type=None):
    """Resolve the model-wide mode while preserving legacy `rope` configs.

    Gated MLA's explicit boolean override selects RoPE or no positional
    encoding. Without the override, `rope=0` retains learned positions.
    """
    from .position_family import resolve_position_family
    family = resolve_position_family(cfg)
    if family is not None:
        if attn_type not in (None, 'mha'):
            raise ValueError('Named position family requires fused MHA')
        return family
    if attn_type is None:
        attn_type = getattr(cfg, "attn_type", None)
    if attn_type == "kda":
        # KDA supplies position-sensitive mixing through convolution/recurrence.
        return "none"
    if attn_type == "gated_mla" and hasattr(cfg, "enable_rope"):
        enabled = cfg.enable_rope
        if not isinstance(enabled, bool):
            raise ValueError("enable_rope must be boolean for gated_mla")
        return "rope" if enabled else "none"
    return "rope" if getattr(cfg, "rope", 0) == 1 else "learned"


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponential (cos + i*sin).
    dim: head_dim
    end: max_seq_len (block_size)
    """
    # base.exp(-2i/d)
    i = torch.arange(0, dim, 2)[: (dim//2)].float()
    freqs = 1.0 / (theta ** (i / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # (max_seq_len, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    """
    x: (B, n_heads, seq_len, head_dim)
    freqs_cis: (seq_len, head_dim // 2)
    """
    B, n_heads, seq_len, head_dim = x.shape
    cos = freqs_cos.view(1, 1, seq_len, head_dim//2)
    sin = freqs_sin.view(1, 1, seq_len, head_dim//2)
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for rotary embeddings")
    x = x.view(B, n_heads, seq_len, head_dim // 2, 2)
    x1s = x[..., 0]
    x2s = x[..., 1]
    x1s_new = x1s * cos - x2s * sin
    x2s_new = x1s * sin + x2s * cos
    x_new = torch.stack([x1s_new, x2s_new], dim=-1).flatten(3)
    return x_new
