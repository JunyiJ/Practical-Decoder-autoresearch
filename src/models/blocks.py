import torch.nn as nn

from ..attention.mha import MHA
from ..attention.gqa import GQA
from ..attention.mla import MLA, GatedMLA
from ..moe.moe_block import MoEBlock

from .rope import precompute_freqs_cis, resolve_position_encoding
from .mlp import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from .ffn_hidden_norm import validate_ffn_hidden_norm_contract, resolve_ffn_hidden_norm
        validate_ffn_hidden_norm_contract(cfg)
        self.dim = cfg.dim
        self.n_heads = getattr(cfg, "n_heads", 1)
        if self.dim % self.n_heads != 0:
            raise ValueError(f"cfg.dim must be divisible by cfg.n_heads (got {self.dim}, {self.n_heads})")
        self.head_dim = self.dim // self.n_heads
        self.head_dim_r = getattr(cfg, "head_dim_r", self.head_dim)
        self.norm_type = getattr(cfg, "norm_type", "layernorm")
        if self.norm_type == "layernorm":
            self.ln1 = nn.LayerNorm(cfg.dim)
            self.ln2 = nn.LayerNorm(cfg.dim)
        elif self.norm_type == "rmsnorm":
            self.ln1 = nn.RMSNorm(cfg.dim)
            self.ln2 = nn.RMSNorm(cfg.dim)
        else:
            raise ValueError(f"Unknown norm_type: {cfg.norm_type}")

        if cfg.attn_type == "mha":
            self.attn = MHA(cfg)
        elif cfg.attn_type == "gqa":
            self.attn = GQA(cfg)
        elif cfg.attn_type == "mla":
            self.attn = MLA(cfg)
        elif cfg.attn_type == "gated_mla":
            self.attn = GatedMLA(cfg)
        else:
            raise ValueError(f"Unknown attn_type: {cfg.attn_type}")
        # TODO add more variants later
        policy = getattr(cfg, "ffn_width_policy", "dense_4x")
        if policy == "swiglu_21_8":
            if (cfg.mlp_type != "dense" or getattr(cfg, "ffn", "gelu") != "swiglu"
                    or cfg.dim % 64 or hasattr(cfg, "hidden_dim")
                    or getattr(cfg, "ffn_bias", False) is not False):
                raise ValueError("swiglu_21_8 requires bias-free SwiGLU, width multiple of64, no hidden_dim")
            self.hidden_dim = 21 * cfg.dim // 8
        elif policy == "swiglu_21_4":
            if (cfg.mlp_type != "dense" or getattr(cfg,"ffn",None) != "swiglu"
                    or cfg.dim not in {64,128,256} or hasattr(cfg,"hidden_dim")
                    or getattr(cfg,"ffn_bias",None) is not False):
                raise ValueError("swiglu_21_4 requires explicit bias-free wide dense SwiGLU")
            self.hidden_dim = 21 * cfg.dim // 4
        elif policy == "moe_swiglu_21_16":
            from ..moe.scaling import validate_moe_contract
            self.hidden_dim = validate_moe_contract(cfg)
        elif policy == "dense_4x":
            self.hidden_dim = getattr(cfg, "hidden_dim", cfg.dim * 4)
        else:
            raise ValueError("Unknown ffn_width_policy")
        ffn_bias = getattr(cfg, "ffn_bias", True)
        if not isinstance(ffn_bias, bool):
            raise ValueError("ffn_bias must be boolean")
        if cfg.mlp_type == "dense":
            self.ffn = getattr(cfg, "ffn", "gelu")
            if self.ffn in {"gelu", "silu"}:
                self.mlp = nn.Sequential(
                    nn.Linear(self.dim, self.hidden_dim, bias=ffn_bias),
                    nn.GELU() if self.ffn == "gelu" else nn.SiLU(),
                    nn.Linear(self.hidden_dim, self.dim, bias=ffn_bias),
                    nn.Dropout(cfg.dropout)
                )
            elif self.ffn == "swiglu":
                self.mlp = SwiGLU(self.dim, self.hidden_dim, cfg.dropout,
                                  hidden_norm=resolve_ffn_hidden_norm(cfg))
            else:
                raise ValueError(f"cfg.ffn has unsupported type: {self.ffn}")
        elif cfg.mlp_type == "moe":
            self.mlp = MoEBlock(cfg)
        else:
            raise ValueError(f"Unknown mlp_type: {cfg.mlp_type}")
        self.enable_rope = resolve_position_encoding(cfg) == "rope"
        if self.enable_rope:
            head_dim = self.head_dim_r if cfg.attn_type in {"mla", "gated_mla"} else self.head_dim
            if head_dim % 2 != 0:
                raise ValueError("head_dim must be even when rope is enabled")
            freq_cos, freq_sin = precompute_freqs_cis(head_dim, cfg.block_size)
            self.register_buffer("freq_cos", freq_cos)
            self.register_buffer("freq_sin", freq_sin)
        else:
            self.freq_cos = None
            self.freq_sin = None

    def forward(self, x, cache=None, start_pos=0):
        residual = x
        x, _ = self.attn(self.ln1(x), cache=cache, start_pos=start_pos, freq_cos=self.freq_cos, freq_sin=self.freq_sin)
        x = residual + x
        mlp_out = self.mlp(self.ln2(x))

        # MoE aux loss if present
        aux_loss = None
        if isinstance(mlp_out, tuple):
            mlp_out, aux_loss = mlp_out
        x = x + mlp_out
        return x, aux_loss
