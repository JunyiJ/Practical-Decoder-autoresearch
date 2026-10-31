"""Prospective headwise output gate; not integrated or registered for training.

Input x is the existing pre-normalized attention input. Apply this gate to
the per-head weighted values, before concatenation and the output projection.
The fixed-half arm controls the learned gate's initial branch amplitude.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


OUTPUT_GATE_KINDS = ('none', 'static_half', 'sigmoid_headwise_zero')


def resolve_output_gate(cfg):
    kind = getattr(cfg, 'attention_output_gate', 'none')
    if type(kind) is not str or kind not in OUTPUT_GATE_KINDS:
        raise ValueError('Unknown attention_output_gate')
    return kind


class HeadwiseOutputGate(nn.Module):
    """Fixed-half or learned per-query/head sigmoid, with no extra cache."""

    def __init__(self, dim, n_heads, kind):
        super().__init__()
        if (type(dim) is not int or type(n_heads) is not int
                or dim not in (64, 128, 256) or dim != 32 * n_heads):
            raise ValueError('Output gating requires reviewed widths and head32')
        if kind not in ('static_half', 'sigmoid_headwise_zero'):
            raise ValueError('Construct gates only for explicit nondefault kinds')
        self.dim, self.n_heads, self.kind = dim, n_heads, kind
        if kind == 'sigmoid_headwise_zero':
            # Unlike constructing and then zeroing nn.Linear, this consumes no
            # RNG. Every common initialization draw remains in its old place.
            self.weight = nn.Parameter(torch.zeros(n_heads, dim))
        else:
            self.register_parameter('weight', None)

    def forward(self, x, y):
        if (x.ndim != 3 or y.ndim != 4 or x.shape[-1] != self.dim
                or y.shape != (x.shape[0], self.n_heads, x.shape[1], 32)):
            raise ValueError('Expected x[B,S,D] and y[B,H,S,32]')
        if self.kind == 'static_half':
            return y * .5
        scores = torch.sigmoid(F.linear(x, self.weight))
        return y * scores.transpose(1, 2).unsqueeze(-1)


def make_output_gate(cfg):
    validate_output_gate_contract(cfg)
    kind = resolve_output_gate(cfg)
    if kind == 'none':
        return None
    return HeadwiseOutputGate(cfg.dim, cfg.n_heads, kind)


def validate_output_gate_contract(cfg):
    """A separate named policy for the reviewed four-layer MHA intervention."""
    kind = resolve_output_gate(cfg)
    if kind == 'none':
        if hasattr(cfg, 'output_gate_width_policy'):
            raise ValueError('Default attention must not carry an output-gate policy')
        return 0
    expected = dict(n_layers=4, mlp_type='dense', ffn='swiglu', ffn_bias=False,
                    ffn_width_policy='swiglu_21_8', attn_type='mha',
                    parameterization='mup', base_width=64, norm_type='rmsnorm',
                    rope=1, dropout=0., attn_dropout=0., block_size=128,
                    output_gate_width_policy='mha_head32_output_gate')
    if any(getattr(cfg, key, None) != value for key, value in expected.items()):
        raise ValueError('Output-gate recipe differs from its explicit contract')
    from .qk_norm import resolve_qk_norm, validate_qkrms_contract
    qk = resolve_qk_norm(cfg)
    if qk != 'none':
        validate_qkrms_contract(cfg)
    elif (getattr(cfg, 'depth_width_policy', None) != 'depth4_unscaled'
          or getattr(cfg, 'attention_width_policy', None) != 'mha_head32'):
        raise ValueError('Output-gate standard MHA policies differ')
    if (type(cfg.n_layers) is not int or type(cfg.dim) is not int
            or type(cfg.n_heads) is not int or cfg.dim not in (64, 128, 256)
            or cfg.dim != 32 * cfg.n_heads or cfg.ffn_bias is not False
            or type(cfg.base_width) is not int):
        raise ValueError('Output-gate shape/layer/base contract changed')
    forbidden = ('hidden_dim', 'n_kv_heads', 'n_groups', 'moe', 'dimckv', 'dimcq',
                 'head_dim_r', 'attn_bias', 'causal', 'residual_scale',
                 'residual_scaling', 'position_encoding', 'gate_bias',
                 'gate_temperature', 'gate_init', 'gate_placement', 'gate_dim',
                 'qk_norm_eps', 'qk_norm_affine', 'qk_norm_placement', 'qk_scale')
    if any(hasattr(cfg, key) for key in forbidden):
        raise ValueError('Output-gate recipe rejects unreviewed overrides')
    return cfg.n_layers * cfg.dim * cfg.n_heads if kind == 'sigmoid_headwise_zero' else 0
