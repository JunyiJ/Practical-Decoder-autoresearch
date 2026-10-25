"""Explicit per-head RMS normalization for a separately reviewed MHA recipe.

Queries and keys are normalized before RoPE. Each learned gain has head_dim
entries shared across heads, positions and batch examples; Q and K have
separate gains. The existing 1/sqrt(head_dim) attention scale is unchanged.
This is an RMS variant, not Henry et al.'s L2-plus-scalar QKNorm reproduction.
"""
import torch.nn as nn

QK_RMS_EPS = 1e-6
QK_RMS_KIND = 'rms_head32'


def resolve_qk_norm(cfg):
    kind = getattr(cfg, 'qk_norm', 'none')
    if kind not in ('none', QK_RMS_KIND):
        raise ValueError('qk_norm must be none or rms_head32')
    if kind == QK_RMS_KIND:
        if (getattr(cfg, 'attn_type', 'mha') != 'mha'
                or type(cfg.dim) is not int or type(cfg.n_heads) is not int
                or cfg.n_heads < 1 or cfg.dim != 32 * cfg.n_heads):
            raise ValueError('rms_head32 requires MHA with head dimension32')
    return kind


def make_qk_norms(cfg):
    if resolve_qk_norm(cfg) == 'none':
        return None, None
    # Ones initialization consumes no RNG and keeps preexisting weight draws.
    return (nn.RMSNorm(32, eps=QK_RMS_EPS),
            nn.RMSNorm(32, eps=QK_RMS_EPS))


def validate_qkrms_contract(cfg):
    """Narrow quality-study contract; width algebra is not an LR guarantee."""
    expected = dict(n_layers=4, mlp_type='dense', ffn='swiglu', ffn_bias=False,
                    ffn_width_policy='swiglu_21_8',
                    depth_width_policy='depth4_qkrms_unscaled', attn_type='mha',
                    attention_width_policy='mha_head32_qkrms', qk_norm=QK_RMS_KIND,
                    parameterization='mup', base_width=64, norm_type='rmsnorm',
                    rope=1, dropout=0., attn_dropout=0., block_size=128)
    if any(getattr(cfg, key, None) != value for key, value in expected.items()):
        raise ValueError('QK RMS recipe differs from its explicit contract')
    resolve_qk_norm(cfg)
    if (type(cfg.n_layers) is not int or cfg.dim not in (64, 128, 256)
            or cfg.ffn_bias is not False or type(cfg.base_width) is not int):
        raise ValueError('QK RMS width/layer/base contract changed')
    forbidden = ('hidden_dim', 'n_kv_heads', 'n_groups', 'moe', 'dimckv', 'dimcq',
                 'head_dim_r', 'attn_bias', 'causal', 'residual_scale',
                 'residual_scaling', 'position_encoding', 'qk_norm_eps',
                 'qk_norm_affine', 'qk_norm_placement', 'qk_scale')
    if any(hasattr(cfg, key) for key in forbidden):
        raise ValueError('QK RMS recipe rejects unreviewed overrides')
    return 2 * 32 * cfg.n_layers
