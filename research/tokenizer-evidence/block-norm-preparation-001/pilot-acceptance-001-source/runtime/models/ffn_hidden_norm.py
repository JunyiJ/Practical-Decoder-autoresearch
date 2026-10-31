"""Prospective SwiGLU-product normalization; not registered for research yet."""
import torch.nn as nn


KINDS = ('none', 'rms', 'layer')
POLICY = 'swiglu_product_norm_21_8'
EPSILON = 1e-5


def resolve_ffn_hidden_norm(cfg):
    kind = getattr(cfg, 'ffn_hidden_norm', 'none')
    if type(kind) is not str or kind not in KINDS:
        raise ValueError('Unknown ffn_hidden_norm')
    return kind


def validate_ffn_hidden_norm_contract(cfg):
    kind = resolve_ffn_hidden_norm(cfg)
    if kind == 'none':
        if hasattr(cfg, 'ffn_hidden_norm_policy'):
            raise ValueError('Default FFN must not carry a hidden-norm policy')
        return 0
    required = dict(n_layers=4, mlp_type='dense', ffn='swiglu', ffn_bias=False,
                    ffn_width_policy='swiglu_21_8', attn_type='mha',
                    attention_width_policy='mha_head32', depth_width_policy='depth4_unscaled',
                    parameterization='mup', base_width=64, norm_type='rmsnorm',
                    rope=1, dropout=0., attn_dropout=0., block_size=128,
                    ffn_hidden_norm_policy=POLICY)
    if any(getattr(cfg, k, None) != v for k, v in required.items()):
        raise ValueError('FFN hidden norm requires the reviewed four-layer backbone')
    if (type(cfg.dim) is not int or cfg.dim not in (64, 128, 256)
            or type(cfg.n_heads) is not int or cfg.n_heads != cfg.dim // 32
            or type(cfg.n_layers) is not int or type(cfg.base_width) is not int
            or cfg.ffn_bias is not False):
        raise ValueError('FFN hidden norm shape/base contract changed')
    forbidden = ('hidden_dim', 'n_kv_heads', 'n_groups', 'moe', 'dimckv', 'dimcq',
                 'head_dim_r', 'attn_bias', 'causal', 'residual_scale',
                 'residual_scaling', 'position_encoding', 'ffn_norm_eps',
                 'ffn_norm_affine', 'ffn_norm_bias', 'ffn_norm_placement',
                 'ffn_norm_gain_init', 'ffn_norm_width', 'qk_norm', 'qk_norm_eps',
                 'qk_norm_affine', 'qk_norm_placement', 'qk_scale')
    if any(hasattr(cfg, k) for k in forbidden):
        raise ValueError('FFN hidden norm rejects unreviewed overrides')
    from ..attention.output_gate import validate_output_gate_contract
    validate_output_gate_contract(cfg)
    hidden = 21 * cfg.dim // 8
    return cfg.n_layers * hidden * (2 if kind == 'layer' else 1)


def make_ffn_hidden_norm(hidden_dim, kind):
    if type(kind) is not str or kind not in KINDS:
        raise ValueError('Unknown ffn_hidden_norm')
    if kind == 'none':
        return None
    if type(hidden_dim) is not int or hidden_dim not in (168, 336, 672):
        raise ValueError('Unreviewed SwiGLU hidden width')
    if kind == 'rms':
        return nn.RMSNorm(hidden_dim, eps=EPSILON, elementwise_affine=True)
    return nn.LayerNorm(hidden_dim, eps=EPSILON, elementwise_affine=True, bias=True)
