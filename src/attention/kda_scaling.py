"""Explicit prospective KDA geometry; not approved for research launch yet.

The human reference defaults remain unchanged. This policy scales the alpha
bottleneck with model width, while convolution fan-in and head size stay fixed.
It specifies parameter roles, not an empirical optimal-LR transfer guarantee.
"""

POLICY = 'kda_head32_alpha_quarter'
DEPTH_POLICY = 'depth4_kda_unscaled'


def has_kda_policy(cfg):
    return (getattr(cfg, 'attention_width_policy', None) == POLICY
            or getattr(cfg, 'depth_width_policy', None) == DEPTH_POLICY)


def resolve_alpha_dimension(cfg):
    from ..models.tokenizer_hybrid import hybrid_contract
    hybrid = hybrid_contract(cfg)
    if hybrid is not None:
        return hybrid['kda_alpha_rank']
    if has_kda_policy(cfg):
        validate_kda_contract(cfg)
        return cfg.dim // 4
    return getattr(cfg, 'dim_alpha', max(1, (cfg.dim // getattr(cfg, 'n_heads', 1)) // 4))


def validate_kda_contract(cfg):
    required = dict(attn_type='kda', n_layers=4, mlp_type='dense', ffn='swiglu',
                    ffn_width_policy='swiglu_21_8', ffn_bias=False,
                    attention_width_policy=POLICY, depth_width_policy=DEPTH_POLICY,
                    parameterization='mup', base_width=64, norm_type='rmsnorm',
                    rope=0, dropout=0., attn_dropout=0., block_size=128,
                    attn_bias=True, causal=True)
    if any(getattr(cfg, key, None) != value for key, value in required.items()):
        raise ValueError('KDA requires the reviewed explicit four-layer recipe')
    if (type(cfg.dim) is not int or cfg.dim not in (64, 128, 256)
            or type(cfg.n_heads) is not int or cfg.n_heads != cfg.dim // 32
            or type(cfg.n_layers) is not int or type(cfg.base_width) is not int
            or cfg.ffn_bias is not False or cfg.attn_bias is not True
            or cfg.causal is not True):
        raise ValueError('KDA head, alpha, layer or base-width contract changed')
    forbidden = ('dim_alpha', 'hidden_dim', 'n_kv_heads', 'n_groups', 'moe', 'dimckv', 'dimcq',
                 'head_dim_r', 'residual_scale', 'residual_scaling',
                 'position_encoding', 'qk_norm', 'qk_norm_eps', 'qk_scale',
                 'ffn_hidden_norm', 'ffn_hidden_norm_policy',
                 'attention_output_gate', 'output_gate_width_policy',
                 'conv_size', 'g_min', 'state_dtype', 'alpha_init', 'gate_init',
                 'enable_rope', 'head_norm_eps')
    if any(hasattr(cfg, key) for key in forbidden):
        raise ValueError('KDA policy rejects unreviewed overrides')
    d, h, a = cfg.dim, cfg.n_heads, cfg.dim // 4
    return dict(attention_parameters_per_layer=5*d*d + d*h + 2*d*a + 16*d + 2*h + 32,
                recurrent_elements_per_layer_per_example=32*d,
                convolution_history_elements_per_layer_per_example=9*d)
