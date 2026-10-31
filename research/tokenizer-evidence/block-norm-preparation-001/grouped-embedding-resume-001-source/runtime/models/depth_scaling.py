"""Narrow six-layer dense recipe for a separately bounded capacity study.

No depth-dependent rescaling is inferred from width scaling. The explicit
contract retains the existing unscaled residual paths and initializers and
requires empirical LR selection at this fixed depth.
"""


def validate_depth6_dense_contract(cfg):
    expected = dict(n_layers=6, mlp_type='dense', ffn='swiglu', ffn_bias=False,
                    ffn_width_policy='swiglu_21_8', depth_width_policy='depth6_dense_unscaled',
                    attn_type='mha', attention_width_policy='mha_head32',
                    parameterization='mup', base_width=64, norm_type='rmsnorm',
                    rope=1, dropout=0., attn_dropout=0., block_size=128)
    if any(getattr(cfg, key, None) != value for key, value in expected.items()):
        raise ValueError('Six-layer dense recipe differs from its explicit contract')
    if (type(cfg.n_layers) is not int or type(cfg.dim) is not int
            or cfg.dim not in (64,128,256) or type(cfg.n_heads) is not int
            or cfg.n_heads != cfg.dim // 32 or cfg.ffn_bias is not False
            or type(cfg.base_width) is not int):
        raise ValueError('Six-layer dense width/head/layer contract changed')
    forbidden = ('hidden_dim','n_kv_heads','n_groups','moe','dimckv','dimcq',
                 'head_dim_r','attn_bias','causal','residual_scale','residual_scaling','position_encoding')
    if any(hasattr(cfg, key) for key in forbidden):
        raise ValueError('Six-layer dense recipe rejects unreviewed overrides')
    return 21 * cfg.dim // 8
