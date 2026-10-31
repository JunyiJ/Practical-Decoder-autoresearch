"""Explicit finite-width MoE contract; not a proof of optimal LR transfer."""

def validate_moe_contract(cfg):
    expected = dict(num_experts=4, top_k=2, shared_num_experts=0,
                    ffn='swiglu', aux_loss_weight=.01, record_expert_hist=True,
                    width_policy='moe4_top2_half_readout')
    moe = getattr(cfg, 'moe', None)
    if (moe is None or set(moe) != set(expected)
            or any(moe[k] != v for k,v in expected.items())
            or cfg.mlp_type != 'moe' or cfg.attn_type != 'mha'
            or cfg.dim not in {64,128,256} or cfg.n_heads != cfg.dim//32
            or cfg.n_layers != 4 or getattr(cfg,'base_width',None) != 64
            or getattr(cfg,'parameterization',None) != 'mup'
            or getattr(cfg,'depth_width_policy',None) != 'depth4_moe_unscaled'
            or getattr(cfg,'attention_width_policy',None) != 'mha_head32'
            or getattr(cfg,'ffn_width_policy',None) != 'moe_swiglu_21_16'
            or getattr(cfg,'ffn',None) != 'swiglu'
            or getattr(cfg,'ffn_bias',None) is not False
            or any(hasattr(cfg,k) for k in ('hidden_dim','n_kv_heads','n_groups'))):
        raise ValueError('MoE requires the explicit four-layer top2-of4 half-expert width contract')
    return 21*cfg.dim//16
