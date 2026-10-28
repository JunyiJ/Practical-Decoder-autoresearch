"""Explicit four-layer NoPE KDA/gated-MLA adaptation; no quality claim."""
from collections.abc import Mapping

POLICY = 'tokenizer_k3_hybrid_nope_v1'
GEOMETRY = 'hybrid_head32_quarter'
WIDE_POLICY = 'tokenizer_wide_k3_hybrid_nope_v1'
WIDE_GEOMETRY = 'wide_hybrid_head32_quarter'
RANK_POLICY = 'tokenizer_wide_k3_hybrid_rank_nope_v1'
RANK_GEOMETRY = 'wide_hybrid_head32_rank'
LAYOUTS = (('kda', 'kda', 'kda', 'gated_mla'), ('gated_mla',)*4, ('kda',)*4)


def get(cfg, key, default=None):
    return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)


def hybrid_contract(cfg):
    policy = get(cfg, 'tokenizer_architecture_policy')
    if policy not in (POLICY, WIDE_POLICY, RANK_POLICY):
        if get(cfg, 'attn_type') == 'hybrid' or get(cfg, 'attention_layout') is not None:
            raise ValueError('Layerwise attention requires its explicit tokenizer hybrid policy')
        return None
    from .tokenizer_architecture import FIXED, VARIABLE
    fixed = dict(FIXED, tokenizer_architecture_policy=policy, rope=0)
    keys = set(cfg.keys()) if hasattr(cfg, 'keys') else set(vars(cfg))
    rank_fields = {'attention_bottleneck_divisor'} if policy == RANK_POLICY else set()
    if keys != set(fixed) | VARIABLE | {'attention_layout'} | rank_fields:
        raise ValueError('Hybrid requires its exact explicit field set')
    for key, value in fixed.items():
        actual = get(cfg, key)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f'Hybrid changed fixed field {key}')
    width, heads = get(cfg, 'dim'), get(cfg, 'n_heads')
    widths = (64, 256) if policy == POLICY else (512, 1024)
    geometry = {POLICY: GEOMETRY, WIDE_POLICY: WIDE_GEOMETRY, RANK_POLICY: RANK_GEOMETRY}[policy]
    divisor = get(cfg, 'attention_bottleneck_divisor') if policy == RANK_POLICY else 4
    if type(divisor) is not int or divisor not in (2, 4, 8):
        raise ValueError('Hybrid rank policy requires integer divisor2/4/8')
    if type(width) is not int or width not in widths or type(heads) is not int or heads != width//32:
        raise ValueError(f'Hybrid requires widths{widths} with head32')
    if (get(cfg, 'attn_type') != 'hybrid' or get(cfg, 'attention_width_policy') != geometry
            or get(cfg, 'depth_width_policy') != 'depth4_hybrid_unscaled'):
        raise ValueError('Hybrid geometry/depth contract mismatch')
    layout = get(cfg, 'attention_layout')
    if isinstance(layout, str) or not isinstance(layout, (list, tuple)):
        # OmegaConf ListConfig intentionally supports sequence access, not list inheritance.
        from omegaconf import ListConfig
        if not isinstance(layout, ListConfig):
            raise TypeError('Hybrid layout must be a four-layer sequence')
    layout = tuple(layout)
    if layout not in LAYOUTS:
        raise ValueError('Undeclared hybrid/control layout')
    return {'policy': policy, 'family': 'hybrid', 'layout': layout, 'width': width, 'heads': heads,
            'position': 'none', 'score_scale': .5, 'latent_rank': width//divisor, 'shared_key_dim': 32,
            'kda_alpha_rank': width//divisor, 'kda_conv': 4, 'kda_g_min': -5., 'kda_output_gate': 'full_rank',
            'prior_sha256': get(get(cfg, 'output_bias_prior'), 'prior_sha256')}


def layer_family(cfg, layer_index=None):
    contract = hybrid_contract(cfg)
    if contract is None:
        return get(cfg, 'attn_type')
    if type(layer_index) is not int or not 0 <= layer_index < 4:
        raise ValueError('Hybrid attention/cache requires an explicit layer index')
    return contract['layout'][layer_index]


def validate_hybrid_model(model, contract):
    from torch import nn

    from ..attention.kda import KDA
    from ..attention.mla import GatedMLA
    if (len(model.transformer.h) != 4 or 'wpe' in model.transformer
            or model.position_encoding != 'none' or model.enable_rope
            or type(model.transformer.ln_f) is not nn.LayerNorm
            or model.transformer.ln_f.eps != 1e-5 or model.transformer.ln_f.bias is None
            or model.lm_head is None or model.output_bias is not None or model.input_embedding_scale != 1.):
        raise ValueError('Hybrid position/readout contract mismatch')
    d, h = contract['width'], contract['heads']
    for i, (block, family) in enumerate(zip(model.transformer.h, contract['layout'])):
        a = block.attn
        if (block.layer_index != i or block.attention_family != family or block.enable_rope
                or block.freq_cos is not None or block.freq_sin is not None
                or type(block.ln1) is not nn.RMSNorm or type(block.ln2) is not nn.RMSNorm
                or block.residual_topology_kind != 'serial' or block.residual_gain_kind != 'unit'
                or block.mlp.gate_activation != 'silu' or block.mlp.hidden_norm is not None
                or block.hidden_dim != 21*d//8 or a.dim != d or a.n_heads != h or a.head_dim != 32):
            raise ValueError('Hybrid constructed block differs from contract')
        if family == 'kda':
            if (type(a) is not KDA or a.dim_alpha != contract['kda_alpha_rank'] or a.g_min != -5.
                    or a.conv_size != 4 or a.head_norm.eps != 1e-6
                    or tuple(a.output_gate.weight.shape) != (d, d)
                    or tuple(a.z1.weight.shape) != (contract['kda_alpha_rank'], d)
                    or tuple(a.z2.weight.shape) != (d, contract['kda_alpha_rank'])):
                raise ValueError('Hybrid KDA geometry/gate/decay mismatch')
        elif (type(a) is not GatedMLA or a.enable_rope or a.dimc_q != contract['latent_rank']
              or a.dimc_kv != contract['latent_rank'] or a.head_dim_r != 32 or a.attention_logit_scale != .5
              or tuple(a.dq.weight.shape) != (contract['latent_rank'], d)
              or tuple(a.dkv.weight.shape) != (contract['latent_rank'], d)
              or any(tuple(getattr(a, name).weight.shape) != (d, contract['latent_rank'])
                     for name in ('uq', 'uk', 'uv', 'qr'))
              or tuple(a.gate.weight.shape) != (d, d)):
            raise ValueError('Hybrid gated MLA geometry/position/score mismatch')


def validate_hybrid_caches(model, caches, batch, start_pos):
    """Validate every layer before any recurrent state can be mutated."""
    contract = hybrid_contract(model.cfg)
    if contract is None:
        return
    if type(start_pos) is not int or start_pos < 0:
        raise ValueError('Hybrid cache requires a nonnegative integer offset')
    import torch
    if model.training or torch.is_grad_enabled():
        raise ValueError('Hybrid cache requires eval mode with gradients disabled')
    if model.tokenizer_architecture_signature != contract:
        raise ValueError('Hybrid model config changed after construction')
    d, h = contract['width'], contract['heads']
    for i, (cache, family) in enumerate(zip(caches, contract['layout'])):
        if (getattr(cache, 'hybrid_signature', None) != contract
                or getattr(cache, 'layer_index', None) != i or cache.attn_type != family
                or cache.position != start_pos):
            raise ValueError('Hybrid cache layer/contract/contiguous position mismatch')
        shapes = ([tuple(cache.state.shape), tuple(cache.q_history.shape),
                   tuple(cache.k_history.shape), tuple(cache.v_history.shape)] if family == 'kda'
                  else [tuple(cache.c_cache.shape), tuple(cache.r_cache.shape)])
        expected = ([(batch, h, 32, 32)] + [(batch, d, 3)]*3 if family == 'kda'
                    else [(batch, 128, contract['latent_rank']), (batch, 128, 32)])
        if shapes != expected:
            raise ValueError('Hybrid cache batch/geometry mismatch')
