"""Explicit D256 MLA block-normalization policy in the isolated runtime only."""
from collections.abc import Mapping
import torch
from torch import nn
from ...spec import Spec, decode, encode, make, checked_forward, validate_eight, require

POLICY = 'tokenizer_mla_block_norm_v1'
FIELD = 'block_normalization'
PLACEMENTS = tuple(f'transformer.h.{i}.ln{j}' for i in range(4) for j in (1, 2))


def get(cfg, key, default=None):
    return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)


def contract(cfg):
    if get(cfg, 'tokenizer_architecture_policy') != POLICY:
        require(get(cfg, FIELD) is None, 'Block normalization requires its named architecture policy')
        return None
    from .tokenizer_architecture import FIXED, VARIABLE
    keys = set(cfg.keys()) if hasattr(cfg, 'keys') else set(vars(cfg))
    require(keys == set(FIXED) | VARIABLE | {FIELD}, 'Block normalization requires exact explicit fields')
    spec = decode(get(cfg, FIELD))
    fixed = dict(FIXED, tokenizer_architecture_policy=POLICY,
                 norm_type='layernorm' if spec.bias else 'rmsnorm', dim=256, n_heads=8,
                 attn_type='mla', attention_width_policy='mla_quarter_head32',
                 depth_width_policy='depth4_unscaled')
    for key, expected in fixed.items():
        value = get(cfg, key)
        require(type(value) is type(expected) and value == expected,
                f'Block normalization changed fixed field {key}')
    return dict(policy=POLICY, family='mla', width=256, heads=8, score_scale=.5,
                prior_sha256=get(get(cfg, 'output_bias_prior'), 'prior_sha256'),
                normalization=encode(spec), placements=list(PLACEMENTS),
                final_normalization=dict(kind='layer_affine', epsilon=1e-5),
                gate_kind='silu', hidden_width=672, residual='serial_unit', tables='untied')


def resolve(cfg):
    value = contract(cfg)
    return None if value is None else decode(value['normalization'])


def validate_model(model, value):
    spec = decode(value['normalization'])
    require(model.block_norm_spec == spec, 'Constructed block norm specification differs')
    normalizers = [model.get_submodule(name) for name in PLACEMENTS]
    validate_eight(spec, normalizers)
    parameters = list(model.parameters())
    require(all(p.dtype == torch.float32 for p in parameters), 'Block norm model requires FP32 parameters')
    require(len({p.device for p in parameters}) == 1, 'Block norm model devices differ')
    require(sum(p.numel() for p in parameters) == 28492416 + (2048 if spec.bias else 0),
            'Block norm model parameter inventory differs')
    final = model.transformer.ln_f
    require(type(final) is nn.LayerNorm and final.normalized_shape == (256,)
            and final.eps == 1e-5 and final.elementwise_affine is True
            and final.weight is not None and final.bias is not None
            and final.weight.requires_grad and final.bias.requires_grad,
            'Final affine LayerNorm differs')
    require(model.lm_head is not None and model.lm_head.bias is None
            and model.lm_head.weight.shape == (50257, 256)
            and model.transformer.wte.weight.shape == (50257, 256)
            and model.lm_head.weight.untyped_storage().data_ptr()
            != model.transformer.wte.weight.untyped_storage().data_ptr(), 'Untied tables differ')
    from .mlp import SwiGLU
    for i, block in enumerate(model.transformer.h):
        require(block.block_norm_spec == spec and block.layer_index == i
                and block.norm_type == ('layernorm' if spec.bias else 'rmsnorm'), 'Block norm placement differs')
        require(block.residual_gain_kind == 'unit'
                and not any(hasattr(block, n) for n in ('attention_residual_gain', 'ffn_residual_gain')),
                'Unit residual gains differ')
        require(type(block.mlp) is SwiGLU and block.hidden_dim == 672
                and block.mlp.dim == 256 and block.mlp.hidden_dim == 672
                and block.mlp.dropout.p == 0., 'SwiGLU dimensions/dropout differ')
        shapes = {'linear_gate': (672, 256), 'linear_up': (672, 256), 'linear_down': (256, 672)}
        for name, shape in shapes.items():
            layer = getattr(block.mlp, name)
            require(type(layer) is nn.Linear and tuple(layer.weight.shape) == shape
                    and layer.bias is None, 'SwiGLU projection differs')
        require(block.attn.causal is True and block.attn.attn_dropout.p == 0.
                and block.attn.out_dropout.p == 0., 'MLA causal/dropout semantics differ')
