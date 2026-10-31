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


CHECKPOINT_KEY = 'block_normalization_signature'


def _same_typed(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same_typed(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same_typed(x, y) for x, y in zip(a, b))
    return a == b


def checkpoint_signature(model):
    """Bind validated constructed semantics, not only mutable config labels."""
    value = contract(model.cfg)
    if value is None:
        require(getattr(model, 'block_norm_spec', None) is None,
                'New block normalization model cannot drop its policy')
        return None
    from .tokenizer_architecture import validate_architecture_model
    validate_architecture_model(model)
    return dict(version=1, architecture=value,
                parameter_names=[name + '.' + parameter for name in PLACEMENTS
                                 for parameter in ('weight', 'bias') if parameter == 'weight'
                                 or model.block_norm_spec.bias])


def validate_checkpoint(model, checkpoint):
    expected = checkpoint_signature(model)
    if expected is None:
        require(CHECKPOINT_KEY not in checkpoint, 'Legacy model cannot accept block normalization signature')
        return
    saved = checkpoint.get(CHECKPOINT_KEY)
    require(_same_typed(saved, expected), 'Saved constructed block normalization signature differs or is missing')
    cfg = get(checkpoint.get('cfg', {}), 'model', {})
    require(_same_typed(contract(cfg), expected['architecture']), 'Saved block normalization config differs')
    state = checkpoint.get('model_state')
    current = model.state_dict()
    require(type(state) in (dict, type(current)) and state.keys() == current.keys(),
            'Block normalization checkpoint tensor inventory differs')
    for name, value in state.items():
        target = current[name]
        require(type(value) is torch.Tensor and value.shape == target.shape and value.dtype == target.dtype,
                'Block normalization checkpoint tensor shape/dtype differs: ' + name)
        require(bool(torch.isfinite(value).all()), 'Nonfinite block normalization checkpoint tensor: ' + name)



def validate_caches(model, caches, batch, start_pos):
    """Validate every request cache before any layer can mutate it."""
    from ..attention.cache import KVCache
    expected = contract(model.cfg)
    if expected is None:
        return
    require(type(start_pos) is int and start_pos >= 0, 'Block norm cache start position invalid')
    require(len(caches) == 4 and len({id(c) for c in caches}) == 4, 'Four distinct caches required')
    pointers = []
    for i, cache in enumerate(caches):
        require(type(cache) is KVCache and cache.layer_index == i
                and _same_typed(cache.block_norm_signature, expected)
                and cache.attn_type == 'mla' and cache.is_mla is True,
                'Block norm cache policy/layer differs')
        require(type(cache.position) is int and cache.position == start_pos,
                'Block norm cache requires contiguous positions')
        for value, width in ((cache.c_cache, 64), (cache.r_cache, 32)):
            require(type(value) is torch.Tensor and tuple(value.shape) == (batch, 128, width)
                    and value.dtype == torch.float32 and value.device == model.transformer.wte.weight.device,
                    'Block norm cache tensor shape/dtype/device differs')
            pointers.append(value.untyped_storage().data_ptr())
    require(len(set(pointers)) == 8, 'Aliased request cache storage forbidden')
