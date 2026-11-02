"""Explicit MLA readout policy; the old MHA-only contract remains unchanged."""
from torch import nn
from research.readout_centering_v1.component import ReadoutNormalization
from research.block_norm_v1.runtime.models.readout_norm import (
    readout_norm_contract as historical_contract,
    build_readout_norm as historical_build,
)
from .block_norm import READOUT_POLICY, contract, get, require


def readout_norm_contract(cfg):
    if get(cfg, 'tokenizer_architecture_policy') != READOUT_POLICY:
        return historical_contract(cfg)
    value = contract(cfg)['final_normalization']
    return dict(readout_norm_kind=value['kind'], readout_norm_epsilon=value['epsilon'],
                readout_norm_width_policy='mla4_readout_centering_affine_v1')


def readout_norm_signature(cfg):
    value = readout_norm_contract(cfg)
    return ('layer_affine', 1e-5) if value is None else (
        value['readout_norm_kind'], value['readout_norm_epsilon'])


def build_readout_norm(cfg):
    if get(cfg, 'tokenizer_architecture_policy') != READOUT_POLICY:
        return historical_build(cfg)
    value = readout_norm_contract(cfg)
    if value['readout_norm_kind'] == 'layer_affine':
        return nn.LayerNorm(256, eps=1e-5, elementwise_affine=True, bias=True)
    return ReadoutNormalization(256, 'rms_affine_bias')


def validate_final(model):
    value = readout_norm_contract(model.cfg)
    require(get(model.cfg, 'tokenizer_architecture_policy') == READOUT_POLICY,
            'Explicit readout architecture required')
    kind = value['readout_norm_kind']
    final = model.transformer.ln_f
    if kind == 'layer_affine':
        require(type(final) is nn.LayerNorm and final.normalized_shape == (256,)
                and final.elementwise_affine is True, 'Constructed LayerNorm differs')
    else:
        require(type(final) is ReadoutNormalization and final.width == 256
                and final.kind == 'rms_affine_bias', 'Constructed affine RMS differs')
    require(final.eps == 1e-5 and set(final._parameters) == {'weight', 'bias'}
            and not final._buffers and set(final.state_dict()) == {'weight', 'bias'},
            'Exact final readout tensor inventory/epsilon required')
    for name in ('weight', 'bias'):
        p = getattr(final, name)
        require(isinstance(p, nn.Parameter) and tuple(p.shape) == (256,)
                and p.requires_grad and final._parameters[name] is p,
                'Learned final readout affine parameter differs')
    require(final.weight is not final.bias, 'Distinct scale and bias required')
    require(model.readout_norm_signature == readout_norm_signature(model.cfg),
            'Constructed readout identity differs')


def validate_readout_norm_load(model, checkpoint):
    cfg = model.cfg
    saved = get(checkpoint.get('cfg', {}), 'model', {})
    require(readout_norm_contract(cfg) == readout_norm_contract(saved),
            'Checkpoint readout semantics differ')
    require(model.readout_norm_signature == readout_norm_signature(cfg),
            'Constructed final normalization differs from config')
    if get(cfg, 'tokenizer_architecture_policy') == READOUT_POLICY:
        validate_final(model)
