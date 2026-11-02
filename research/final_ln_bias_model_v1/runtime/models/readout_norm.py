"""Explicit final LayerNorm bias policy, isolated from historical guards."""
from torch import nn
from research.readout_centering_model_v1.runtime.models.readout_norm import (
    readout_norm_contract as historical_contract, build_readout_norm as historical_build,
)
from .block_norm import READOUT_POLICY, contract, get, require


def readout_norm_contract(cfg):
    if get(cfg, 'tokenizer_architecture_policy') != READOUT_POLICY:
        return historical_contract(cfg)
    value = contract(cfg)['final_normalization']
    return dict(readout_norm_kind='layer_affine', readout_norm_epsilon=1e-5,
                readout_norm_width_policy='mla4_final_ln_bias_v1', final_ln_bias_kind=value['bias_kind'])


def readout_norm_signature(cfg):
    value = readout_norm_contract(cfg)
    return ('layer_affine', 1e-5) if value is None else tuple(sorted(value.items()))


def build_readout_norm(cfg):
    if get(cfg, 'tokenizer_architecture_policy') != READOUT_POLICY:
        return historical_build(cfg)
    value = contract(cfg)['final_normalization']
    return nn.LayerNorm(256, eps=1e-5, elementwise_affine=True, bias=value['bias'])


def validate_final(model):
    require(get(model.cfg, 'tokenizer_architecture_policy') == READOUT_POLICY, 'Explicit final-bias architecture required')
    value = contract(model.cfg)['final_normalization']
    norm = model.transformer.ln_f
    require(type(norm) is nn.LayerNorm and norm.normalized_shape == (256,)
            and norm.eps == 1e-5 and norm.elementwise_affine is True, 'Final LayerNorm semantics differ')
    require(set(norm._parameters) == {'weight', 'bias'} and not norm._buffers
            and set(norm.state_dict()) == ({'weight','bias'} if value['bias'] else {'weight'}), 'Final affine inventory differs')
    require(isinstance(norm.weight, nn.Parameter) and norm.weight.shape == (256,)
            and norm.weight.requires_grad, 'Final learned scale differs')
    if value['bias']:
        require(isinstance(norm.bias, nn.Parameter) and norm.bias.shape == (256,)
                and norm.bias.requires_grad and norm.weight is not norm.bias, 'Learned bias differs')
    else:
        require(norm.bias is None and norm._parameters['bias'] is None, 'Fixed zero must not be a parameter or buffer')
    require(model.readout_norm_signature == readout_norm_signature(model.cfg), 'Constructed final-bias identity differs')


def validate_readout_norm_load(model, checkpoint):
    saved = get(checkpoint.get('cfg', {}), 'model', {})
    require(readout_norm_contract(model.cfg) == readout_norm_contract(saved), 'Checkpoint final-bias policy differs')
    validate_final(model)
