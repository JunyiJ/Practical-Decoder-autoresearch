"""Independent normalization algebra and primitive contracts, awaiting execution."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from src.models.ffn_hidden_norm import (
    EPSILON, POLICY, make_ffn_hidden_norm, resolve_ffn_hidden_norm,
    validate_ffn_hidden_norm_contract,
)


@pytest.mark.parametrize('hidden', [168, 336, 672])
@pytest.mark.parametrize('kind', ['rms', 'layer'])
@pytest.mark.parametrize('regime', ['random', 'constant', 'zero', 'small'])
def test_independent_forward_and_gradients(hidden, kind, regime):
    rng = np.random.default_rng(9330)
    z = rng.normal(size=(2, 3, hidden))
    if regime == 'constant': z.fill(.7)
    if regime == 'zero': z.fill(0.)
    if regime == 'small': z *= 1e-8
    dy = rng.normal(size=z.shape)
    gain = np.linspace(.5, 1.5, hidden)
    bias = np.linspace(-.1, .1, hidden)
    center = z - z.mean(axis=-1, keepdims=True) if kind == 'layer' else z
    inverse = (np.mean(center ** 2, axis=-1, keepdims=True) + EPSILON) ** -.5
    unit = center * inverse
    expected = unit * gain + (bias if kind == 'layer' else 0.)
    v = dy * gain
    dx = inverse * (v - center * inverse ** 2 * np.mean(v * center, axis=-1, keepdims=True)
                    - (v.mean(axis=-1, keepdims=True) if kind == 'layer' else 0.))
    dg = np.sum(dy * unit, axis=(0, 1))
    module = make_ffn_hidden_norm(hidden, kind).double()
    with torch.no_grad():
        module.weight.copy_(torch.from_numpy(gain))
        if kind == 'layer': module.bias.copy_(torch.from_numpy(bias))
    x = torch.from_numpy(z.copy()).requires_grad_()
    y = module(x)
    y.backward(torch.from_numpy(dy))
    for actual, wanted in [(y.detach().numpy(), expected), (x.grad.numpy(), dx),
                           (module.weight.grad.numpy(), dg)]:
        np.testing.assert_allclose(actual, wanted, atol=2e-10, rtol=2e-10)
    if kind == 'layer':
        np.testing.assert_allclose(module.bias.grad.numpy(), dy.sum(axis=(0, 1)),
                                   atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize('kind', ['rms', 'layer'])
@pytest.mark.parametrize('hidden', [168, 336, 672])
def test_init_does_not_consume_rng_and_counts(kind, hidden):
    torch.manual_seed(9330);before = torch.get_rng_state().clone()
    m = make_ffn_hidden_norm(hidden, kind)
    assert torch.equal(before, torch.get_rng_state())
    assert sum(p.numel() for p in m.parameters()) == hidden * (2 if kind == 'layer' else 1)
    assert torch.equal(m.weight, torch.ones(hidden))
    assert m.eps == 1e-5
    if kind == 'layer': assert torch.equal(m.bias, torch.zeros(hidden))


def config(kind='rms', width=64):
    return SimpleNamespace(n_layers=4, dim=width, n_heads=width//32, mlp_type='dense',
        ffn='swiglu', ffn_bias=False, ffn_width_policy='swiglu_21_8', attn_type='mha',
        attention_width_policy='mha_head32', depth_width_policy='depth4_unscaled',
        parameterization='mup', base_width=64, norm_type='rmsnorm', rope=1,
        dropout=0., attn_dropout=0., block_size=128, ffn_hidden_norm=kind,
        ffn_hidden_norm_policy=POLICY)


@pytest.mark.parametrize('kind', ['rms', 'layer'])
@pytest.mark.parametrize('width', [64, 128, 256])
@pytest.mark.parametrize('gate', ['none', 'static_half', 'sigmoid_headwise_zero'])
def test_explicit_shape_contract(kind, width, gate):
    cfg = config(kind, width)
    if gate != 'none':
        cfg.attention_output_gate = gate
        cfg.output_gate_width_policy = 'mha_head32_output_gate'
    assert validate_ffn_hidden_norm_contract(cfg) == 4 * (21*width//8) * (2 if kind == 'layer' else 1)


@pytest.mark.parametrize('field,value', [
    ('dim', 65), ('dim', 64.), ('n_heads', True), ('n_layers', 6), ('base_width', 128),
    ('ffn', 'gelu'), ('ffn_bias', True), ('ffn_width_policy', 'swiglu_21_4'),
    ('attn_type', 'gqa'), ('attention_width_policy', 'mha_head16'),
    ('norm_type', 'layernorm'), ('block_size', 512), ('dropout', .1),
    ('hidden_dim', 168), ('qk_norm', 'rms_head32'), ('ffn_norm_eps', 1e-5),
    ('ffn_norm_affine', True), ('ffn_norm_bias', False),
    ('ffn_norm_placement', 'before_product'), ('ffn_norm_gain_init', 1.),
    ('ffn_norm_width', 168), ('residual_scale', .5), ('ffn_hidden_norm_policy', 'other'),
])
def test_rejects_unreviewed_contract(field, value):
    cfg = config();setattr(cfg, field, value)
    with pytest.raises(ValueError): validate_ffn_hidden_norm_contract(cfg)


@pytest.mark.parametrize('bad', [None, False, True, 0, 'RMS', 'rmsnorm', ''])
def test_rejects_unknown_kind(bad):
    with pytest.raises(ValueError): resolve_ffn_hidden_norm(SimpleNamespace(ffn_hidden_norm=bad))
    with pytest.raises(ValueError): make_ffn_hidden_norm(168, bad)


def test_default_contract():
    assert resolve_ffn_hidden_norm(SimpleNamespace()) == 'none'
    assert validate_ffn_hidden_norm_contract(SimpleNamespace()) == 0
    assert make_ffn_hidden_norm(256, 'none') is None
    with pytest.raises(ValueError):
        validate_ffn_hidden_norm_contract(SimpleNamespace(ffn_hidden_norm_policy=POLICY))
