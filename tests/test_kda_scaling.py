"""Prospective KDA role/shape gates; execute only after active training ends."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from src.attention.cache import KVCache
from src.attention.kda_scaling import POLICY, DEPTH_POLICY, validate_kda_contract
from src.models.gpt import GPT
from src.train.optimizers import build_optimizer


def config(width=64, **changes):
    values = dict(attn_type='kda', dim=width, n_heads=width//32,
                  n_layers=4, mlp_type='dense', ffn='swiglu',
                  ffn_width_policy='swiglu_21_8', ffn_bias=False,
                  attention_width_policy=POLICY, depth_width_policy=DEPTH_POLICY,
                  parameterization='mup', base_width=64, norm_type='rmsnorm',
                  rope=0, dropout=0., attn_dropout=0., block_size=128,
                  attn_bias=True, causal=True, vocab_size=65)
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize('width', [64, 128, 256])
def test_saved_geometry_and_constant_cache_storage(width):
    cfg = config(width)
    expected = validate_kda_contract(cfg)
    model = GPT(cfg)
    assert model.position_encoding == 'none'
    assert 'wpe' not in model.transformer
    total = (4*expected['attention_parameters_per_layer']
             + 4*3*width*(21*width//8) + 140*width)
    assert sum(p.numel() for p in model.parameters()) == total
    for block in model.transformer.h:
        layer = block.attn
        assert layer.dim_alpha == width//4
        assert layer.qkv.weight.shape == (3*width, width)
        assert layer.z1.weight.shape == (width//4, width)
        assert layer.z2.weight.shape == (width, width//4)
        assert layer.beta_proj.weight.shape == (width//32, width)
        assert layer.head_norm.weight.shape == (32,)
        assert layer.A.shape == (width//32,)
        assert torch.count_nonzero(layer.A) == 0
        assert layer.g_min == -5. and layer.conv_size == 4
        assert sum(p.numel() for p in layer.parameters()) == expected['attention_parameters_per_layer']
    cache = KVCache(SimpleNamespace(model=cfg), 'cpu', batch_size=8)
    assert cache.state.shape == (8, width//32, 32, 32)
    assert sum(x.numel() for x in cache.buffers()) == 8*41*width
    assert all(x.dtype == torch.float32 for x in cache.buffers())
    bigger = deepcopy(cfg); bigger.block_size = 512
    other = KVCache(SimpleNamespace(model=bigger), 'cpu', batch_size=8)
    assert sum(x.numel() for x in other.buffers()) == sum(x.numel() for x in cache.buffers())


@pytest.mark.parametrize('width', [64, 256])
def test_optimizer_roles_scales_and_independent_decay(width):
    model = GPT(config(width))
    training = SimpleNamespace(optimizer='muon', decay_mode='independent',
                               learning_rate=.005, weight_decay=.00006)
    opt = build_optimizer(model, training)
    roles = {}
    for child in opt.optimizers:
        for group in child.param_groups:
            assert len(group['params']) == len(group['parameter_names']) == 1
            name, p = group['parameter_names'][0], group['params'][0]
            assert name not in roles
            roles[name] = type(child).__name__
            matrix = name.startswith('transformer.h.') and p.ndim == 2
            assert type(child).__name__ == ('Muon' if matrix else 'AdamW')
            scale = (width/64)**(-.5) if matrix else 1.
            assert group['lr'] == pytest.approx(.005*scale)
            assert group['width_lr_scale'] == pytest.approx(scale)
            assert group['weight_decay'] == 0.
            group['lr'] = 0.
    assert set(roles) == set(dict(model.named_parameters()))
    for i in range(4):
        for item in ('q_conv.weight', 'k_conv.weight', 'v_conv.weight', 'A', 'head_norm.weight'):
            assert roles[f'transformer.h.{i}.attn.{item}'] == 'AdamW'
        for item in ('qkv.weight', 'beta_proj.weight', 'z1.weight', 'z2.weight',
                     'output_gate.weight', 'out_proj.weight'):
            assert roles[f'transformer.h.{i}.attn.{item}'] == 'Muon'
    before = {n:p.detach().clone() for n,p in model.named_parameters()}
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    for name,p in model.named_parameters():
        torch.testing.assert_close(p, before[name]*(1-.00006), rtol=0, atol=0)
        assert torch.isfinite(p).all()


@pytest.mark.parametrize('width', [64, 256])
def test_mup_initialization_preserves_documented_parameter_roles(width):
    cfg = config(width)
    torch.manual_seed(9361); mup = GPT(cfg)
    sp_cfg = deepcopy(cfg)
    del sp_cfg.attention_width_policy
    del sp_cfg.depth_width_policy
    sp_cfg.parameterization = 'sp'
    sp_cfg.dim_alpha = width//4
    torch.manual_seed(9361); sp = GPT(sp_cfg)
    expected = dict(sp.named_parameters())
    for name,p in mup.named_parameters():
        bias = name.startswith('transformer.h.') and name.endswith('.bias')
        # All KDA/FFN block biases belong to Linear modules; block RMSNorm has none.
        factor = (width/64)**.5 if name == 'lm_head.weight' or bias else 1.
        torch.testing.assert_close(p, expected[name]*factor, rtol=0, atol=0)


@pytest.mark.parametrize('change', [
    dict(dim=96), dict(dim=True), dict(n_heads=1), dict(n_heads=True),
    dict(dim_alpha=8), dict(dim_alpha=True), dict(n_layers=2),
    dict(attn_type='mha'), dict(rope=1), dict(parameterization='sp'),
    dict(base_width=32), dict(ffn_bias=True), dict(attn_bias=False),
    dict(causal=False), dict(dropout=.1), dict(block_size=512),
    dict(attention_width_policy='mha_head32'), dict(depth_width_policy='depth4_unscaled'),
    dict(hidden_dim=168), dict(n_kv_heads=2), dict(position_encoding='none'),
    dict(qk_norm='none'), dict(ffn_hidden_norm='none'), dict(attention_output_gate='none'),
    dict(conv_size=4), dict(g_min=-5.), dict(state_dtype='float32'),
])
def test_unreviewed_overrides_rejected(change):
    with pytest.raises(ValueError, match='KDA'):
        validate_kda_contract(config(**change))


def test_default_kda_stays_available_but_unreviewed_muon_is_rejected():
    cfg = config()
    del cfg.attention_width_policy
    del cfg.depth_width_policy
    model = GPT(cfg)
    assert model.transformer.h[0].attn.dim_alpha == 8
    training = SimpleNamespace(optimizer='muon', decay_mode='independent',
                               learning_rate=.005, weight_decay=.00006)
    with pytest.raises(ValueError, match='KDA'):
        build_optimizer(model, training)


@pytest.mark.parametrize('width', [64, 256])
def test_all_projection_and_convolution_gradients_match_matrix_reference(width):
    from src.attention.kda import KDA
    from tests.test_kda import matrix_reference
    torch.set_num_threads(1)
    torch.manual_seed(9362)
    actual_layer = KDA(config(width)).double()
    reference_layer = deepcopy(actual_layer)
    x = torch.randn(2, 5, width, dtype=torch.float64, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    probe = torch.randn_like(x)
    actual = actual_layer(x)[0]
    reference, _ = matrix_reference(reference_layer, xr)
    torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-9)
    actual_grad = torch.autograd.grad((actual*probe).sum(), (x, *actual_layer.parameters()))
    reference_grad = torch.autograd.grad((reference*probe).sum(), (xr, *reference_layer.parameters()))
    for ga, gr in zip(actual_grad, reference_grad):
        assert torch.isfinite(ga).all() and torch.isfinite(gr).all()
        torch.testing.assert_close(ga, gr, atol=1e-9, rtol=1e-7)
