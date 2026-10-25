"""Tiny CPU model/attention equations and grouping, without optimizer steps."""
import copy
import math
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F
from src.attention.mha import MHA
from src.attention.cache import KVCache
from src.attention.output_gate import validate_output_gate_contract
from src.models.gpt import GPT
from src.models.rope import precompute_freqs_cis
from src.train.optimizers import build_optimizer
from tests.test_qk_norm_model_contract import model_config
from tests.test_qk_norm_attention import independent_rotate


def config(width=64, normalized=False, kind='sigmoid_headwise_zero'):
    cfg = model_config(width, normalized)
    if kind != 'none':
        cfg.attention_output_gate = kind
        cfg.output_gate_width_policy = 'mha_head32_output_gate'
    return cfg


@pytest.mark.parametrize('width', [64, 256])
@pytest.mark.parametrize('normalized', [False, True])
def test_common_initialization_and_optimizer_parameter_coverage(width, normalized):
    torch.set_num_threads(1)
    models = []
    for kind in ('none', 'static_half', 'sigmoid_headwise_zero'):
        torch.manual_seed(9311)
        m = GPT(config(width, normalized, kind)); models.append(m)
        if kind == 'none':
            rng = torch.get_rng_state().clone()
        else:
            assert torch.equal(rng, torch.get_rng_state())
        named = dict(m.named_parameters())
        for name, value in models[0].named_parameters():
            assert torch.equal(value, named[name])
        expected = 4 * width * (width // 32) if kind == 'sigmoid_headwise_zero' else 0
        assert sum(p.numel() for p in m.parameters()) - sum(p.numel() for p in models[0].parameters()) == expected
        opt = build_optimizer(m, SimpleNamespace(optimizer='muon', decay_mode='independent', learning_rate=.005, weight_decay=.00006))
        seen = []
        for child in opt.optimizers:
            assert not child.state
            for group in child.param_groups:
                name, = group['parameter_names']; seen.append(name)
                matrix = name.startswith('transformer.h.') and named[name].ndim == 2
                assert isinstance(child, torch.optim.Muon if matrix else torch.optim.AdamW)
                assert group['width_lr_scale'] == (1 / math.sqrt(width / 64) if matrix else 1.)
                if name.endswith('output_gate.weight'):
                    assert matrix and named[name].shape == (width // 32, width)
        assert len(seen) == len(set(seen)) == len(named) and set(seen) == set(named)
    tokens = torch.arange(7).reshape(1, 7)
    with torch.no_grad():
        assert torch.equal(models[1](tokens)[0], models[2](tokens)[0])


@pytest.mark.parametrize('width', [64, 256])
@pytest.mark.parametrize('normalized', [False, True])
@pytest.mark.parametrize('kind', ['static_half', 'sigmoid_headwise_zero'])
def test_independent_attention_formula_and_gradients(width, normalized, kind):
    torch.set_num_threads(1); torch.manual_seed(9312)
    model = MHA(config(width, normalized, kind)).double()
    if kind == 'sigmoid_headwise_zero':
        with torch.no_grad():
            model.output_gate.weight.copy_(torch.linspace(-.2, .3, model.output_gate.weight.numel()).reshape_as(model.output_gate.weight))
    x = torch.randn(1, 5, width, dtype=torch.float64, requires_grad=True)
    cos, sin = [a.double() for a in precompute_freqs_cis(32, 5)]
    actual = model(x, freq_cos=cos, freq_sin=sin)[0]
    q, k, v = [z.reshape(1, 5, width // 32, 32).transpose(1, 2)
               for z in F.linear(x, model.qkv.weight, model.qkv.bias).chunk(3, -1)]
    if normalized:
        q = q / torch.sqrt(q.square().mean(-1, keepdim=True) + 1e-6) * model.q_norm.weight
        k = k / torch.sqrt(k.square().mean(-1, keepdim=True) + 1e-6) * model.k_norm.weight
    q, k = independent_rotate(q, cos, sin), independent_rotate(k, cos, sin)
    scores = torch.einsum('bhtd,bhsd->bhts', q, k) / math.sqrt(32)
    scores = scores.masked_fill(torch.arange(5)[:, None] < torch.arange(5)[None, :], -torch.inf)
    value_mix = torch.einsum('bhts,bhsd->bthd', scores.softmax(-1), v)
    if kind == 'static_half':
        value_mix = value_mix * .5
    else:
        gate = torch.einsum('btd,hd->bth', x, model.output_gate.weight).sigmoid()
        value_mix = value_mix * gate[..., None]
    expected = F.linear(value_mix.reshape(1, 5, width), model.out.weight, model.out.bias)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    coefficient = torch.linspace(-1, 1, actual.numel()).reshape_as(actual)
    params = (x, *model.parameters())
    ga = torch.autograd.grad((actual * coefficient).sum(), params, retain_graph=True)
    gb = torch.autograd.grad((expected * coefficient).sum(), params)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('normalized', [False, True])
@pytest.mark.parametrize('width', [64, 256])
@pytest.mark.parametrize('kind', ['static_half', 'sigmoid_headwise_zero'])
def test_model_full_cached_causality_and_strict_reload(tmp_path, normalized, width, kind):
    torch.set_num_threads(1); torch.manual_seed(9313)
    c = config(width, normalized, kind); model = GPT(c).eval()
    if kind == 'sigmoid_headwise_zero':
        with torch.no_grad():
            for block in model.transformer.h:
                w = block.attn.output_gate.weight
                w.copy_(torch.linspace(-.1, .1, w.numel()).reshape_as(w))
    idx = torch.arange(7).reshape(1, 7)
    with torch.no_grad():
        full = model(idx)[0]
        caches = [KVCache(SimpleNamespace(model=c), 'cpu', batch_size=1) for _ in range(4)]
        parts = [model(idx[:, :3], cache=caches, start_pos=0)[0], model(idx[:, 3:5], cache=caches, start_pos=3)[0], model(idx[:, 5:], cache=caches, start_pos=5)[0]]
        torch.testing.assert_close(torch.cat(parts, 1), full, atol=1e-6, rtol=1e-5)
        changed = idx.clone(); changed[:, 4:] += 10
        assert torch.equal(model(changed)[0][:, :4], full[:, :4])
        path = tmp_path / 'state.pt'; torch.save(model.state_dict(), path)
        fresh = GPT(copy.deepcopy(c)).eval()
        fresh.load_state_dict(torch.load(path, weights_only=True), strict=True)
        assert torch.equal(fresh(idx)[0], full)
        if kind == 'sigmoid_headwise_zero':
            with pytest.raises(RuntimeError):
                GPT(config(width, normalized, 'none')).load_state_dict(model.state_dict(), strict=True)


@pytest.mark.parametrize('field,value', [('attn_type', 'gqa'), ('n_layers', 6), ('dim', 512),
    ('n_heads', 4), ('ffn', 'gelu'), ('ffn_bias', True), ('dropout', .01),
    ('gate_bias', False), ('gate_temperature', 1.), ('gate_init', 'zeros'),
    ('gate_placement', 'after_output'), ('gate_dim', 2), ('attn_bias', False),
    ('residual_scale', .5), ('output_gate_width_policy', 'unknown'),
    ('qk_norm_eps', 1e-6), ('qk_norm_affine', True), ('qk_norm_placement', 'pre_rope'),
    ('qk_scale', 1.)])
def test_unreviewed_contract_rejected(field, value):
    c = config(); setattr(c, field, value)
    with pytest.raises(ValueError):
        GPT(c)


def test_default_cannot_smuggle_new_policy():
    c = config(kind='none'); c.output_gate_width_policy = 'mha_head32_output_gate'
    with pytest.raises(ValueError):
        validate_output_gate_contract(c)
