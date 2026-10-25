"""CPU attention equations/cache checks; no optimizer or corpus evaluation."""
import copy
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.attention.cache import KVCache
from src.attention.mha import MHA, MHA_NAIVE
from src.models.rope import precompute_freqs_cis


def config(width=64, kind='rms_head32'):
    return SimpleNamespace(dim=width, n_heads=width // 32, attn_type='mha',
                           block_size=128, dropout=0., attn_dropout=0., qk_norm=kind)


def independent_rotate(x, cos, sin):
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


def reference(model, x, cos, sin, normalized):
    b, t, width = x.shape
    projected = F.linear(x, model.qkv.weight, model.qkv.bias)
    q, k, v = [a.reshape(b, t, width // 32, 32).transpose(1, 2)
               for a in projected.chunk(3, -1)]
    if normalized:
        q = q / torch.sqrt(q.square().mean(-1, keepdim=True) + 1e-6) * model.q_norm.weight
        k = k / torch.sqrt(k.square().mean(-1, keepdim=True) + 1e-6) * model.k_norm.weight
    q, k = independent_rotate(q, cos, sin), independent_rotate(k, cos, sin)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(32)
    future = torch.arange(t)[:, None] < torch.arange(t)[None, :]
    weights = scores.masked_fill(future, -torch.inf).softmax(-1)
    y = (weights @ v).transpose(1, 2).reshape(b, t, width)
    return F.linear(y, model.out.weight, model.out.bias), k, v


@pytest.mark.parametrize('width', [64, 128, 256])
@pytest.mark.parametrize('kind', ['none', 'rms_head32'])
def test_forward_and_parameter_gradients_match_independent_equation(width, kind):
    torch.set_num_threads(1)
    torch.manual_seed(41)
    model = MHA(config(width, kind)).double()
    if kind != 'none':
        with torch.no_grad():
            model.q_norm.weight.copy_(torch.linspace(.4, 1.3, 32))
            model.k_norm.weight.copy_(torch.linspace(1.2, .6, 32))
    x = torch.randn(2, 7, width, dtype=torch.float64, requires_grad=True)
    cos, sin = [a.double() for a in precompute_freqs_cis(32, 7)]
    actual, _ = model(x, freq_cos=cos, freq_sin=sin)
    expected, _, _ = reference(model, x, cos, sin, kind != 'none')
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    coeff = torch.linspace(-1., 1., actual.numel()).reshape(actual.shape).double()
    params = (x, *model.parameters())
    gradients = torch.autograd.grad((actual * coeff).sum(), params, retain_graph=True)
    ref_gradients = torch.autograd.grad((expected * coeff).sum(), params)
    for a, b in zip(gradients, ref_gradients):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('width', [64, 256])
@pytest.mark.parametrize('batch', [1, 2])
@pytest.mark.parametrize('chunks', [(1, 1, 1, 1, 1, 1, 1), (3, 2, 2)])
def test_cache_keys_normalized_rotated_once_and_values_unchanged(width, batch, chunks):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg = config(width)
    model = MHA(cfg).double().eval()
    with torch.no_grad():
        model.k_norm.weight.copy_(torch.linspace(.5, 1.5, 32))
        x = torch.randn(batch, 7, width, dtype=torch.float64)
        cos, sin = [a.double() for a in precompute_freqs_cis(32, 128)]
        expected, keys, values = reference(model, x, cos[:7], sin[:7], True)
        cache = KVCache(SimpleNamespace(model=cfg), 'cpu', batch_size=batch, dtype=torch.float64)
        outputs = []
        start = 0
        for size in chunks:
            y, _ = model(x[:, start:start + size], cache=cache, start_pos=start,
                         freq_cos=cos, freq_sin=sin)
            outputs.append(y)
            start += size
            torch.testing.assert_close(cache.k_cache[:, :, :start], keys[:, :, :start], rtol=1e-11, atol=1e-11)
            torch.testing.assert_close(cache.v_cache[:, :, :start], values[:, :, :start], rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(torch.cat(outputs, 1), expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize('width', [64, 256])
def test_default_and_explicit_none_preserve_rng_weights_and_outputs(width):
    cfg = config(width, 'none')
    missing = copy.deepcopy(cfg)
    del missing.qk_norm
    torch.manual_seed(43)
    legacy = MHA(missing)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(43)
    disabled = MHA(cfg)
    assert torch.equal(rng, torch.get_rng_state())
    torch.manual_seed(43)
    enabled = MHA(config(width))
    assert torch.equal(rng, torch.get_rng_state())
    assert set(enabled.state_dict()) - set(legacy.state_dict()) == {'q_norm.weight', 'k_norm.weight'}
    for key, value in legacy.state_dict().items():
        assert torch.equal(value, disabled.state_dict()[key])
        assert torch.equal(value, enabled.state_dict()[key])
    x = torch.randn(1, 5, width)
    assert torch.equal(legacy(x)[0], disabled(x)[0])
    with pytest.raises(RuntimeError):
        enabled.load_state_dict(legacy.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        legacy.load_state_dict(enabled.state_dict(), strict=True)


def test_normalized_attention_remains_causal():
    torch.manual_seed(44)
    model = MHA(config()).double().eval()
    x = torch.randn(2, 7, 64, dtype=torch.float64)
    changed = x.clone()
    changed[:, 4:] *= 100
    cos, sin = [a.double() for a in precompute_freqs_cis(32, 7)]
    a = model(x, freq_cos=cos, freq_sin=sin)[0]
    b = model(changed, freq_cos=cos, freq_sin=sin)[0]
    assert torch.equal(a[:, :4], b[:, :4])


def test_naive_implementation_rejects_unimplemented_normalization():
    with pytest.raises(ValueError, match='only in fused MHA'):
        MHA_NAIVE(config())
