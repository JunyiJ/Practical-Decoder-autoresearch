"""Check KDA against the matrix recurrence and across cached chunk boundaries."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.attention.cache import KVCache
from src.attention.kda import KDA
from src.models.gpt import GPT
from src.utils.generate import generate


def config(**kwargs):
    values = dict(dim=12, n_heads=3, dim_alpha=3, attn_bias=True,
                  dropout=0.0, causal=True, attn_type="kda", rope=1,
                  n_layers=2, vocab_size=29, block_size=16, mlp_type="dense")
    values.update(kwargs)
    return SimpleNamespace(**values)


def cache_for(cfg, batch=2, dtype=torch.float32):
    return KVCache(SimpleNamespace(model=cfg), "cpu", batch_size=batch, dtype=dtype)


def matrix_reference(layer, x):
    b, s, _ = x.shape
    features = []
    # Explicit causal windows independently check Conv1d padding and alignment.
    for name, projected in zip(("q", "k", "v"), layer.qkv(x).chunk(3, dim=-1)):
        padded = F.pad(projected.transpose(1, 2), (3, 0))
        windows = padded.unfold(-1, 4, 1)
        weights = getattr(layer, name + "_conv").weight[:, 0]
        y = F.silu((windows * weights[None, :, None, :]).sum(-1))
        features.append(y.transpose(1, 2).reshape(b, s, layer.n_heads, layer.head_dim))
    q, k, v = features
    q, k = F.normalize(q, dim=-1, eps=1e-6), F.normalize(k, dim=-1, eps=1e-6)
    z = layer.z2(layer.z1(x)).reshape_as(q)
    alpha = torch.exp(-5 * torch.sigmoid(layer.A.exp()[None, None, :, None] * z))
    beta = torch.sigmoid(layer.beta_proj(x))
    state = x.new_zeros(b, layer.n_heads, layer.head_dim, layer.head_dim)
    eye = torch.eye(layer.head_dim, dtype=x.dtype)
    out = []
    for t in range(s):
        key = k[:, t].unsqueeze(-1)
        # Direct paper equation, rather than implementation's prediction/error form.
        transition = (eye - beta[:, t, :, None, None] * (key @ key.transpose(-1, -2)))
        state = transition @ torch.diag_embed(alpha[:, t]) @ state
        state = state + beta[:, t, :, None, None] * (key @ v[:, t].unsqueeze(-2))
        out.append((state.transpose(-1, -2) @ q[:, t].unsqueeze(-1)).squeeze(-1))
    out = torch.stack(out, dim=1)
    out = out * torch.rsqrt(out.square().mean(-1, keepdim=True) + 1e-6)
    out = out * layer.head_norm.weight
    out = out.reshape(b, s, layer.dim) * torch.sigmoid(layer.output_gate(x))
    return layer.out_proj(out), state


@pytest.mark.parametrize("heads", [1, 3])
@pytest.mark.parametrize("bias", [False, True])
def test_output_and_state_match_matrix_equation(heads, bias):
    torch.manual_seed(24)
    cfg = config(n_heads=heads, attn_bias=bias)
    layer = KDA(cfg).double().eval()
    with torch.no_grad():
        layer.A.copy_(torch.linspace(-0.7, 0.8, heads))
        layer.z2.bias.copy_(torch.linspace(-3, 2, cfg.dim))
        x = torch.randn(2, 7, cfg.dim, dtype=torch.float64)
        expected, final_state = matrix_reference(layer, x)
        cache = cache_for(cfg, dtype=torch.float64)
        actual, aux = layer(x, cache=cache)
    assert aux is None
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(cache.state, final_state, rtol=1e-9, atol=1e-10)


def test_closed_write_gate_produces_zero_output():
    layer = KDA(config()).double()
    with torch.no_grad():
        layer.beta_proj.weight.zero_()
        layer.beta_proj.bias.fill_(-1000)
    actual, _ = layer(torch.randn(2, 6, 12, dtype=torch.float64))
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)


@pytest.mark.parametrize("chunks", [(1,) * 8, (2, 1, 5), (5, 3)])
@pytest.mark.parametrize("head_dim", [3, 4])
def test_model_cache_causality_reset_and_generation(chunks, head_dim):
    torch.manual_seed(25)
    cfg = config(dim=3 * head_dim)
    model = GPT(cfg).eval()
    assert model.position_encoding == "none"
    assert "wpe" not in model.transformer
    assert all(block.freq_cos is None and block.freq_sin is None for block in model.transformer.h)
    caches = [cache_for(cfg) for _ in range(cfg.n_layers)]
    # Constant-size recurrent/conv storage, independent of context capacity.
    bigger = cache_for(config(dim=cfg.dim, block_size=1024))
    assert sum(v.numel() for v in bigger.buffers()) == sum(v.numel() for v in caches[0].buffers())
    tokens = torch.randint(cfg.vocab_size, (2, 8))
    with torch.no_grad():
        full = model(tokens)[0]
        pieces, start = [], 0
        for length in chunks:
            pieces.append(model(tokens[:, start:start + length], cache=caches, start_pos=start)[0])
            start += length
        torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=1e-5, atol=1e-6)
        changed = tokens.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % cfg.vocab_size
        torch.testing.assert_close(model(changed)[0][:, :4], full[:, :4], rtol=0, atol=0)
        for cache in caches:
            assert cache.position == 8
            cache.reset()
            assert cache.position == 0
            assert all(torch.count_nonzero(v) == 0 for v in cache.buffers())
        torch.testing.assert_close(model(tokens, cache=caches)[0], full, rtol=1e-5, atol=1e-6)
        for cache in caches:
            cache.reset()
        uncached = generate(model, tokens[:, :3], 4, greedy=True)
        cached = generate(model, tokens[:, :3], 4, cache=caches, greedy=True)
        torch.testing.assert_close(cached, uncached, rtol=0, atol=0)


def test_gradients_and_checkpoint_round_trip():
    torch.manual_seed(26)
    cfg = config()
    model = GPT(cfg)
    tokens = torch.randint(cfg.vocab_size, (2, 8))
    out, loss = model(tokens, targets=tokens)
    loss.backward()
    for block in model.transformer.h:
        for name, parameter in block.attn.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
    clone = GPT(cfg)
    clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(clone(tokens)[0], out, rtol=0, atol=0)


def test_input_gradient_matches_numerical_gradient():
    torch.manual_seed(27)
    layer = KDA(config(dim=4, n_heads=2, dim_alpha=2)).double()
    x = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda value: layer(value)[0], (x,), fast_mode=True)


def test_mixed_precision_keeps_recurrent_cache_float32():
    torch.manual_seed(28)
    cfg = config()
    layer = KDA(cfg).eval()
    cache = cache_for(cfg, dtype=torch.bfloat16)
    x = torch.randn(2, 6, cfg.dim)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        full = layer(x)[0]
        first = layer(x[:, :2], cache=cache)[0]
        second = layer(x[:, 2:], cache=cache, start_pos=2)[0]
    assert cache.state.dtype == torch.float32
    assert torch.isfinite(cache.state).all()
    assert torch.isfinite(full).all()
    torch.testing.assert_close(torch.cat((first, second), dim=1), full, rtol=0.03, atol=0.003)


def test_cache_rejects_gaps_reuse_overflow_wrong_batch_and_training():
    cfg = config(block_size=4)
    layer = KDA(cfg).eval()
    cache = cache_for(cfg)
    x = torch.randn(2, 2, cfg.dim)
    with pytest.raises(ValueError, match="gradients disabled"):
        layer(x, cache=cache)
    with torch.no_grad():
        layer(x, cache=cache)
        for position in (0, 1, 3):
            with pytest.raises(ValueError, match="contiguous"):
                layer(x, cache=cache, start_pos=position)
        with pytest.raises(ValueError, match="size exceeded"):
            layer(torch.randn(2, 3, cfg.dim), cache=cache, start_pos=2)
        with pytest.raises(ValueError, match="batch/head shape"):
            layer(x[:1], cache=cache, start_pos=2)
        with pytest.raises(ValueError, match="requires a KDA cache"):
            layer(x, cache=cache_for(config(attn_type="mha")))
        with pytest.raises(ValueError, match="Nonzero start_pos"):
            layer(x, start_pos=1)
        with pytest.raises(ValueError, match="not RoPE"):
            layer(x, freq_cos=torch.ones(2, 2))
        assert cache.position == 2


@pytest.mark.parametrize("kwargs", [dict(n_heads=0), dict(dim=11), dict(dim_alpha=0), dict(causal=False)])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        KDA(config(**kwargs))
