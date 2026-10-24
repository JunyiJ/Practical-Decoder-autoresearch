"""Correctness tests for sigmoid-gated MLA with decoupled RoPE."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.attention.cache import KVCache
from src.attention.mla import GatedMLA, MLA
from src.models.gpt import GPT
from src.models.rope import precompute_freqs_cis


def _config(**overrides):
    values = dict(
        dim=32,
        dimcq=12,
        dimckv=10,
        n_heads=4,
        head_dim_r=6,  # Deliberately different from the content head width (8).
        block_size=8,
        dropout=0.0,
        attn_dropout=0.0,
        attn_bias=True,
        causal=True,
        attn_type="gated_mla",
        rope=1,
        n_layers=2,
        vocab_size=41,
        mlp_type="dense",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _rotate(x, cos, sin):
    # Independent complex-number reference for the interleaved RoPE pairs.
    pairs = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
    rotated = pairs * torch.complex(cos, sin)
    return torch.view_as_real(rotated).flatten(-2)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_matches_per_head_sdpa_reference(causal, bias):
    torch.manual_seed(11)
    cfg = _config(causal=causal, attn_bias=bias)
    attn = GatedMLA(cfg).eval()
    x = torch.randn(2, 6, cfg.dim)
    cos, sin = precompute_freqs_cis(cfg.head_dim_r, x.size(1))
    cq, ckv = attn.dq(x), attn.dkv(x)
    content_q = attn.uq(cq).chunk(cfg.n_heads, dim=-1)
    content_k = attn.uk(ckv).chunk(cfg.n_heads, dim=-1)
    values = attn.uv(ckv).chunk(cfg.n_heads, dim=-1)
    rotary_q = attn.qr(cq).chunk(cfg.n_heads, dim=-1)
    rotary_k = _rotate(attn.kr(x), cos, sin)
    heads = []
    for q, k, v, qr in zip(content_q, content_k, values, rotary_q):
        heads.append(F.scaled_dot_product_attention(
            torch.cat((q, _rotate(qr, cos, sin)), dim=-1),
            torch.cat((k, rotary_k), dim=-1),
            v,
            is_causal=causal,
        ))
    expected = attn.out(torch.sigmoid(attn.gate(x)) * torch.cat(heads, dim=-1))
    actual, aux_loss = attn(x, freq_cos=cos, freq_sin=sin)
    assert aux_loss is None
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_zero_gate_logits_halve_attention_before_output_projection():
    torch.manual_seed(12)
    cfg = _config()
    gated = GatedMLA(cfg).eval()
    plain = MLA(cfg).eval()
    plain.load_state_dict({k: v for k, v in gated.state_dict().items()
                           if not k.startswith("gate.")})
    with torch.no_grad():
        gated.gate.weight.zero_()
        gated.gate.bias.zero_()
        # Nonzero output bias distinguishes gating before vs. after projection.
        gated.out.bias.fill_(0.3)
        plain.out.bias.copy_(gated.out.bias)
    x = torch.randn(2, 6, cfg.dim)
    cos, sin = precompute_freqs_cis(cfg.head_dim_r, x.size(1))
    actual = gated(x, freq_cos=cos, freq_sin=sin)[0]
    baseline = plain(x, freq_cos=cos, freq_sin=sin)[0]
    expected = 0.5 * (baseline - plain.out.bias) + gated.out.bias
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_rope_is_active_and_future_tokens_do_not_change_prefix():
    torch.manual_seed(13)
    # Exercise mask growth beyond its initial allocation as well.
    cfg = _config(block_size=2)
    attn = GatedMLA(cfg).eval()
    x = torch.randn(2, 6, cfg.dim)
    cos, sin = precompute_freqs_cis(cfg.head_dim_r, x.size(1))
    full = attn(x, freq_cos=cos, freq_sin=sin)[0]
    changed = x.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 10
    other = attn(changed, freq_cos=cos, freq_sin=sin)[0]
    torch.testing.assert_close(full[:, :3], other[:, :3], rtol=0, atol=0)
    assert not torch.allclose(full, attn(x)[0])


@pytest.mark.parametrize("chunks", [(1, 1, 1, 1, 1, 1), (3, 2, 1)])
def test_gpt_cached_decoding_matches_full_forward(chunks):
    torch.manual_seed(14)
    cfg = _config()
    model = GPT(cfg).eval()
    assert all(isinstance(block.attn, GatedMLA) for block in model.transformer.h)
    assert model.transformer.h[0].freq_cos.shape == (cfg.block_size, cfg.head_dim_r // 2)
    cache_cfg = SimpleNamespace(model=cfg)
    caches = [KVCache(cache_cfg, "cpu", batch_size=2) for _ in range(cfg.n_layers)]
    for cache in caches:
        assert cache.c_cache.shape == (2, cfg.block_size, cfg.dimckv)
        assert cache.r_cache.shape == (2, cfg.block_size, cfg.head_dim_r)
        assert cache.k_cache is None and cache.v_cache is None
    tokens = torch.randint(cfg.vocab_size, (2, sum(chunks)))
    with torch.no_grad():
        full = model(tokens)[0]
        pieces = []
        start = 0
        for length in chunks:
            pieces.append(model(tokens[:, start:start + length],
                                cache=caches, start_pos=start)[0])
            start += length
        torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=1e-5, atol=1e-6)
        for cache in caches:
            cache.reset()
            assert torch.count_nonzero(cache.c_cache) == 0
            assert torch.count_nonzero(cache.r_cache) == 0
        torch.testing.assert_close(model(tokens, cache=caches)[0], full, rtol=1e-5, atol=1e-6)


def test_gpt_backward_and_state_dict_round_trip():
    torch.manual_seed(15)
    cfg = _config()
    model = GPT(cfg)
    tokens = torch.randint(cfg.vocab_size, (2, 6))
    targets = torch.randint(cfg.vocab_size, (2, 6))
    logits, loss = model(tokens, targets)
    assert logits.shape == (2, 6, cfg.vocab_size)
    assert torch.isfinite(loss)
    loss.backward()
    for block in model.transformer.h:
        for name, parameter in block.attn.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            if name.endswith("weight"):
                assert parameter.grad.abs().sum() > 0, name
    clone = GPT(cfg)
    clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(clone(tokens)[0], logits, rtol=0, atol=0)
