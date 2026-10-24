from types import SimpleNamespace
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Allow `from src...` imports when running pytest from repo root.
    sys.path.append(str(PROJECT_ROOT))

import torch
import pytest

from src.attention.cache import KVCache
from src.models.gpt import GPT


def _build_cfg(attn_type: str, n_heads: int, n_kv_heads: int) -> SimpleNamespace:
    model = SimpleNamespace(
        dim=16,
        n_layers=2,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        vocab_size=32,
        block_size=8,
        dropout=0.0,
        attn_dropout=0.0,
        attn_bias=True,
        causal=True,
        attn_type=attn_type,
        mlp_type="dense",
        moe=SimpleNamespace(num_experts=2, top_k=1, aux_loss_weight=0.0),
    )
    training = SimpleNamespace(batch_size=2)
    return SimpleNamespace(model=model, training=training)


@pytest.mark.parametrize(
    "attn_type,n_heads,n_kv_heads",
    [
        ("mha", 4, 4),
        ("gqa", 4, 2),
    ],
)
def test_kv_cache_matches_full_forward(attn_type: str, n_heads: int, n_kv_heads: int) -> None:
    torch.manual_seed(0)
    cfg = _build_cfg(attn_type, n_heads, n_kv_heads)
    model = GPT(cfg.model)
    model.eval()

    batch = 2
    seq_len = 6
    idx = torch.randint(0, cfg.model.vocab_size, (batch, seq_len))
    full_logits, _ = model(idx)

    caches = [
        KVCache(cfg, device="cpu", batch_size=batch, dtype=next(model.parameters()).dtype)
        for _ in range(cfg.model.n_layers)
    ]
    step_logits = []
    for t in range(seq_len):
        logits, _ = model(idx[:, t:t + 1], cache=caches, start_pos=t)
        step_logits.append(logits[:, -1, :])
    cached_logits = torch.stack(step_logits, dim=1)

    torch.testing.assert_close(full_logits, cached_logits, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('width', [64, 128, 256])
def test_policy_gqa_cache_shape_and_prefill_decode(width):
    from tests.test_gqa_study import gqa_config
    torch.set_num_threads(1)
    cfg = SimpleNamespace(model=gqa_config(width))
    model = GPT(cfg.model).eval()
    cache = [KVCache(cfg, 'cpu', batch_size=2) for _ in model.transformer.h]
    assert all(c.k_cache.shape[1] == width // 64 for c in cache)
    x = torch.randint(0, cfg.model.vocab_size, (2, 8))
    with torch.no_grad():
        full = model(x)[0]
        prefill = model(x[:, :4], cache=cache)[0]
        decoded = [model(x[:, i:i+1], cache=cache, start_pos=i)[0] for i in range(4, 8)]
    torch.testing.assert_close(torch.cat([prefill, *decoded], dim=1), full, atol=2e-6, rtol=1e-5)
    with pytest.raises(ValueError, match='Cache KV heads'):
        KVCache(cfg, 'cpu', batch_size=2, n_kv_heads=width // 32)


def test_legacy_gqa_group_alias_cache_matches_attention():
    cfg = _build_cfg('gqa', 4, 2)
    del cfg.model.n_kv_heads
    cfg.model.n_groups = 2
    assert KVCache(cfg, 'cpu').k_cache.shape[1] == GPT(cfg.model).transformer.h[0].attn.n_kv_heads == 2
