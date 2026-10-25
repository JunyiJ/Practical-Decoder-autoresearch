"""CPU model construction/inference and optimizer grouping, without update steps."""
import copy
import math
from types import SimpleNamespace

import pytest
import torch

from src.attention.cache import KVCache
from src.models.gpt import GPT
from src.train.optimizers import build_optimizer
from src.train.weight_averaging import TailWeightAverages
from tests.test_qk_norm_math import cfg


def model_config(width=64, normalized=True):
    c = cfg(dim=width, n_heads=width // 32, vocab_size=65)
    if not normalized:
        c.depth_width_policy = 'depth4_unscaled'
        c.attention_width_policy = 'mha_head32'
        del c.qk_norm
    return c


@pytest.mark.parametrize('width', [64, 128, 256])
def test_common_initialization_rng_exact_counts_and_optimizer_roles(width):
    torch.set_num_threads(1)
    torch.manual_seed(51)
    original = GPT(model_config(width, False))
    rng = torch.get_rng_state().clone()
    torch.manual_seed(51)
    normalized = GPT(model_config(width))
    assert torch.equal(rng, torch.get_rng_state())
    old, new = dict(original.named_parameters()), dict(normalized.named_parameters())
    gains = {f'transformer.h.{layer}.attn.{which}_norm.weight'
             for layer in range(4) for which in ('q', 'k')}
    assert set(new) - set(old) == gains
    for name, value in old.items():
        assert torch.equal(value, new[name])
    assert sum(p.numel() for p in new.values()) - sum(p.numel() for p in old.values()) == 256
    for name in gains:
        assert new[name].shape == (32,)
        assert torch.equal(new[name], torch.ones(32))
    training = SimpleNamespace(optimizer='muon', decay_mode='independent',
                               learning_rate=.005, weight_decay=.00006)
    optimizer = build_optimizer(normalized, training)
    seen = set()
    for child in optimizer.optimizers:
        for group in child.param_groups:
            name, = group['parameter_names']
            assert name not in seen
            seen.add(name)
            param, = group['params']
            matrix = name.startswith('transformer.h.') and param.ndim == 2
            assert isinstance(child, torch.optim.Muon if matrix else torch.optim.AdamW)
            scale = 1 / math.sqrt(width / 64) if matrix else 1.
            assert group['lr'] == .005 * scale
            assert group['width_lr_scale'] == scale and group['weight_decay'] == 0
            if name in gains:
                assert not matrix and scale == 1
    assert seen == set(new) and optimizer.decay == .00006
    assert all(not child.state for child in optimizer.optimizers)  # No steps executed.


@pytest.mark.parametrize('width', [64, 256])
def test_model_cache_and_strict_checkpoint_roundtrip(tmp_path, width):
    torch.set_num_threads(1)
    torch.manual_seed(52)
    c = model_config(width)
    model = GPT(c).eval()
    idx = torch.randint(65, (1, 7))
    with torch.no_grad():
        full = model(idx)[0]
        caches = [KVCache(SimpleNamespace(model=c), 'cpu', batch_size=1) for _ in range(4)]
        parts = [model(idx[:, :3], cache=caches, start_pos=0)[0],
                 model(idx[:, 3:5], cache=caches, start_pos=3)[0],
                 model(idx[:, 5:], cache=caches, start_pos=5)[0]]
        torch.testing.assert_close(torch.cat(parts, 1), full, atol=1e-6, rtol=1e-5)
        path = tmp_path / 'state.pt'
        torch.save(model.state_dict(), path)
        reload = GPT(copy.deepcopy(c)).eval()
        reload.load_state_dict(torch.load(path, weights_only=True), strict=True)
        assert torch.equal(full, reload(idx)[0])
        with pytest.raises(RuntimeError):
            GPT(model_config(width, False)).load_state_dict(model.state_dict(), strict=True)


def test_gains_are_included_in_shadow_parameter_inventory():
    from omegaconf import OmegaConf
    model = GPT(model_config())
    training = OmegaConf.create(dict(max_iters=30, weight_averaging=dict(kind='spaced_tail_ema',
        total_steps=30, start_step=20, interval=10, half_lives=[200])))
    tracker = TailWeightAverages(model, training)
    assert {f'transformer.h.{layer}.attn.{which}_norm.weight'
            for layer in range(4) for which in ('q', 'k')} <= {name for name, _ in tracker.parameters}
