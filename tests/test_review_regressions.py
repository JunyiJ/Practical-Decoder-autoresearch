from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from src.attention.mla import MLA
from src.models.rope import precompute_freqs_cis
from src.data.loaders import TinyDataLoader
from src.utils.generate import generate


def test_rope_known_frequencies():
    cos, sin = precompute_freqs_cis(8, 2)
    angles = torch.tensor([1., .1, .01, .001])
    torch.testing.assert_close(cos[1], angles.cos())
    torch.testing.assert_close(sin[1], angles.sin())


def test_mla_matches_per_head_reference():
    torch.manual_seed(12)
    cfg = SimpleNamespace(dim=12, n_heads=3, dimckv=5, dimcq=7,
                          head_dim_r=2, dropout=0., block_size=8)
    model = MLA(cfg).eval()
    x = torch.randn(2, 4, 12)
    cq, ckv = model.dq(x), model.dkv(x)
    expected_heads = []
    for head in range(3):
        q = torch.cat([model.uq(cq)[..., head*4:(head+1)*4],
                       model.qr(cq)[..., head*2:(head+1)*2]], dim=-1)
        k = torch.cat([model.uk(ckv)[..., head*4:(head+1)*4], model.kr(x)], dim=-1)
        v = model.uv(ckv)[..., head*4:(head+1)*4]
        expected_heads.append(F.scaled_dot_product_attention(q, k, v, is_causal=True))
    expected = model.out(torch.cat(expected_heads, dim=-1))
    actual, _ = model(x)
    torch.testing.assert_close(actual, expected)


def test_loader_accepts_exactly_one_window(tmp_path):
    path = tmp_path / 'data.txt'
    path.write_text('abcdefghij' * 10)
    loader = TinyDataLoader(path, 2, 9, 'cpu')  # validation has ten chars
    x, y = loader.get_batch('val')
    assert x.shape == (2, 9)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_eval_sampling_does_not_change_training_batches(tmp_path):
    path = tmp_path / 'data.txt'
    path.write_text('abcdefghij' * 100)
    a = TinyDataLoader(path, 2, 9, 'cpu', seed=42)
    b = TinyDataLoader(path, 2, 9, 'cpu', seed=42)
    for _ in range(3):
        a.get_batch('val')
    assert torch.equal(a.get_batch('train')[0], b.get_batch('train')[0])


def test_generation_rejects_empty_prompt():
    with pytest.raises(ValueError, match='non-empty'):
        generate(None, torch.empty(1, 0, dtype=torch.long), 1)


def tiny_cfg():
    from omegaconf import OmegaConf
    return OmegaConf.create({'model': {
        'dim': 16, 'n_heads': 4, 'n_layers': 2, 'vocab_size': 10,
        'block_size': 8, 'dropout': 0.1, 'rope': 1, 'norm_type': 'rmsnorm',
        'attn_type': 'mla', 'dimckv': 4, 'dimcq': 4, 'head_dim_r': 2,
        'mlp_type': 'moe', 'moe': {'num_experts': 2, 'top_k': 2,
            'shared_num_experts': 1, 'aux_loss_weight': .01}},
        'data': {'path': ''}, 'training': {'device': 'cpu', 'seed': 42,
            'batch_size': 2, 'learning_rate': .001, 'weight_decay': .1,
            'grad_clip': .01, 'max_iters': 3, 'eval_every': 1, 'eval_iters': 2,
            'checkpoint_dir': '', 'checkpoint_every': 0}})


def test_mla_rotary_cache_matches_full_and_chunks():
    from src.models.gpt import GPT
    from src.attention.cache import KVCache
    cfg = tiny_cfg()
    model = GPT(cfg.model).eval()
    idx = torch.randint(10, (2, 7))
    caches = [KVCache(cfg, 'cpu') for _ in range(2)]
    with torch.no_grad():
        full, _ = model(idx)
        first, _ = model(idx[:, :3], cache=caches)
        second, _ = model(idx[:, 3:], cache=caches, start_pos=3)
    torch.testing.assert_close(full, torch.cat([first, second], dim=1))


def test_training_eval_cadence_is_independent_and_clipping_applied(tmp_path, monkeypatch):
    from src.train.train import run_training
    path = tmp_path / 'corpus.txt'
    path.write_text('abcdefghij' * 100)
    cfg = tiny_cfg()
    cfg.data.path = str(path)
    cfg.training.checkpoint_dir = str(tmp_path / 'a')
    calls = []
    original_clip = torch.nn.utils.clip_grad_norm_
    def record_clip(parameters, max_norm, **kwargs):
        params = list(parameters)
        result = original_clip(params, max_norm, **kwargs)
        norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in params if p.grad is not None]))
        assert norm <= max_norm + 1e-6
        calls.append(max_norm)
        return result
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', record_clip)
    a = run_training(cfg)
    state_a = torch.load(a['final_checkpoint'], weights_only=True)['model_state']
    cfg.training.eval_every = 2
    cfg.training.checkpoint_dir = str(tmp_path / 'b')
    b = run_training(cfg)
    state_b = torch.load(b['final_checkpoint'], weights_only=True)['model_state']
    assert calls == [.01] * 6
    for key in state_a:
        torch.testing.assert_close(state_a[key], state_b[key], rtol=0, atol=0)
    assert a['final_metric']['val_loss'] == b['final_metric']['val_loss']
    assert a['final_metric']['tokens_processed'] == 48


def test_benchmark_near_cache_limit_uses_bounded_warmup():
    from src.models.gpt import GPT
    from src.utils.bench_compare import _benchmark
    cfg = tiny_cfg()
    model = GPT(cfg.model).eval()
    throughput, _ = _benchmark(model, cfg, torch.zeros(1, 7, dtype=torch.long),
                              1, True, 1., None, True)
    assert throughput > 0


def test_generation_invalid_top_k():
    with pytest.raises(ValueError, match='top_k'):
        generate(None, torch.ones(1, 1, dtype=torch.long), 1, top_k=0)


def test_resume_matches_uninterrupted_training(tmp_path):
    from src.train.train import run_training
    from src.utils.checkpoint import read_checkpoint
    path = tmp_path / 'corpus.txt'
    path.write_text('abcdefghij' * 100)
    cfg = tiny_cfg()
    cfg.data.path = str(path)
    cfg.training.checkpoint_dir = str(tmp_path / 'full')
    cfg.training.max_iters = 4
    full = run_training(cfg)
    cfg.training.checkpoint_dir = str(tmp_path / 'split')
    cfg.training.max_iters = 2
    first = run_training(cfg)
    cfg.training.resume_from = first['final_checkpoint']
    cfg.training.max_iters = 4
    resumed = run_training(cfg)
    a, b = [read_checkpoint(r['final_checkpoint']) for r in (full, resumed)]
    for key in a['model_state']:
        torch.testing.assert_close(a['model_state'][key], b['model_state'][key], rtol=0, atol=0)
    assert resumed['completed_steps'] == 4
    assert resumed['final_metric']['val_loss'] == full['final_metric']['val_loss']
    best = read_checkpoint(resumed['best_checkpoint'])
    assert best['step'] == resumed['best_metric']['step'] + 1
    assert best['training_state']['best_metric']['val_loss'] == resumed['best_metric']['val_loss']
    cfg.training.learning_rate *= 2
    with pytest.raises(ValueError, match='matching'):
        run_training(cfg)


def test_time_budget_saves_actual_completed_step(tmp_path):
    from src.train.train import run_training
    from src.utils.checkpoint import read_checkpoint
    path = tmp_path / 'corpus.txt'
    path.write_text('abcdefghij' * 100)
    cfg = tiny_cfg()
    cfg.data.path = str(path)
    cfg.training.checkpoint_dir = str(tmp_path / 'timed')
    cfg.training.max_train_seconds = 1e-12
    result = run_training(cfg)
    assert result['completed_steps'] == 1
    assert result['stop_reason'] == 'time_budget'
    assert read_checkpoint(result['final_checkpoint'])['step'] == 1
    assert result['final_metric']['step'] == 0


def test_legacy_checkpoint_rejected(tmp_path):
    from src.utils.checkpoint import read_checkpoint
    path = tmp_path / 'old.pt'
    torch.save({'model_state': {}}, path)
    with pytest.raises(ValueError, match='Legacy'):
        read_checkpoint(path)


def test_cpu_memory_is_labeled_as_process_peak():
    from src.utils.device import memory_measurement
    value, label = memory_measurement('cpu')
    assert value > 0
    assert 'lifetime peak RSS' in label
