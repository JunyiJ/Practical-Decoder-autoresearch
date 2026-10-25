"""Actual CPU trainer integration. Run only after the active research worker exits."""
import copy
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.train.train import run_training
from src.utils.checkpoint import read_checkpoint
from tests.test_schedule_study import plan
from src.research.schedule_study import suite_spec


def cfg_for(tmp_path, family, width, backbone='dense'):
    cfg = OmegaConf.create(suite_spec(plan(backbone), family, .0025)['candidate']['config'])
    corpus = tmp_path / 'synthetic.txt'
    corpus.write_text('abcdefghijklmno\n' * 1024)
    cfg.data.path = str(corpus)
    cfg.model.update(dim=width, n_heads=width // 32)
    cfg.training.update(max_iters=8, device='cpu', strict_device=True, seed=42,
                        eval_seed=2026, eval_every=2, eval_iters=2, checkpoint_every=4,
                        checkpoint_dir=str(tmp_path / 'full'), keep_best_checkpoint=False)
    if family != 'constant':
        cfg.training.lr_schedule.total_steps = 8
        if family == 'linear_cooldown': cfg.training.lr_schedule.start_step = 5
    return cfg


def equal_nested(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal_nested(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal_nested(x, y)
    else: assert a == b


@pytest.mark.parametrize('backbone', ['dense', 'moe4_top2'])
@pytest.mark.parametrize('width', [64, 256])
@pytest.mark.parametrize('family', ['constant', 'cosine', 'linear_cooldown'])
def test_full_trainer_resume_exact_with_bound_schedule(tmp_path, width, family, backbone):
    torch.set_num_threads(1)
    cfg = cfg_for(tmp_path, family, width, backbone)
    full = run_training(copy.deepcopy(cfg))
    middle = list((tmp_path / 'full').glob('step_000004_*.pt'))
    assert len(middle) == 1
    cfg.training.checkpoint_dir = str(tmp_path / 'resumed')
    cfg.training.resume_from = str(middle[0])
    resumed = run_training(copy.deepcopy(cfg))
    a, b = [read_checkpoint(r['final_checkpoint']) for r in [full, resumed]]
    for key in ('model_state', 'optimizer_state'):
        equal_nested(a[key], b[key])
    equal_nested(a['training_state']['loader_rng'], b['training_state']['loader_rng'])
    equal_nested(a['training_state']['torch_rng'], b['training_state']['torch_rng'])
    for key in ('train_ce_loss', 'val_ce_loss', 'lr_multiplier', 'scheduled_learning_rate'):
        assert full['final_metric'][key] == resumed['final_metric'][key]
    assert b['step'] == 8
    expected = 1 if family == 'constant' else .1
    assert b['training_state']['metrics'][-1]['lr_multiplier'] == expected
    for optimizer in b['optimizer_state']['optimizers']:
        for group in optimizer['param_groups']:
            assert group['lr'] == .0025 * group['width_lr_scale'] * expected
    if family != 'constant':
        cfg.training.max_iters = 9
        with pytest.raises(ValueError, match='total_steps'): run_training(copy.deepcopy(cfg))
        cfg.training.lr_schedule.total_steps = 9
        with pytest.raises(ValueError, match='Resume requires matching'): run_training(copy.deepcopy(cfg))


@pytest.mark.parametrize('backbone', ['dense', 'moe4_top2'])
def test_explicit_constant_has_identical_legacy_training_math(tmp_path, backbone):
    torch.set_num_threads(1)
    cfg = cfg_for(tmp_path, 'constant', 64, backbone)
    explicit = run_training(copy.deepcopy(cfg))
    del cfg.training.lr_schedule
    cfg.training.checkpoint_dir = str(tmp_path / 'legacy')
    legacy = run_training(copy.deepcopy(cfg))
    a, b = [read_checkpoint(r['final_checkpoint']) for r in [explicit, legacy]]
    for key in ('model_state', 'optimizer_state'):
        equal_nested(a[key], b[key])
    equal_nested(a['training_state']['loader_rng'], b['training_state']['loader_rng'])
    assert 'lr_schedule' not in b['training_state']['contract']
    assert 'lr_multiplier' not in legacy['final_metric']
    assert explicit['final_metric']['val_ce_loss'] == legacy['final_metric']['val_ce_loss']
