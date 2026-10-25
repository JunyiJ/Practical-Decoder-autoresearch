"""CPU scalar/optimizer correctness; no model training or accelerator work."""
import copy
import math
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src.train.lr_schedule import LearningRateSchedule, multiplier, schedule_contract
from src.train.optimizers import HybridOptimizer


def config(kind='cosine', total=12):
    schedule = {'kind': kind}
    if kind != 'constant':
        schedule.update(total_steps=total, min_ratio=.1)
    if kind == 'linear_cooldown':
        schedule['start_step'] = 2 * total // 3
    return OmegaConf.create({'max_iters': total, 'lr_schedule': schedule})


@pytest.mark.parametrize('kind', ['constant', 'cosine', 'linear_cooldown'])
def test_full_budget_sequence(kind):
    c = config(kind, 3000)
    s = schedule_contract(c)
    values = [multiplier(s, step) for step in range(3000)]
    assert values[0] == 1
    assert values[-1] == (1 if kind == 'constant' else .1)
    assert all(.1 <= v <= 1 for v in values)
    assert all(a >= b for a, b in zip(values, values[1:]))
    if kind == 'linear_cooldown':
        assert values[:2001] == [1.] * 2001
        assert values[2001] == 1 - .9 / 999
    if kind == 'cosine':
        assert values[1500] == pytest.approx(.1 + .9 * (1 + math.cos(math.pi * 1500 / 2999)) / 2)


@pytest.mark.parametrize('change', [
    {'kind': 'unknown'}, {'extra': 1}, {'total_steps': 13}, {'total_steps': 12.},
    {'total_steps': True}, {'min_ratio': 0}, {'min_ratio': -1}, {'min_ratio': 2},
    {'min_ratio': float('nan')}, {'min_ratio': float('inf')}, {'min_ratio': True},
])
def test_bad_schedule_rejected(change):
    c = config(); c.lr_schedule.update(change)
    with pytest.raises(ValueError): schedule_contract(c)


@pytest.mark.parametrize('start', [-1, 11, 12, 1., True])
def test_bad_cooldown_boundary_rejected(start):
    c = config('linear_cooldown'); c.lr_schedule.start_step = start
    with pytest.raises(ValueError): schedule_contract(c)


@pytest.mark.parametrize('step', [-1, True, 1., 12])
def test_bad_update_rejected(step):
    with pytest.raises(ValueError): multiplier(schedule_contract(config()), step)


def make_optimizer(width=64):
    matrix = torch.nn.Parameter(torch.tensor([[.2, -.1], [.4, .3]]))
    vector = torch.nn.Parameter(torch.tensor([.2, .3]))
    scale = 1 / math.sqrt(width / 64)
    muon = torch.optim.Muon([{'params': [matrix], 'lr': .005 * scale,
                             'width_lr_scale': scale, 'parameter_names': ['hidden']}],
                            lr=.005, weight_decay=0, adjust_lr_fn='match_rms_adamw')
    adam = torch.optim.AdamW([{'params': [vector], 'lr': .005,
                              'width_lr_scale': 1., 'parameter_names': ['vector']}],
                            lr=.005, weight_decay=0)
    return HybridOptimizer([muon, adam], .00006), [matrix, vector]


@pytest.mark.parametrize('width', [64, 128, 256])
@pytest.mark.parametrize('kind', ['constant', 'cosine', 'linear_cooldown'])
def test_resume_uses_actual_child_groups_and_original_rates(width, kind):
    # Deterministic synthetic gradients test optimizer state, not model quality.
    torch.set_num_threads(1)
    cfg = config(kind)
    direct, direct_params = make_optimizer(width)
    direct_schedule = LearningRateSchedule(direct, cfg)
    snapshot = None
    history = []
    for step in range(12):
        ratio = direct_schedule.apply(step)
        for p in direct_params: p.grad = p.detach().square() + .01 * (step + 1)
        direct.step()
        history.append([p.detach().clone() for p in direct_params])
        if step == 4:
            snapshot = copy.deepcopy(direct.state_dict())
        assert direct.param_groups[0]['lr'] == .005 * (1 / math.sqrt(width / 64)) * ratio
        assert direct.param_groups[1]['lr'] == .005 * ratio
    resumed, resumed_params = make_optimizer(width)
    resumed_schedule = LearningRateSchedule(resumed, cfg)  # Before load.
    with torch.no_grad():
        for target, saved in zip(resumed_params, history[4]): target.copy_(saved)
    resumed.load_state_dict(snapshot)
    assert all(a is b for a, b in zip(resumed.param_groups,
               [g for o in resumed.optimizers for g in o.param_groups]))
    for step in range(5, 12):
        ratio = resumed_schedule.apply(step)
        rates = [g['lr'] for g in resumed.param_groups]
        resumed_schedule.apply(step)
        assert rates == [g['lr'] for g in resumed.param_groups]  # No compounding.
        for p in resumed_params: p.grad = p.detach().square() + .01 * (step + 1)
        resumed.step()
        assert all(torch.equal(a, b) for a, b in zip(resumed_params, history[step]))
    assert all(torch.equal(a, b) for a, b in zip(resumed_params, direct_params))


def test_default_does_not_modify_rates_or_require_new_contract():
    optimizer, _ = make_optimizer()
    cfg = SimpleNamespace(max_iters=12)
    assert schedule_contract(cfg) is None
    scheduler = LearningRateSchedule(optimizer, cfg)
    optimizer.param_groups[0]['lr'] = .123
    assert scheduler.apply(3) is None
    assert optimizer.param_groups[0]['lr'] == .123


def test_group_reordering_is_rejected():
    optimizer, _ = make_optimizer()
    scheduler = LearningRateSchedule(optimizer, config())
    optimizer.param_groups.reverse()
    with pytest.raises(ValueError, match='groups changed'): scheduler.apply(0)


def test_nonconstant_horizon_is_bound_but_constant_can_extend():
    cfg = config(); frozen = schedule_contract(cfg)
    cfg.max_iters = 24
    with pytest.raises(ValueError): schedule_contract(cfg)
    cfg.lr_schedule.total_steps = 24
    assert schedule_contract(cfg) != frozen
    cfg = config('constant'); frozen = schedule_contract(cfg)
    cfg.max_iters = 24
    assert schedule_contract(cfg) == frozen


def test_training_resume_contract_includes_complete_schedule(tmp_path):
    from src.train.train import _training_contract, _contracts_match
    corpus = tmp_path / 'text.txt'; corpus.write_text('synthetic contract fixture')
    cfg = OmegaConf.create({'data': {'path': str(corpus)}, 'model': {'dim': 64},
                            'training': {'max_iters': 12, 'batch_size': 8,
                                         'learning_rate': .005, 'weight_decay': .00006,
                                         'optimizer': 'muon', 'decay_mode': 'independent'}})
    loader = SimpleNamespace(batch_layout='paired_blocks512', train_fraction=.8,
                             val_fraction=.1, eval_seed=2026)
    legacy = _training_contract(cfg, 'cpu', loader)
    assert 'lr_schedule' not in legacy
    cfg.training.lr_schedule = config('linear_cooldown').lr_schedule
    contract = _training_contract(cfg, 'cpu', loader)
    assert contract['lr_schedule'] == dict(kind='linear_cooldown', total_steps=12,
                                          min_ratio=.1, start_step=8)
    assert not _contracts_match(legacy, contract)
    for key, value in [('min_ratio', .2), ('start_step', 7), ('total_steps', 24)]:
        changed = copy.deepcopy(contract); changed['lr_schedule'][key] = value
        assert not _contracts_match(changed, contract)
    assert _contracts_match(copy.deepcopy(contract), contract)


@pytest.mark.parametrize('corrupt', [None, 'missing', 'lr_multiplier', 'base_learning_rate',
                                     'scheduled_learning_rate', 'step', 'history','omitted_update'])
def test_worker_result_requires_matching_schedule_metrics(tmp_path, corrupt):
    from src.research.runner import validate_result
    t = dict(max_iters=12, batch_size=8, block_size=128, learning_rate=.005,eval_every=6,
             lr_schedule=dict(config().lr_schedule))
    request = {'stage': 'confirmation', 'config': {'training': t}}
    plan = {'device': 'cpu', 'training': t}
    path = tmp_path / 'final.pt'; path.write_bytes(b'not loaded in contract test')
    metrics = []
    for step in (0, 6, 11):
        ratio = multiplier(t['lr_schedule'], step)
        metrics.append(dict(step=step, tokens_processed=(step + 1) * 8 * 128,
                            val_ce_loss=1.5, rss_mb=1., accel_mem_mb=None, accel_mem_kind=None,
                            lr_multiplier=ratio, base_learning_rate=.005,
                            scheduled_learning_rate=.005 * ratio))
    result = dict(device='cpu', completed_steps=12, stop_reason='max_iters',
                  final_metric=metrics[-1], metrics=metrics, final_checkpoint=str(path),
                  training_time_sec=1.)
    if corrupt == 'missing': result.pop('metrics')
    elif corrupt == 'history': metrics[1]['lr_multiplier'] = .123
    elif corrupt == 'omitted_update': del metrics[1]
    elif corrupt is not None: result['final_metric'][corrupt] = .123
    if corrupt is None:
        assert validate_result(result, request, plan, tmp_path)['metric'] == 1.5
    else:
        with pytest.raises(ValueError): validate_result(result, request, plan, tmp_path)
