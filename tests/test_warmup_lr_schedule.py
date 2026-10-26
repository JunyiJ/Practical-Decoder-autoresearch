"""Scalar and metadata checks only: no torch, model, optimizer steps or MPS."""
import math
from fractions import Fraction
from types import SimpleNamespace

import pytest

from src.train import lr_schedule as new


def legacy_multiplier(kind, step):
    if kind in (None, 'constant'):
        return 1.0
    if step == 2999:
        return .1
    if kind == 'cosine':
        return .1 + .9 * (1 + math.cos(math.pi * step / 2999)) / 2
    return 1.0 if step <= 2000 else 1 - .9 * (step - 2000) / 999


def config(warmup=100):
    return SimpleNamespace(max_iters=3000, lr_schedule=dict(
        kind='warmup_linear_cooldown', total_steps=3000, min_ratio=.1,
        start_step=2000, warmup_steps=warmup, initial_ratio=.1))


@pytest.mark.parametrize('warmup', [2, 100, 300, 2001])
def test_entire_sequence_against_independent_rational_segments(warmup):
    spec = new.schedule_contract(config(warmup))
    expected = []
    for i in range(3000):
        if i < warmup:
            value = Fraction(1, 10) + Fraction(9 * i, 10 * (warmup - 1))
        elif i <= 2000:
            value = Fraction(1)
        else:
            value = Fraction(1) - Fraction(9 * (i - 2000), 9990)
        expected.append(float(value))
    actual = [new.multiplier(spec, i) for i in range(3000)]
    assert actual == pytest.approx(expected, abs=2e-15, rel=0)
    assert actual[0] == .1 and actual[warmup - 1] == 1 and actual[-1] == .1
    assert all(.1 <= value <= 1 for value in actual)
    assert sum(actual) == pytest.approx(2550 - .45 * warmup, abs=1e-9)


@pytest.mark.parametrize('kind', [None, 'constant', 'cosine', 'linear_cooldown'])
def test_historical_behavior_bitwise_unchanged(kind):
    c = SimpleNamespace(max_iters=3000)
    if kind:
        c.lr_schedule = {'kind': kind}
        if kind != 'constant':
            c.lr_schedule.update(total_steps=3000, min_ratio=.1)
        if kind == 'linear_cooldown':
            c.lr_schedule['start_step'] = 2000
    after = new.schedule_contract(c)
    assert after == (None if kind is None else c.lr_schedule)
    assert [legacy_multiplier(kind, i) for i in range(3000)] == [
        new.multiplier(after, i) for i in range(3000)]


@pytest.mark.parametrize('change', [
    {'warmup_steps': 0}, {'warmup_steps': 1}, {'warmup_steps': 2002},
    {'warmup_steps': True}, {'warmup_steps': 100.0}, {'warmup_steps': '100'},
    {'initial_ratio': 0}, {'initial_ratio': -1}, {'initial_ratio': 1},
    {'initial_ratio': True}, {'initial_ratio': float('nan')},
    {'initial_ratio': float('inf')}, {'initial_ratio': '.1'},
    {'start_step': 98}, {'start_step': 2999}, {'start_step': True},
    {'total_steps': 2999}, {'total_steps': True}, {'min_ratio': 0},
    {'min_ratio': True}, {'min_ratio': float('nan')}, {'extra': 1},
])
def test_invalid_contract_rejected(change):
    c = config(); c.lr_schedule.update(change)
    with pytest.raises(ValueError): new.schedule_contract(c)


@pytest.mark.parametrize('field', ['warmup_steps', 'initial_ratio', 'start_step', 'total_steps'])
def test_required_fields_cannot_disappear(field):
    c = config(); del c.lr_schedule[field]
    with pytest.raises(ValueError): new.schedule_contract(c)


@pytest.mark.parametrize('step', [-1, True, 1.0, 3000])
def test_update_bounds_rejected(step):
    with pytest.raises(ValueError): new.multiplier(new.schedule_contract(config()), step)


@pytest.mark.parametrize('width', [64, 128, 256])
@pytest.mark.parametrize('restored_step', [0, 49, 99, 2000, 2998])
def test_absolute_group_rates_survive_restored_scheduled_values(width, restored_step):
    scale = (width / 64) ** -.5
    groups = [dict(lr=.005 * scale, params=[object()], width_lr_scale=scale,
                   parameter_names=['matrix']),
              dict(lr=.005, params=[object()], width_lr_scale=1.,
                   parameter_names=['vector'])]
    optimizer = SimpleNamespace(param_groups=groups)
    scheduler = new.LearningRateSchedule(optimizer, config())
    # Mimic serialized scheduled rates, with the base rates already captured.
    for base, group in zip((.005 * scale, .005), groups):
        group['lr'] = base * new.multiplier(scheduler.spec, restored_step)
    next_step = restored_step + 1
    expected = [.005 * scale * new.multiplier(scheduler.spec, next_step),
                .005 * new.multiplier(scheduler.spec, next_step)]
    scheduler.apply(next_step)
    assert [g['lr'] for g in groups] == expected
    scheduler.apply(next_step)
    assert [g['lr'] for g in groups] == expected


def test_identity_change_rejected():
    opt = SimpleNamespace(param_groups=[dict(lr=.005, params=[object()])])
    scheduler = new.LearningRateSchedule(opt, config())
    opt.param_groups[0]['params'] = [object()]
    with pytest.raises(ValueError): scheduler.apply(1)
