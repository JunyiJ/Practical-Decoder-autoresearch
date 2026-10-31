import copy

import numpy as np
from omegaconf import OmegaConf
import pytest

from src.research import tokenizer_activation_rules as r
from src.research.tokenizer_activation_config import config
from src.models.tokenizer_architecture import architecture_contract
from src.train.lr_schedule import schedule_contract
from src.train.weight_averaging import averaging_contract


def scored(spec, ce=5.):
    return dict(spec, status='completed', h200_ce=ce)


def pilots():
    return [dict(s, status='completed') for s in r.pilot_specs()]


def screen():
    return [scored(s, 5. + abs(s['lr'] - (.00125 if s['family'] == 'silu' else .000625)))
            for s in r.screen_specs()]


def test_complete_serial_prefix_and_explicit_boundaries():
    p, rows, selection = [], [], None
    for spec in r.pilot_specs():
        assert r.next_action(p, rows) == dict(kind='pilot', spec=spec)
        p.append(dict(spec, status='completed'))
    assert r.next_action(p, rows)['kind'] == 'review_pilot_boundary'
    for row in screen():
        action = r.next_action(p, rows, pilots_released=True)
        assert action == dict(kind='quality', spec={k: row[k] for k in r.screen_specs()[0]})
        rows.append(row)
    selected = r.next_action(p, rows, pilots_released=True)
    assert selected == dict(kind='save_selection', value={'silu': .00125, 'gelu_exact': .000625})
    selection = selected['value']
    assert r.next_action(p, rows, selection, pilots_released=True)['kind'] == 'review_screen_boundary'
    for spec in r.confirmation_specs(selection):
        assert r.next_action(p, rows, selection, pilots_released=True, losers_released=True) == dict(kind='quality', spec=spec)
        rows.append(scored(spec))
    assert len(p) == 2 and len(rows) == 12
    assert r.next_action(p, rows, selection, pilots_released=True, losers_released=True) == dict(kind='review_final', adopted=False)
    with pytest.raises(ValueError):
        r.next_action(p, rows + [rows[-1]], selection, pilots_released=True, losers_released=True)


def test_ties_and_retention_preserve_both_selected_pairs():
    rows = [scored(s) for s in r.screen_specs()]
    selected = r.choose_rates(rows[::-1])
    assert selected == dict.fromkeys(r.FAMILIES, .000625)
    retained = r.screening_retention(rows, selected)
    assert retained == dict(selected_indices=[0, 1], losing_indices=[2, 3, 4, 5],
                            fresh_pairs_always_protected=True, deletion_authorized=False)
    for family in r.FAMILIES:
        wrong = dict(selected); wrong[family] = .0025
        with pytest.raises(ValueError):
            r.screening_retention(rows, wrong)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'failed', 'nan', 'bool', 'raw', 'seed_float', 'extra'])
def test_bad_scores_fail_closed(mutation):
    rows = screen()
    if mutation == 'missing': rows.pop()
    elif mutation == 'duplicate': rows[-1] = rows[0]
    elif mutation == 'failed': rows[0]['status'] = 'failed'
    elif mutation == 'nan': rows[0]['h200_ce'] = float('nan')
    elif mutation == 'bool': rows[0]['h200_ce'] = True
    elif mutation == 'raw': rows[0]['raw_ce'] = rows[0].pop('h200_ce')
    elif mutation == 'seed_float': rows[0]['seed'] = float(rows[0]['seed'])
    elif mutation == 'extra': rows[0]['ignored'] = 1
    with pytest.raises(ValueError):
        r.choose_rates(rows)


def test_prefix_reorder_selection_tampering_and_early_dispatch_rejected():
    p, rows = pilots(), screen()
    selected = r.choose_rates(rows)
    invalid = [
        (p[::-1], [], None, {}),
        (p[:1], rows[:1], None, {}),
        (p, rows[:1], None, {}),
        (p, rows[:1][::-1] + rows[:1], None, {'pilots_released': True}),
        (p, rows[:2][::-1], None, {'pilots_released': True}),
        (p, rows[:2], selected, {'pilots_released': True}),
        (p, rows, dict.fromkeys(r.FAMILIES, .0025), {'pilots_released': True}),
        (p, rows + [scored(r.confirmation_specs(selected)[0])], selected, {'pilots_released': True}),
        (p, rows, None, {'pilots_released': True, 'losers_released': True}),
        (p, [], None, {'pilots_released': 1}),
    ]
    for a, b, c, flags in invalid:
        with pytest.raises(ValueError):
            r.next_action(a, b, c, **flags)


def test_exact_requests_and_pilot_exclusion():
    selection = dict.fromkeys(r.FAMILIES, .00125)
    specs = r.pilot_specs() + r.screen_specs() + r.confirmation_specs(selection)
    assert len(specs) == 14
    assert set(r.FRESH_SEEDS).isdisjoint({r.PILOT_SEED, r.SCREEN_SEED})
    assert [s['family'] for s in specs[-6:]] == ['silu', 'gelu_exact', 'gelu_exact', 'silu', 'silu', 'gelu_exact']
    for spec in specs:
        cfg = OmegaConf.create(config('/tmp/activation-config-only', spec))
        assert architecture_contract(cfg.model)['gate_kind'] == spec['family']
        assert cfg.training.device == 'mps' and cfg.training.strict_device is True
        assert cfg.training.resume_from is None and cfg.training.checkpoint_every == 0
        assert cfg.training.keep_best_checkpoint is False
        assert cfg.training.seed == spec['seed'] and cfg.training.max_iters == spec['steps']
        assert schedule_contract(cfg.training)['start_step'] == (13 if spec['stage'] == 'pilot' else 800)
        avg = averaging_contract(cfg.training)
        assert avg['half_lives'] == [200]
        assert avg['interval'] == (1 if spec['stage'] == 'pilot' else 10)
    for patch in ({'seed': True}, {'steps': 600}, {'lr': .005}, {'width': 64}, {'extra': 1}):
        with pytest.raises(ValueError): config('/tmp/activation-config-only', dict(specs[-1], **patch))
    with pytest.raises(ValueError): r.choose_rates([scored(s) for s in r.pilot_specs()])


def loss_panel():
    n = np.array([2048] * 8 + [23])
    base = np.stack([np.stack([n * (5 + i*.01), n * (5 + i*.01)]) for i in range(3)])
    delta = np.array([-.1, .04, -.03, .02, -.08, -.01, -.06, .01, .5])
    for i in range(3): base[i, 1] += (delta - i*.002) * n
    return base, n


def test_weighted_tail_statistics_against_independent_loop_oracle():
    loss, n = loss_panel()
    actual = r.paired_statistics(loss, n)
    deltas = [(loss[s, 1].sum() - loss[s, 0].sum()) / sum(n) for s in range(3)]
    np.testing.assert_allclose(actual['seed_deltas'], deltas, rtol=0, atol=1e-14)
    assert actual['mean_delta'] == pytest.approx(sum(deltas)/3, abs=1e-14)
    for size in r.BLOCKS:
        groups = [list(range(i, min(i+size//2048, len(n)))) for i in range(0, len(n), size//2048)]
        counts = np.array([sum(n[i] for i in group) for group in groups])
        totals = np.array([sum(loss[s, 1, i]-loss[s, 0, i] for s in range(3) for i in group)/3 for group in groups])
        rng = np.random.Generator(np.random.PCG64(20261031))
        draws = []
        for _ in range(2000):
            selected = rng.integers(0, len(groups), size=len(groups))
            draws.append(sum(totals[i] for i in selected)/sum(counts[i] for i in selected))
        expected = np.quantile(draws, [1-.05/3, .025, .975], method='linear')
        row = actual['bounds'][str(size)]
        np.testing.assert_allclose([row['gate_upper'], *row['descriptive_95pct']], expected, rtol=0, atol=1e-14)
    assert r.paired_statistics(loss, n) == actual


def test_strict_only_quality_thresholds_and_finite_panel():
    n = np.array([2048, 2048, 2048, 17])
    loss = np.tile(np.stack([n*5., n*4.97]), (3, 1, 1))
    stats = r.paired_statistics(loss, n)
    assert r.quality_gate(stats) == dict(strict_quality=True, adopted=False)
    for deltas in ([-.02]*3, [-.04, -.04, 0.], [.001]*3):
        changed = copy.deepcopy(stats)
        changed.update(seed_deltas=deltas, mean_delta=sum(deltas)/3)
        assert r.quality_gate(changed)['strict_quality'] is False
    changed = copy.deepcopy(stats); changed['bounds']['4096']['gate_upper'] = 0.
    assert r.quality_gate(changed)['strict_quality'] is False
    for changed in (dict(stats, mean_delta=0), dict(stats, seed_deltas=[-.03, -.03]),
                    dict(stats, mean_delta=True), dict(stats, bounds={})):
        with pytest.raises(ValueError): r.quality_gate(changed)


@pytest.mark.parametrize('case', ['count_float', 'count_bool', 'interior_tail', 'negative', 'nan', 'missing_seed', 'missing_arm', 'loss_bool'])
def test_invalid_statistics_fail_closed(case):
    loss, n = loss_panel()
    if case == 'count_float': n = n.astype(float)
    elif case == 'count_bool': n = np.ones(len(n), dtype=bool)
    elif case == 'interior_tail': n[0] = 17
    elif case == 'negative': loss[0, 0, 0] = -1
    elif case == 'nan': loss[0, 0, 0] = np.nan
    elif case == 'missing_seed': loss = loss[:2]
    elif case == 'missing_arm': loss = loss[:, :1]
    elif case == 'loss_bool': loss = np.ones_like(loss, dtype=bool)
    with pytest.raises(ValueError): r.paired_statistics(loss, n)
