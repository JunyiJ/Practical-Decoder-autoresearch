"""Synthetic prospective rules only: no model, continuation, training or data."""
import json
import math
import numpy as np
import pytest
from src.research import tokenizer_wide_hybrid_rules as r


def scores(specs, ce=5.):
    return [dict(s, status='completed', ce=ce) for s in specs]


def stats(d=-.03):
    return {c: dict(seed_deltas=[d]*3, mean_delta=d,
                   bounds={str(b): dict(gate_upper=d, descriptive_95pct=[d, d]) for b in r.BLOCKS})
            for c in r.CONTROLS}


def memory(hybrid=90):
    return {(s, b): dict(hybrid=hybrid, gated_mla=100, kda=100)
            for s in r.SEEDS for b in (1, 8)}


def test_complete_staged_selection_and_fresh_order():
    lr = scores(r.lr_specs())
    assert [s['family'] for s in lr] == ['hybrid', 'gated_mla', 'kda', 'kda', 'hybrid', 'gated_mla',
                                       'gated_mla', 'kda', 'hybrid']
    assert r.choose_rates(lr) == dict.fromkeys(r.FAMILIES, .000625)
    lr[3]['ce'] = 4.
    lr[6]['ce'] = 4.
    rates = r.choose_rates(lr)
    assert rates == dict(hybrid=.000625, gated_mla=.0025, kda=.00125)
    ranks = scores(r.rank_specs(rates), 4.)
    ranks[-1]['ce'] = 3. # Hybrid selects D/2; tied controls choose D/8.
    selected = r.choose_ranks(lr, ranks)
    assert selected == {f: dict(lr=rates[f], divisor=2 if f == 'hybrid' else 8) for f in r.FAMILIES}
    confirm = r.confirmation_specs(selected)
    assert len(lr) + len(ranks) + len(confirm) + 3 == 27
    assert [s['family'] for s in confirm] == [s['family'] for s in lr]
    assert {s['seed'] for s in confirm} == set(r.SEEDS)
    assert not {s['seed'] for s in confirm} & {s['seed'] for s in lr+ranks}
    assert all(s['lr'] == selected[s['family']]['lr'] and
               s['divisor'] == selected[s['family']]['divisor'] for s in confirm)
    # Record order cannot change selection.
    assert r.choose_ranks(lr[::-1], ranks[::-1]) == selected


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'nan', 'inf', 'negative',
                                      'failed', 'seed', 'bool', 'float_seed', 'extra'])
def test_reject_corrupt_lr_panel(corruption):
    rows = scores(r.lr_specs())
    if corruption == 'missing': rows.pop()
    elif corruption == 'duplicate': rows[-1] = rows[0]
    elif corruption == 'extra': rows.append(rows[0])
    elif corruption == 'failed': rows[0]['status'] = 'failed'
    elif corruption == 'seed': rows[0]['seed'] = 3511
    elif corruption == 'float_seed': rows[0]['seed'] = 3501.
    else: rows[0]['ce'] = dict(nan=float('nan'), inf=float('inf'), negative=-1, bool=True)[corruption]
    with pytest.raises(ValueError): r.choose_rates(rows)


def test_rank_stage_rejects_wrong_lr_or_anchor_replacement():
    lr = scores(r.lr_specs()); rates = r.choose_rates(lr)
    rank = scores(r.rank_specs(rates))
    rank[0]['lr'] = .0025
    with pytest.raises(ValueError): r.choose_ranks(lr, rank)
    rank = scores(r.rank_specs(rates)); rank[0]['divisor'] = 4
    with pytest.raises(ValueError): r.choose_ranks(lr, rank)
    with pytest.raises(ValueError): r.choose_ranks(lr, rank[:-1])
    with pytest.raises(ValueError): r.rank_specs({'hybrid': .000625})
    with pytest.raises(ValueError): r.confirmation_specs(dict.fromkeys(r.FAMILIES, dict(lr=.00125, divisor=True)))


@pytest.mark.parametrize('q,expected', [(0, 0), (1, 1999), (.5, 999.5),
                                     (r.GATE_QUANTILE, 1982.3416666666667)])
def test_quantile_independent_order_statistic(q, expected):
    assert r.linear_quantile(list(range(1999, -1, -1)), q) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize('data,q', [([], .5), ([1, float('nan')], .5), ([1], -1), ([1], float('nan'))])
def test_bad_quantile(data, q):
    with pytest.raises(ValueError): r.linear_quantile(data, q)


def test_paired_bootstrap_weighted_tail_and_independent_reference():
    counts = np.array([2048]*8+[31])
    loss = np.full((3, 3, 9), 5.)*counts
    # A common shift cancels; distinct seed effects are averaged, not resampled.
    for s in range(3):
        loss[s, 0] += (np.arange(9)*.002 - .04 + s*.001)*counts
        loss[s, 2] += .02*counts
    got = r.paired_statistics(loss, counts)
    again = r.paired_statistics(loss, counts)
    assert got == again
    assert r.quality_gates(json.loads(json.dumps(got))) == r.quality_gates(got)
    for ci, c in enumerate(r.CONTROLS, 1):
        expected_d = [(loss[s, 0].sum()-loss[s, ci].sum())/counts.sum() for s in range(3)]
        assert got[c]['seed_deltas'] == pytest.approx(expected_d)
        for b in r.BLOCKS:
            f = b//2048
            chunks = [list(range(i, min(i+f, len(counts)))) for i in range(0, len(counts), f)]
            rng = np.random.default_rng(20261028)
            samples = []
            for _ in range(2000):
                chosen = rng.integers(0, len(chunks), size=len(chunks))
                ids = [i for j in chosen for i in chunks[j]]
                samples.append(sum(float((loss[s, 0, ids]-loss[s, ci, ids]).sum())/
                                   float(counts[ids].sum()) for s in range(3))/3)
            # Numpy's independent implementation cross-checks explicit interpolation.
            expected = np.quantile(samples, 1-.05/6, method='linear')
            assert got[c]['bounds'][str(b)]['gate_upper'] == pytest.approx(expected, abs=1e-12)
            assert got[c]['bounds'][str(b)]['gate_upper'] >= got[c]['bounds'][str(b)]['descriptive_95pct'][1]
    shifted = r.paired_statistics(loss + 3*counts, counts)
    for c in r.CONTROLS:
        assert shifted[c]['seed_deltas'] == pytest.approx(got[c]['seed_deltas'])


@pytest.mark.parametrize('bad', ['shape', 'nonfinite', 'negative', 'counts_float', 'short_middle', 'zero_tail'])
def test_bad_paired_inputs(bad):
    counts = np.array([2048]*4+[31]); loss = np.full((3, 3, 5), 5.)*counts
    if bad == 'shape': loss = loss[:2]
    if bad == 'nonfinite': loss[0, 0, 0] = np.inf
    if bad == 'negative': loss[0, 0, 0] = -1
    if bad == 'counts_float': counts = counts.astype(float)
    if bad == 'short_middle': counts[0] = 31
    if bad == 'zero_tail': counts[-1] = 0
    with pytest.raises(ValueError): r.paired_statistics(loss, counts)


def test_gate_boundaries_and_both_controls():
    for d, strict, neutral in [(-.021, True, True), (-.02, False, True), (.01, False, True),
                               (math.nextafter(.01, math.inf), False, False)]:
        g = r.quality_gates(stats(d))
        assert g['strict_quality'] is strict and g['neutral_quality'] is neutral
        assert g['adopted'] is False
    s = stats(); s['kda']['bounds']['8192']['gate_upper'] = .011
    assert not r.quality_gates(s)['neutral_quality']
    # Mean gain cannot conceal one failed seed.
    s = stats(); s['gated_mla']['seed_deltas'] = [-.2, -.2, .011]
    s['gated_mla']['mean_delta'] = np.mean(s['gated_mla']['seed_deltas'])
    assert not r.quality_gates(s)['neutral_quality']
    s = stats(); s['kda']['bounds']['2048']['gate_upper'] = 0.
    assert not r.quality_gates(s)['strict_quality']
    s = stats(); s['kda']['mean_delta'] = -.1
    with pytest.raises(ValueError): r.quality_gates(s)
    with pytest.raises(ValueError): r.quality_gates({'kda': stats()['kda']})


def test_memory_gate_all_cells_and_exact_threshold():
    assert r.memory_gate(memory())
    for key in memory():
        for control in r.CONTROLS:
            m = memory(); m[key][control] = 99
            assert not r.memory_gate(m)
    assert not r.memory_gate(memory(91))
    for bad in (0, -1, True, 100., float('nan')):
        m = memory(); m[(3511, 1)]['kda'] = bad
        with pytest.raises(ValueError): r.memory_gate(m)
    m = memory(); m.pop((3513, 8))
    with pytest.raises(ValueError): r.memory_gate(m)
