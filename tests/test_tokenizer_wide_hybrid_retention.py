"""Pure v2 planning tests; no checkpoint deletion, corpus access or training."""
import copy
import itertools
import json
from pathlib import Path

import pytest

from src.research import tokenizer_wide_hybrid_retention as retention
from src.research import tokenizer_wide_hybrid_rules as rules

ROOT = Path(__file__).resolve().parents[1]
PROJECTION = ROOT/'research/tokenizer-evidence/wide-hybrid-preparation-001/rank-storage-projection.json'


def rows(specs, ce=5.0):
    return [dict(s, status='completed', ce=ce) for s in specs]


def panel(divisors=(4, 4, 4)):
    lr = rows(rules.lr_specs())
    rates = rules.choose_rates(lr)
    rank = rows(rules.rank_specs(rates), ce=6.0)
    for row in rank:
        if row['divisor'] == dict(zip(rules.FAMILIES, divisors))[row['family']]:
            row['ce'] = 4.0
    records = lr+rank
    return records, rates, rules.choose_ranks(lr, rank)


def test_full_envelope_matches_independent_retained_projection():
    value = retention.storage_envelope(json.loads(PROJECTION.read_text()))
    reference = json.loads((ROOT/'research/tokenizer-evidence/checkpoint-cleanup-002/prospective-retention-projection.json').read_text())
    assert value['required_free_bytes'] == reference['required_free_bytes'] == 30_631_281_412
    assert value['all_27_final_rank_selections'] == reference['all_27_final_rank_selections']
    assert list(value['phase_checkpoint_allowance_bytes'].values()) == list(reference['phase_checkpoint_allowance_bytes'].values())
    assert value['reserve_bytes'] == 8*1024**3
    assert value['metadata_allowance_bytes'] == 256*1024**2
    assert value['executable'] is False


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'bool_rank', 'float_bytes', 'negative', 'pilot_bound', 'reserve', 'total', 'count'])
def test_corrupt_projection(mutation):
    doc = json.loads(PROJECTION.read_text())
    r = next(r for r in doc['records'] if r['width'] == 512 and r['bottleneck_divisor'] == 2)
    if mutation == 'missing': doc['records'].remove(r)
    elif mutation == 'duplicate': doc['records'].append(copy.deepcopy(r))
    elif mutation == 'bool_rank': r['bottleneck_divisor'] = True
    elif mutation == 'float_bytes': r['per_trajectory_allowance_bytes'] = float(r['per_trajectory_allowance_bytes'])
    elif mutation == 'negative': r['per_trajectory_allowance_bytes'] = -1
    elif mutation == 'pilot_bound': r['per_trajectory_allowance_bytes'] = 1
    elif mutation == 'reserve': doc['reserve_bytes'] -= 1
    elif mutation == 'total': doc['panels']['512']['required_free_bytes'] -= 1
    else: doc['panels']['512']['trajectories'] = 26
    with pytest.raises(ValueError): retention.storage_envelope(doc)


def test_pilot_candidates_require_all_three_and_never_approve_deletion():
    records = rows(retention._pilot_specs())
    value = retention.boundary_candidates('pilot', records)
    assert value['candidate_indices'] == value['audit_required_indices'] == [0, 1, 2]
    assert value['retained_indices'] == []
    assert value['deletion_authorized'] is False
    for prefix in (records[:1], records[:2], records[::-1]):
        with pytest.raises(ValueError): retention.boundary_candidates('pilot', prefix)


def test_lr_ties_keep_lower_rate_and_all_nine_require_audit():
    records, rates, _ = panel()
    value = retention.boundary_candidates('lr', records[:9], rates)
    assert value['retained_indices'] == [0, 1, 2]
    assert value['candidate_indices'] == list(range(3, 9))
    assert value['audit_required_indices'] == list(range(9))
    assert value['deletion_authorized'] is False


@pytest.mark.parametrize('divisors', list(itertools.product((8, 4, 2), repeat=3)))
def test_all_rank_selections_protect_winners_and_every_confirmation(divisors):
    records, rates, selection = panel(divisors)
    value = retention.boundary_candidates('rank', records, rates, selection)
    assert len(value['candidate_indices']) == 6
    assert len(value['retained_indices']) == 3
    assert len(value['audit_required_indices']) == 9
    for i in value['retained_indices']:
        row = records[i]
        assert row['divisor'] == dict(zip(rules.FAMILIES, divisors))[row['family']]
    assert set(value['candidate_indices']).isdisjoint(value['retained_indices'])
    # Even one fresh confirmation makes this ineligible as a cleanup boundary.
    confirm = rows(rules.confirmation_specs(selection))
    for n in (1, 9):
        with pytest.raises(ValueError):
            retention.boundary_candidates('rank', records+confirm[:n], rates, selection)
    with pytest.raises(ValueError):
        retention.boundary_candidates('confirm', records+confirm, rates, selection)


@pytest.mark.parametrize('mutation', ['missing', 'order', 'failed', 'nan', 'seed_type', 'rates', 'rank', 'early', 'unknown'])
def test_invalid_boundary_never_produces_candidates(mutation):
    records, rates, selection = panel()
    boundary = 'rank'
    if mutation == 'missing': records.pop()
    elif mutation == 'order': records[0], records[1] = records[1], records[0]
    elif mutation == 'failed': records[12]['status'] = 'failed'
    elif mutation == 'nan': records[12]['ce'] = float('nan')
    elif mutation == 'seed_type': records[12]['seed'] = 3501.0
    elif mutation == 'rates': rates['hybrid'] = .0025
    elif mutation == 'rank': selection['hybrid']['divisor'] = 8
    elif mutation == 'early': selection = None
    else: boundary = 'unknown'
    with pytest.raises(ValueError): retention.boundary_candidates(boundary, records, rates, selection)


def test_rank_tie_uses_original_larger_divisor_rule():
    lr = rows(rules.lr_specs())
    rates = rules.choose_rates(lr)
    records = lr + rows(rules.rank_specs(rates))
    selection = rules.choose_ranks(records[:9], records[9:])
    value = retention.boundary_candidates('rank', records, rates, selection)
    assert value['retained_indices'] == [9, 10, 11]
    assert all(selection[f]['divisor'] == 8 for f in rules.FAMILIES)
