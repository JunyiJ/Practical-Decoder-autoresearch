"""Draft v2 retention decisions and storage envelope; never deletes or launches.

These pure decisions are NOT evidence that reviews passed or files were removed.
A v2 controller must bind complete audits, persisted selections and exact file
hashes, hold both locks, journal deletion and verify protected evidence. Existing
v1 controllers deliberately do not consume this contract.
"""
import itertools

from . import tokenizer_wide_hybrid_rules as rules
from .tokenizer_wide_hybrid_simulation import next_action

VERSION = 'wide_hybrid_retention_v2'
RESERVE = 8 * 1024**3
METADATA = 256 * 1024**2
DIVISORS = (8, 4, 2)


def allocations(projection):
    """Validate original full-trajectory allowances without loading a model."""
    rows = [r for r in projection['records'] if r['width'] == 512]
    keys = [(r['family'], r['bottleneck_divisor']) for r in rows]
    expected = set(itertools.product(rules.FAMILIES, DIVISORS))
    if (len(keys) != 9 or set(keys) != expected
            or any(type(r['bottleneck_divisor']) is not int for r in rows)):
        raise ValueError('Incomplete or duplicate width512 allocation records')
    budgets = dict(zip(keys, (r['per_trajectory_allowance_bytes'] for r in rows)))
    if any(type(n) is not int or n <= 0 for n in budgets.values()):
        raise ValueError('Expected positive integer trajectory allowances')
    if any(budgets[f, 2] < max(budgets[f, d] for d in DIVISORS) for f in rules.FAMILIES):
        raise ValueError('Pilot D/2 must bound every rank in its family')
    original = RESERVE + 4*1024**2 + sum(
        3*budgets[f, 4] + budgets[f, 8] + budgets[f, 2] + 4*budgets[f, 2]
        for f in rules.FAMILIES)
    panel = projection['panels']['512']
    if (type(projection['reserve_bytes']) is not int or projection['reserve_bytes'] != RESERVE
            or type(panel['trajectories']) is not int or panel['trajectories'] != 27
            or type(panel['required_free_bytes']) is not int
            or panel['required_free_bytes'] != original):
        raise ValueError('Original retain-all projection does not reconcile')
    return budgets


def storage_envelope(projection):
    """Complete campaign high-water, assuming VERIFIED boundary cleanup.

    Includes original temporary/optimizer/export allowances even though only
    raw/H200 pairs remain on disk. This is intentionally conservative. No credit
    for an intended deletion may be used as an observed free-space measurement.
    """
    b = allocations(projection)
    phases = {
        'pilot': sum(b[f, 2] for f in rules.FAMILIES),
        'lr': 3*sum(b[f, 4] for f in rules.FAMILIES),
        'rank': sum(b[f, d] for f in rules.FAMILIES for d in DIVISORS),
    }
    possibilities = []
    for ds in itertools.product(DIVISORS, repeat=3):
        possibilities.append(dict(divisors=dict(zip(rules.FAMILIES, ds)),
                                  checkpoint_allowance_bytes=4*sum(
                                      b[f, d] for f, d in zip(rules.FAMILIES, ds))))
    phases['confirm'] = max(p['checkpoint_allowance_bytes'] for p in possibilities)
    return dict(version=VERSION, trajectories=27, reserve_bytes=RESERVE,
                metadata_allowance_bytes=METADATA, phase_checkpoint_allowance_bytes=phases,
                all_27_final_rank_selections=possibilities,
                required_free_bytes=RESERVE+METADATA+max(phases.values()),
                executable=False)


def _pilot_specs():
    return [dict(stage='pilot', family=f, width=512, seed=3491,
                 divisor=2, lr=.0025, steps=20) for f in rules.FAMILIES]


def boundary_candidates(boundary, records, rates=None, selection=None):
    """Return indices of candidate weight pairs and retained pairs, NOT approval.

    records is the complete ordered pilot or quality prefix, with scalar scores
    from retained result evidence. LR and rank decisions must have already been
    persisted and hash-bound by the caller. Every stage member needs full audits,
    including selected models. All non-checkpoint evidence remains protected.
    """
    if boundary == 'pilot':
        if rates is not None or selection is not None or len(records) != 3:
            raise ValueError('Pilot boundary requires exactly three pilots and no selection')
        for record, spec in zip(records, _pilot_specs()):
            rules._checked([record], [spec])
        retained = []
        stage = list(range(3))
    elif boundary in ('lr', 'rank'):
        expected_n = 9 if boundary == 'lr' else 15
        if len(records) != expected_n or rates is None:
            raise ValueError('Complete stage and persisted selection required')
        if (boundary == 'lr' and selection is not None) or (boundary == 'rank' and selection is None):
            raise ValueError('Incorrect selection boundary')
        # Recompute both scientific decisions, enforcing exact order and types.
        action = next_action(records, rates, selection)
        if action['kind'] != 'trial':
            raise ValueError('Persisted scientific decision is missing')
        anchors = [i for i, r in enumerate(records[:9]) if r['lr'] == rates[r['family']]]
        stage = list(range(9)) if boundary == 'lr' else anchors + list(range(9, 15))
        retained = anchors if boundary == 'lr' else [i for i in stage if
            records[i]['lr'] == selection[records[i]['family']]['lr'] and
            records[i]['divisor'] == selection[records[i]['family']]['divisor']]
        if len(retained) != 3 or {records[i]['family'] for i in retained} != set(rules.FAMILIES):
            raise ValueError('Exactly one retained screen per family required')
    else:
        raise ValueError('Only pilot, LR and rank boundaries may prune weights')
    return dict(version=VERSION, boundary=boundary,
                audit_required_indices=sorted(stage),
                candidate_indices=sorted(set(stage)-set(retained)),
                retained_indices=sorted(retained),
                all_non_checkpoint_evidence_protected=True,
                deletion_authorized=False)
