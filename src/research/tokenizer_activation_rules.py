"""Prospective finite activation rules; no launcher, artifact audit or adoption.

Pure scores and booleans cannot prove checkpoint integrity or completed cleanup.
Production integration must bind reviewed receipts before using these decisions.
"""
import math

import numpy as np

FAMILIES = ('silu', 'gelu_exact')
RATES = (.000625, .00125, .0025)
PILOT_SEED = 3591
SCREEN_SEED = 3601
FRESH_SEEDS = (3611, 3612, 3613)
BLOCKS = (2048, 4096, 8192)
BOOTSTRAP_SEED = 20261031
DRAWS = 2000
GATE_QUANTILE = 1 - .05 / 3  # One contrast, three predeclared block sizes.


def _spec(stage, family, seed, lr, steps=1200):
    return dict(stage=stage, family=family, seed=seed, lr=lr, width=256, steps=steps)


def pilot_specs():
    return [_spec('pilot', f, PILOT_SEED, .0025, 20) for f in FAMILIES]


def screen_specs():
    orders = (FAMILIES, FAMILIES[::-1], FAMILIES)
    return [_spec('screen', f, SCREEN_SEED, lr) for lr, order in zip(RATES, orders) for f in order]


def validate_selection(selection):
    if (type(selection) is not dict or set(selection) != set(FAMILIES)
            or any(type(v) is not float or v not in RATES for v in selection.values())):
        raise ValueError('Both family choices must be declared floating-point LRs')


def confirmation_specs(selection):
    validate_selection(selection)
    return [_spec('confirm', f, seed, selection[f])
            for seed, order in zip(FRESH_SEEDS, (FAMILIES, FAMILIES[::-1], FAMILIES))
            for f in order]


def checked(records, specs, *, quality=True):
    """Require complete unique records with exact schema, types and endpoint."""
    remaining = list(specs)
    for record in records:
        if type(record) is not dict:
            raise ValueError('Expected a score record')
        matches = [s for s in remaining if all(type(record.get(k)) is type(v)
                   and record[k] == v for k, v in s.items())]
        extra = {'status', 'h200_ce'} if quality else {'status'}
        if (len(matches) != 1 or set(record) != set(matches[0]) | extra
                or record['status'] != 'completed'):
            raise ValueError('Incomplete, duplicate, failed or unexpected record')
        if quality:
            ce = record['h200_ce']
            if type(ce) not in (int, float) or not math.isfinite(ce) or ce < 0:
                raise ValueError('Expected finite nonnegative H200 CE')
        remaining.remove(matches[0])
    if remaining:
        raise ValueError('Incomplete stage')


def choose_rates(records):
    checked(records, screen_specs())
    return {f: min((r for r in records if r['family'] == f),
                  key=lambda r: (r['h200_ce'], r['lr']))['lr'] for f in FAMILIES}


def _prefix(records, specs, *, quality):
    if len(records) > len(specs):
        raise ValueError('Extra records beyond the finite plan')
    for row, spec in zip(records, specs):
        checked([row], [spec], quality=quality)


def next_action(pilots, records, selection=None, *, pilots_released=False, losers_released=False):
    """Suggest one action from an exact accepted prefix; never dispatch it.

    Release flags represent audited boundary receipts in a future controller,
    not an authorization to delete anything or bypass the frozen resource gate.
    """
    if type(pilots_released) is not bool or type(losers_released) is not bool:
        raise ValueError('Boundary status must be boolean')
    _prefix(pilots, pilot_specs(), quality=False)
    if len(pilots) < 2:
        if records or selection is not None or pilots_released or losers_released:
            raise ValueError('Quality or cleanup before complete pilots')
        return dict(kind='pilot', spec=pilot_specs()[len(pilots)])
    if not pilots_released:
        if records or selection is not None or losers_released:
            raise ValueError('Quality before reviewed pilot boundary')
        return dict(kind='review_pilot_boundary')
    screen = screen_specs()
    _prefix(records[:6], screen, quality=True)
    if len(records) < 6:
        if selection is not None or losers_released:
            raise ValueError('Selection before complete screen')
        return dict(kind='quality', spec=screen[len(records)])
    chosen = choose_rates(records[:6])
    if selection is None:
        if len(records) != 6 or losers_released:
            raise ValueError('Fresh runs or cleanup before persisted selection')
        return dict(kind='save_selection', value=chosen)
    validate_selection(selection)
    if selection != chosen:
        raise ValueError('Persisted selection differs from frozen rule')
    if not losers_released:
        if len(records) != 6:
            raise ValueError('Fresh runs before reviewed screen boundary')
        return dict(kind='review_screen_boundary')
    queue = screen + confirmation_specs(selection)
    _prefix(records, queue, quality=True)
    return (dict(kind='quality', spec=queue[len(records)]) if len(records) < len(queue)
            else dict(kind='review_final', adopted=False))


def screening_retention(records, selection):
    """Index-only proposal. All paths/hashes/reviews/locks remain mandatory."""
    _prefix(records, screen_specs(), quality=True)
    chosen = choose_rates(records)
    validate_selection(selection)
    if selection != chosen:
        raise ValueError('Retention cannot change scientific selection')
    selected = [i for i, row in enumerate(records) if row['lr'] == selection[row['family']]]
    return dict(selected_indices=selected,
                losing_indices=[i for i in range(6) if i not in selected],
                fresh_pairs_always_protected=True, deletion_authorized=False)


def _quantile(values, q):
    a = np.sort(np.asarray(values, dtype=np.float64))
    pos = (len(a) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    return float(a[low] + (a[high] - a[low]) * (pos - low))


def paired_statistics(loss_sums, counts):
    """[fresh seed3611/12/13, SiLU/GELU, aligned 2048-token block sums].

    Caller must verify target identity/order/coverage and chosen-checkpoint
    provenance. Text resampling conditions on the trained seeds and selection.
    This function never reads a model, corpus, holdout or experimental result.
    """
    raw = np.asarray(loss_sums)
    n = np.asarray(counts)
    if (n.ndim != 1 or len(n) < 4 or n.dtype.kind not in 'iu'
            or np.any(n <= 0) or np.any(n > 2048) or np.any(n[:-1] != 2048)):
        raise ValueError('Aligned 2048-token counts with only a short final tail required')
    if (raw.dtype.kind not in 'fiu' or raw.shape != (3, 2, len(n))
            or not np.isfinite(raw).all() or np.any(raw < 0)):
        raise ValueError('All three finite nonnegative fresh pairs required')
    loss = raw.astype(np.float64)
    counts_float = n.astype(np.float64)
    delta = loss[:, 1] - loss[:, 0]
    seed_deltas = delta.sum(axis=1) / counts_float.sum()
    bounds = {}
    for size in BLOCKS:
        starts = np.arange(0, len(n), size // 2048)
        grouped_counts = np.add.reduceat(counts_float, starts)
        grouped_delta = np.add.reduceat(delta.mean(axis=0), starts)
        rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
        indices = rng.integers(0, len(starts), size=(DRAWS, len(starts)))
        draws = grouped_delta[indices].sum(axis=1) / grouped_counts[indices].sum(axis=1)
        bounds[str(size)] = dict(gate_upper=_quantile(draws, GATE_QUANTILE),
                                descriptive_95pct=[_quantile(draws, .025), _quantile(draws, .975)])
    return dict(seed_deltas=seed_deltas.tolist(), mean_delta=float(seed_deltas.mean()), bounds=bounds)


def quality_gate(statistics):
    """Strict arithmetic eligibility only; never an adoption decision."""
    if set(statistics) != {'seed_deltas', 'mean_delta', 'bounds'}:
        raise ValueError('Unexpected statistics schema')
    ds = np.asarray(statistics['seed_deltas'])
    mean = statistics['mean_delta']
    if (ds.shape != (3,) or ds.dtype.kind not in 'fiu' or not np.isfinite(ds).all()
            or type(mean) not in (float, int) or not math.isfinite(mean)
            or not math.isclose(mean, float(ds.mean()), rel_tol=0, abs_tol=1e-12)
            or set(statistics['bounds']) != {str(b) for b in BLOCKS}):
        raise ValueError('Missing or inconsistent seed/block panel')
    upper = []
    for row in statistics['bounds'].values():
        if set(row) != {'gate_upper', 'descriptive_95pct'}:
            raise ValueError('Invalid bound schema')
        values = [row['gate_upper'], *row['descriptive_95pct']]
        if (len(values) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or values[1] > values[2] or values[0] < values[2]):
            raise ValueError('Invalid conditional interval')
        upper.append(values[0])
    return dict(strict_quality=bool(mean < -.02 and np.all(ds < 0) and np.all(np.array(upper) < 0)),
                adopted=False)
