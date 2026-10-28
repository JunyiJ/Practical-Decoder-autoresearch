"""Pure draft selection/statistics; no dispatcher, artifact audit or adoption.

These rules are prospective preparation, not a frozen launch contract.
"""
import math
import numpy as np

FAMILIES = ('hybrid', 'gated_mla', 'kda')
CONTROLS = FAMILIES[1:]
RATES = (.000625, .00125, .0025)
SEEDS = (3511, 3512, 3513)
BLOCKS = (2048, 4096, 8192)
GATE_QUANTILE = 1 - .05/6


def _spec(stage, family, seed, lr, divisor):
    return dict(stage=stage, family=family, seed=seed, lr=lr, divisor=divisor,
                width=512, steps=1200)


def lr_specs():
    orders = (FAMILIES, ('kda', 'hybrid', 'gated_mla'), ('gated_mla', 'kda', 'hybrid'))
    return [_spec('lr', f, 3501, lr, 4) for lr, order in zip(RATES, orders) for f in order]


def _rates(rates):
    if set(rates) != set(FAMILIES) or any(type(v) is not float or v not in RATES for v in rates.values()):
        raise ValueError('Incomplete or undeclared family LR choices')


def rank_specs(rates):
    _rates(rates)
    return [_spec('rank', f, 3501, rates[f], d) for d, order in
            ((8, FAMILIES), (2, ('kda', 'gated_mla', 'hybrid'))) for f in order]


def confirmation_specs(selection):
    if set(selection) != set(FAMILIES):
        raise ValueError('Incomplete final choices')
    _rates({f: selection[f]['lr'] for f in FAMILIES})
    for choice in selection.values():
        if (set(choice) != {'lr', 'divisor'} or type(choice['divisor']) is not int
                or choice['divisor'] not in (2, 4, 8)):
            raise ValueError('Undeclared final rank choice')
    orders = (FAMILIES, ('kda', 'hybrid', 'gated_mla'), ('gated_mla', 'kda', 'hybrid'))
    return [_spec('confirm', f, seed, **selection[f]) for seed, order in zip(SEEDS, orders) for f in order]


def _checked(records, specs):
    # Exact types prevent bools masquerading as integers and float seed aliases.
    remaining = list(specs)
    for record in records:
        matches = [s for s in remaining if all(type(record.get(k)) is type(v)
                   and record[k] == v for k, v in s.items())]
        ce = record.get('ce')
        if (len(matches) != 1 or record.get('status') != 'completed'
                or type(ce) not in (float, int) or not math.isfinite(ce) or ce < 0):
            raise ValueError('Invalid, duplicate or unexpected completed score')
        remaining.remove(matches[0])
    if remaining:
        raise ValueError('Incomplete score panel')


def choose_rates(records):
    _checked(records, lr_specs())
    return {f: min((r for r in records if r['family'] == f),
                  key=lambda r: (r['ce'], r['lr']))['lr'] for f in FAMILIES}


def choose_ranks(lr_records, rank_records):
    rates = choose_rates(lr_records)
    _checked(rank_records, rank_specs(rates))
    anchors = [r for r in lr_records if r['lr'] == rates[r['family']]]
    selected = {}
    for f in FAMILIES:
        r = min((r for r in anchors + list(rank_records) if r['family'] == f),
                key=lambda r: (r['ce'], -r['divisor']))
        selected[f] = dict(lr=r['lr'], divisor=r['divisor'])
    return selected


def linear_quantile(values, q):
    """Sorted linear interpolation at (n-1)*q, explicitly fixed for gate bounds."""
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or a.size == 0 or not np.isfinite(a).all() or not 0 <= q <= 1:
        raise ValueError('Invalid quantile data/probability')
    a = np.sort(a)
    pos = (len(a)-1)*q
    lo, hi = math.floor(pos), math.ceil(pos)
    return float(a[lo] + (a[hi]-a[lo])*(pos-lo))


def paired_statistics(loss_sums, counts):
    """Input [seed3511/12/13, family hybrid/gated_mla/kda, base block].

    Common 2048-token blocks (only the tail may be short), aligned targets and
    seeds are required. Caller must audit that identity from saved artifacts.
    Same resamples for every model/seed/contrast; weighted tail, not mean of
    block means. No model, corpus, checkpoint or held-out data is loaded here.
    """
    loss = np.asarray(loss_sums, dtype=np.float64)
    raw_counts = np.asarray(counts)
    if (raw_counts.ndim != 1 or len(raw_counts) < 4 or raw_counts.dtype.kind not in 'iu'
            or np.any(raw_counts <= 0) or np.any(raw_counts > 2048)
            or np.any(raw_counts[:-1] != 2048)):
        raise ValueError('Expected aligned positive 2048-token blocks and a tail')
    n = raw_counts.astype(np.float64)
    if loss.shape != (3, 3, len(n)) or not np.isfinite(loss).all() or np.any(loss < 0):
        raise ValueError('Expected finite nonnegative loss sums for all fresh triples')
    delta_sums = loss[:, :1, :] - loss[:, 1:, :]
    seed_delta = delta_sums.sum(axis=2)/n.sum()
    report = {c: dict(seed_deltas=seed_delta[:, i].tolist(),
                     mean_delta=float(seed_delta[:, i].mean()), bounds={})
              for i, c in enumerate(CONTROLS)}
    for size in BLOCKS:
        factor = size//2048
        starts = np.arange(0, len(n), factor)
        grouped_n = np.add.reduceat(n, starts)
        # Average seed-paired loss sums before common text resampling.
        grouped_delta = np.add.reduceat(delta_sums.mean(axis=0), starts, axis=1)
        rng = np.random.default_rng(20261028)
        indices = rng.integers(0, len(starts), size=(2000, len(starts)))
        draws = grouped_delta[:, indices].sum(axis=2)/grouped_n[indices].sum(axis=1)
        for i, c in enumerate(CONTROLS):
            report[c]['bounds'][str(size)] = dict(
                gate_upper=linear_quantile(draws[i], GATE_QUANTILE),
                descriptive_95pct=[linear_quantile(draws[i], .025), linear_quantile(draws[i], .975)])
    return report


def quality_gates(statistics):
    """Arithmetic eligibility only; deliberately never returns adopted=True."""
    if set(statistics) != set(CONTROLS):
        raise ValueError('Both homogeneous controls are required')
    per_control = {}
    for c in CONTROLS:
        r = statistics[c]
        ds = np.asarray(r['seed_deltas'], dtype=np.float64)
        if ds.shape != (3,) or not np.isfinite(ds).all() or set(r['bounds']) != {str(b) for b in BLOCKS}:
            raise ValueError('Incomplete/non-finite seed or block panel')
        mean = float(r['mean_delta'])
        upper = np.array([r['bounds'][str(b)]['gate_upper'] for b in BLOCKS], dtype=np.float64)
        if (not math.isfinite(mean) or not np.isfinite(upper).all()
                or not math.isclose(mean, float(ds.mean()), rel_tol=0, abs_tol=1e-12)):
            raise ValueError('Invalid mean arithmetic or conditional bounds')
        per_control[c] = dict(strict=bool(mean < -.02 and np.all(ds < 0) and np.all(upper < 0)),
                              neutral=bool(mean <= .01 and np.all(ds <= .01) and np.all(upper <= .01)))
    return dict(strict_quality=all(r['strict'] for r in per_control.values()),
                neutral_quality=all(r['neutral'] for r in per_control.values()),
                per_control=per_control, adopted=False)


def memory_gate(state_bytes):
    """All six fresh seed/batch cells, all three families; integer actual bytes."""
    expected = {(s, b) for s in SEEDS for b in (1, 8)}
    if set(state_bytes) != expected:
        raise ValueError('Missing/extra fresh seed-batch state measurements')
    for row in state_bytes.values():
        if set(row) != set(FAMILIES) or any(type(v) is not int or v <= 0 for v in row.values()):
            raise ValueError('All models require positive measured state bytes')
    # Exact integer arithmetic includes precisely 10% saving without float drift.
    return all(10*r['hybrid'] <= 9*r[c] for r in state_bytes.values() for c in CONTROLS)
