"""Saved inference measurement audit; no checkpoint loading, dispatch or adoption.

Only arithmetic/sequence/input consistency is established. Model/checkpoint,
cache numerical and device provenance must be verified by the future wrapper.
"""
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from .protocol import digest
from .tokenizer_wide_hybrid_rules import CONTROLS, confirmation_specs, memory_gate
from .tokenizer_wide_hybrid_timing import (
    FAMILIES, SEEDS, BATCHES, WARMUPS, REPEATS, order, token_digest,
    _correctness, _measurement,
)

LIMIT = 4*1024**2
SAMPLE_KEYS = {'prefill_seconds', 'decode_seconds', 'state_bytes',
               'observed_mps_current_bytes', 'observed_mps_driver_bytes'}


def _same(actual, expected):
    # Canonical JSON distinguishes bool/int and integer/float identity aliases.
    if digest(actual) != digest(expected):
        raise ValueError('Saved inference identity or sequence mismatch')


def review_result(result, seed, selection, token_sha256):
    order(seed, 0)
    specs = [s for s in confirmation_specs(selection) if s['seed'] == seed]
    expected = dict(status='measurements_complete_pending_artifact_audit', seed=seed,
                    selected_specs=specs, token_sha256=token_sha256,
                    token_hash_encoding='uint32-little-endian', context_tokens=128,
                    prefill_tokens=32, decode_tokens=64, warmups=WARMUPS,
                    repeats=REPEATS, adopted=False)
    extras = {'fixed_train_token_ids', 'correctness', 'warmup_records', 'records', 'scope'}
    if not isinstance(result, dict) or set(result) != set(expected) | extras:
        raise ValueError('Incomplete/extra inference report fields')
    _same({k: result[k] for k in expected}, expected)
    if not isinstance(result['scope'], str) or not result['scope']:
        raise ValueError('Missing measurement scope')
    if token_digest(result['fixed_train_token_ids']) != token_sha256:
        raise ValueError('Saved timing tokens differ from independently expected hash')
    checks = result['correctness']
    expected_checks = [(b, f) for b in BATCHES for f in order(seed, 0)]
    if not isinstance(checks, list) or len(checks) != len(expected_checks):
        raise ValueError('Incomplete cache correctness panel')
    for row, (batch, family) in zip(checks, expected_checks):
        if set(row) != {'batch', 'family', 'passed', 'max_absolute_error'}:
            raise ValueError('Incomplete cache check fields')
        _same([row['batch'], row['family']], [batch, family])
        _correctness({k: row[k] for k in ('passed', 'max_absolute_error')})
    state = {}
    for field, phase, count in (('warmup_records', 'warmup', WARMUPS), ('records', 'measured', REPEATS)):
        rows = result[field]
        sequence = [(b, r, p, f) for b in BATCHES for r in range(count)
                    for p, f in enumerate(order(seed, r))]
        if not isinstance(rows, list) or len(rows) != len(sequence):
            raise ValueError('Incomplete warmup/measurement sequence')
        for row, (batch, repeat, position, family) in zip(rows, sequence):
            identity = dict(batch=batch, phase=phase, repeat=repeat, position=position, family=family)
            if set(row) != SAMPLE_KEYS | set(identity) | {'decode_tokens_per_second'}:
                raise ValueError('Incomplete timing sample fields')
            _same({k: row[k] for k in identity}, identity)
            sample = _measurement({k: row[k] for k in SAMPLE_KEYS}, batch)
            rate = row['decode_tokens_per_second']
            if (type(rate) not in (int, float) or not math.isfinite(rate)
                    or not math.isclose(rate, sample['decode_tokens_per_second'], rel_tol=1e-12, abs_tol=0)):
                raise ValueError('Saved throughput arithmetic mismatch')
            key = (family, batch)
            if key in state and state[key] != sample['state_bytes']:
                raise ValueError('Saved state allocation changed')
            state[key] = sample['state_bytes']
    # Summaries exclude warmups and are computed independently of saved aggregates.
    cells = []
    for batch in BATCHES:
        for family in FAMILIES:
            rows = [r for r in result['records'] if r['batch'] == batch and r['family'] == family]
            cells.append(dict(seed=seed, batch=batch, family=family, state_bytes=state[family, batch],
                median_prefill_seconds=median(r['prefill_seconds'] for r in rows),
                median_decode_seconds=median(r['decode_seconds'] for r in rows)))
    return dict(status='measurement_contract_reviewed_pending_provenance', seed=seed,
                token_sha256=token_sha256, cells=cells, adopted=False)


def _unique_keys(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError('Duplicate JSON key in saved inference evidence')
        result[k] = v
    return result


def review_file(path, expected_sha256, seed, selection, token_sha256):
    path = Path(path)
    with path.open('rb') as stream:
        raw = stream.read(LIMIT+1)
    if len(raw) > LIMIT or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Inference file exceeds bound or changed hash')
    result = json.loads(raw, object_pairs_hook=_unique_keys)
    review = review_result(result, seed, selection, token_sha256)
    return dict(review, result_sha256=expected_sha256)


def summarize(results, selection, token_sha256):
    if len(results) != len(SEEDS):
        raise ValueError('All three fresh inference triples are required')
    # Require canonical seed order, not a caller-selected subset or duplicates.
    reviews = [review_result(r, s, selection, token_sha256) for r, s in zip(results, SEEDS)]
    cells = [c for r in reviews for c in r['cells']]
    indexed = {(c['seed'], c['batch'], c['family']): c for c in cells}
    state = {(s, b): {f: indexed[s, b, f]['state_bytes'] for f in FAMILIES}
             for s in SEEDS for b in BATCHES}
    ratios = []
    for batch in BATCHES:
        for control in CONTROLS:
            for metric in ('prefill', 'decode'):
                key = 'median_'+metric+'_seconds'
                paired = [indexed[s, batch, 'hybrid'][key]/indexed[s, batch, control][key] for s in SEEDS]
                if not all(math.isfinite(x) and x > 0 for x in paired):
                    raise ValueError('Nonfinite paired latency ratio')
                ratios.append(dict(batch=batch, control=control, metric=metric,
                                   seed_ratios=paired, median_ratio=median(paired)))
    return dict(status='measurement_panel_reviewed_pending_provenance', cells=cells,
                paired_latency_ratios=ratios, memory_gate=memory_gate(state), adopted=False,
                aggregation='Median of three seed-paired ratios of eight-repeat median latency; '
                            'warmups excluded. No speed-win adoption gate. '
                            'Memory gate alone cannot authorize adoption.')
