"""One NEW deterministic literal TOP target proof; no model/data/backend calls."""
import argparse
import copy
from decimal import Decimal, localcontext
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
R = Path(__file__).with_suffix('')
SR = R.parent / 'top-target-source-review001'
CASES = R.parent / 'top-target-component-source-plan001.json'
CAP = 2 * 1024**2
SERIAL = 0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(name, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    retained = CASES.stat().st_size + sum(p.stat().st_size for folder in (R, SR)
                  for p in folder.iterdir() if p.is_file())
    assert retained + len(raw) + 65536 <= CAP, 'Whole metadata cap exceeded'
    with (R / name).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def typed(value):
    if type(value) in (tuple, list):
        return dict(type=type(value).__name__, items=[typed(x) for x in value])
    if type(value) is dict:
        return dict(type='dict', items={k: typed(v) for k, v in value.items()})
    return dict(type=type(value).__name__, value=value)


def record(label, passed, **details):
    global SERIAL
    SERIAL += 1
    publish(f'{SERIAL:03d}-{label}.json', dict(id=label, passed=passed, **details))
    assert passed, label  # Original report is durable BEFORE its fixed assertion.


def decimal_oracle(literal_row):
    # Receives ONLY presaved literal expected offsets, never component output.
    with localcontext() as ctx:
        ctx.prec = 80
        masses = [(Decimal(1) - Decimal(offset)).exp() for _, offset in literal_row]
        denominator = sum(masses, Decimal(0))
        return tuple((token, float(mass / denominator))
                     for (token, _), mass in zip(literal_row, masses))


def negative_payloads():
    return {
        'tr_context_bool': ('rows', ([1, 2, 3],), {'context': True, 'window': 2, 'vocabulary': 31}),
        'tr_window_bool': ('rows', ([1, 2, 3],), {'context': 1, 'window': True, 'vocabulary': 31}),
        'tr_vocabulary_bool': ('rows', ([1, 2, 3],), {'context': 1, 'window': 2, 'vocabulary': True}),
        'tr_sequence_string': ('rows', ('123',), {'context': 1, 'window': 2, 'vocabulary': 31}),
        'tr_sequence_short': ('rows', ([1, 2],), {'context': 1, 'window': 2, 'vocabulary': 31}),
        'tr_sequence_long': ('rows', ([1, 2, 3, 4],), {'context': 1, 'window': 2, 'vocabulary': 31}),
        'tr_id_bool': ('rows', ([1, True, 3],), {'context': 1, 'window': 2, 'vocabulary': 31}),
        'tr_id_negative': ('rows', ([1, -1, 3],), {'context': 1, 'window': 2, 'vocabulary': 31}),
        'tr_id_OOV': ('rows', ([1, 31, 3],), {'context': 1, 'window': 2, 'vocabulary': 31}),
        'tr_context_zero': ('rows', ([1, 2, 3],), {'context': 0, 'window': 2, 'vocabulary': 31}),
        'tr_window_129': ('rows', ([1, 2, 3],), {'context': 1, 'window': 129, 'vocabulary': 31}),
        'tr_vocabulary_50258': ('rows', ([1, 2, 3],), {'context': 1, 'window': 2, 'vocabulary': 50258}),
        'tm_row_list': ('probabilities', ([(2, 1), (3, 2)],), {}),
        'tm_row_empty': ('probabilities', ((),), {}),
        'tm_pair_list': ('probabilities', (([2, 1], (3, 2)),), {}),
        'tm_pair_length3': ('probabilities', (((2, 1, 7), (3, 2)),), {}),
        'tm_id_bool': ('probabilities', (((True, 1), (3, 2)),), {}),
        'tm_id_negative': ('probabilities', (((-1, 1), (3, 2)),), {}),
        'tm_id_50257': ('probabilities', (((50257, 1), (3, 2)),), {}),
        'tm_duplicate_id': ('probabilities', (((2, 1), (2, 2)),), {}),
        'tm_offset_bool': ('probabilities', (((2, True), (3, 2)),), {}),
        'tm_offset_zero': ('probabilities', (((2, 0), (3, 2)),), {}),
        'tm_offset_nonincreasing': ('probabilities', (((2, 1), (3, 1)),), {}),
        'tm_missing_offset1': ('probabilities', (((2, 2), (3, 3)),), {}),
        'tm_offset_129': ('probabilities', (((2, 1), (3, 129)),), {}),
    }


def positive(C, case):
    sequence = copy.deepcopy(case['sequence'])
    before = copy.deepcopy(sequence)
    expected = tuple(tuple(tuple(pair) for pair in row) for row in case['expected_rows'])
    actual = C.rows(sequence, context=case['context'], window=case['window'],
                    vocabulary=case['vocabulary'])
    original_actual = copy.deepcopy(actual)
    masses = tuple(C.probabilities(row) for row in actual)
    oracle = tuple(decimal_oracle(row) for row in case['expected_rows'])
    integer_exact = actual == expected
    mass_layout = (len(masses) == len(oracle) and all(len(a) == len(b) and
                    tuple(x[0] for x in a) == tuple(x[0] for x in b)
                    for a, b in zip(masses, oracle)))
    errors = [abs(a[1] - b[1]) for ar, br in zip(masses, oracle) for a, b in zip(ar, br)]
    mass_gate = mass_layout and all(math.isfinite(a[1]) and a[1] > 0 and
                abs(a[1] - b[1]) <= 1e-14 + 4e-14 * abs(b[1])
                for ar, br in zip(masses, oracle) for a, b in zip(ar, br))
    sums = [abs(math.fsum(x[1] for x in row) - 1.) for row in masses]
    sum_gate = len(sums) == case['context'] and all(x <= 1e-14 for x in sums)
    onehot = all(len(ar) != 1 or ar[0][1] == 1. for ar in masses)
    immutable = (type(actual) is tuple and all(type(row) is tuple and
                 all(type(pair) is tuple and all(type(x) is int for x in pair) for pair in row)
                 for row in actual) and type(masses) is tuple and
                 all(type(row) is tuple and all(type(pair) is tuple for pair in row) for row in masses))
    preserved = sequence == before and case['sequence'] == before
    sequence[0] = -1  # Output remains independent after mutation of caller-owned input.
    independent = actual == original_actual
    far = True
    far_relative_error = None
    if case['id'] == 'tp12_far_nonzero_mass':
        a, b = masses[0][-1][1], oracle[0][-1][1]
        far_relative_error = abs(a - b) / b
        far = a > 0 and far_relative_error <= 4e-14
    record(case['id'], integer_exact and mass_gate and sum_gate and onehot and immutable
           and preserved and independent and far, actual_rows=actual, expected_rows=expected,
           actual_masses=masses, Decimal80_expected_masses=oracle, integer_exact=integer_exact,
           all_elementwise_mass_predicates=mass_gate, all_mass_sum_predicates=sum_gate,
           single_support_exact_onehot=onehot, input_preserved=preserved,
           immutable_outputs=immutable, output_independent_after_input_mutation=independent,
           far_relative_predicate=far, far_relative_error=far_relative_error,
           max_absolute_mass_error=max(errors, default=0.), max_mass_sum_error=max(sums, default=0.))


def negative(C, label, payload):
    api, args, kwargs = copy.deepcopy(payload)
    before = typed((args, kwargs))
    rejected, error = False, None
    try:
        getattr(C, api)(*args, **kwargs)
    except ValueError as exc:
        rejected, error = True, repr(exc)
    preserved = typed((args, kwargs)) == before
    record(label, rejected and preserved, api=api, typed_payload=before,
           ValueError_rejected=rejected, caller_input_preserved=preserved, error=error)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan-sha', required=True)
    arg = ap.parse_args()
    assert sha(R / 'plan.json') == arg.plan_sha
    assert {p.name for p in R.iterdir()} == {'plan.json', 'source-review.json', 'admission.json'}
    plan = json.loads((R / 'plan.json').read_text())
    admission = json.loads((R / 'admission.json').read_text())
    assert admission['plan_sha256'] == arg.plan_sha and admission['scope'] == 'literal_target_math_only'
    lock = (ROOT / 'research_state/tokenizer-worker-lock/.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    publish('attempt.json', dict(pid=os.getpid(), plan_sha256=arg.plan_sha, no_retry=True))
    try:
        with CpuPrerequisiteGuard(R / 'guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
            assert shutil.disk_usage(ROOT).free >= plan['complete_target_bytes']
            assert not any(k.split('.')[0] in {'torch', 'numpy', 'tiktoken', 'triton', 'random'} for k in sys.modules)
            for path, wanted in plan['sources'].items():
                assert sha(ROOT / path) == wanted, path
            assert sha(CASES) == plan['case_plan_sha256']
            cases = json.loads(CASES.read_text())
            labels = [c['id'] for c in cases['positive_cases'] + cases['negative_cases']]
            payloads = negative_payloads()
            assert len(labels) == len(set(labels)) == 37 and list(payloads) == labels[12:]
            from research import top_proximity_targets_v1 as C
            for case in cases['positive_cases']:
                positive(C, case)
            for label, payload in payloads.items():
                negative(C, label, payload)
            assert SERIAL == 37
            for path, wanted in plan['sources'].items():
                assert sha(ROOT / path) == wanted, path
            assert sha(CASES) == plan['case_plan_sha256']
            assert not any(k.split('.')[0] in {'torch', 'numpy', 'tiktoken', 'triton', 'random'} for k in sys.modules)
            publish('result.json', dict(status='complete_original_literal_TOP_target_math',
                    positive_cases=12, negative_transactions=25, reports=SERIAL,
                    all_integer_entries_and_elementwise_mass_predicates=True,
                    Decimal_precision=80, atol=1e-14, rtol=4e-14, sum_atol=1e-14,
                    far_positive_relative_rtol=4e-14, source_pins_unchanged=True,
                    no_Torch_corpus_model_sampler_RNG_or_backend_calls=True,
                    scope='literal_target_math_only_NOT_model_quality_or_upstream_reproduction'))
        publish('STOP', dict(status='complete_original_literal_TOP_target_math', no_retry=True, no_replay=True))
    except BaseException as error:
        if not (R / 'STOP').exists():
            publish('STOP', dict(status='closed_failed', error=repr(error), no_retry=True, no_replay=True))
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
