"""Descriptive accepted-run cost analysis; standard library and saved data only."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

START = time.perf_counter()
OUT = Path(__file__).resolve().parent
Q = OUT.parent / 'quality001'
BINDINGS = {}


def read(path):
    data = path.read_bytes()
    BINDINGS[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


plan = read(Q / 'plan.json')
assert BINDINGS[str(Q / 'plan.json')] == 'cd0dcc19ae94e09e3d14aa071043f6c89b498e482e942bc0bfeb65bda15d8ddd'
driver = next(r for r in plan['sources'] if r['path'].endswith('/research/byte_input_quality_v1/driver.py'))
assert hashlib.sha256(Path(driver['path']).read_bytes()).hexdigest() == driver['sha256']
BINDINGS[driver['path']] = driver['sha256']
selection = read(Q / 'selection.json')
for row in selection['screens']:
    review = read(Q / 'slots' / f"{row['slot']:02}" / 'review.json')
    assert BINDINGS[str(Q / 'slots' / f"{row['slot']:02}" / 'review.json')] == row['review_sha256']
    assert review['accepted'] and review['ce'] == row['ce']
selected = {}
for variant, lr in selection['learning_rates'].items():
    winner = min((r for r in selection['screens'] if r['variant'] == variant),
                 key=lambda r: (r['ce'], r['learning_rate']))
    assert winner['learning_rate'] == lr
    selected[variant] = winner['slot']

rows = []
for slot in [selected[v] for v in ('native', 'raw_bytes', 'nonlexical_codes')] + [9, 10, 11]:
    root = Q / 'slots' / f'{slot:02}'
    acceptance = read(root / 'acceptance.json')
    assert acceptance['accepted'] and acceptance['slot'] == slot
    assert set(acceptance['evidence']) == {'STOP', 'artifacts.json', 'attempt.json', 'guard.json',
                                         'inference.json', 'result.json', 'review.json', 'validation.json'}
    for filename, digest in acceptance['evidence'].items():
        path = root / filename
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == digest
        BINDINGS[str(path)] = actual
    result, review, artifacts, guard, inference = [read(root / name) for name in
        ('result.json', 'review.json', 'artifacts.json', 'guard.json', 'inference.json')]
    assert guard['status'] == 'finished' and guard['reason'] is None
    assert result['accepted'] and result['completed_steps'] == 1200 and result['targets_seen'] == 1228800
    assert review['averaging_snapshots'] == 41 and review['validation_targets'] == 34278
    assert result['spec']['learning_rate'] == selection['learning_rates'][result['spec']['variant']]
    assert abs(result['mean_update_seconds'] * 1200 - result['total_update_seconds']) < 1e-8
    assert result['spec']['bundle'] == ('screen' if slot < 9 else 'fresh0')
    serving = {}
    for batch in (1, 8):
        timings = [t for t in inference['timings'] if t['batch'] == batch]
        assert [t['repeat'] for t in timings] == list(range(7))
        assert [t['warmup'] for t in timings] == [True, True, False, False, False, False, False]
        serving[batch] = {}
        for metric in ('full_seconds', 'prefill96_seconds', 'decode32_seconds'):
            values = [t[metric] for t in timings if not t['warmup']]
            assert all(math.isfinite(v) and v > 0 for v in values)
            serving[batch][metric] = dict(n=5, minimum=min(values), median=statistics.median(values), maximum=max(values))
    rows.append(dict(spec=result['spec'], ce=review['ce'], sampler_hashes=review['sampler_hashes'],
                     parameters=artifacts['parameter_elements'], update_seconds=result['mean_update_seconds'],
                     run_seconds=guard['elapsed_seconds'], step_end_driver_max_bytes=result['max_mps_driver_bytes'],
                     h200_bytes=result['h200']['size'], serving=serving))

ratios = {}
for bundle in ('screen', 'fresh0'):
    group = {r['spec']['variant']: r for r in rows if r['spec']['bundle'] == bundle}
    assert len(group) == 3
    assert all(r['sampler_hashes'] == group['native']['sampler_hashes'] for r in group.values())
    native, raw = group['native'], group['raw_bytes']
    ratios[bundle] = dict(raw_minus_native_ce=raw['ce']-native['ce'],
                         raw_over_native_update=raw['update_seconds']/native['update_seconds'],
                         raw_over_native_serving={b: {m: raw['serving'][b][m]['median']/native['serving'][b][m]['median']
                             for m in raw['serving'][b]} for b in (1, 8)})

output = dict(scope='Selected screens and first complete fresh bundle, kept separate; descriptive costs only.',
              rows=rows, raw_over_native=ratios, evidence_sha256=BINDINGS,
              analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              elapsed_seconds=time.perf_counter()-START,
              no_model_execution=True, no_new_scientific_gate=True, adopted=False)
with (OUT / 'result.json').open('x') as f:
    json.dump(output, f, indent=2, sort_keys=True); f.write('\n')
print(json.dumps(dict(elapsed_seconds=output['elapsed_seconds'], ratios=ratios), indent=2))
