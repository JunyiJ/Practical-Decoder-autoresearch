"""Read-only final panel review, outside frozen quality-v1 training sources.

Reuses accepted per-run numerical reviews; independently rebuilds selection and
conditional-text statistics from saved target losses. No Torch/model imports,
training, inference, new scientific seeds, or incumbent modification.
Run only after quality001/STOP reports successful complete-panel termination.
"""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/tokenizer-evidence/byte-composition-boundary-001/quality001'
PLAN_SHA = 'cd0dcc19ae94e09e3d14aa071043f6c89b498e482e942bc0bfeb65bda15d8ddd'
ARMS = ('native', 'raw_bytes', 'nonlexical_codes')
EVIDENCE = {'attempt.json', 'result.json', 'artifacts.json', 'validation.json',
            'inference.json', 'review.json', 'guard.json', 'STOP'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def verify_file(record):
    path = Path(record['path'])
    assert path.is_file() and not path.is_symlink()
    assert path.stat().st_size == record['size'] and sha(path) == record['sha256']


def contrast(pairs, control, seed, threshold):
    """Scalar sums from target losses, independent of stored block aggregates."""
    import numpy as np  # Only used for the protocol's fixed PCG64 draws.
    deltas = [(math.fsum(p['raw_bytes']) - math.fsum(p[control])) / 34278
              for p in pairs]
    bounds = []
    for width in (2048, 4096, 8192):
        starts = list(range(0, 34278, width))
        counts = [min(width, 34278 - start) for start in starts]
        differences = [[math.fsum(p['raw_bytes'][i:i+width]) -
                        math.fsum(p[control][i:i+width]) for i in starts]
                       for p in pairs]
        draws = np.random.Generator(np.random.PCG64(seed)).integers(
            0, len(counts), (2000, len(counts)))
        values = []
        for draw in draws:
            count = sum(counts[k] for k in draw)
            values.append(math.fsum(math.fsum(row[k] for k in draw) / count
                                    for row in differences) / 3)
        values.sort()
        position = 1999 * (1 - .05 / 6)
        lo = math.floor(position)
        upper = values[lo] + (position - lo) * (values[lo+1] - values[lo])
        bounds.append(dict(block_tokens=width, upper=upper))
    mean = math.fsum(deltas) / 3
    gates = dict(mean_below_threshold=mean < threshold,
                 every_seed_negative=all(d < 0 for d in deltas),
                 all_bounds_negative=all(b['upper'] < 0 for b in bounds))
    return dict(deltas=deltas, mean_delta=mean, threshold=threshold,
                bounds=bounds, gates=gates, passes=all(gates.values()))


def run():
    assert sha(BASE / 'plan.json') == PLAN_SHA
    plan = read(BASE / 'plan.json')
    assert read(BASE / 'STOP')['status'] == 'complete_original_quality_panel_pending_adoption_review'
    statistics_guard = read(BASE / 'statistics-guard.json')
    assert statistics_guard['status'] == 'finished' and statistics_guard['reason'] is None
    final = read(BASE / 'final.json')
    assert final['accepted_complete_panel'] and final['trajectories'] == 18 and final['updates'] == 21600
    assert final['design'] == plan['design'] and final['bootstrap_seed'] == plan['bootstrap_seed']
    assert final['adopted'] is False and final['independent_final_acceptance_required'] is True
    for source in plan['sources']:
        verify_file(source)
    rows, losses, receipts = [], [], []
    for index, spec in enumerate(plan['slots']):
        out = BASE / 'slots' / f'{index:02d}'
        receipt = read(out / 'acceptance.json')
        assert receipt['accepted'] and receipt['slot'] == index and receipt['no_adoption']
        assert set(receipt['evidence']) == EVIDENCE
        for name, digest in receipt['evidence'].items():
            assert sha(out / name) == digest
        assert read(out / 'STOP')['status'] == 'complete_original_scientific_slot'
        guard = read(out / 'guard.json')
        assert guard['status'] == 'finished' and guard['reason'] is None
        row = read(out / 'review.json')
        assert row['accepted'] and row['completed_steps'] == 1200 and row['targets_seen'] == 1228800
        assert row['averaging_snapshots'] == 41 and row['validation_targets'] == 34278
        assert not row['quality_selected']
        for key in ('slot', 'stage', 'bundle', 'variant'):
            assert row[key] == spec[key]
        attempt = read(out / 'attempt.json')
        assert attempt['plan_sha256'] == PLAN_SHA
        assert attempt['spec'] == {key: row[key] for key in spec}
        for key in ('parent', 'h200'):
            verify_file(row[key])
        saved = read(out / 'validation.json')
        report, loss = saved['report'], saved['per_target_losses']
        assert report['target'] == plan['target'] and report['binding'] == row['h200']
        assert len(loss) == 34278 and all(type(x) is float and math.isfinite(x) and x >= 0 for x in loss)
        assert abs(math.fsum(loss) - report['total_loss']) < 1e-8
        assert row['ce'] == report['ce'] == report['total_loss'] / 34278
        rows.append(row)
        losses.append(loss)
        receipts.append(dict(slot=index, sha256=sha(out / 'acceptance.json')))
    assert len(rows) == 18 and final['all_slot_acceptances'] == receipts
    choices = {}
    for arm in ARMS:
        candidates = [r for r in rows[:9] if r['variant'] == arm]
        assert sorted(r['learning_rate'] for r in candidates) == [.000625, .00125, .0025]
        choices[arm] = min(candidates, key=lambda r: (r['ce'], r['learning_rate']))['learning_rate']
    selection = dict(learning_rates=choices, screens=[dict(
        slot=r['slot'], variant=r['variant'], learning_rate=r['learning_rate'], ce=r['ce'],
        review_sha256=sha(BASE / 'slots' / f"{r['slot']:02d}" / 'review.json')) for r in rows[:9]])
    assert selection == read(BASE / 'selection.json') == final['selection']
    for start, length in ((0, 9), (9, 3), (12, 3), (15, 3)):
        assert all(r['sampler_hashes'] == rows[start]['sampler_hashes'] for r in rows[start:start+length])
    for row, spec in zip(rows, plan['slots']):
        assert row['learning_rate'] == (spec['learning_rate'] if row['stage'] == 'screen' else choices[row['variant']])
    pairs = [{rows[i]['variant']: losses[i] for i in range(start, start+3)} for start in (9, 12, 15)]
    independent = {}
    for control, threshold in (('native', -.02), ('nonlexical_codes', 0.)):
        rebuilt = contrast(pairs, control, plan['bootstrap_seed'], threshold)
        stored = final['contrasts'][control]
        for a, b in zip(rebuilt['deltas'], stored['deltas'], strict=True):
            assert abs(a - b) < 1e-12
        assert abs(rebuilt['mean_delta'] - stored['mean_delta']) < 1e-12
        for a, b in zip(rebuilt['bounds'], stored['bounds'], strict=True):
            assert a['block_tokens'] == b['block_tokens'] and abs(a['upper'] - b['upper']) < 1e-12
        for key in ('threshold', 'gates', 'passes'):
            assert rebuilt[key] == stored[key]
        independent[control] = rebuilt
    assert final['primary_passes'] == independent['native']['passes']
    assert final['lexical_content_supported'] == independent['nonlexical_codes']['passes']
    return dict(accepted=True, plan_sha256=PLAN_SHA, final_sha256=sha(BASE / 'final.json'),
                reviewer_sha256=sha(Path(__file__)), selection=choices, contrasts=independent,
                practical_eligible=independent['native']['passes'],
                lexical_content_supported=independent['nonlexical_codes']['passes'], adopted=False,
                scope='Reuses immutable accepted per-run numerical reviews; independently checks full-panel selection, saved losses, statistical decisions, source and retained-weight hashes. No model execution or incumbent edit.',
                interpretation=plan['design']['uncertainty'])


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Provide a new output JSON path outside the frozen panel.'
    output = Path(sys.argv[1]).resolve()
    assert not output.is_relative_to(BASE) and not output.exists()
    with (ROOT / 'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
        with CpuPrerequisiteGuard(output.with_suffix('.guard.json'), ROOT,
                                  timeout_seconds=300, rss_limit_bytes=1024**3,
                                  reserve_bytes=8*1024**3):
            result = run()
        with output.open('x') as stream:
            json.dump(result, stream, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
