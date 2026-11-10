"""Apply the frozen exposure decision to all six verified final endpoints."""
import datetime
import json
from pathlib import Path
import numpy as np
from research.general_rsi_exposure_review_v1 import review
from research.general_rsi_training_v1 import sha

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / 'research/tokenizer-evidence/general-rsi-exposure-001'


def read(path):
    return json.loads(path.read_text())


def analyze():
    plan = read(P / 'plan.json')
    assert sha(P / 'plan.json') == 'f4bacb74f0b82627e1a5639890f53b5911d2d2bba05fa0c8b0276c17fde19ad6'
    reviews, times, results, scores, bindings = {}, {}, {}, {}, {}
    for slot, attempt in enumerate([2, 1, 1, 1, 2, 2]):
        d = P / f'slot-{slot:02d}'
        reviews[slot] = review(slot, attempt)
        saved = read(d / 'review.json')
        assert saved == reviews[slot]
        times[slot] = sum(read(f)['elapsed_seconds'] for f in d.glob('guard-*.json'))
        assert times[slot] < plan['slot_timeout_seconds']
        results[slot] = read(d / 'result.json')
        for endpoint in ['raw', 'h200']:
            scores[slot, endpoint] = read(d / f'{endpoint}-validation.json')
        for f in d.glob('*.json*'):
            bindings[str(f.relative_to(ROOT))] = sha(f)
    assert sum(times.values()) < plan['tranche_seconds']
    # Preserve and reverify original failures, without replaying numerical work.
    for number in [1, 2]:
        rec = ROOT / f'research/tokenizer-evidence/general-rsi-exposure-recovery-{number:03d}'
        prep = read(rec / 'preparation.json')
        replay = read(rec / 'replay-verification.json')
        assert replay['passed'] and replay['replay_updates'] == prep['replayed_updates']
        for name, row in prep['original_files'].items():
            f = rec / 'attempt-01-original' / name
            assert f.stat().st_size == row['size'] and sha(f) == row['sha256']
        bindings[str((rec / 'replay-verification.json').relative_to(ROOT))] = sha(rec / 'replay-verification.json')
    inventory = [(x['document'], x['source'], x['count']) for x in scores[0, 'h200']['documents']]
    assert len(inventory) == 108
    for v in scores.values():
        assert [(x['document'], x['source'], x['count']) for x in v['documents']] == inventory
    pairs, deltas = [], []
    for short, long in [(0, 1), (2, 3), (4, 5)]:
        pairs.append(dict(short_slot=short, long_slot=long,
                          h200_delta=scores[long, 'h200']['ce']-scores[short, 'h200']['ce'],
                          raw_delta=scores[long, 'raw']['ce']-scores[short, 'raw']['ce'],
                          guard_time_ratio=times[long]/times[short]))
        deltas.append([b['loss_sum']-a['loss_sum'] for a, b in
                       zip(scores[short, 'h200']['documents'], scores[long, 'h200']['documents'])])
    counts = np.array([x[2] for x in inventory])
    means = np.mean(deltas, axis=0)
    mean_delta = float(means.sum()/counts.sum())
    assert abs(mean_delta-np.mean([x['h200_delta'] for x in pairs])) < 1e-12
    rng = np.random.Generator(np.random.PCG64(plan['bootstrap_seed']))
    ids = rng.integers(0, 108, size=(plan['bootstrap_resamples'], 108))
    draws = means[ids].sum(axis=1)/counts[ids].sum(axis=1)
    upper = float(np.quantile(draws, plan['bootstrap_upper_quantile'], method='linear'))
    sources = {source:float(np.mean([
        results[b]['endpoints'][1]['sources'][source]['ce']-results[a]['endpoints'][1]['sources'][source]['ce']
        for a, b in [(0, 1), (2, 3), (4, 5)]])) for source in ['fineweb_edu', 'wikipedia']}
    gates = dict(all_three_negative=all(x['h200_delta'] < 0 for x in pairs),
                 mean_threshold=mean_delta <= plan['paired_delta_mean_threshold'],
                 bootstrap_upper_negative=upper < 0,
                 source_guard=max(sources.values()) <= plan['source_regression_limit'])
    return dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                plan_sha256=sha(P/'plan.json'), analysis_source_sha256=sha(__file__),
                all_six_executions_verified=True, all_three_batch_prefixes_verified=True,
                original_failure_archives_verified=True, pairs=pairs, mean_h200_delta=mean_delta,
                mean_h200_ce={str(h):float(np.mean([scores[i, 'h200']['ce'] for i in slots]))
                              for h, slots in [(10240, [0, 2, 4]), (20480, [1, 3, 5])]},
                bootstrap=dict(seed=plan['bootstrap_seed'], resamples=len(draws),
                               upper_quantile=plan['bootstrap_upper_quantile'], upper=upper,
                               method=plan['bootstrap_method'],
                               scope='Conditional on these108documents and three paired seeds; not population-seed confidence'),
                source_mean_deltas=sources, gates=gates,
                selected_horizon=20480 if all(gates.values()) else 10240,
                guard_seconds_by_slot_including_failures=times,
                tranche_guard_seconds=sum(times.values()),
                quality_scope='Own-horizon training recipe, including cooldown; not equal-time superiority or generation coherence.',
                test_opened=False, evidence_sha256=bindings)


if __name__ == '__main__':
    result = analyze()
    with (P/'decision.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'evidence_sha256'}, indent=2))
