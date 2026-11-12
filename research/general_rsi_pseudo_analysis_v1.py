"""Apply the frozen pseudo-repetition quality gates only after all six execution reviews pass."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

from research.general_rsi_equal_time_analysis_v1 import bootstrap

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT/'research/tokenizer-evidence/general-rsi-pseudo-repetition-001'
BINDING = None
PAIRS = ((0, 1), (2, 3), (4, 5))


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decide(deltas, counts, sources, contract):
    """Paired pseudo-minus-CE document loss sums. Apply both frozen directions."""
    deltas = np.asarray(deltas, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    assert deltas.shape == (3, len(counts)) and np.isfinite(deltas).all()
    assert (counts > 0).all() and len(sources) == len(counts)
    assert set(sources) == {'fineweb_edu', 'wikipedia'}
    paired = deltas.sum(axis=1)/counts.sum()
    mean, draws = bootstrap(deltas, counts, contract['bootstrap_seed'], contract['replicates'])
    assert abs(mean-float(paired.mean())) < 1e-12
    source_deltas = {}
    for source in ('fineweb_edu', 'wikipedia'):
        mask = np.asarray([x == source for x in sources])
        source_deltas[source] = float(deltas[:, mask].sum(axis=1).mean()/counts[mask].sum())
    directions = {}
    for arm, sign in [('native_pseudo_repetition', 1), ('native_ce', -1)]:
        upper = float(np.quantile(sign*draws, contract['upper_quantile'], method='linear'))
        sd = {k:sign*v for k,v in source_deltas.items()}
        gates = dict(all_three_deltas_negative=bool((sign*paired < 0).all()),
            mean_threshold=sign*mean <= -contract['minimum_mean_gain'],
            bootstrap_upper_negative=upper < 0,
            source_guard=max(sd.values()) <= contract['source_regression_max'])
        directions[arm] = dict(candidate_minus_other=sign*mean, bootstrap_upper_99=upper,
            source_deltas=sd, gates=gates, accepted=all(gates.values()))
    winners = [k for k,v in directions.items() if v['accepted']]
    assert len(winners) <= 1
    return dict(paired_deltas=paired.tolist(), mean_delta_pseudo_minus_ce=mean,
        source_deltas=source_deltas, directions=directions,
        selected_arm=winners[0] if winners else None,
        outcome='superiority' if winners else 'inconclusive; no equivalence claim')


def analyze():
    global BINDING
    BINDING = read(PANEL/'admission.json')['plan_sha256']
    assert sha(PANEL/'plan.json') == BINDING
    plan = read(PANEL/'plan.json')
    # Require the complete panel before reading any endpoint scores.
    for slot in range(6):
        d = PANEL/f'slot-{slot:02d}'
        assert (d/'result.json').exists() and (d/'review.json').exists(), 'Six accepted endpoints required'
    for name, digest in plan['sources'].items(): assert sha(ROOT/name) == digest, name
    # The frozen source manifest includes prerequisite evidence.
    results, scores, evidence, times = {}, {}, {}, {}
    for slot in range(6):
        d = PANEL/f'slot-{slot:02d}'
        r, review = read(d/'result.json'), read(d/'review.json')
        assert review['accepted_execution'] and review['result_sha256'] == sha(d/'result.json')
        assert r['status'] == 'completed' and r['slot'] == plan['slots'][slot]
        assert r['binding'] == review['binding'] == BINDING and not r['test_opened']
        assert [e['endpoint'] for e in r['endpoints']] == ['raw', 'time_ema']
        times[slot] = sum(read(p)['elapsed_seconds'] for p in d.glob('guard-*.json'))
        assert 0 < times[slot] <= plan['slots'][slot]['timeout_seconds']
        results[slot] = r
        for p in [d/'result.json', d/'review.json']:
            evidence[str(p.relative_to(ROOT))] = sha(p)
        for endpoint in r['endpoints']:
            receipt = endpoint['weights']; path = Path(receipt['path'])
            assert path.stat().st_size == receipt['size'] and sha(path) == receipt['sha256']
            p = d/f"{endpoint['endpoint']}-validation.json"
            scores[slot, endpoint['endpoint']] = read(p)
            evidence[str(p.relative_to(ROOT))] = sha(p)
    assert sum(times.values()) <= plan['tranche_seconds']
    inventory = [(r['document'], r['source'], r['count']) for r in scores[0, 'time_ema']['documents']]
    assert [r[0] for r in inventory] == list(range(108)) and sum(r[2] for r in inventory) == 100416
    for (slot, name), score in scores.items():
        assert [(r['document'], r['source'], r['count']) for r in score['documents']] == inventory
        assert np.isfinite([r['loss_sum'] for r in score['documents']]).all()
        assert abs(sum(r['loss_sum'] for r in score['documents'])/100416-score['ce']) < 1e-12
        endpoint = next(e for e in results[slot]['endpoints'] if e['endpoint'] == name)
        assert score['ce'] == endpoint['ce'] and score['count'] == endpoint['count'] == 100416
        for source in ('fineweb_edu', 'wikipedia'):
            rows = [r for r in score['documents'] if r['source'] == source]
            count = sum(r['count'] for r in rows)
            expected_ce = sum(r['loss_sum'] for r in rows)/count
            assert endpoint['sources'][source]['count'] == count
            assert abs(endpoint['sources'][source]['ce']-expected_ce) < 1e-12
    pairs, deltas = [], []
    for bundle, (short, long) in enumerate(PAIRS, 1):
        a, b = plan['slots'][short], plan['slots'][long]
        assert a['bundle'] == b['bundle'] == bundle and a['roles'] == b['roles']
        assert a['recipe'] == 'native_ce' and b['recipe'] == 'native_pseudo_repetition'
        assert a['parent'] == b['parent'] and a['pseudo_seed'] == b['pseudo_seed']
        assert a['budget_ns'] == b['budget_ns'] == 1200*10**9
        pairs.append(dict(bundle=bundle, ce_slot=short, pseudo_slot=long,
            time_ema_delta=scores[long, 'time_ema']['ce']-scores[short, 'time_ema']['ce'],
            raw_delta=scores[long, 'raw']['ce']-scores[short, 'raw']['ce']))
        deltas.append([y['loss_sum']-x['loss_sum'] for x,y in zip(
            scores[short, 'time_ema']['documents'], scores[long, 'time_ema']['documents'])])
    decision = decide(deltas, [r[2] for r in inventory], [r[1] for r in inventory], plan['acceptance'])
    assert np.allclose(decision['paired_deltas'], [p['time_ema_delta'] for p in pairs], rtol=0, atol=1e-12)
    return dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), plan_sha256=BINDING,
        analysis_source_sha256=sha(__file__), bootstrap_source_sha256=sha(ROOT/'research/general_rsi_equal_time_analysis_v1.py'),
        all_six_execution_reviews_bound=True, pairs=pairs, decision=decision,
        mean_ce={h:{e:float(np.mean([scores[i,e]['ce'] for i in slots])) for e in ('raw','time_ema')}
                 for h,slots in [('native_ce',[0,2,4]), ('native_pseudo_repetition',[1,3,5])]},
        bootstrap=dict(seed=plan['acceptance']['bootstrap_seed'], replicates=plan['acceptance']['replicates'], upper_quantile=.99,
            scope='Conditional on108documents and three paired seeds; not population-seed confidence.'),
        guard_seconds_by_slot=times, total_guard_seconds=sum(times.values()),
        training_seconds_by_slot={i:r['training_ns']/1e9 for i,r in results.items()},
        updates_by_slot={i:r['steps'] for i,r in results.items()}, evidence_sha256=evidence,
        scope='Paired native continuation recipes from three selected parents at1200s each. Ordinary CE versus alternating CE/pseudo repetition; natural exposure differs. RAW secondary. No fresh-pretraining, semantic-generation or transfer claim.', test_opened=False)


if __name__ == '__main__':
    from research.general_rsi_pseudo_worker_v1 import put
    result = analyze()
    put(PANEL/'decision.json', result)
    print(json.dumps(result['decision'], indent=2))
