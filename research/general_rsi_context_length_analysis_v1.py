"""Apply the frozen context gates only after all six execution reviews pass."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

from research.general_rsi_equal_time_analysis_v1 import bootstrap

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT/'research/tokenizer-evidence/general-rsi-context-length-quality-001'
BINDING = '60b4addc99cef4cb2825b792699275ac490cf240ff478a35391a03a55699838e'
PAIRS = ((0, 1), (3, 2), (4, 5))


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decide(deltas, counts, sources, contract):
    """Deltas contain paired long-minus-short document loss sums, not CE means."""
    deltas = np.asarray(deltas, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    assert deltas.shape == (3, len(counts)) and np.isfinite(deltas).all()
    assert (counts > 0).all() and len(sources) == len(counts)
    assert set(sources) == {'fineweb_edu', 'wikipedia'}
    paired = deltas.sum(axis=1)/counts.sum()
    mean, draws = bootstrap(deltas, counts, contract['bootstrap']['seed'], contract['bootstrap']['replicates'])
    assert abs(mean-float(paired.mean())) < 1e-12
    upper = float(np.quantile(draws, .99, method='linear'))
    source_deltas = {}
    for source in ('fineweb_edu', 'wikipedia'):
        mask = np.asarray([x == source for x in sources])
        source_deltas[source] = float(deltas[:, mask].sum(axis=1).mean()/counts[mask].sum())
    gates = dict(all_three_deltas_negative=bool((paired < 0).all()),
        mean_threshold=mean <= contract['mean_delta_at_most'],
        bootstrap_upper_negative=upper < 0,
        source_guard=max(source_deltas.values()) <= contract['source_regression_at_most'])
    accepted = all(gates.values())
    return dict(paired_deltas=paired.tolist(), mean_delta_256_minus_128=mean,
        bootstrap_upper_99=upper, source_deltas=source_deltas, gates=gates,
        context256_accepted=accepted, retained_recipe='context256' if accepted else 'context128')


def analyze():
    assert sha(PANEL/'plan.json') == BINDING
    plan = read(PANEL/'plan.json')
    # Require the complete panel before reading any endpoint scores.
    for slot in range(6):
        d = PANEL/f'slot-{slot:02d}'
        assert (d/'result.json').exists() and (d/'review.json').exists(), 'Six accepted endpoints required'
    for name, digest in plan['sources'].items(): assert sha(ROOT/name) == digest, name
    for name, digest in plan['qualification_evidence'].items(): assert sha(ROOT/name) == digest, name
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
        assert a['context'] == 128 and b['context'] == 256
        assert a['budget_ns'] == 14400*10**9 and b['budget_ns'] == 14400*10**9
        pairs.append(dict(bundle=bundle, short_slot=short, long_slot=long,
            time_ema_delta=scores[long, 'time_ema']['ce']-scores[short, 'time_ema']['ce'],
            raw_delta=scores[long, 'raw']['ce']-scores[short, 'raw']['ce']))
        deltas.append([y['loss_sum']-x['loss_sum'] for x,y in zip(
            scores[short, 'time_ema']['documents'], scores[long, 'time_ema']['documents'])])
    common = {}
    for endpoint in ('raw', 'time_ema'):
        rows = []
        for short, long in PAIRS:
            path = PANEL/f'slot-{long:02d}'/f'{endpoint}-common128-validation.json'
            score = read(path)
            assert [(r['document'],r['source'],r['count']) for r in score['documents']] == inventory
            evidence[str(path.relative_to(ROOT))] = sha(path)
            rows.append(dict(control_slot=short,candidate_slot=long,control_ce=scores[short,endpoint]['ce'],candidate_ce=score['ce'],delta=score['ce']-scores[short,endpoint]['ce']))
        common[endpoint] = dict(pairs=rows,mean_delta=float(np.mean([r['delta'] for r in rows])),scope='Fixed secondary contrast; cannot rescue primary gate or isolate causal mechanisms.')
    decision = decide(deltas, [r[2] for r in inventory], [r[1] for r in inventory], plan['acceptance'])
    assert np.allclose(decision['paired_deltas'], [p['time_ema_delta'] for p in pairs], rtol=0, atol=1e-12)
    return dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), plan_sha256=BINDING,
        analysis_source_sha256=sha(__file__), bootstrap_source_sha256=sha(ROOT/'research/general_rsi_equal_time_analysis_v1.py'),
        all_six_execution_reviews_bound=True, pairs=pairs, decision=decision, common128=common,
        mean_ce={h:{e:float(np.mean([scores[i,e]['ce'] for i in slots])) for e in ('raw','time_ema')}
                 for h,slots in [('context128',[0,3,4]), ('context256',[1,2,5])]},
        bootstrap=dict(**plan['acceptance']['bootstrap'], upper_quantile=.99,
            scope='Conditional on108documents and three paired seeds; not population-seed confidence.'),
        guard_seconds_by_slot=times, total_guard_seconds=sum(times.values()),
        training_seconds_by_slot={i:r['training_ns']/1e9 for i,r in results.items()},
        updates_by_slot={i:r['steps'] for i,r in results.items()}, evidence_sha256=evidence,
        scope='Native complete context128/context256 recipes at equal14400-second training time. RAW and common128 secondary. No isolated training-context effect, long-prompt ability or size-transfer claim.', test_opened=False)


if __name__ == '__main__':
    from research.general_rsi_native_horizon_worker_v1 import put
    result = analyze()
    put(PANEL/'decision.json', result)
    print(json.dumps(result['decision'], indent=2))
