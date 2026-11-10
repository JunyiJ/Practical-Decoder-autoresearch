"""Apply the frozen equal-time gates to all three accepted fresh pairs."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / 'research/tokenizer-evidence/general-rsi-equal-time-001'
BINDING = '4f51d54569d8b649c9609d25d42ba5bbaefdfd390db99b3283d5efc1c2c1165d'

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def bootstrap(deltas, counts, seed, replicates):
    means = np.mean(deltas, axis=0)
    counts = np.asarray(counts)
    ids = np.random.Generator(np.random.PCG64(seed)).integers(0, len(counts), size=(replicates, len(counts)))
    return float(means.sum()/counts.sum()), means[ids].sum(axis=1)/counts[ids].sum(axis=1)

def analyze():
    assert sha(P/'plan.json') == BINDING
    plan = read(P/'plan.json'); contract = plan['acceptance']
    for path, digest in plan['sources'].items():
        assert sha(ROOT/path) == digest, path
    results, scores, bindings, times = {}, {}, {}, {}
    for slot in range(10):
        d = P/f'slot-{slot:02d}'; r = read(d/'result.json'); v = read(d/'review.json')
        assert v['accepted_execution'] and v['result_sha256'] == sha(d/'result.json')
        assert r['binding'] == v['binding'] == BINDING and not r['test_opened']
        times[slot] = sum(read(f)['elapsed_seconds'] for f in d.glob('guard-*.json'))
        assert times[slot] <= plan['slots'][slot]['timeout_seconds']
        for e in r['endpoints']:
            w = e['weights']; assert Path(w['path']).stat().st_size == w['size'] and sha(w['path']) == w['sha256']
        results[slot] = r
        for f in [d/'result.json', d/'review.json']:
            bindings[str(f.relative_to(ROOT))] = sha(f)
        if slot >= 4:
            for endpoint in ['raw','time_ema']:
                f = d/f'{endpoint}-validation.json'; scores[slot,endpoint] = read(f)
                bindings[str(f.relative_to(ROOT))] = sha(f)
    assert sum(times.values()) <= plan['tranche_seconds']
    inventory = [(x['document'],x['source'],x['count']) for x in scores[4,'time_ema']['documents']]
    assert len(inventory) == 108 and sum(x[2] for x in inventory) == 100416
    for (slot,endpoint), v in scores.items():
        assert [(x['document'],x['source'],x['count']) for x in v['documents']] == inventory
        assert abs(sum(x['loss_sum'] for x in v['documents'])/100416-v['ce']) < 1e-12
        assert v['ce'] == next(e['ce'] for e in results[slot]['endpoints'] if e['endpoint']==endpoint)
    pairs = []; deltas = []
    for a,b in [(4,5),(6,7),(8,9)]:
        pairs.append(dict(bundle=(a-2)//2,native_slot=a,byte_slot=b,
            time_ema_delta_byte_minus_native=scores[b,'time_ema']['ce']-scores[a,'time_ema']['ce'],
            raw_delta_byte_minus_native=scores[b,'raw']['ce']-scores[a,'raw']['ce']))
        deltas.append([y['loss_sum']-x['loss_sum'] for x,y in zip(scores[a,'time_ema']['documents'],scores[b,'time_ema']['documents'])])
    mean, draws = bootstrap(deltas,[x[2] for x in inventory],contract['bootstrap']['seed'],contract['bootstrap']['replicates'])
    assert abs(mean-np.mean([x['time_ema_delta_byte_minus_native'] for x in pairs])) < 1e-12
    source_deltas = {source:float(np.mean([results[b]['endpoints'][1]['sources'][source]['ce']-results[a]['endpoints'][1]['sources'][source]['ce'] for a,b in [(4,5),(6,7),(8,9)]])) for source in ['fineweb_edu','wikipedia']}
    directions = {}
    for arm, sign in [('raw_bytes',1),('native',-1)]:
        upper = float(np.quantile(sign*draws,.99,method='linear'))
        sources = {k:sign*v for k,v in source_deltas.items()}
        gates = dict(all_three_deltas_negative=all(sign*p['time_ema_delta_byte_minus_native']<0 for p in pairs),
            mean_threshold=sign*mean<=contract['mean_delta_at_most'],bootstrap_upper_negative=upper<0,
            source_guard=max(sources.values())<=contract['source_regression_at_most'])
        directions[arm] = dict(candidate_minus_other=sign*mean,bootstrap_upper_99=upper,source_deltas=sources,gates=gates,accepted=all(gates.values()))
    winners = [arm for arm,v in directions.items() if v['accepted']]; assert len(winners)<=1
    return dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=BINDING,
        analysis_source_sha256=sha(__file__),all_ten_execution_reviews_bound=True,pairs=pairs,
        mean_ce={arm:{e:float(np.mean([scores[i,e]['ce'] for i in slots])) for e in ['raw','time_ema']} for arm,slots in [('native',[4,6,8]),('raw_bytes',[5,7,9])]},
        mean_delta_byte_minus_native=mean,directions=directions,selected_equal_time_arm=winners[0] if winners else None,
        bootstrap=dict(**contract['bootstrap'],upper_quantile=.99,scope='Conditional on these108documents and three paired seeds; not population-seed confidence.'),
        guard_seconds_by_slot=times,tranche_guard_seconds=sum(times.values()),
        training_seconds_by_slot={i:r['training_ns']/1e9 for i,r in results.items()},
        updates_by_slot={i:r['steps'] for i,r in results.items()},
        scope='Final time-EMA loss at7200 synchronized training seconds per arm on this Mac and corpus. Includes budget-matched schedule and EMA; excludes checkpoint/evaluation/generation I/O from training clock. Not a generation-quality or transfer claim.',
        test_opened=False,evidence_sha256=bindings)

if __name__=='__main__':
    r=analyze()
    with (P/'decision.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in r.items() if k!='evidence_sha256'},indent=2))
