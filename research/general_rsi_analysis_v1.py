"""Audit the completed frozen panel and analyze saved scores; no model execution."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / 'research/tokenizer-evidence/general-rsi-panel-001'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def analyze():
    plan = read(P/'plan.json')
    assert sha(P/'plan.json') == 'ab70714782ccc514809a70832c8ca650c884ca03719eb39616df37e1c3d876b2'
    for path, digest in plan['sources'].items():
        assert sha(ROOT/path) == digest, path
    scores, results, guards, histories, curves, bindings = {}, {}, {}, {}, {}, {}
    inventory = None
    for slot in range(4,10):
        d = P/f'slot-{slot:02d}'
        result, guard = read(d/'result.json'), read(d/'guard-01.json')
        expected = dict(plan['slots'][slot]); expected['lr'] = .0025
        assert result['slot'] == expected
        assert result['status'] == 'completed' and result['test_opened'] is False
        assert result['plan_sha256'] == sha(P/'plan.json')
        assert result['steps'] == 10240 and result['ema_snapshots'] == 342
        assert guard['status'] == 'finished' and guard['reason'] is None
        assert guard['observation']['peak_rss_bytes'] < plan['rss_limit_bytes']
        assert guard['elapsed_seconds'] < plan['slot_timeout_seconds']
        assert read(d/'latest.json')['status'] == 'retired_after_completion'
        steps = [json.loads(x) for x in (d/'steps.jsonl').read_text().splitlines()]
        assert [x['step'] for x in steps] == list(range(1,10241))
        assert steps[-1]['lr_ratio'] == .1
        for x in steps:
            assert math.isfinite(x['loss']) and math.isfinite(x['clip']['pre_norm'])
            assert math.isfinite(x['clip']['post_norm']) and x['clip']['post_norm'] <= 1.00001
            assert x['mps_driver_bytes'] < plan['mps_limit_bytes']
            if slot % 2: assert x['clip']['gradient_tensors'] == 105
        histories[slot] = steps
        for e in result['endpoints']:
            w=e['weights']; assert Path(w['path']).stat().st_size == w['size']
            assert sha(w['path']) == w['sha256']
            v=read(d/f"{e['endpoint']}-validation.json")
            docs=v['documents']; inv=[(x['document'],x['source'],x['count']) for x in docs]
            assert [x[0] for x in inv] == list(range(108))
            if inventory is None: inventory=inv
            assert inventory == inv
            assert sum(x['count'] for x in docs) == v['count'] == e['count'] == 100416
            assert abs(sum(x['loss_sum'] for x in docs)/v['count']-v['ce']) < 1e-12
            assert v['ce'] == e['ce']
            for source, count in [('fineweb_edu',80354),('wikipedia',20062)]:
                sub=[x for x in docs if x['source']==source]
                assert sum(x['count'] for x in sub)==count
                assert abs(sum(x['loss_sum'] for x in sub)/count-e['sources'][source]['ce'])<1e-12
            scores[slot,e['endpoint']] = v
        results[slot], guards[slot] = result,guard
        # The step2048 diagnostic is full validation; all other saved probes use eight docs.
        curve=[]
        for step in range(1024,10240,1024):
            v=read(d/f'validation-{step:05d}.json')
            docs=v['documents']
            probe=[x for x in docs if x['document'] in plan['probe_documents']]
            assert [x['document'] for x in probe]==plan['probe_documents']
            curve.append({'step':step,'probe_ce':sum(x['loss_sum'] for x in probe)/sum(x['count'] for x in probe)})
        v=scores[slot,'raw']; probe=[x for x in v['documents'] if x['document'] in plan['probe_documents']]
        curve.append({'step':10240,'probe_ce':sum(x['loss_sum'] for x in probe)/sum(x['count'] for x in probe)})
        curves[slot]={'probe_curve':curve,'full_raw_2048':read(d/'validation-02048.json')['ce'],
            'full_raw_10240':v['ce'],'training_first1024_mean':float(np.mean([x['loss'] for x in steps[:1024]])),
            'training_last1024_mean':float(np.mean([x['loss'] for x in steps[-1024:]]))}
        for f in d.glob('*.json*'): bindings[str(f.relative_to(ROOT))]=sha(f)
    pairs=[]; document_deltas=[]
    for n,b in [(4,5),(6,7),(8,9)]:
        assert [x['batch_sha256'] for x in histories[n]] == [x['batch_sha256'] for x in histories[b]]
        delta=scores[b,'h200']['ce']-scores[n,'h200']['ce']
        pairs.append({'native_slot':n,'byte_slot':b,'h200_delta':delta,
            'raw_delta':scores[b,'raw']['ce']-scores[n,'raw']['ce'],
            'guard_time_ratio':guards[b]['elapsed_seconds']/guards[n]['elapsed_seconds']})
        document_deltas.append([y['loss_sum']-x['loss_sum'] for x,y in zip(scores[n,'h200']['documents'],scores[b,'h200']['documents'])])
    counts=np.array([x[2] for x in inventory]); means=np.mean(document_deltas,axis=0)
    mean_delta=float(means.sum()/counts.sum())
    assert abs(mean_delta-np.mean([x['h200_delta'] for x in pairs]))<1e-12
    rng=np.random.default_rng(plan['bootstrap_seed'])
    ids=rng.integers(0,108,size=(plan['bootstrap_resamples'],108))
    draws=means[ids].sum(axis=1)/counts[ids].sum(axis=1)
    # Use the expressly frozen ONE-SIDED .99 upper quantile, not a newly chosen CI.
    upper=float(np.quantile(draws,plan['bootstrap_upper_quantile'],method='linear'))
    native_upper=float(np.quantile(-draws,plan['bootstrap_upper_quantile'],method='linear'))
    sources={s:float(np.mean([results[b]['endpoints'][1]['sources'][s]['ce']-results[n]['endpoints'][1]['sources'][s]['ce'] for n,b in [(4,5),(6,7),(8,9)]])) for s in ['fineweb_edu','wikipedia']}
    gates={'all_three_negative':all(x['h200_delta']<0 for x in pairs),'mean_threshold':mean_delta<=plan['paired_delta_mean_threshold'],
        'bootstrap_upper_negative':upper<0,'source_guard':max(sources.values())<=plan['source_regression_limit']}
    native_gates=[all(x['h200_delta']>0 for x in pairs),-mean_delta<=plan['paired_delta_mean_threshold'],native_upper<0,max(-x for x in sources.values())<=plan['source_regression_limit']]
    decision='raw_bytes' if all(gates.values()) else ('native' if all(native_gates) else 'unresolved')
    return {'plan_sha256':sha(P/'plan.json'),'analysis_source_sha256':sha(__file__),'source_bindings_verified':len(plan['sources']),
        'all_six_executions_verified':True,'all_three_10240_batch_pairs_verified':True,'pairs':pairs,'mean_h200_delta':mean_delta,
        'mean_h200_ce':{a:float(np.mean([scores[i,'h200']['ce'] for i in slots])) for a,slots in [('native',[4,6,8]),('raw_bytes',[5,7,9])]},
        'bootstrap':{'seed':plan['bootstrap_seed'],'resamples':len(draws),'upper_quantile':plan['bootstrap_upper_quantile'],'upper':upper,'native_upper':native_upper,
            'method':'NumPy PCG64 default_rng; average paired seed document loss sums, resample108 IDs, divide by resampled counts; linear quantile',
            'scope':'Conditional on these validation documents and three seeds; not population-seed confidence'},
        'source_mean_deltas':sources,'byte_gates':gates,'selected_architecture':decision,'curves':curves,
        'guard_seconds_by_slot':{i:g['elapsed_seconds'] for i,g in guards.items()},
        'test_opened':False,'evidence_sha256':bindings}

if __name__=='__main__':
    import sys
    result=analyze()
    target=Path(sys.argv[1])
    with target.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['curves','evidence_sha256']},indent=2))
