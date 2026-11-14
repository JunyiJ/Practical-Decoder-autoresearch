"""Report retained broader scores, accounting and curves without new training."""
import json,hashlib,math,statistics
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent/'tokenizer-evidence/general-rsi-cross-size-quality-001'
ROOT=P.parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,j):
    with p.open('x') as f:json.dump(j,f,indent=2);f.write('\n')
def run():
    plan=read(P/'plan.json');binding=sha(P/'plan.json');flops=read(P/'flops.json')
    assert all(sha(ROOT/k)==v for k,v in plan['sources'].items())
    estimates={(r['depth'],r['arm']):r for r in flops['rows']}
    allocation=read(ROOT/'research/tokenizer-evidence/general-rsi-cross-size-boundary-001/allocation.json')
    inventory=allocation['baseline_parameter_inventory']
    base={a:sum(r['elements'] for r in rows.values()) for a,rows in inventory.items()}
    results={};broader={};costs=[];curves=[];evidence={}
    for s in plan['slots']:
        d=P/f"slot-{s['slot']:02d}";r=read(d/'result.json');v=read(d/'review.json')
        assert v['accepted_execution'] and v['result_sha256']==sha(d/'result.json') and r['binding']==binding
        key=(s['depth'],s['bundle'],s['arm']);results[key]=r
        for e in ['raw','h200']:
            broader[key+(e,)]=read(d/f'{e}-broader.json')
            evidence[str((d/f'{e}-broader.json').relative_to(ROOT))]=sha(d/f'{e}-broader.json')
        steps=[]
        for q in sorted(d.glob('steps-*.jsonl')):steps += [json.loads(x) for x in q.read_text().splitlines()]
        assert len(steps)==10240 and [x['step'] for x in steps]==list(range(1,10241))
        est=estimates[s['depth'],s['arm']]
        costs.append(dict(slot=s['slot'],depth=s['depth'],arm=s['arm'],bundle=s['bundle'],parameters=base[s['arm']]+(s['depth']-4)*allocation['per_block_parameters'],targets=r['executed_targets'],training_seconds=r['training_seconds'],guard_seconds=v['guard_seconds'],median_update_seconds_after_first128=statistics.median(x['seconds'] for x in steps[128:]),peak_mps_bytes=v['peak_mps_bytes'],peak_rss_bytes=v['peak_rss_bytes'],estimated_accounted_flops=est['estimated_accounted_flops_per_run']))
        for n in plan['probe_steps']:
            q=d/f'probe-{n:05d}.json';pr=read(q)
            curves.append(dict(slot=s['slot'],depth=s['depth'],arm=s['arm'],bundle=s['bundle'],step=n,targets=n*1024,estimated_accounted_flops=n*est['estimated_accounted_flops_per_update'],training_seconds=sum(x['seconds'] for x in steps[:n]),probe_ce=pr['ce'],probe_targets=pr['count']))
            evidence[str(q.relative_to(ROOT))]=sha(q)
        for name in ['result.json','review.json','steps-01.jsonl']:evidence[str((d/name).relative_to(ROOT))]=sha(d/name)
    summaries=[];parents=[]
    for source,seed in plan['broader_bootstrap_seeds'].items():
        rng=np.random.default_rng(seed);ids=rng.integers(0,8,size=(plan['bootstrap_resamples'],8))
        for depth in [4,8,16]:
            for endpoint in ['raw','h200']:
                paired=[]
                for bundle in [1,2,3]:
                    n=broader[depth,bundle,'native',endpoint];b=broader[depth,bundle,'raw_bytes',endpoint]
                    assert [(x['document'],x['source'],x['count']) for x in n['documents']]==[(x['document'],x['source'],x['count']) for x in b['documents']]
                    ns=[x for x in n['documents'] if x['source']==source];bs=[x for x in b['documents'] if x['source']==source];assert len(ns)==len(bs)==8
                    assert all(x['count']==2048 for x in ns+bs)
                    delta=[(y['loss_sum']-x['loss_sum'])/2048 for x,y in zip(ns,bs)];paired.append(delta)
                    for x,y,z in zip(ns,bs,delta):parents.append(dict(depth=depth,bundle=bundle,endpoint=endpoint,source=source,parent_document=x['document'],native_ce=x['loss_sum']/2048,byte_ce=y['loss_sum']/2048,byte_minus_native=z))
                a=np.asarray(paired);d=a.mean(axis=0);draws=d[ids].mean(axis=1)
                summaries.append(dict(depth=depth,endpoint=endpoint,source=source,paired_seed_deltas=a.mean(axis=1).tolist(),mean=float(a.mean()),interval95=np.quantile(draws,plan['broader_interval_quantiles'],method='linear').tolist(),parent_count=8,targets_per_seed=16384,bootstrap_seed=seed,bootstrap_resamples=plan['bootstrap_resamples']))
    raw=[];ratios=[]
    for depth in [4,8,16]:
        ds=[];times=[]
        for bundle in [1,2,3]:
            n=results[depth,bundle,'native'];b=results[depth,bundle,'raw_bytes']
            ds.append(b['endpoints'][0]['ce']-n['endpoints'][0]['ce']);times.append(b['training_seconds']/n['training_seconds'])
        raw.append(dict(depth=depth,paired_deltas=ds,mean=statistics.mean(ds)))
        ratios.append(dict(depth=depth,paired_ratios=times,paired_log_ratios=[math.log(x) for x in times],geometric_mean=math.exp(statistics.mean(math.log(x) for x in times))))
    total=dict(training_seconds=sum(x['training_seconds'] for x in costs),worker_guard_seconds=sum(x['guard_seconds'] for x in costs),completed_engineering_seconds=plan['completed_engineering_seconds'],total_targets=sum(x['targets'] for x in costs),estimated_accounted_flops=sum(x['estimated_accounted_flops'] for x in costs),scientific_failed_or_replayed_updates=0,prelaunch_failure='Prelaunch11 receipt-path failure started no training; original failure and repaired attempt retained separately. Earlier prelaunch guard failures likewise charged zero scientific updates.',warmup_policy='Posthoc descriptive latency excludes first128 updates; full training totals include every update.',flop_limits=flops['assumptions'])
    put(P/'descriptive-results.json',dict(binding=binding,source_sha256=sha(Path(__file__)),input_sha256=evidence,flops_sha256=sha(P/'flops.json'),broader=summaries,broader_parent_rows=parents,raw_validation=raw,runtime=ratios,costs=costs,totals=total,broader_limits='Descriptive95% parent bootstrap conditional on3 seeds and8 parents per domain. Narrow PG19/Python prose probe; exact-span exclusion misses paraphrases. No effect on primary gates.',test_opened=False))
    put(P/'probe-curves.json',dict(rows=curves,source_sha256=sha(Path(__file__)),policy='Fixed documents0/1 (4503FineWeb-Edu targets); intermediate model state, not terminal H200. Axes use actual successful targets and synchronized training time or explicitly partial matrix/Muon FLOPs.',test_opened=False))
    print(json.dumps(dict(broader=summaries,raw=raw,runtime=ratios,totals=total),indent=2))
if __name__=='__main__':run()
