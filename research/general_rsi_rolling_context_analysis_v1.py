"""Analyze the prospectively bound six-model stride-one development diagnostic."""
import datetime, fcntl, json, math, time
import numpy as np
from research.general_rsi_rolling_context_v1 import ROOT, OUT, read, put, sha, verify_bindings, data


def bootstrap(doc_effects, counts, seed, draws):
    rng=np.random.Generator(np.random.PCG64(seed));values=[];checked=0;worst=0.
    for start in range(0,draws,1000):
        indexes=rng.integers(0,len(counts),size=(min(1000,draws-start),len(counts)))
        denominator=counts[indexes].sum(axis=1)
        current=doc_effects[:,indexes].sum(axis=2)/denominator
        # Independently count resampled document multiplicities for128 draws.
        for j in range(min(len(indexes),max(0,128-checked))):
            weights=np.bincount(indexes[j],minlength=len(counts))
            for k in range(3):
                reference=math.fsum(float(doc_effects[k,i])*int(weights[i]) for i in range(len(counts)))/sum(int(counts[i])*int(weights[i]) for i in range(len(counts)))
                worst=max(worst,abs(reference-current[k,j]));assert abs(reference-current[k,j])<1e-12
            checked+=1
        values.append(current)
    return np.concatenate(values,axis=1),dict(draws_independently_checked=checked,max_abs_error=worst)


def main():
    start=time.monotonic()
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=read(OUT/'plan.json');verify_bindings(plan)
        scoring=OUT/'scoring-01';assert read(scoring/'result.json')['completed']
        tokens,docs=data('val');groups={4:[],16:[]};evidence={};counts=np.array([max(0,d['tokens']-256) for d in docs],dtype=np.int64)
        assert sum(counts)==plan['target_count']
        for e in plan['endpoints']:
            folder=scoring/f"d{e['depth']}-b{e['bundle']}";completion=read(folder/'result.json')
            assert completion['parameters_unchanged'] and completion['endpoint']==e
            sums={128:[],256:[]}
            for i,doc in enumerate(docs):
                path=folder/f'doc-{i:03d}.json';digest=sha(path);assert digest==completion['files'][path.name]
                evidence[str(path.relative_to(ROOT))]=digest;r=read(path)
                assert r['document']==i and r['source']==doc['source'] and r['offset']==doc['offset']
                assert r['positions']==list(range(256,doc['tokens']))
                expected=tokens[doc['offset']+256:doc['offset']+doc['tokens']].tolist()
                assert r['target_ids']==expected and r['count']==counts[i]
                for c in (128,256):
                    losses=r['losses'][str(c)];assert len(losses)==counts[i] and all(math.isfinite(v) for v in losses)
                    total=math.fsum(losses);assert abs(total-r['loss_sums'][str(c)])<1e-10
                    sums[c].append(total)
            delta=np.array(sums[256])-np.array(sums[128]);total=int(sum(counts))
            sources={}
            for source in sorted({d['source'] for d in docs}):
                indexes=[i for i,d in enumerate(docs) if d['source']==source];n=int(sum(counts[indexes]))
                sources[source]=dict(count=n,delta=math.fsum(delta[i] for i in indexes)/n)
            groups[e['depth']].append(dict(bundle=e['bundle'],slot=e['slot'],count=total,
                 ce128=math.fsum(sums[128])/total,ce256=math.fsum(sums[256])/total,
                 delta=math.fsum(delta)/total,sources=sources,document_delta_loss_sums=delta.tolist()))
        d4=np.mean([r['document_delta_loss_sums'] for r in groups[4]],axis=0)
        d16=np.mean([r['document_delta_loss_sums'] for r in groups[16]],axis=0)
        effects=np.array([d4,d16,d16-d4]);samples,check=bootstrap(effects,counts,plan['bootstrap_seed'],plan['bootstrap_resamples'])
        summaries={}
        for k,name in enumerate(('depth4','depth16','depth16_minus_depth4')):
            lo,hi=np.quantile(samples[k],plan['interval_quantiles']).tolist()
            paired=[groups[4][i]['delta'] if k==0 else groups[16][i]['delta'] if k==1 else groups[16][i]['delta']-groups[4][i]['delta'] for i in range(3)]
            mean=math.fsum(effects[k])/sum(counts)
            assert abs(mean-math.fsum(paired)/3)<1e-12
            summaries[name]=dict(mean_delta=mean,paired_deltas=paired,interval=[lo,hi],
                                 consistent_predictive_benefit=bool(k<2 and hi<0 and all(x<0 for x in paired)),
                                 interval_direction='negative' if hi<0 else 'positive' if lo>0 else 'unresolved_not_equivalent')
        put(OUT/'analysis.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            plan_sha256=sha(OUT/'plan.json'),source_sha256=sha(__file__),models=groups,contrasts=summaries,
            bootstrap_check=check,counts=counts.tolist(),evidence=evidence,
            scoring_seconds=read(scoring/'result.json')['seconds'],analysis_seconds=time.monotonic()-start,
            test_opened=False,training_updates=0,interpretation='Conditional reused-development diagnostic. Fixed256-trained EMA weights; full128 vs256 histories. No recipe adoption or scaling law.'))
        print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
