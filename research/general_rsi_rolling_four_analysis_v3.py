"""Same frozen analysis with all D paths consistently selecting attempt02."""
import datetime, fcntl, hashlib, math, time
import numpy as np
from research.general_rsi_rolling_four_v1 import ROOT, OUT, BC, read, put, sha, bindings, base


def bootstrap(effects,counts,plan):
    rng=np.random.Generator(np.random.PCG64(plan['bootstrap_seed']));chunks=[];checked=0;worst=0.
    for start in range(0,plan['bootstrap_resamples'],1000):
        ix=rng.integers(0,len(counts),size=(min(1000,plan['bootstrap_resamples']-start),len(counts)))
        values=effects[:,ix].sum(axis=2)/counts[ix].sum(axis=1)
        for j in range(min(len(ix),128-checked)):
            freq=np.bincount(ix[j],minlength=len(counts))
            denominator=sum(int(counts[i])*int(freq[i]) for i in range(len(counts)))
            for k in range(len(effects)):
                ref=math.fsum(float(effects[k,i])*int(freq[i]) for i in range(len(counts)))/denominator
                worst=max(worst,abs(ref-values[k,j]));assert abs(ref-values[k,j])<1e-12
            checked+=1
        chunks.append(values)
    return np.concatenate(chunks,axis=1),dict(independent_frequency_draws=checked,maximum_absolute_error=worst)


def main():
    started=time.monotonic()
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.check_stop();plan=read(OUT/'plan.json');bindings(plan)
        assert read(OUT/'scoring-02/result.json')['completed'] and read(BC/'scoring-01/result.json')['completed']
        tokens,docs=base.data('val');values=tokens.numpy();counts=np.array([d['tokens']-256 for d in docs]);assert sum(counts)==72876
        permutations={};inputs={}
        # Independent scalar input reconstruction, without the production shuffle helper.
        for doc,d in enumerate(docs):
            ph=hashlib.sha256();ih=hashlib.sha256()
            for p in range(256,d['tokens']):
                key=f'rolling-older-shuffle-v1|621116220001|val|{doc}|{p}'.encode()
                seed=int.from_bytes(hashlib.sha256(key).digest()[:16],'big')
                perm=np.random.Generator(np.random.PCG64(seed)).permutation(128).astype(np.uint8)
                assert sorted(perm.tolist())==list(range(128));a=d['offset']+p
                x=np.concatenate((values[a-256:a-128][perm.astype(np.int64)],values[a-128:a]))
                assert np.array_equal(x[128:],values[a-128:a]) and np.array_equal(np.sort(x[:128]),np.sort(values[a-256:a-128]))
                ph.update(perm.tobytes());ih.update(x.tobytes())
            permutations[doc]=ph.hexdigest();inputs[doc]=ih.hexdigest()
        rows={4:[],16:[]};evidence={};control_error=0.;control_count=0
        for e in plan['endpoints']:
            name=f"d{e['depth']}-b{e['bundle']}";bd=BC/'scoring-01'/name;dd=OUT/'scoring-02'/name
            br=read(bd/'result.json');dr=read(dd/'result.json')
            assert br['endpoint']==dr['endpoint']==e and br['parameters_unchanged'] and dr['parameters_unchanged']
            ar=next(a for a in plan['saved_a'] if a['depth']==e['depth'] and a['bundle']==e['bundle'])
            ap=ROOT/ar['path'];assert sha(ap)==ar['sha256'];evidence[ar['path']]=ar['sha256'];a={}
            for chunk in read(ap)['chunks']:
                for j,(tid,loss) in enumerate(zip(chunk['target_ids'],chunk['token_losses'])):
                    key=(chunk['document'],chunk['target_start']+j);assert key not in a;a[key]=(tid,loss)
            sums={k:[] for k in 'ABCD'}
            for doc,d in enumerate(docs):
                bp=bd/f'doc-{doc:03d}.json';dp=dd/bp.name
                for path,receipt in ((bp,br),(dp,dr)):
                    digest=sha(path);assert digest==receipt['files'][path.name];evidence[str(path.relative_to(ROOT))]=digest
                b=read(bp);sh=read(dp);expected=list(range(256,d['tokens']));target_ids=values[d['offset']+256:d['offset']+d['tokens']].tolist()
                assert b['positions']==sh['positions']==expected and b['target_ids']==sh['target_ids']==target_ids
                assert b['document']==sh['document']==doc and b['offset']==sh['offset']==d['offset'] and b['source']==sh['source']==d['source']
                assert sh['permutations_sha256']==permutations[doc] and sh['inputs_sha256']==inputs[doc]
                aa=[]
                for j,p in enumerate(expected):
                    tid,loss=a[doc,d['offset']+p];assert tid==target_ids[j];aa.append(loss)
                    if p%128==0:
                        control_count+=1;control_error=max(control_error,abs(loss-b['losses']['128'][j]))
                arrays=dict(A=aa,B=b['losses']['128'],C=b['losses']['256'],D=sh['losses'])
                for k,arr in arrays.items():
                    assert len(arr)==counts[doc] and all(math.isfinite(x) for x in arr);sums[k].append(math.fsum(arr))
                assert abs(sums['D'][-1]-sh['loss_sum'])<1e-10
            ds=np.array([np.array(sums[u])-np.array(sums[v]) for u,v in [('B','A'),('C','B'),('C','D'),('D','B')]])
            src={}
            for source in sorted({d['source'] for d in docs}):
                ix=[i for i,d in enumerate(docs) if d['source']==source];n=int(sum(counts[ix]))
                src[source]=dict(count=n,ce={k:math.fsum(sums[k][i] for i in ix)/n for k in 'ABCD'},deltas={k:math.fsum(ds[j,i] for i in ix)/n for j,k in enumerate(plan['contrasts'])})
            rows[e['depth']].append(dict(bundle=e['bundle'],slot=e['slot'],ce={k:math.fsum(v)/sum(counts) for k,v in sums.items()},
                 deltas={k:math.fsum(ds[j])/sum(counts) for j,k in enumerate(plan['contrasts'])},sources=src,document_delta_loss_sums=ds.tolist()))
        assert control_error<1e-5,(control_error,control_count)
        means={d:np.mean([r['document_delta_loss_sums'] for r in rows[d]],axis=0) for d in (4,16)}
        effects=np.concatenate([means[4],means[16],means[16]-means[4]],axis=0)
        samples,check=bootstrap(effects,counts,plan);summaries={}
        for gi,group in enumerate(('depth4','depth16','depth16_minus_depth4')):
            summaries[group]={}
            for j,contrast in enumerate(plan['contrasts']):
                k=gi*4+j;lo,hi=np.quantile(samples[k],plan['interval_quantiles']).tolist()
                paired=[rows[4][b]['deltas'][contrast] if gi==0 else rows[16][b]['deltas'][contrast] if gi==1 else rows[16][b]['deltas'][contrast]-rows[4][b]['deltas'][contrast] for b in range(3)]
                mean=math.fsum(effects[k])/sum(counts);assert abs(mean-math.fsum(paired)/3)<1e-12
                summaries[group][contrast]=dict(mean_delta=mean,paired_deltas=paired,interval=[lo,hi],
                    consistently_negative=bool(hi<0 and all(x<0 for x in paired)),consistently_positive=bool(lo>0 and all(x>0 for x in paired)),
                    interval_direction='negative' if hi<0 else 'positive' if lo>0 else 'unresolved_not_equivalent')
        put(OUT/'analysis.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(OUT/'plan.json'),source_sha256=sha(__file__),
             rows=rows,contrasts=summaries,bootstrap_check=check,A_B_same_input_control=dict(targets=control_count,max_abs_loss_difference=control_error),
             shuffled_inputs_independently_verified=True,shared_permutations_across_models_verified=True,evidence=evidence,
             D_scoring_seconds=read(OUT/'scoring-02/result.json')['seconds'],analysis_seconds=time.monotonic()-started,test_opened=False,training_updates=0,
             scope='Reused development targets, fixed six context256-trained EMA endpoints and one fixed token permutation per target. Twelve-contrast family separate from original B/C family.'))
        print(__import__('json').dumps(summaries,indent=2))
if __name__=='__main__':main()
