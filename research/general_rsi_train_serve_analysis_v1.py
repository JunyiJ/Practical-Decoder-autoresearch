"""Frozen paired analysis for the 2x3 training/serving context table."""
import datetime,fcntl,math,time
import numpy as np
from research.general_rsi_train_serve_context_v1 import ROOT,OUT,read,put,sha,verify,base,name,HISTORIES


def contrasts(cells):
    out={}
    for h in HISTORIES:out[f'train256_minus128_at_h{h}']=cells[256,h]-cells[128,h]
    for t in (128,256):
        out[f'serve256_minus128_t{t}']=cells[t,256]-cells[t,128]
        out[f'serve512_minus256_t{t}']=cells[t,512]-cells[t,256]
    for lo,hi in [(128,256),(256,512)]:
        out[f'train_by_serve_{lo}_to_{hi}']=(cells[256,hi]-cells[256,lo])-(cells[128,hi]-cells[128,lo])
    for short,long in [(128,256),(128,512),(256,512)]:
        out[f'policy_t256_h{short}_minus_t128_h{long}']=cells[256,short]-cells[128,long]
    assert len(out)==12;return out


def main():
    started=time.monotonic()
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.check_stop();plan=read(OUT/'plan.json');verify(plan)
        assert read(OUT/'scoring-01/result.json')['completed']
        values,all_docs=base.data('val');docs=[(i,d) for i,d in enumerate(all_docs) if d['tokens']>512]
        counts=np.array([d['tokens']-512 for _,d in docs]);assert len(counts)==65 and sum(counts)==51457
        cells={};rows=[];evidence={};reuse_cells=0
        for e in plan['endpoints']:
            folder=OUT/'scoring-01'/name(e);r=read(folder/'result.json')
            assert r['completed'] and r['endpoint']==e and r['parameters_and_buffers_unchanged']
            sums={h:[] for h in HISTORIES}
            for doc,d in docs:
                path=folder/f'doc-{doc:03d}.json';digest=sha(path);assert digest==r['files'][path.name];evidence[str(path.relative_to(ROOT))]=digest
                v=read(path);assert v['positions']==list(range(512,d['tokens']))
                assert v['target_ids']==values[d['offset']+512:d['offset']+d['tokens']].tolist()
                assert v['source']==d['source'] and v['offset']==d['offset'] and v['document']==doc
                if e['train_context']==256:
                    old=read(ROOT/v['reuse']['path']);assert sha(ROOT/v['reuse']['path'])==v['reuse']['sha256']
                    ix=[j for j,p in enumerate(old['positions']) if p>=512]
                    for h in (128,256):assert v['losses'][str(h)]==[old['losses'][str(h)][j] for j in ix]
                for h in HISTORIES:
                    ls=v['losses'][str(h)];assert len(ls)==d['tokens']-512 and all(math.isfinite(x) for x in ls)
                    total=math.fsum(ls);assert abs(total-v['loss_sums'][str(h)])<1e-10;sums[h].append(total)
            if e['train_context']==256:reuse_cells+=2
            for h in HISTORIES:cells[e['depth'],e['bundle'],e['train_context'],h]=np.array(sums[h])
            sources={}
            for src in sorted({d['source'] for _,d in docs}):
                ix=[i for i,(_,d) in enumerate(docs) if d['source']==src];n=int(sum(counts[ix]))
                sources[src]=dict(count=n,ce={h:math.fsum(sums[h][i] for i in ix)/n for h in HISTORIES})
            rows.append(dict(depth=e['depth'],bundle=e['bundle'],train_context=e['train_context'],slot=e['slot'],
                             ce={h:math.fsum(sums[h])/51457 for h in HISTORIES},sources=sources,new_scoring_seconds=r['seconds']))
        assert reuse_cells==12
        per_pair={};means={};keys=None
        for depth in (4,16):
            per_pair[depth]=[contrasts({(t,h):cells[depth,b,t,h] for t in (128,256) for h in HISTORIES}) for b in (1,2,3)]
            keys=list(per_pair[depth][0]);means[depth]=np.mean([[r[k] for k in keys] for r in per_pair[depth]],axis=0)
        effects=np.concatenate([means[4],means[16],means[16]-means[4]])
        rng=np.random.Generator(np.random.PCG64(plan['bootstrap_seed']));chunks=[];worst=0.;checked=0
        for _ in range(plan['bootstrap_resamples']//1000):
            ix=rng.integers(0,65,size=(1000,65));sample=effects[:,ix].sum(axis=2)/counts[ix].sum(axis=1)
            for j in range(min(1000,128-checked)):
                freq=np.bincount(ix[j],minlength=65)
                for k in range(36):
                    ref=math.fsum(float(effects[k,i])*int(freq[i]) for i in range(65))/sum(int(counts[i])*int(freq[i]) for i in range(65))
                    worst=max(worst,abs(ref-sample[k,j]));assert abs(ref-sample[k,j])<1e-12
                checked+=1
            chunks.append(sample)
        samples=np.concatenate(chunks,axis=1);results={}
        for gi,group in enumerate(['depth4','depth16','depth16_minus4']):
            results[group]={}
            for j,k in enumerate(keys):
                pairs=[math.fsum(per_pair[4][b][k] if gi==0 else per_pair[16][b][k] if gi==1 else per_pair[16][b][k]-per_pair[4][b][k])/51457 for b in range(3)]
                lo,hi=np.quantile(samples[gi*12+j],plan['interval_quantiles']).tolist();mean=math.fsum(effects[gi*12+j])/51457
                assert abs(mean-math.fsum(pairs)/3)<1e-12
                results[group][k]=dict(mean_delta=mean,paired_deltas=pairs,interval=[lo,hi],consistently_negative=hi<0 and all(p<0 for p in pairs),consistently_positive=lo>0 and all(p>0 for p in pairs))
        put(OUT/'analysis.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(OUT/'plan.json'),source_sha256=sha(__file__),
             rows=rows,contrasts=results,bootstrap_check=dict(independent_draws=checked,max_error=worst),
             document_counts=counts.tolist(),paired_document_effects={str(d):[{k:v.tolist() for k,v in row.items()} for row in per_pair[d]] for d in (4,16)},
             matched_targets=51457,eligible_documents=65,reused_cells_exact=True,evidence=evidence,seconds=time.monotonic()-started,
             test_opened=False,training_updates=0,scope='Conditional on reused development documents and12 saved EMA models; direct unscaled RoPE extension.'))
        print('Complete36-contrast analysis with exact matched targets and reused cells.',flush=True)

if __name__=='__main__':main()
