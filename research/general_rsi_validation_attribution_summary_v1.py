"""Fixed descriptive additive token categories; no scientific decision revision."""
import datetime
import math
import numpy as np
from research.general_rsi_validation_attribution_v1 import OUT,PRIOR,DATA,ROOT,NAMES,read,put,sha,inputs

def main():
    cfg=read(OUT/'plan.json');dest=OUT/'attempt-01';assert read(dest/'complete.json')['slots']==[4,5,6,7,8,9]
    assert cfg['analysis_sha256']==sha(__file__)
    for f,h in cfg['inputs_sha256'].items():assert sha(ROOT/f)==h
    _,rows=inputs();labeldata=np.load(OUT/'labels.npz');labels=labeldata['labels'];offsets=labeldata['offsets'];docids=labeldata['document_ids'];sources=np.array([r['source'] for r in rows])[docids]
    results={};old={};bindings={};max_error=0.
    for slot in range(4,10):
        path=dest/f'slot-{slot:02d}.json';r=read(path);f=dest/f'slot-{slot:02d}-losses.npz';a=np.load(f)['losses'];expected=read(PRIOR/f'slot-{slot:02d}/time_ema-validation.json')
        assert r['worker_sha256']==cfg['worker_sha256'] and r['plan_sha256']==sha(OUT/'plan.json') and sha(f)==r['losses_sha256']
        assert a.shape==(100416,) and a.dtype==np.dtype('<f4') and np.isfinite(a).all() and (a>=0).all()
        for i,row in enumerate(expected['documents']):
            error=abs(math.fsum(map(float,a[offsets[i]:offsets[i+1]]))-row['loss_sum'])/row['count'];max_error=max(max_error,error);assert error<1e-6
        for source,g in r['groups'].items():
            mask=np.ones(len(a),dtype=bool) if source=='all' else sources==source
            for k,name in enumerate(NAMES):assert g[name]['count']==int((mask&(labels==k)).sum()) and abs(math.fsum(map(float,a[mask&(labels==k)]))-g[name]['loss_sum'])<1e-8
        results[slot]=r;old[slot]=expected;bindings[str(path.relative_to(ROOT))]=sha(path)
    summary={}
    for source in ['all','fineweb_edu','wikipedia']:
        total=100416 if source=='all' else int((sources==source).sum());groups={}
        for name in NAMES:
            count=results[4]['groups'][source][name]['count'];means={arm:float(np.mean([results[s]['groups'][source][name]['loss_sum'] for s in slots])) for arm,slots in [('native',[4,6,8]),('raw_bytes',[5,7,9])]}
            diff=means['native']-means['raw_bytes']
            groups[name]=dict(count=count,target_fraction=count/total,mean_ce={a:v/count if count else None for a,v in means.items()},native_minus_byte_category_ce=diff/count if count else None,contribution_to_source_delta=diff/total,pair_contributions=[(results[n]['groups'][source][name]['loss_sum']-results[b]['groups'][source][name]['loss_sum'])/total for n,b in [(4,5),(6,7),(8,9)]])
        ce=lambda r:r['ce'] if source=='all' else r['sources'][source]['ce']
        original=float(np.mean([ce(old[n])-ce(old[b]) for n,b in [(4,5),(6,7),(8,9)]]));summed=math.fsum(g['contribution_to_source_delta'] for g in groups.values())
        assert sum(g['count'] for g in groups.values())==total and abs(original-summed)<1e-6
        summary[source]=dict(targets=total,original_native_minus_byte_ce=original,sum_category_contributions=summed,categories=groups)
    guard=read(OUT/'guard-01.json');assert guard['status']=='finished'
    put(OUT/'summary.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(OUT/'plan.json'),analysis_sha256=sha(__file__),input_receipts_sha256=bindings,results=summary,max_original_parent_ce_abs_delta=max_error,guard=guard,scope='Descriptive reuse of original validation and all six fixed checkpoints. No new independent confirmation, category significance test or revised adoption gate. Other includes mixed whitespace/text, punctuation, EOT and markup; it is not semantic content or prose-only.',test_opened=False))
    print(summary)
if __name__=='__main__':main()
