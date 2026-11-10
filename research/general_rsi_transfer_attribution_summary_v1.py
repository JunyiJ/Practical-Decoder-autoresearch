"""Additive attribution of existing transfer losses, as fixed in attribution-plan."""
import datetime
import math
import numpy as np
from research.general_rsi_transfer_data_v1 import ROOT,P,D,read,put,sha
from research.general_rsi_transfer_token_attribution_v1 import OUT,NAMES

def main():
    cfg=read(OUT/'attribution-plan.json');dest=OUT/'attribution-attempt-01'
    assert read(dest/'complete.json')['slots']==[4,5,6,7,8,9]
    dataset=read(P/'dataset.json');labels=np.load(OUT/'target-labels.npz')['labels'];decision=read(P/'decision.json')
    for f,h in cfg['inputs_sha256'].items():assert sha(ROOT/f)==h
    arrays={};receipts={};bindings={}
    for slot in range(4,10):
        f=dest/f'slot-{slot:02d}.json';r=read(f);path=dest/f'slot-{slot:02d}-losses.npz'
        assert r['worker_sha256']==cfg['worker_sha256'] and r['plan_sha256']==sha(OUT/'attribution-plan.json') and r['losses_sha256']==sha(path)
        a=np.load(path)['losses'];assert a.shape==labels.shape==(64,3200) and a.dtype==np.dtype('<f4') and np.isfinite(a).all() and (a>=0).all()
        assert r['max_original_parent_ce_abs_delta']<1e-6
        arrays[slot]=a.astype('f8');receipts[slot]=r;bindings[str(f.relative_to(ROOT))]=sha(f)
        for domain in ['pg19_books','python_library_rst']:
            rows=np.array([i for i,d in enumerate(dataset['documents']) if d['source']==domain]);lab=labels[rows]
            for k,name in enumerate(NAMES):
                g=r['groups'][domain][name];assert g['count']==int((lab==k).sum()) and abs(math.fsum(arrays[slot][rows][lab==k].tolist())-g['loss_sum'])<1e-8
    results={}
    delta=np.mean([arrays[n]-arrays[b] for n,b in [(4,5),(6,7),(8,9)]],axis=0)
    for domain in ['pg19_books','python_library_rst']:
        rows=np.array([i for i,d in enumerate(dataset['documents']) if d['source']==domain]);lab=labels[rows];groups={}
        for k,name in enumerate(NAMES):
            count=int((lab==k).sum());arm_means={arm:float(np.mean([receipts[s]['groups'][domain][name]['loss_sum'] for s in slots])) for arm,slots in [('native',[4,6,8]),('raw_bytes',[5,7,9])]}
            diff=arm_means['native']-arm_means['raw_bytes']
            groups[name]=dict(count=count,target_fraction=count/102400,mean_ce={a:(v/count if count else None) for a,v in arm_means.items()},native_minus_byte_category_ce=diff/count if count else None,contribution_to_domain_delta=diff/102400,pair_contributions=[(receipts[n]['groups'][domain][name]['loss_sum']-receipts[b]['groups'][domain][name]['loss_sum'])/102400 for n,b in [(4,5),(6,7),(8,9)]])
        summed=math.fsum(g['contribution_to_domain_delta'] for g in groups.values());old=decision['directions']['native']['domains'][domain]['mean_delta']
        assert sum(g['count'] for g in groups.values())==102400 and abs(summed-old)<1e-6
        results[domain]=dict(original_domain_delta=old,summed_contributions=summed,categories=groups)
    docs=[]
    for i,d in enumerate(dataset['documents']):
        parts={name:dict(count=int((labels[i]==k).sum()),contribution_to_parent_delta=math.fsum(delta[i][labels[i]==k].tolist())/3200) for k,name in enumerate(NAMES)}
        docs.append(dict(document=i,source=d['source'],name=d['name'],categories=parts,total_delta=math.fsum(x['contribution_to_parent_delta'] for x in parts.values())))
    audit=read(OUT/'composer-collision-audit-v2.json')
    assert audit['dataset_sha256']==sha(P/'dataset.json')
    collision_ids=[t['token'] for g in audit['groups'] for t in g['tokens']]
    values=np.fromfile(D/'excerpts.bin',dtype='<u2')
    inputs=np.stack([values[r['offset']:r['offset']+3200] for r in dataset['documents']])
    # A target sees only its causal input prefix inside its original128-token chunk.
    collision_context=np.cumsum(np.isin(inputs,collision_ids).reshape(64,25,128),axis=2).reshape(64,3200)>0
    context_summary={}
    for domain in ['pg19_books','python_library_rst']:
        rows=np.array([i for i,d in enumerate(dataset['documents']) if d['source']==domain]);mask=collision_context[rows]
        context_summary[domain]={}
        for name,m in [('collision_in_causal_chunk_prefix',mask),('no_collision_in_causal_chunk_prefix',~mask)]:
            count=int(m.sum());total=math.fsum(delta[rows][m].tolist())
            context_summary[domain][name]=dict(count=count,contribution_to_domain_delta=total/102400,mean_delta=total/count if count else None)
    guard=read(OUT/'attribution-guard-01.json');assert guard['status']=='finished' 
    put(OUT/'attribution-summary.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),analysis_sha256=sha(__file__),plan_sha256=sha(OUT/'attribution-plan.json'),input_receipts_sha256=bindings,results=results,documents=docs,posthoc_collision_context=context_summary,collision_audit_sha256=sha(OUT/'composer-collision-audit-v2.json'),max_original_parent_ce_abs_delta=max(r['max_original_parent_ce_abs_delta'] for r in receipts.values()),guard=guard,scope='Post-result descriptive loss partition on the same inputs and checkpoints. All targets retained. Other is not prose-only; mixed/straddling tokens enter other. Target localization is not a causal input-architecture or optimization test. No new adoption or significance claim.',test_opened=False))
    print(results)
    print('Posthoc collision context',context_summary)
if __name__=='__main__':main()
