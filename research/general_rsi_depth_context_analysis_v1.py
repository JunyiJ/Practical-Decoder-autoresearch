"""Apply three frozen context contrasts after all12execution reviews."""
import argparse,json,math
from pathlib import Path
import numpy as np
from research.general_rsi_training_v2 import sha
from research.general_rsi_depth_context_worker_v1 import put
ROOT=Path(__file__).resolve().parents[1]
PANEL=ROOT/'research/tokenizer-evidence/general-rsi-depth-context-quality-001'
def read(p):return json.loads(p.read_text())
def decisions(deltas,counts,sources,plan):
    a=np.asarray(deltas,dtype=np.float64);counts=np.asarray(counts,dtype=np.int64)
    assert a.shape==(2,3,len(counts)) and np.isfinite(a).all() and (counts>0).all()
    # Dimensions: size, paired seed, document. Interaction is formed BEFORE
    # bootstrap and uses exactly the same paired seed/document observations.
    contrasts=np.concatenate([a,(a[1]-a[0])[None]],axis=0)
    paired=contrasts.sum(axis=2)/counts.sum();means=paired.mean(axis=1)
    sums=contrasts.mean(axis=1);rng=np.random.default_rng(plan['bootstrap_seed']);draws=[]
    for first in range(0,plan['bootstrap_resamples'],1000):
        ids=rng.integers(0,len(counts),size=(min(1000,plan['bootstrap_resamples']-first),len(counts)))
        draws.append(sums[:,ids].sum(axis=2)/counts[ids].sum(axis=1)[None,:])
    draws=np.concatenate(draws,axis=1)
    bounds=np.quantile(draws,plan['primary_interval_quantiles'],axis=1,method='linear').T
    rows=[]
    for i,name in enumerate(plan['primary_contrasts']):
        row=dict(contrast=name,paired_deltas=paired[i].tolist(),mean=float(means[i]),seed_sd=float(paired[i].std(ddof=1)),seed_range=[float(paired[i].min()),float(paired[i].max())],interval=bounds[i].tolist())
        negative=bool((paired[i]<0).all() and means[i]<=-plan['practical_ce_threshold'] and bounds[i,1]<0)
        positive=bool((paired[i]>0).all() and means[i]>=plan['practical_ce_threshold'] and bounds[i,0]>0)
        if i<2:
            sd={s:float(contrasts[i,:,np.array(sources)==s].sum()/3/counts[np.array(sources)==s].sum()) for s in sorted(set(sources))}
            row['source_deltas']=sd
            row['decision']='context256_superior' if negative and max(sd.values())<=plan['source_regression_guard'] else 'context128_superior' if positive and min(sd.values())>=-plan['source_regression_guard'] else 'unresolved'
        else:row['decision']='context_benefit_strengthens' if negative else 'context_benefit_weakens' if positive else 'unresolved_not_equivalent'
        rows.append(row)
    return rows

def fixture():
    plan=dict(bootstrap_seed=7,bootstrap_resamples=2000,primary_interval_quantiles=[.01/6,1-.01/6],primary_contrasts=['delta4','delta16','delta16_minus_delta4'],practical_ce_threshold=.02,source_regression_guard=.05)
    counts=np.array([100,200]);sources=['fineweb_edu','wikipedia']
    a=np.array([[[-.08,-.08]]*3,[[-.03,-.03]]*3])*counts
    r=decisions(a,counts,sources,plan);assert [x['decision'] for x in r]==['context256_superior']*2+['context_benefit_weakens']
    r=decisions(-a,counts,sources,plan);assert [x['decision'] for x in r]==['context128_superior']*2+['context_benefit_strengthens']
    a[1]=a[0];assert decisions(a,counts,sources,plan)[2]['decision']=='unresolved_not_equivalent'
    a[0,0]*=-1;assert decisions(a,counts,sources,plan)[0]['decision']=='unresolved'
    a[:]=np.array([.06,-.2])*counts;assert decisions(a,counts,sources,plan)[0]['decision']=='unresolved'
    return dict(accepted=True,cases=['paired_interaction_not_separate_significance','symmetric_native_rule','zero_interaction_not_equivalence','one_unfavorable_seed_blocks','source_regression_blocks'],scientific_scores_used=0)

def analyze():
    plan=read(PANEL/'plan.json');binding=sha(PANEL/'plan.json')
    assert all(sha(ROOT/n)==h for n,h in plan['sources'].items())
    # Require every accepted slot before reading the scientific score panel.
    for i in range(12):
        p=PANEL/f'slot-{i:02d}';review=read(p/'review.json')
        assert review['accepted_execution'] and review['binding']==binding and review['result_sha256']==sha(p/'result.json')
    scores={};results={};evidence={}
    for s in plan['slots']:
        out=PANEL/f"slot-{s['slot']:02d}"
        key=(s['depth'],s['bundle'],s['arm']);results[key]=read(out/'result.json')
        scores[key]=read(out/'time_ema-validation.json')
        for n in ('result.json','review.json','time_ema-validation.json'):
            evidence[str((out/n).relative_to(ROOT))]=sha(out/n)
    inventory=[(x['document'],x['source'],x['count']) for x in next(iter(scores.values()))['documents']]
    assert len(inventory)==108 and sum(x[2] for x in inventory)==100416
    assert all([(x['document'],x['source'],x['count']) for x in v['documents']]==inventory for v in scores.values())
    a=[];runtime=[]
    for d in [4,16]:
        ds=[];ratios=[]
        for b in [1,2,3]:
            ds.append([y['loss_sum']-x['loss_sum'] for x,y in zip(scores[d,b,'context128']['documents'],scores[d,b,'context256']['documents'])])
            ratios.append(results[d,b,'context256']['steps']/results[d,b,'context128']['steps'])
        a.append(ds);runtime.append(ratios)
    outcomes=decisions(a,[x[2] for x in inventory],[x[1] for x in inventory],plan)
    ratio_change=np.asarray(runtime)[1]/np.asarray(runtime)[0]
    output=dict(binding=binding,contrasts=outcomes,context256_transfer_to_depth16=all(row['decision']=='context256_superior' for row in outcomes[:2]),sampled_token_ratio_context256_over128=runtime,token_ratio16_over_ratio4=ratio_change.tolist(),evidence_sha256=evidence,uncertainty='Document bootstrap conditional on these3bundles. Two-depth transfer, not scaling law, population-seed confidence or equivalence.',secondary_and_generation='Common128, RAW, curves and generation required separately; no primary rescue.',test_opened=False)
    put(PANEL/'decision.json',output)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fixture','analyze']);a=p.parse_args()
    if a.action=='fixture':print(json.dumps(fixture()))
    else:analyze()
