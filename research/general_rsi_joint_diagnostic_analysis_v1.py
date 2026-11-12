"""Frozen descriptive contrasts; no adoption or best-horizon selection."""
import json,math,statistics
from pathlib import Path
from research.general_rsi_pseudo_update_diagnostic_v1 import sha,put
from research.general_rsi_joint_diagnostic_v1 import OUT,ARMS
METRICS=['ce','penalty','mean_probability','mean_previous_probability','probability_ratio']
CONTRASTS=[('joint_minus_full','ce_pseudo_mean','ce_full'),('joint_minus_half','ce_pseudo_mean','ce_half'),('half_minus_full','ce_half','ce_full')]
def flat(p):return dict(ce=p['scores']['natural']['ce'],**{k:p['scores']['repetition'][k] for k in METRICS if k!='ce'})
def main():
    plan=json.loads((OUT/'plan.json').read_text());A=OUT/'attempt-01';result=json.loads((A/'result.json').read_text());assert result['plan_sha256']==sha(OUT/'plan.json')
    rows=[];clipping=[];endpoints=[]
    for c in plan['parents']:
        slot=c['parent']['slot'];initial=json.loads((A/f'p{slot:02d}-initial.json').read_text())
        for n in plan['probes']:
            v={arm:flat(json.loads((A/f'p{slot:02d}-{arm}-n{n:03d}.json').read_text())) for arm in ARMS}
            for name,a,b in CONTRASTS:rows.append(dict(parent_slot=slot,n=n,contrast=name,delta={k:v[a][k]-v[b][k] for k in METRICS}))
            if n==100:
                endpoints.append(dict(parent_slot=slot,initial=flat(initial),arms=v,
                    ce_drift={arm:v[arm]['ce']-initial['scores']['natural']['ce'] for arm in ARMS}))
        for arm in ARMS:
            history=[json.loads(l) for l in (A/f'p{slot:02d}-{arm}-steps.jsonl').read_text().splitlines()];assert len(history)==100
            clipping.append(dict(parent_slot=slot,arm=arm,updates=100,clipped=sum(r['clip']['coefficient']<1 for r in history),
                median_pre_norm=statistics.median(r['clip']['pre_norm'] for r in history),mean_ce_training_loss=statistics.mean(r['ce_loss'] for r in history),
                saturated_targets=sum(r['saturated_targets'] for r in history)))
    pooled=[]
    for n in plan['probes']:
        for name,_,_ in CONTRASTS:
            selected=[r for r in rows if r['n']==n and r['contrast']==name];assert len(selected)==3
            pooled.append(dict(n=n,contrast=name,mean={k:statistics.mean(r['delta'][k] for r in selected) for k in METRICS},
                negative_parents={k:sum(r['delta'][k]<0 for r in selected) for k in METRICS}))
    required=['ce','penalty','mean_probability','probability_ratio']
    directional=all(r['delta'][k]<0 for r in rows if r['n']==100 and r['contrast'] in ['joint_minus_full','joint_minus_half'] for k in required)
    out=dict(plan_sha256=sha(OUT/'plan.json'),result_sha256=sha(A/'result.json'),paired_curves=rows,pooled_curves=pooled,
        endpoints=endpoints,clipping=clipping,directional_mechanism_support=directional,
        criterion='At n100 only, joint minus both CE controls negative for CE, pseudo penalty, repeated probability and aggregate repeat/previous probability ratio in each parent.',
        interpretation='Descriptive short-horizon training-split mechanism diagnostic. No quality adoption, significance claim, independent holdout or equal-time comparison.',
        test_opened=False,quality_claim=False)
    put(OUT/'analysis.json',out)
    print(json.dumps(dict(directional_support=directional,endpoints=[r for r in pooled if r['n']==100]),indent=2))
if __name__=='__main__':main()
