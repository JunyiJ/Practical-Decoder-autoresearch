"""Descriptive cohort ratio instability from closed scalar records only."""
import hashlib,json,math,statistics
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT,require
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.tokenizer_moe_worker_v1.execution import idle
E=Path(__file__).resolve().parent
def ref(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def checked(r):
    p=Path(r['path']);require(ref(p)=={k:r[k] for k in ('path','sha256')},'Closed metadata reference changed');return json.loads(p.read_text())
def correlation(x,y):
    a=statistics.mean(x);b=statistics.mean(y);xx=math.fsum((v-a)**2 for v in x);yy=math.fsum((v-b)**2 for v in y)
    return math.fsum((v-a)*(w-b) for v,w in zip(x,y))/math.sqrt(xx*yy) if xx*yy>0 else None
def spread(x):
    return dict(min=min(x),max=max(x),max_over_min=max(x)/min(x) if min(x)>0 else None,descriptive_sample_cv=statistics.stdev(x)/statistics.mean(x) if statistics.mean(x)!=0 else None)

with session_lock(GLOBAL_LOCK.parent):
    idle();require(not(E/'existing-analysis002.json').exists() and not(E/'existing-analysis002-guard.json').exists(),'New closed-metadata analysis only;no replay')
    with CpuPrerequisiteGuard(E/'existing-analysis002-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
        review_ref=ref(E/'cohort-review001.json');review=checked(review_ref);stop_ref=review['terminal'];checked(stop_ref)
        result=checked(review['result']);plan=checked(review['plan']);require(review['status']=='complete_training_free_gradient_cohorts_independently_reviewed','Completed independent scalar review required')
        rows=[];norms=[];ces=[]
        for item in result['endpoints']:
            d=checked(item);ds=d['draws'];cohorts=d['cohorts'];combined=d['combined'];n=[math.fsum(x['gradient_squared_norms'].values()) for x in ds];ce=[x['ce'] for x in ds]
            require(len(n)==128 and len(cohorts)==4,'Complete existing records required')
            q=math.fsum(n);ordered=sorted(n,reverse=True);norms.append(n);ces.append(ce)
            cv=[c['statistics']['trace_sample_covariance'] for c in cohorts];sg=[c['statistics']['signal_squared_unbiased'] for c in cohorts];ratios=[c['statistics']['ratio_in_independent_B8_draw_units'] for c in cohorts]
            rows.append(dict(slot=d['slot'],covariance_spread=spread(cv),corrected_signal_spread=spread(sg),ratio_spread=spread(ratios) if all(v is not None for v in ratios) else None,undefined_cohort_ratios=sum(v is None for v in ratios),combined_signal=combined['signal_squared_unbiased'],combined_ratio=combined['ratio_in_independent_B8_draw_units'],perdraw_squared_norm=dict(mean=statistics.mean(n),median=statistics.median(n),max=max(n),top1_fraction=ordered[0]/q,top8_fraction=math.fsum(ordered[:8])/q),norm_ce_correlation=correlation(n,ce),cohort_mean_train_ce=[statistics.mean(ce[i*32:(i+1)*32]) for i in range(4)],scope='Descriptive finite shared-input cohorts at fixed endpoints;no confidence interval or causal attribution.'))
        pairs=[dict(slots=[rows[i]['slot'],rows[j]['slot']],shared_input_gradient_norm_correlation=correlation(norms[i],norms[j]),shared_input_train_ce_correlation=correlation(ces[i],ces[j])) for i,j in ((0,1),(0,2),(1,2))]
        for p,h in plan['sources'].items():require(ref(ROOT/p)['sha256']==h,'Frozen source changed')
        require(ref(E/'cohort-review001.json')==review_ref and ref(Path(stop_ref['path']))==stop_ref,'Closed evidence changed')
    out=dict(status='closed_complete_existing_cohort_scalar_analysis',source=ref(Path(__file__)),cohort_review=review_ref,guard=ref(E/'existing-analysis002-guard.json'),rows=rows,pairs=pairs,models_loaded=0,gradient_calls=0,optimizer_steps=0,validation_encoded=False,holdout_encoded=False,bootstrap_draws=0,quality_result=False,scope='Separate covariance from bias-corrected denominator variability using all existing cohorts. Correlation is descriptive under shared sampled inputs;not evidence of causal batch benefit. No old-probe pooling or diagnostic replay.')
    with (E/'existing-analysis002.json').open('x') as f:json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(rows=rows,pairs=pairs)),flush=True)
