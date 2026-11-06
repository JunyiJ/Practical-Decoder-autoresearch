"""Predeclared two-contrast native-token inference; no quality-selected panels."""
import math
import numpy as np
from research.body_allocation_model_v1.contract import require,fingerprint
from research.tokenizer_moe_worker_v1.scoring import validate_report
ARMS=('dense','wide','moe');CONTROLS=ARMS[:2];LRS=(.000625,.00125,.0025);WIDTHS=(2048,4096,8192)
N=34278;DRAWS=2000;QUANTILE=1-.05/6
TARGET_SHA='a826853819874af37dab506709d45d834f01dd8bd4351e703c0e667c237b1498'

def report_contract(report,arm):
    require(arm in ARMS and fingerprint(report['target'])==TARGET_SHA,'Exact original native target metadata required')
    validate_report(report,report['target'])
    windows=report['window_loss_sums']
    for i,value in enumerate(report['block_loss_sums']):
        require(abs(math.fsum(windows[16*i:16*(i+1)])-value)<1e-8,'Native window/block loss conservation differs')
    return report

def select(records):
    require(type(records) is list and len(records)==9,'All nine equal-allowance LR screens required');observed={}
    for row in records:
        require(row['status']=='accepted' and row['arm'] in ARMS and type(row['peak_lr']) is float and row['peak_lr'] in LRS,'Accepted finite screen required')
        key=(row['arm'],row['peak_lr']);require(key not in observed,'Duplicate LR screen rejected');observed[key]=report_contract(row['h200'],row['arm'])['ce']
    require(set(observed)=={(a,r) for a in ARMS for r in LRS},'Exact equal finite tuning set required')
    return {arm:min(LRS,key=lambda lr:(observed[arm,lr],lr)) for arm in ARMS}

def groups(report,width):
    require(width in WIDTHS,'Predeclared token block width required')
    stride=width//2048;small=report['block_loss_sums']
    return [math.fsum(small[i:i+stride]) for i in range(0,len(small),stride)]

def review(panel,bootstrap_seed):
    require(type(panel) is list and len(panel)==3 and all(type(p) is dict and set(p)=={'seed',*ARMS} and type(p['seed']) is int and p['seed']>=0 for p in panel),'Exactly three complete fresh triples required')
    require(len({p['seed'] for p in panel})==3 and type(bootstrap_seed) is int and bootstrap_seed>=0 and bootstrap_seed not in {p['seed'] for p in panel},'Distinct explicit frozen seed roles required')
    for p in panel:
        for a in ARMS:report_contract(p[a],a)
    contrasts={}
    for a in CONTROLS:
        ds=[(p['moe']['total_loss']-p[a]['total_loss'])/N for p in panel]
        contrasts[a]=dict(candidate_minus_control_ce=ds,mean_ce_delta=math.fsum(ds)/3,intervals={})
    for width in WIDTHS:
        counts=np.asarray([min(width,N-i) for i in range(0,N,width)],dtype=np.int64)
        rng=np.random.Generator(np.random.PCG64(bootstrap_seed));indices=rng.integers(0,len(counts),size=(DRAWS,len(counts)))
        for a in CONTROLS:
            difference=np.asarray([np.asarray(groups(p['moe'],width))-np.asarray(groups(p[a],width)) for p in panel],dtype=np.float64)
            per_draw=difference.mean(axis=0)[indices].sum(axis=1)/counts[indices].sum(axis=1)
            require(np.isfinite(per_draw).all(),'Finite complete bootstrap required')
            contrasts[a]['intervals'][str(width)]=dict(upper=float(np.quantile(per_draw,QUANTILE,method='linear')),draws=DRAWS,quantile=QUANTILE,token_counts=counts.tolist(),draw_seed=bootstrap_seed,pcg64_reinitialized_each_width=True,same_indices_across_seeds_and_contrasts=True)
    gates=dict(both_mean_ce_below_minus_point02=all(c['mean_ce_delta']<-.02 for c in contrasts.values()),every_paired_seed_delta_negative=all(d<0 for c in contrasts.values() for d in c['candidate_minus_control_ce']),all_six_simultaneous_conditional_text_upper_bounds_negative=all(v['upper']<0 for c in contrasts.values() for v in c['intervals'].values()))
    return dict(contrasts=contrasts,practical_threshold=-.02,gates=gates,quality_gates_pass=all(gates.values()),scope='Conditional on trained seeds and reused34278native validation targets;not holdout or population-seed confidence. Both controls required;no wide fallback or cost-only adoption. Separate adoption/consumer review required.')
