"""One predeclared native/8K contrast,not per-token CE or population-seed inference."""
import math
import numpy as np
from research.body_allocation_model_v1.contract import require,fingerprint
from research.gpt2_prefix_worker_v1.scoring import validate_report
ARMS=('native','prefix8192');LRS=(.000625,.00125,.0025);WIDTHS=(4096,8192,16384)
TARGET_BYTES=111475;TARGET_SHA='ad7da8cdcacc24b5312461c23f2d00aead59d544fcc2fad4fb2de40da4964fe8'
PRACTICAL_BPB=-.01;DRAWS=2000;QUANTILE=1-.05/3
# Actual target-policy metadata only,no model losses or selection data.
TARGET_CONTRACT_SHA={'native': '75c803553c9066de07fd6d92ffdb105fe852e3e8ed88e5102d5c1a155ec2419d', 'prefix8192': 'e254ccad95134430341d581f282ce9c55420836f7e148b63d724d21992e8e40c'}

def report_contract(report,arm):
    require(arm in ARMS,'Declared native/8K arm required');target=report['target'];tokens={'native':34257,'prefix8192':39963}[arm]
    require(target['target_sha256']==TARGET_SHA and target['target_bytes']==TARGET_BYTES and target['target_tokens']==tokens,'Original common byte and own token target coverage required')
    require(fingerprint(target)==TARGET_CONTRACT_SHA[arm],'Exact codec/boundary/context target metadata required')
    validate_report(report,target,None);return report

def select(records):
    require(type(records) is list and len(records)==6,'All six equal-allowance LR screens required');observed={}
    for row in records:
        require(row['status']=='accepted' and row['arm'] in ARMS and type(row['peak_lr']) is float and row['peak_lr'] in LRS,'Accepted finite screen required')
        key=(row['arm'],row['peak_lr']);require(key not in observed,'Duplicate LR screen rejected');observed[key]=report_contract(row['h200'],row['arm'])['bits_per_byte']
    require(set(observed)=={(a,r) for a in ARMS for r in LRS},'Exact equal finite tuning set required')
    return {arm:min(LRS,key=lambda lr:(observed[arm,lr],lr)) for arm in ARMS}

def review(panel,bootstrap_seed):
    require(type(panel) is list and len(panel)==3 and len({p['seed'] for p in panel})==3,'Exactly three distinct fresh paired seeds required')
    require(all(type(p['seed']) is int and p['seed']>=0 and set(p)=={'seed',*ARMS} for p in panel),'Exact paired panel inventory required')
    require(type(bootstrap_seed) is int and bootstrap_seed>=0,'Explicit frozen bootstrap seed required')
    for p in panel:
        for a in ARMS:report_contract(p[a],a)
    deltas=[(p['prefix8192']['total_nats']-p['native']['total_nats'])/(math.log(2)*TARGET_BYTES) for p in panel];mean=math.fsum(deltas)/3;intervals={}
    for width in WIDTHS:
        counts=np.array([min(width,TARGET_BYTES-i) for i in range(0,TARGET_BYTES,width)],dtype=np.int64)
        difference=[]
        for p in panel:
            a=p['native']['raw_byte_blocks'][str(width)];b=p['prefix8192']['raw_byte_blocks'][str(width)]
            require(a['byte_counts']==b['byte_counts']==counts.tolist(),'Same raw-byte block counts required')
            difference.append(np.asarray(b['loss_sums'],dtype=np.float64)-np.asarray(a['loss_sums'],dtype=np.float64))
        rng=np.random.Generator(np.random.PCG64(bootstrap_seed));indices=rng.integers(0,len(counts),size=(DRAWS,len(counts)))
        per_draw=np.asarray(difference).mean(axis=0)[indices].sum(axis=1)/counts[indices].sum(axis=1)/math.log(2)
        require(np.isfinite(per_draw).all(),'Finite complete bootstrap required');upper=float(np.quantile(per_draw,QUANTILE,method='linear'))
        intervals[str(width)]=dict(upper=upper,draws=DRAWS,quantile=QUANTILE,blocks=len(counts),raw_byte_counts=counts.tolist(),draw_seed=bootstrap_seed,pcg64_reinitialized_each_width=True)
    gates=dict(mean_bpb_below_minus_point01=mean<PRACTICAL_BPB,every_fresh_delta_negative=all(d<0 for d in deltas),all_simultaneous_conditional_text_upper_bounds_negative=all(r['upper']<0 for r in intervals.values()))
    return dict(candidate_minus_native_bpb=deltas,mean_bpb_delta=mean,practical_threshold=PRACTICAL_BPB,intervals=intervals,gates=gates,quality_gates_pass=all(gates.values()),
        scope='Conditional on these trained seeds and reused111475validation bytes;not holdout or population-seed confidence. Deterministic token coding likelihood;uniform-within-token raw-byte loss allocation is bookkeeping. No cost-only adoption;separate adoption/serving-consumer review required.')
