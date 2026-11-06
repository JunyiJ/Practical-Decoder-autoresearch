"""Two predeclared contrasts, simultaneous conditional-text bounds, no selection."""
import numpy as np
from research.body_allocation_model_v1.contract import require

def review(pairs,seed):
    require(type(pairs) is list and len(pairs)==3,'Exactly three fresh triplets required')
    names=('incumbent','shallow','deeper');counts=None;totals={n:[] for n in names};sums={n:[] for n in names}
    for pair in pairs:
        require(set(pair)==set(names),'All three arms required')
        for n in names:
            r=pair[n];c=r['target']['block_counts'];v=np.asarray(r['block_loss_sums'],dtype=np.float64)
            require(c==([2048]*16+[1510]) and len(v)==17 and np.isfinite(v).all() and (v>=0).all(),'Complete finite target blocks required')
            if counts is None:counts=np.asarray(c,dtype=np.float64)
            require(r['target']==pair['incumbent']['target'],'Target identities differ across arms')
            require(abs(float(v.sum())-r['total_loss'])<1e-8 and r['ce']==r['total_loss']/34278,'Loss arithmetic differs')
            sums[n].append(v);totals[n].append(r['ce'])
    arrays={n:np.asarray(v) for n,v in sums.items()};result={}
    for control in ('incumbent','shallow'):
        delta=np.asarray(totals['deeper'])-np.asarray(totals[control]);bounds=[]
        for width in (2048,4096,8192):
            group=width//2048;gc=np.asarray([counts[i:i+group].sum() for i in range(0,17,group)])
            differences=arrays['deeper']-arrays[control]
            grouped=np.stack([differences[:,i:i+group].sum(axis=1) for i in range(0,17,group)],axis=1)
            draws=np.random.Generator(np.random.PCG64(seed)).integers(0,len(gc),(2000,len(gc)))
            values=(grouped[:,draws].sum(axis=2)/gc[draws].sum(axis=1)).mean(axis=0)
            bounds.append(dict(block_tokens=width,upper=float(np.quantile(values,1-.05/6,method='linear'))))
        mean=float(delta.mean());gates=dict(mean_below_minus_point02=mean<-.02,every_seed_negative=bool((delta<0).all()),all_simultaneous_bounds_negative=all(b['upper']<0 for b in bounds))
        result[control]=dict(deltas=delta.tolist(),mean_delta=mean,conditional_text_upper_bounds=bounds,gates=gates,passes=all(gates.values()))
    return dict(contrasts=result,all_strict_gates_pass=all(r['passes'] for r in result.values()),candidate='deeper',
        adoption_requires_separate_review=True,bootstrap_seed=seed,draws=2000,upper_quantile=1-.05/6,
        interpretation='Conditional reused-text intervals given three trained seeds; not population-seed uncertainty or holdout confidence. No shallow fallback or cost-only adoption.')
