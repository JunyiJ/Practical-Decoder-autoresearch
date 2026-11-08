"""Frozen two-contrast conditional-text inference; no candidate selection."""
import numpy as np
from .contract import DESIGN,gates

def calculate(pairs,seed):
 assert len(pairs)==3
 counts=np.array([2048]*16+[1510],dtype=np.float64);arrays={};totals={};target=pairs[0]['native']['target']
 for arm in ('native','raw_bytes','nonlexical_codes'):
  rows=[p[arm] for p in pairs]
  assert all(r['target']==target and len(r['block_loss_sums'])==17 for r in rows)
  arrays[arm]=np.asarray([r['block_loss_sums'] for r in rows],dtype=np.float64)
  totals[arm]=np.asarray([r['ce'] for r in rows],dtype=np.float64)
  assert np.isfinite(arrays[arm]).all() and (arrays[arm]>=0).all()
 assert target['targets']==34278 and target['block_counts']==counts.tolist() and not target['holdout_scored']
 result={}
 for control,threshold in [('native',-.02),('nonlexical_codes',0.)]:
  delta=totals['raw_bytes']-totals[control];differences=arrays['raw_bytes']-arrays[control];bounds=[]
  for width in (2048,4096,8192):
   group=width//2048;gc=np.asarray([counts[i:i+group].sum() for i in range(0,17,group)])
   grouped=np.stack([differences[:,i:i+group].sum(axis=1) for i in range(0,17,group)],axis=1)
   draws=np.random.Generator(np.random.PCG64(seed)).integers(0,len(gc),(2000,len(gc)))
   values=(grouped[:,draws].sum(axis=2)/gc[draws].sum(axis=1)).mean(axis=0)
   upper=float(np.quantile(values,1-.05/6,method='linear'))
   # Separate scalar aggregation/linear-quantile implementation on all draws.
   scalar=sorted(sum(sum(float(grouped[s,k]) for k in draw)/sum(float(gc[k]) for k in draw) for s in range(3))/3 for draw in draws)
   pos=(len(scalar)-1)*(1-.05/6);lo=int(pos);hi=min(lo+1,len(scalar)-1)
   independent=scalar[lo]+(pos-lo)*(scalar[hi]-scalar[lo]);assert abs(independent-upper)<1e-12
   bounds.append(dict(block_tokens=width,upper=upper,independent_upper=independent))
  gate=gates(delta.tolist(),[b['upper'] for b in bounds],threshold)
  result[control]=dict(deltas=delta.tolist(),mean_delta=float(delta.mean()),threshold=threshold,bounds=bounds,gates=gate,passes=all(gate.values()))
 return dict(contrasts=result,primary_passes=result['native']['passes'],lexical_content_supported=result['nonlexical_codes']['passes'],
  adopted=False,independent_final_acceptance_required=True,design=DESIGN,bootstrap_seed=seed,
  interpretation=DESIGN['uncertainty'])
