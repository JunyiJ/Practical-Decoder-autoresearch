"""Separate explicit per-draw arithmetic,grouping and linear quantile verification."""
import math
import numpy as np
from research.body_allocation_model_v1.contract import require
from research.gpt2_prefix_statistics_v1.statistics import report_contract

def verify(panel,seeds,bootstrap,primary):
 require(type(panel) is list and len(panel)==3 and [p['seed'] for p in panel]==seeds and len(set(seeds))==3,'All original fresh pairs required')
 for pair in panel:
  for arm in ('native','prefix8192'):report_contract(pair[arm],arm)
 deltas=[(p['prefix8192']['total_nats']-p['native']['total_nats'])/(111475*math.log(2)) for p in panel];mean=math.fsum(deltas)/3;upper={}
 for width in (4096,8192,16384):
  counts=[min(width,111475-i) for i in range(0,111475,width)];rng=np.random.Generator(np.random.PCG64(bootstrap));samples=[]
  for _ in range(2000):
   draw=rng.integers(0,len(counts),len(counts));denom=sum(counts[int(i)] for i in draw);values=[]
   for pair in panel:
    a=pair['native']['raw_byte_blocks'][str(width)];b=pair['prefix8192']['raw_byte_blocks'][str(width)];require(a['byte_counts']==b['byte_counts']==counts,'Actual common-byte grouping differs')
    values.append(math.fsum(b['loss_sums'][int(i)]-a['loss_sums'][int(i)] for i in draw)/(denom*math.log(2)))
   samples.append(math.fsum(values)/3)
  ordered=sorted(samples);position=(1-.05/3)*(len(ordered)-1);lo=math.floor(position);hi=math.ceil(position);fraction=position-lo
  upper[str(width)]=ordered[lo]*(1-fraction)+ordered[hi]*fraction
  require(abs(upper[str(width)]-primary['intervals'][str(width)]['upper'])<1e-12,'Separate bootstrap/linear quantile differs')
 require(max(abs(a-b) for a,b in zip(deltas,primary['candidate_minus_native_bpb']))<1e-12 and abs(mean-primary['mean_bpb_delta'])<1e-12,'Separate paired mean differs')
 gates=dict(mean_bpb_below_minus_point01=mean<-.01,every_fresh_delta_negative=all(v<0 for v in deltas),all_simultaneous_conditional_text_upper_bounds_negative=all(v<0 for v in upper.values()))
 require(gates==primary['gates'] and all(gates.values())==primary['quality_gates_pass'],'Independent strict decisions differ')
 return dict(status='independent_complete_panel_statistics_agreed',deltas=deltas,mean=mean,upper_bounds=upper,gates=gates,quality_gates_pass=all(gates.values()),atol=1e-12,adopted=False,
  scope='Conditional trained-seed/reused-validation-text inference only;not holdout or population-seed confidence. Separate adoption/consumer integration review required.')
