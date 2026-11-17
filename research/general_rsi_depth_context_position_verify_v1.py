"""Independent ordered-vector check of retained token-loss position summaries."""
import json,math,time,resource,fcntl
from pathlib import Path
import numpy as np
from research.general_rsi_depth_context_position_v1 import ROOT,P,OUT,read,sha

def main():
 start=time.perf_counter()
 with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  a=read(OUT/'analysis.json');complete=read(P/'completion-review.json');plan=read(P/'plan.json')
  for path,h in a['evidence_sha256'].items():assert h==complete['evidence_sha256'][path] and sha(ROOT/path)==h
  slots={(s['depth'],s['bundle'],s['context']):s['slot'] for s in plan['slots']}
  def vec(path):
   x=read(path);loss=[];ids=[];docs=[];positions=[]
   for c in x['chunks']:
    loss.extend(c['token_losses']);ids.extend(c['target_ids']);docs.extend([c['document']]*c['targets']);positions.extend(range(c['target_start'],c['target_start']+c['targets']))
   return np.array(loss),ids,np.array(docs),positions
  maxerror=0.;shares=[]
  for row in a['rows']:
   d,b,e=row['depth'],row['bundle'],row['endpoint'];p256=P/f'slot-{slots[d,b,256]:02d}';p128=P/f'slot-{slots[d,b,128]:02d}'
   n,ids,docs,positions=vec(p256/f'{e}-validation.json');c,ids2,docs2,pos2=vec(p256/f'{e}-common128-validation.json');r,ids3,docs3,pos3=vec(p128/f'{e}-validation.json')
   assert ids==ids2==ids3 and positions==pos2==pos3 and np.array_equal(docs,docs2) and np.array_equal(docs,docs3)
   ordinal=np.concatenate([np.arange(np.count_nonzero(docs==i)) for i in range(108)])%256
   masks={'shared_1_128':ordinal<128,'reset_129_160':(ordinal>=128)&(ordinal<160),'middle_161_192':(ordinal>=160)&(ordinal<192),'late_193_256':ordinal>=192,'remaining96':ordinal>=160}
   for label,mask in masks.items():
    expected=row['bins'][label];assert mask.sum()==expected['count']
    for k,delta in [('evaluation_context_delta',n-c),('common128_weight_delta',c-r),('native_delta',n-r)]:
     err=abs(float(delta[mask].mean())-expected[k]);maxerror=max(maxerror,err);assert err<1e-12
   assert np.array_equal(n[masks['shared_1_128']],c[masks['shared_1_128']])
   shares.append(dict(depth=d,bundle=b,endpoint=e,first32_share_of_evaluation_gain=float((n-c)[masks['reset_129_160']].sum()/(n-c).sum()),first32_fraction_of_all_targets=float(masks['reset_129_160'].mean())))
  groups=[]
  for g in a['groups']:
   q=g['bins']['reset_129_160'];groups.append(dict(depth=g['depth'],endpoint=g['endpoint'],first32_share_of_mean_evaluation_gain=q['count']*q['evaluation_context_delta']/(100416*g['total']['evaluation_context_delta']),same_weight_evaluation_share_of_native_delta=g['total']['evaluation_context_delta']/g['total']['native_delta']))
  out=dict(accepted=True,input_hashes_match_completed_panel=True,input_files=len(a['evidence_sha256']),independent_ordered_vector_max_abs_error=maxerror,shared_prefix_token_losses_exact=True,all_target_ids_and_global_positions_match=True,descriptive_shares=shares,group_shares=groups,analysis_sha256=sha(OUT/'analysis.json'),source_sha256=sha(Path(__file__)),elapsed_seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,test_opened=False)
  assert out['peak_rss_bytes']<4*2**30
  with (OUT/'verification.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
  print(json.dumps({k:v for k,v in out.items() if k!='descriptive_shares'},indent=2))
if __name__=='__main__':main()
