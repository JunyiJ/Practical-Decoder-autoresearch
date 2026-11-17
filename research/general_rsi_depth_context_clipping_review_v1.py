"""Describe retained clipping coefficients; never infer or replay optimizer updates."""
import fcntl,json,math,resource,shutil,time
from pathlib import Path
import numpy as np
from research.general_rsi_depth_context_position_v1 import ROOT,P,read,sha
OUT=ROOT/'research/tokenizer-evidence/general-rsi-depth-context-clipping-review-001'
SOURCES=['research/general_rsi_equal_time_training_v1.py','research/general_rsi_depth_context_training_v1.py','research/general_rsi_depth_context_v1/optimizer.py','research/gradient_clip_model_v1/clipping.py','.venv/lib/python3.13/site-packages/torch/optim/_muon.py']
HISTORY=['research/tokenizer-evidence/gradient-clip-preparation-001/quality002-final-review.json','research/tokenizer-evidence/gradient-clip-preparation-001/quality002-closure.json','research/tokenizer-evidence/general-rsi-update-response-001/analysis.json','research/tokenizer-evidence/general-rsi-update-response-001/completion-review.json','research/tokenizer-evidence/general-rsi-momentum-precision-001/summary.json']
def main():
 start=time.perf_counter();plan=read(P/'plan.json');review=read(P/'completion-review.json');rows=[];hashes={};maxerror=0.;total=0
 assert shutil.disk_usage(ROOT).free>=12*2**30+16*2**20
 with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for name in SOURCES:
   h=sha(ROOT/name);assert h==plan['sources'][name];hashes[name]=h
  for name in HISTORY:hashes[name]=sha(ROOT/name)
  for s in plan['slots']:
   path=P/f"slot-{s['slot']:02d}/steps-01.jsonl";name=str(path.relative_to(ROOT));hashes[name]=sha(path);assert hashes[name]==review['evidence_sha256'][name]
   bins={i:[] for i in range(4)}
   with path.open() as f:
    for line in f:
     x=json.loads(line);c=x['clip'];total+=1
     assert c['threshold']==1.0 and 0<c['coefficient']<=1 and math.isfinite(c['pre_norm'])
     maxerror=max(maxerror,abs(c['coefficient']-min(1.,1./(c['pre_norm']+1e-6))))
     phase=min(2,x['begin_ns']//4800000000000);bins[phase].append((c['coefficient'],c['pre_norm']))
     if x['begin_ns']>=14280000000000:bins[3].append((c['coefficient'],c['pre_norm']))
   for i,values in bins.items():
    v=np.asarray(values);coef=v[:,0];change=np.abs(np.diff(coef))
    rows.append(dict(slot=s['slot'],depth=s['depth'],context=s['context'],bundle=s['bundle'],phase=['first_third','middle_third','final_third','final_120s'][i],updates=len(v),clipped_fraction=float((coef<1).mean()),coefficient_mean=float(coef.mean()),coefficient_sd=float(coef.std()),coefficient_min=float(coef.min()),coefficient_quantiles_01_50_99=np.quantile(coef,[.01,.5,.99]).tolist(),pre_norm_quantiles_01_50_99=np.quantile(v[:,1],[.01,.5,.99]).tolist(),adjacent_coefficient_abs_change_mean=float(change.mean()),adjacent_coefficient_abs_change_max=float(change.max())))
   assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*2**30 and time.perf_counter()-start<120
  assert total==384968 and maxerror<1e-6
  result=dict(rows=rows,updates=total,coefficient_formula_max_abs_error=maxerror,source_and_input_hashes=hashes,analysis_source_sha256=sha(Path(__file__)),elapsed_seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,free_bytes=shutil.disk_usage(ROOT).free,limits='Descriptive scalar histories; no per-parameter gradient direction, optimizer moments or counterfactual update measured. No quality intervention admitted.',test_opened=False,new_optimizer_updates=0)
  with (OUT/'analysis.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
  for d in [4,16]:
   for c in [128,256]:
    q=[r for r in rows if r['depth']==d and r['context']==c and r['phase']=='final_third']
    print(d,c,'clipped',[round(r['clipped_fraction'],4) for r in q],'coef_mean',[round(r['coefficient_mean'],4) for r in q],'coef_sd',[round(r['coefficient_sd'],4) for r in q],'coef_min',[round(r['coefficient_min'],4) for r in q])
  print('resources',result['elapsed_seconds'],result['peak_rss_bytes'],'formula_error',maxerror)
if __name__=='__main__':main()
