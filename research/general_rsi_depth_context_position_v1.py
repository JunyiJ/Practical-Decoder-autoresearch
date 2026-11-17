"""Saved-loss decomposition by supplied-history position; no model/data execution."""
import datetime,fcntl,hashlib,json,math,os,resource,shutil,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'research/tokenizer-evidence/general-rsi-depth-context-quality-001'
OUT=ROOT/'research/tokenizer-evidence/general-rsi-depth-context-position-001'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 start=time.perf_counter();stop=threading.Event()
 def guard():
  while not stop.wait(.2):
   if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=4*2**30 or shutil.disk_usage(ROOT).free<12*2**30 or time.perf_counter()-start>300:os._exit(75)
 threading.Thread(target=guard,daemon=True).start()
 assert shutil.disk_usage(ROOT).free>=12*2**30+64*2**20
 with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  plan=read(P/'plan.json');review=read(P/'completion-review.json');assert review['accepted'] and review['binding']==sha(P/'plan.json')
  evidence={};rows=[]
  def flat(path):
   evidence[str(path.relative_to(ROOT))]=sha(path);a=read(path);r={}
   for c in a['chunks']:
    assert len(c['token_losses'])==len(c['target_ids'])==c['targets']
    for j,(token,loss) in enumerate(zip(c['target_ids'],c['token_losses'])):
     key=(c['document'],c['target_start']+j);assert key not in r and math.isfinite(loss)
     r[key]=(token,loss,c['source'])
   assert len(r)==100416
   assert abs(math.fsum(x[1] for x in r.values())/len(r)-a['ce'])<1e-10
   return a['ce'],r
  slots={(s['depth'],s['bundle'],s['context']):s['slot'] for s in plan['slots']}
  for depth in [4,16]:
   for bundle in [1,2,3]:
    for endpoint in ['raw','time_ema']:
     nc,n=flat(P/f'slot-{slots[depth,bundle,256]:02d}'/f'{endpoint}-validation.json')
     cc,c=flat(P/f'slot-{slots[depth,bundle,256]:02d}'/f'{endpoint}-common128-validation.json')
     rc,r=flat(P/f'slot-{slots[depth,bundle,128]:02d}'/f'{endpoint}-validation.json')
     assert n.keys()==c.keys()==r.keys()
     first={}
     for doc,pos in n:first[doc]=min(first.get(doc,pos),pos)
     bins={k:[] for k in ['shared_1_128','reset_129_160','middle_161_192','late_193_256']};bydoc={};bysource={}
     for key,(token,loss,source) in n.items():
      assert token==c[key][0]==r[key][0] and source==c[key][2]==r[key][2]
      pos=(key[1]-first[key[0]])%256
      band='shared_1_128' if pos<128 else 'reset_129_160' if pos<160 else 'middle_161_192' if pos<192 else 'late_193_256'
      values=(loss-c[key][1],c[key][1]-r[key][1],loss-r[key][1])
      bins[band].append(values);bydoc.setdefault(key[0],[]).append(values);bysource.setdefault(source,[]).append(values)
     def summary(vals):
      return dict(count=len(vals),evaluation_context_delta=math.fsum(v[0] for v in vals)/len(vals),common128_weight_delta=math.fsum(v[1] for v in vals)/len(vals),native_delta=math.fsum(v[2] for v in vals)/len(vals))
     bs={k:summary(v) for k,v in bins.items()};bs['remaining96']=summary(bins['middle_161_192']+bins['late_193_256'])
     total=summary([v for band in bins.values() for v in band]);assert abs(total['native_delta']-(nc-rc))<1e-10 and abs(total['evaluation_context_delta']-(nc-cc))<1e-10 and abs(total['common128_weight_delta']-(cc-rc))<1e-10
     assert abs(total['native_delta']-total['evaluation_context_delta']-total['common128_weight_delta'])<1e-10
     rows.append(dict(depth=depth,bundle=bundle,endpoint=endpoint,total=total,bins=bs,shared_prefix_max_abs_difference=max(abs(v[0]) for v in bins['shared_1_128']),first32_more_negative_than_remaining96=bs['reset_129_160']['evaluation_context_delta']<bs['remaining96']['evaluation_context_delta'],documents={str(k):summary(v) for k,v in bydoc.items()},sources={k:summary(v) for k,v in bysource.items()}))
  groups=[]
  for depth in [4,16]:
   for endpoint in ['raw','time_ema']:
    rs=[r for r in rows if r['depth']==depth and r['endpoint']==endpoint]
    groups.append(dict(depth=depth,endpoint=endpoint,bins={b:{k:sum(r['bins'][b][k] for r in rs)/3 for k in ['count','evaluation_context_delta','common128_weight_delta','native_delta']} for b in rs[0]['bins']},all_three_concentrated=all(r['first32_more_negative_than_remaining96'] for r in rs),total={k:sum(r['total'][k] for r in rs)/3 for k in ['evaluation_context_delta','common128_weight_delta','native_delta']},maximum_shared_prefix_abs=max(r['shared_prefix_max_abs_difference'] for r in rs)))
  result=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),kind='posthoc_saved_loss_description',rows=rows,groups=groups,evidence_sha256=evidence,plan_sha256=sha(P/'plan.json'),completion_review_sha256=sha(P/'completion-review.json'),boundary_sha256=sha(ROOT/'research/GENERAL_RSI_DEPTH_CONTEXT_POSITION_BOUNDARY.md'),source_sha256=sha(Path(__file__)),shared_lock_acquired=True,elapsed_seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,free_bytes=shutil.disk_usage(ROOT).free,test_opened=False,new_training_runs=0,new_model_evaluations=0)
  with (OUT/'analysis.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
  assert (OUT/'analysis.json').stat().st_size<16*2**20
  stop.set();print(json.dumps(groups,indent=2))
if __name__=='__main__':main()
