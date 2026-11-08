"""Independent saved training evidence and FP64 EMA oracle; no model/optimizer calls."""
import fcntl,hashlib,json,os,struct
from pathlib import Path
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from .cases import BACKENDS,VARIANTS,REJECTIONS,LIVE_REJECTIONS,IO_REJECTIONS
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'research/tokenizer-evidence/byte-composition-boundary-001'
OUT=BASE/'training-review001';SOURCE=BASE/'training-math001'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  while x:=f.read(1048576):h.update(x)
 return h.hexdigest()
def read(n):return json.loads((SOURCE/n).read_text())
def save(n,v):
 with (OUT/n).open('x')as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def check_batch(r):
 assert r['batch']==r['expected_batch'] and r['batch']['shape']==[8,128]
 assert len(r['batch']['inputs'])==len(r['batch']['labels'])==8
 assert all(len(row)==128 and all(type(x)is int and 0<=x<50257 for x in row)for k in ('inputs','labels')for row in r['batch'][k])
def check_receipt(r):
 p=Path(r['path']);assert p.is_file()and not p.is_symlink()and p.stat().st_nlink==1 and p.stat().st_size==r['size'] and sha(p)==r['sha256'];return p

def main():
 with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not list(OUT.iterdir());save('attempt.json',dict(pid=os.getpid(),no_replay=True))
  try:
   with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=300,rss_limit_bytes=4294967296,reserve_bytes=8589934592):
    assert read('STOP')['status']=='complete_original_training_context'and read('guard.json')['status']=='finished'
    assert read('result.json')['accepted']and read('result.json')['caller_rng_restored']
    # CPU-only deserialization of own hash-bound archives, no candidate import.
    import torch
    import numpy as np
    torch.set_num_threads(4)
    common={};archives=[];middles=[];cases=[];verified=0
    for backend in BACKENDS:
     for variant in VARIANTS:
      prefix=backend+'-'+variant;count=88 if variant=='native'else 105;elements=28492416 if variant=='native'else 16351859
      complete=read(prefix+'-complete.json');assert complete['exact']and complete['horizon']==4 and complete['resume_boundary']==2 and complete['updates_executed']==6 and complete['rejected']==37 and complete['ema_snapshots']==3 and complete['parameter_count']==count
      assert read(prefix+'-empty.json')['exact']
      val=read(prefix+'-validation-initial.json');check_batch(val);assert val['sampler']==val['expected_sampler']
      for stage in ('prefix1','prefix2','control3','control4','resumed3','resumed4'):
       r=read(prefix+'-'+stage+'.json');check_batch(r);assert r['state']['samplers']['train']==r['expected_sampler']
       step=int(stage[-1]);assert r['state']['step']==step and len(r['state']['parameters'])==count and len(r['state']['history'])==step
       if not stage.startswith('resumed'):
        old=common.setdefault(step,r['batch']);assert old==r['batch'],'Identical paired physical inputs across arms/backends'
       if step>=2:
        em=read(prefix+'-'+stage+'-ema-reference.json');assert em['passed']and em['coordinates']==elements and em['parameters']==count and em['atol']==4e-6 and em['rtol']==4e-5
       assert r['state']['averages']['snapshots']==max(0,step-1)
       for child in r['state']['children']:
        assert set(child['state'])=={g['name']for g in child['groups']}
        if child['kind']=='AdamW':
         wanted=hashlib.sha256(struct.pack('<f',float(step))).hexdigest()
         assert all(fields['step']['sha256']==wanted for fields in child['state'].values())
      for step in (3,4):assert read(prefix+'-control'+str(step)+'.json')==read(prefix+'-resumed'+str(step)+'.json')
      boundary=read(prefix+'-restored-boundary.json');assert boundary['expected']==boundary['actual']
      draws=read(prefix+'-continuation-draws.json');assert draws['expected_rng']==draws['actual_rng'] and draws['expected_val']==draws['actual_val'];check_batch(draws['actual_val'])
      for n in REJECTIONS:
       r=read(prefix+'-reject-'+n+'.json');assert r['error']and r['before']==r['after']and r['payload_before']==r['payload_after']
      for n in LIVE_REJECTIONS:
       r=read(prefix+'-live-'+n+'.json');assert r['error']and r['before']==r['after']
      for n in IO_REJECTIONS:
       r=read(prefix+'-io-'+n+'.json');assert r['error']and r['before']==r['after']and r['original_sha256']==r['after_sha256']
      for n in ('horizon','nonpristine'):
       r=read(prefix+'-'+n+'.json');assert r['error']and r['before']==r['after']
      receipts=[read(prefix+'-archive'+str(step)+'.json')for step in (2,4)]
      paths=[check_receipt(r)for r in receipts];archives+=receipts
      middle=read(prefix+'-middle.json');mp=check_receipt(middle);middles.append(middle)
      before=torch.load(paths[0],map_location='cpu',weights_only=True);after=torch.load(paths[1],map_location='cpu',weights_only=True)
      assert before['averages']['snapshots']==1 and after['averages']['snapshots']==3
      with np.load(mp)as mid:
       parameters=before['numerical']['parameters'];assert len(parameters)==count and set(mid.files)==set(parameters)
       assert sum(p.numel()for p in parameters.values())==elements
       maximum=0.;alpha=1.-2.**(-1./200.);state3=read(prefix+'-control3.json')['state']['parameters']
       for name,p in parameters.items():
        assert torch.equal(p,before['averages']['averages']['200'][name])
        assert hashlib.sha256(mid[name].astype('<f4').tobytes()).hexdigest()==state3[name]['sha256']
        initial=p.reshape(-1).numpy();intermediate=mid[name].reshape(-1);final=after['numerical']['parameters'][name].reshape(-1).numpy();actual=after['averages']['averages']['200'][name].reshape(-1).numpy()
        for start in range(0,len(initial),65536):
         part=slice(start,start+65536);ref=initial[part].astype('float64');ref+=alpha*(intermediate[part].astype('float64')-ref);ref+=alpha*(final[part].astype('float64')-ref)
         error=np.abs(actual[part].astype('float64')-ref);assert (np.isfinite(actual[part])&(error<=4e-6+4e-5*np.abs(ref))).all();maximum=max(maximum,float(error.max(initial=0.)));verified+=len(ref)
      for path,r in zip(paths,receipts):assert sha(path)==r['sha256']
      cases.append(dict(backend=backend,variant=variant,exact_continuation=True,independent_final_ema=True,max_ema_error=maximum))
      del before,after
    save('review.json',dict(accepted=True,cases=cases,invalid_transactions=222,updates_executed=36,independent_ema_coordinates=verified,archives=archives,middles=middles,scope='Four-update full-size training-context resume fixture. Exact sampled trajectory, clipping/history, schedule, optimizer/EMA and RNG; independent saved FP64 final EMA. No quality selection or 1200-step campaign/H200-export/scorer qualification.'))
   save('STOP',dict(status='complete_original_training_saved_review',no_replay=True))
  except BaseException as exc:
   save('STOP',dict(status='closed_failed_training_saved_review',error=repr(exc),no_replay=True));raise
if __name__=='__main__':main()
