"""New standalone autograd primitive proof using retained diagnostic arrays."""
import fcntl,hashlib,json,os
from pathlib import Path
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
ROOT=Path.cwd();OUT=Path(__file__).resolve().parent;TORCH=None

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while x:=f.read(1048576):h.update(x)
 return h.hexdigest()
def save(n,v):
 with (OUT/n).open('x') as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
class Guard(CpuPrerequisiteGuard):
 def _sample(self):
  d,r=super()._sample();d['output_bytes']=sum(p.stat().st_size for p in OUT.iterdir())
  if d['output_bytes']>33554432:r='output_cap'
  if TORCH is not None:
   d['mps_driver_bytes']=TORCH.mps.driver_allocated_memory()
   if d['mps_driver_bytes']>=6442450944:r='mps_cap'
  return d,r

with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert {p.name for p in OUT.iterdir()}=={'prove.py','plan.json'}
 plan=json.loads((OUT/'plan.json').read_text());save('attempt.json',dict(pid=os.getpid(),plan_sha256=sha(OUT/'plan.json')))
 try:
  with Guard(OUT/'guard.json',ROOT,timeout_seconds=180,rss_limit_bytes=4294967296,reserve_bytes=8589934592,poll_seconds=.25):
   for p in plan['pins']:assert Path(p['path']).stat().st_size==p['size'] and sha(p['path'])==p['sha256']
   fs=os.statvfs(ROOT);assert fs.f_bavail*fs.f_frsize>=33554432+8589934592+2147483648
   import torch
   import numpy as np
   TORCH=torch;torch.set_num_threads(4);assert torch.backends.mps.is_available()
   from research.byte_embedding_v1 import embedding
   cpu=torch.random.get_rng_state().clone();mps=torch.mps.get_rng_state().clone();records=[];arrays={};rejections=[]
   try:
    for number,path in enumerate(plan['arrays']):
     with np.load(path) as saved:
      index=saved['index'].copy();upstream=saved['upstream'].copy();weight=saved['weight'].copy();reference=saved['reference'].copy()
     # Independently rebuild float64 row sums from every saved upstream coordinate.
     oracle=np.stack([upstream.astype('float64')[index==row].sum(axis=0) for row in range(259)])
     assert np.array_equal(oracle,reference)
     for backend in ('cpu','mps'):
      grads=[]
      for repeat in range(3):
       w=torch.tensor(weight,device=backend,requires_grad=True);ids=torch.tensor(index,device=backend)
       result=embedding(w,ids)
       native=torch.nn.functional.embedding(ids,w)
       assert torch.equal(result,native)
       result.backward(torch.tensor(upstream,device=backend))
       actual=w.grad.detach().cpu().numpy().copy();grads.append(actual)
       key=f'case{number}-{backend}-{repeat}';arrays[key]=actual
       passed=bool((np.isfinite(actual)&(np.abs(actual.astype('float64')-oracle)<=4e-6+4e-5*np.abs(oracle))).all())
       records.append(dict(key=key,forward_exact=True,reference_pass=passed,count=actual.size,max_error=float(np.abs(actual.astype('float64')-oracle).max())))
       del result,native,w;torch.mps.synchronize();torch.mps.empty_cache()
      assert all(np.array_equal(grads[0],x)for x in grads[1:])
    for backend in ('cpu','mps'):
     for case in ('float_ids','negative_ids','high_ids','empty_ids','weight_shape','weight_dtype','weight_nonfinite'):
      w=torch.tensor(weight,device=backend,requires_grad=True);ids=torch.tensor(index,device=backend)
      if case=='float_ids':ids=ids.float()
      elif case=='negative_ids':ids=ids.clone();ids[0]=-1
      elif case=='high_ids':ids=ids.clone();ids[0]=259
      elif case=='empty_ids':ids=ids[:0]
      elif case=='weight_shape':w=w[:258].detach().requires_grad_()
      elif case=='weight_dtype':w=w.double().detach().requires_grad_()
      elif case=='weight_nonfinite':w=torch.full_like(w,float('nan'),requires_grad=True)
      before=w.detach().cpu().numpy().tobytes();error=None
      try:embedding(w,ids)
      except ValueError as exc:error=repr(exc)
      unchanged=before==w.detach().cpu().numpy().tobytes() and w.grad is None
      rejections.append(dict(backend=backend,case=case,error=error,unchanged=unchanged))
      assert error and unchanged
    with (OUT/'gradients.npz').open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
   finally:
    torch.random.set_rng_state(cpu);torch.mps.set_rng_state(mps)
   preserved=torch.equal(cpu,torch.random.get_rng_state()) and torch.equal(mps,torch.mps.get_rng_state())
   save('result.json',dict(accepted=all(r['reference_pass']for r in records) and preserved,records=records,rejections=rejections,repeat_exact=True,caller_rng_preserved=preserved,gradients_sha256=sha(OUT/'gradients.npz'),scope='Standalone byte lookup first-order autograd only; no composer/model/checkpoint integration qualification'))
   assert all(r['reference_pass']for r in records) and preserved
  save('STOP',dict(status='complete_original_embedding_primitive',no_replay=True))
 except BaseException as exc:
  save('STOP',dict(status='closed_failed_embedding_primitive',error=repr(exc),no_replay=True));raise
