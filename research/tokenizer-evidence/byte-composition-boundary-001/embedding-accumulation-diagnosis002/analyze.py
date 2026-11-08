"""Isolated new primitive diagnosis; never runs a historical model or optimizer."""
import fcntl,hashlib,json,os,sys
from pathlib import Path
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
ROOT=Path.cwd();OUT=Path(__file__).resolve().parent
TORCH=None

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while x:=f.read(1048576):h.update(x)
 return h.hexdigest()
def save(name,data):
 with (OUT/name).open('x') as f:json.dump(data,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
class Guard(CpuPrerequisiteGuard):
 def _sample(self):
  d,r=super()._sample();d['output_bytes']=sum(p.stat().st_size for p in OUT.iterdir())
  if d['output_bytes']>33554432:r='output_cap'
  if TORCH is not None:
   d['mps_driver_bytes']=TORCH.mps.driver_allocated_memory()
   if d['mps_driver_bytes']>=6442450944:r='mps_cap'
  return d,r

def main():
 global TORCH
 with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert {p.name for p in OUT.iterdir()}=={'analyze.py','plan.json'}
  plan=json.loads((OUT/'plan.json').read_text());save('attempt.json',dict(pid=os.getpid(),plan_sha256=sha(OUT/'plan.json'),no_replay=True))
  try:
   with Guard(OUT/'guard.json',ROOT,timeout_seconds=180,rss_limit_bytes=4294967296,reserve_bytes=8589934592,poll_seconds=.25):
    for pin in plan['pins']:
     p=Path(pin['path']);assert p.stat().st_size==pin['size'] and sha(p)==pin['sha256']
    fs=os.statvfs(ROOT);assert fs.f_bavail*fs.f_frsize>=33554432+8589934592+2147483648
    import torch
    import numpy as np
    TORCH=torch;torch.set_num_threads(4);assert torch.backends.mps.is_available()
    from research.byte_input_proof_v1.codebook_reference import raw_rows,expected_codes
    from research.byte_input_proof_v1.cases import inputs
    before_cpu=torch.random.get_rng_state().clone();before_mps=torch.mps.get_rng_state().clone()
    try:
     archive=plan['archive'];assert sha(archive['path'])==archive['sha256']
     payload=torch.load(archive['path'],map_location='cpu',weights_only=True)
     weight=payload['parameters']['composer.byte_weight'].clone();del payload
     rows=raw_rows(*(Path(p['path']).read_bytes() for p in plan['assets']))
     codes,lengths=expected_codes(rows,'raw_bytes',None)
     token_ids,_=inputs(torch,rows,'cpu')
     code=torch.tensor(codes,dtype=torch.int64).reshape(50257,130)[token_ids.reshape(-1)]
     lens=torch.tensor(lengths,dtype=torch.int64)[token_ids.reshape(-1)]
     valid=torch.arange(130)[None,:]<lens[:,None]
     index=(code-(code>256).long()).reshape(-1)
     # Deterministic new upstream, not historical model backward or new RNG roles.
     i=torch.arange(index.numel()*15,dtype=torch.int64).reshape(-1,15)
     phase=i.to(torch.float64)
     upstream=((torch.sin(phase*.13)+torch.cos(phase*.017))*torch.exp2(-10.-(phase%23))).float()
     upstream*=valid.reshape(-1,1)
     ids=index.to('mps');grad=upstream.to('mps');w=weight.to('mps')
     native=[];grouped=[];forward=[]
     for trial in range(3):
      leaf=w.detach().clone().requires_grad_()
      y=torch.nn.functional.embedding(ids,leaf)
      y.backward(grad)
      native.append(leaf.grad.detach().cpu().numpy().copy())
      forward.append(sha_tensor(y.detach()))
      del y,leaf;torch.mps.synchronize();torch.mps.empty_cache()
      # Fixed-order dense row reduction: no repeated-index atomic scatter.
      total=torch.zeros((259,15),dtype=torch.float32,device='mps')
      vocabulary=torch.arange(259,device='mps')
      for start in range(0,index.numel(),2048):
       membership=(ids[start:start+2048,None]==vocabulary[None,:]).float()
       total.add_(membership.T@grad[start:start+2048])
      grouped.append(total.cpu().numpy().copy());del total
     cpu_index=index.numpy();cpu_grad=upstream.numpy().astype(np.float64)
     reference=np.stack([cpu_grad[cpu_index==row].sum(axis=0) for row in range(259)])
     arrays=dict(index=cpu_index,upstream=upstream.numpy(),weight=weight.numpy(),reference=reference)
     arrays.update({f'native_{n}':x for n,x in enumerate(native)});arrays.update({f'grouped_{n}':x for n,x in enumerate(grouped)})
     with (OUT/'arrays.npz').open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
     def inspect(values):
      return dict(repeat_exact=all(np.array_equal(values[0],x)for x in values[1:]),
       changed_coordinates=[int(np.count_nonzero(values[0]!=x))for x in values[1:]],
       maximum_repeat_difference=[float(np.abs(values[0].astype('float64')-x).max())for x in values[1:]],
       reference_pass=[bool((np.abs(x.astype('float64')-reference)<=4e-6+4e-5*np.abs(reference)).all())for x in values],
       maximum_reference_error=[float(np.abs(x.astype('float64')-reference).max())for x in values])
     result=dict(native=inspect(native),grouped=inspect(grouped),forward_repeat_exact=len(set(forward))==1,
        raw_shape=[1024,130,15],gradient_shape=[259,15],trials=3,
        arrays_sha256=sha(OUT/'arrays.npz'),no_model_optimizer_or_historical_fixture_call=True,
        scope='Isolated full-size byte lookup backward with saved weights and new deterministic upstream. Cannot quantify historical gradient error, whose full values were not saved. No real-data quality conclusion.',roles='No initializer or sampler draws; saved weights and explicit arithmetic upstream.')
    finally:
     torch.random.set_rng_state(before_cpu);torch.mps.set_rng_state(before_mps)
    result['caller_rng_preserved']=torch.equal(before_cpu,torch.random.get_rng_state()) and torch.equal(before_mps,torch.mps.get_rng_state())
    save('result.json',result)
    assert result['caller_rng_preserved'] and result['forward_repeat_exact']
    assert result['grouped']['repeat_exact'] and all(result['grouped']['reference_pass'])
   save('STOP',dict(status='complete_original_primitive_diagnosis',no_replay=True))
  except BaseException as exc:
   save('STOP',dict(status='closed_failed_primitive_diagnosis',error=repr(exc),no_replay=True));raise

def sha_tensor(value):
 return hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()
if __name__=='__main__':main()
