"""Generated clipping/order/resume checks; no corpus, model or quality claim."""
import copy,hashlib,io,json,math
from pathlib import Path
import torch
from research.ffn_structure_v1.runtime.train.optimizers import HybridOptimizer
from research.branch_scale_quality_v1.support import idle
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
B=Path(__file__).resolve().parent;ROOT=B.parents[2]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,value):
 with (B/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
plan=json.loads((B/'optimizer-component001-plan.json').read_text())
def verify():
 for path,h in plan['sources'].items():assert sha(ROOT/path)==h,path
 for ref in plan['dependencies'].values():assert sha(ref['path'])==ref['sha256']
def equal(a,b):
 if isinstance(a,torch.Tensor):assert torch.equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b

def setup(dtype,decay=.00006):
 p=[torch.nn.Parameter(torch.arange(32,dtype=dtype).reshape(4,8)/31-.5),torch.nn.Parameter(torch.tensor([.5,-1.,2.,-.25],dtype=dtype)),torch.nn.Parameter(torch.tensor([3.,-4.],dtype=dtype))]
 muon=torch.optim.Muon([p[0]],lr=.000625,weight_decay=0.,adjust_lr_fn='match_rms_adamw')
 adam=torch.optim.AdamW(p[1:],lr=.00125,weight_decay=0.)
 return p,HybridOptimizer([muon,adam],decay)

def gradients(p,norm):
 values=torch.arange(1,37,dtype=torch.float64);values[1::2]*=-1
 values=values/torch.linalg.vector_norm(values)*norm
 p[0].grad=values[:32].reshape(4,8).to(p[0].dtype)
 p[1].grad=values[32:].to(p[1].dtype);p[2].grad=None

def step(p,opt,threshold):
 before=[x.grad.detach().clone() if x.grad is not None else None for x in p]
 exact=math.sqrt(sum(float(g.double().square().sum()) for g in before if g is not None))
 coefficient=min(1.,threshold/(exact+1e-6))
 observed=torch.nn.utils.clip_grad_norm_(p,threshold,error_if_nonfinite=True)
 tolerance=1e-12 if p[0].dtype==torch.float64 else 1e-6
 assert abs(float(observed)-exact)<=tolerance*max(1.,exact)
 error=0.
 for x,g in zip(p,before):
  if g is None:assert x.grad is None;continue
  difference=float((x.grad.double()-g.double()*coefficient).abs().max());error=max(error,difference)
  assert difference<=tolerance
 post=math.sqrt(sum(float(x.grad.double().square().sum()) for x in p if x.grad is not None))
 assert post<=threshold+tolerance
 return dict(pre_norm=float(observed),oracle_norm=exact,post_norm=post,coefficient=coefficient,threshold_exceeded=exact>threshold,max_gradient_error=error)

def restore(payload,threshold,dtype):
 assert payload['clip_threshold']==threshold,'Foreign clipping checkpoint rejected'
 p,o=setup(dtype)
 with torch.no_grad():
  for x,y in zip(p,payload['parameters']):x.copy_(y)
 o.load_state_dict(payload['optimizer']);return p,o

verify();assert not any((B/n).exists() for n in ('optimizer-component001-result.json','optimizer-component001-guard.json','optimizer-component001-failure.json'))
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
 idle()
 try:
  with CpuPrerequisiteGuard(B/'optimizer-component001-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
   initial_rng=torch.get_rng_state().clone();rows=[]
   for dtype in (torch.float64,torch.float32):
    for threshold in (1.,.5):
     p,opt=setup(dtype);manual,reference=setup(dtype,0.);saved=None;observations=[]
     norms=[0.,.25,.5,.75,1.,2.]
     for index,norm in enumerate(norms):
      gradients(p,norm);observation=step(p,opt,threshold);observations.append(observation)
      # Independently step children after shrinkage, using gradients whose
      # clipping arithmetic was checked above against a float64 norm oracle.
      for x,y in zip(manual,p):x.grad=None if y.grad is None else y.grad.detach().clone()
      with torch.no_grad():
       for x in manual:
        if x.grad is not None:x.mul_(1-.00006)
      for child in reference.optimizers:child.step()
      opt.step();equal(p,manual);equal(opt.state_dict()['optimizers'],reference.state_dict()['optimizers']);opt.zero_grad();reference.zero_grad()
      if index==2:
       buf=io.BytesIO();torch.save(dict(clip_threshold=threshold,parameters=[x.detach().clone() for x in p],optimizer=opt.state_dict()),buf);saved=buf.getvalue();assert len(saved)<1024**2
     payload=torch.load(io.BytesIO(saved),weights_only=False);pp,oo=restore(payload,threshold,dtype)
     for norm in norms[3:]:gradients(pp,norm);step(pp,oo,threshold);oo.step();oo.zero_grad()
     equal(pp,p);equal(oo.state_dict(),opt.state_dict());equal(p[2],torch.tensor([3.,-4.],dtype=dtype))
     try:restore(payload,.5 if threshold==1. else 1.,dtype)
     except AssertionError:pass
     else:raise AssertionError('Foreign clipping threshold accepted')
     # Nonfinite gradients must raise before either child optimizer or decay.
     for bad in (float('nan'),float('inf')):
      gradients(pp,2.);pp[0].grad[0,0]=bad;weights=[x.detach().clone() for x in pp];state=copy.deepcopy(oo.state_dict())
      try:torch.nn.utils.clip_grad_norm_(pp,threshold,error_if_nonfinite=True)
      except RuntimeError:pass
      else:raise AssertionError('Nonfinite clipping accepted')
      equal(pp,weights);equal(oo.state_dict(),state)
     equal(torch.get_rng_state(),initial_rng)
     rows.append(dict(dtype=str(dtype),threshold=threshold,observations=observations,exact_child_order=True,exact_serialized_resume=True,foreign_threshold_rejected=True,nonfinite_rejected_before_update=True,no_gradient_unchanged=True,rng_unchanged=True,serialized_bytes=len(saved)))
   verify()
  save('optimizer-component001-result.json',dict(status='passed',rows=rows,source_count=len(plan['sources']),guard='optimizer-component001-guard.json',scope='Four CPU FP64/FP32 generated real Muon/AdamW clipping/order/resume cases. No actual full-model clipping frequency, MPS, EMA/H200/cache or scientific quality evidence.',training_launched=False,scientific_seeds_used=False))
  print(json.dumps(dict(status='passed',cases=len(rows),max_gradient_error=max(x['max_gradient_error'] for r in rows for x in r['observations']))))
 except BaseException as exc:
  save('optimizer-component001-failure.json',dict(status='failed',error=repr(exc),no_retry=True));raise
