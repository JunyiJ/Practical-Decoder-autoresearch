"""Generated AdamW epsilon arithmetic/order/resume checks, without training a model."""
import copy,gc,hashlib,io,json,math
from pathlib import Path
import torch
from research.ffn_structure_v1.runtime.train.optimizers import HybridOptimizer
from research.readout_centering_quality_v1.support import idle
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
B=Path(__file__).resolve().parent;ROOT=B.parents[2]
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,value):
 with (B/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
plan=json.loads((B/'optimizer-component001-plan.json').read_text())
def verify():
 for name,h in plan['sources'].items():assert sha(ROOT/name)==h,name
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

def setup(dtype,epsilon,decay=.00006):
 p=[torch.nn.Parameter(torch.arange(32,dtype=dtype).reshape(4,8)/31-.5),torch.nn.Parameter(torch.tensor([.5,-1.,2.,-.25],dtype=dtype)),torch.nn.Parameter(torch.tensor([3.,-4.],dtype=dtype))]
 muon=torch.optim.Muon([p[0]],lr=.000625,weight_decay=0.,adjust_lr_fn='match_rms_adamw')
 adam=torch.optim.AdamW(p[1:],lr=.00125,betas=(.9,.999),eps=epsilon,weight_decay=0.)
 return p,HybridOptimizer([muon,adam],decay)
def gradients(p,index):
 values=[[1e-9,-2e-9,3e-8,-4e-7],[1.,-2.,3.,-4.],[-.5,.25,-.125,.0625],[0.,0.,0.,0.],[3.,1.,-2.,.5],[-1.,2.,.25,-.75]]
 p[0].grad=(torch.arange(32,dtype=p[0].dtype).reshape(4,8)/31-.5)*(.1*(index+1))
 p[1].grad=torch.tensor(values[index],dtype=p[1].dtype);p[2].grad=None
 torch.nn.utils.clip_grad_norm_(p,1.,error_if_nonfinite=True)
def restore(payload,epsilon,dtype):
 assert payload['beta2']==.999 and payload['beta1']==.9 and payload['epsilon']==epsilon,'Foreign AdamW policy rejected'
 groups=payload['optimizer']['optimizers'][1]['param_groups'];assert len(groups)==1 and groups[0]['betas']==(.9,.999) and groups[0]['eps']==epsilon,'Foreign serialized AdamW group rejected'
 p,o=setup(dtype,epsilon)
 with torch.no_grad():
  for x,y in zip(p,payload['parameters']):x.copy_(y)
 o.load_state_dict(payload['optimizer']);return p,o

verify();assert not any((B/n).exists() for n in ('optimizer-component001-result.json','optimizer-component001-guard.json','optimizer-component001-failure.json'))
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
 idle()
 try:
  with CpuPrerequisiteGuard(B/'optimizer-component001-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
   initial_rng=torch.get_rng_state().clone();rows=[];outcomes={};moment_states={}
   binding=plan['dependencies']['accepted_incumbent_raw'];assert Path(binding['path']).stat().st_size<512*1024**2
   raw=torch.load(binding['path'],map_location='cpu',weights_only=False)
   groups=raw['optimizer_state']['optimizers'][1]['param_groups'];assert len(groups)==44 and all(tuple(g['betas'])==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0. for g in groups)
   assert raw['optimizer_state']['independent_decay']==.00006
   incumbent=dict(adamw_groups=len(groups),betas=[.9,.999],epsilon=1e-8,child_weight_decay=0.,independent_decay=.00006)
   del raw,groups;gc.collect()
   for dtype in (torch.float64,torch.float32):
    for epsilon in (1e-8,1e-6):
     p,opt=setup(dtype,epsilon);manual,reference=setup(dtype,epsilon,0.)
     x=p[1].detach().double().clone();m=torch.zeros_like(x);v=torch.zeros_like(x);errors=[];saved=None
     tolerance=1e-12 if dtype==torch.float64 else 1e-6
     for index in range(6):
      gradients(p,index);g=p[1].grad.detach().double().clone()
      # Float64 recurrence, bias corrections and epsilon outside sqrt are independent of AdamW.step.
      x*=1-.00006;m=.9*m+.1*g;v=.999*v+.001*g.square()
      x-=.00125*(m/(1-.9**(index+1)))/(torch.sqrt(v/(1-.999**(index+1)))+epsilon)
      for a,b in zip(manual,p):a.grad=None if b.grad is None else b.grad.detach().clone()
      with torch.no_grad():
       for a in manual:
        if a.grad is not None:a.mul_(1-.00006)
      for child in reference.optimizers:child.step()
      opt.step();equal(p,manual);equal(opt.state_dict()['optimizers'],reference.state_dict()['optimizers'])
      state=opt.optimizers[1].state[p[1]]
      err=dict(parameter=float((p[1].detach().double()-x).abs().max()),first_moment=float((state['exp_avg'].double()-m).abs().max()),second_moment=float((state['exp_avg_sq'].double()-v).abs().max()))
      assert max(err.values())<=tolerance,err;errors.append(err)
      opt.zero_grad();reference.zero_grad()
      if index==2:
       buf=io.BytesIO();torch.save(dict(beta1=.9,beta2=.999,epsilon=epsilon,parameters=[a.detach().clone() for a in p],optimizer=opt.state_dict()),buf);saved=buf.getvalue();assert len(saved)<1024**2
     payload=torch.load(io.BytesIO(saved),weights_only=False);pp,oo=restore(payload,epsilon,dtype)
     for index in range(3,6):gradients(pp,index);oo.step();oo.zero_grad()
     equal(pp,p);equal(oo.state_dict(),opt.state_dict());equal(p[2],torch.tensor([3.,-4.],dtype=dtype));assert p[2] not in opt.optimizers[1].state
     for kind in ('policy','serialized_group'):
      bad=copy.deepcopy(payload)
      if kind=='policy':bad['epsilon']=1e-6 if epsilon==1e-8 else 1e-8
      else:bad['optimizer']['optimizers'][1]['param_groups'][0]['eps']=1e-6 if epsilon==1e-8 else 1e-8
      try:restore(bad,epsilon,dtype)
      except AssertionError:pass
      else:raise AssertionError('Foreign epsilon accepted')
     equal(torch.get_rng_state(),initial_rng)
     outcomes[(str(dtype),epsilon)]=[a.detach().clone() for a in p]
     moment_states[(str(dtype),epsilon)]=copy.deepcopy(opt.optimizers[1].state[p[1]])
     rows.append(dict(dtype=str(dtype),epsilon=epsilon,oracle_tolerance=tolerance,oracle_errors=errors,exact_shrinkage_before_child_order=True,exact_serialized_resume=True,foreign_policy_and_group_rejected=True,no_gradient_unchanged=True,rng_unchanged=True,serialized_bytes=len(saved)))
   for dtype in (torch.float64,torch.float32):
    equal(moment_states[(str(dtype),1e-8)],moment_states[(str(dtype),1e-6)])
    control=outcomes[(str(dtype),1e-8)];candidate=outcomes[(str(dtype),1e-6)];equal(control[0],candidate[0]);equal(control[2],candidate[2]);assert not torch.equal(control[1],candidate[1])
   verify()
  result=dict(status='passed',rows=rows,incumbent_inspection=incumbent,source_count=len(plan['sources']),guard='optimizer-component001-guard.json',scope='Four generated CPU FP64/FP32 real Muon/AdamW epsilon/order/resume cases and read-only accepted incumbent group inspection. Muon unchanged for prescribed gradients. No actual full-model candidate, MPS, EMA/H200/cache or scientific quality evidence.',training_launched=False,scientific_seeds_used=False)
  save('optimizer-component001-result.json',result);print(json.dumps(dict(status='passed',cases=len(rows),max_oracle_error=max(max(e.values()) for r in rows for e in r['oracle_errors']))))
 except BaseException as exc:
  save('optimizer-component001-failure.json',dict(status='failed',error=repr(exc),no_retry=True));raise
