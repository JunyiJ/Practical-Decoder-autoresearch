"""Generated final-LayerNorm scale oracle plus read-only adopted incumbent audit."""
import gc,hashlib,json,math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from research.final_ln_scale_v1.component import FinalLayerNorm,KINDS
from research.adamw_epsilon_worker_v1 import validation as v
from research.adamw_epsilon_quality_v1.support import idle
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
B=Path(__file__).resolve().parent;R=B.parents[2]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def save(n,d):
 raw=(json.dumps(d,indent=2)+'\n').encode();assert len(raw)<1024**2
 assert sum(p.stat().st_size for p in B.rglob('*') if p.is_file())+len(raw)<8*1024**2
 with (B/n).open('xb') as f:f.write(raw)
def generated(shape,offset):
 x=torch.arange(math.prod(shape),dtype=torch.float64).reshape(shape)
 return torch.sin(x*1.137+offset)+.3*torch.cos(x*.731+offset)+.4
def error(a,b):return float((a.detach().cpu().double()-b.detach().cpu().double()).abs().max())
def oracle(x,w,b):
 z=x-x.sum(-1,keepdim=True)/x.shape[-1]
 return z/(z.square().sum(-1,keepdim=True)/x.shape[-1]+1e-5).sqrt()*w+b
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
 idle();old=R/'research/tokenizer-evidence/adamw-epsilon-preparation-001'
 prior=json.loads((old/'quality001-plan.json').read_text())['sources']
 for n,h in prior.items():assert sha(R/n)==h
 closure=json.loads((old/'quality001-closure.json').read_text());assert closure['status']=='closed_adopted'
 recipe=json.loads((old/'adopted-width256-mla-epsilon-recipe.json').read_text());binding={k:recipe['representative']['files'][k] for k in ('config','parent','h200')}
 files=[Path(__file__),R/'research/final_ln_scale_v1/__init__.py',R/'research/final_ln_scale_v1/component.py'];sources={str(p.relative_to(R)):sha(p) for p in files}
 plan=dict(sources=sources,predecessors=prior,decision=ref(B/'direction-decision.json'),closure=ref(old/'quality001-closure.json'),adopted_recipe=ref(old/'adopted-width256-mla-epsilon-recipe.json'),binding=binding,
  cases=[dict(device=d,dtype=t,batch=b,width=w,kind=k) for d,t in [('cpu','float64'),('cpu','float32'),('mps','float32')] for b in [1,3] for w in [7,256] for k in KINDS],
  normwise_tolerances=dict(float64=1e-10,float32=5e-5),finite_difference=dict(step=1e-6,atol=1e-7,rtol=1e-5),oracle='CPU FP64 explicit mean/variance forward and gradients; normwise max error/max(1,max oracle magnitude). Exact same-backend native forward/all-gradient parity also required.',
  scope='Generated tokenwise LayerNorm only; adopted incumbent inspection. No candidate full-model/optimizer/EMA/cache or quality evidence.',training_allowed=False,scientific_seeds_used=False,output_cap_bytes=8*1024**2)
 save('component001-plan.json',plan);rows=[];observation=None;failure=None
 try:
  with StrictMpsGuard(B/'component001-guard.json',R):
   model,data,cfg=v.load(binding,'cpu');norm=model.transformer.ln_f
   assert type(norm) is nn.LayerNorm and norm.eps==1e-5 and norm.weight.shape==norm.bias.shape==(256,)
   assert cfg.training.adamw_epsilon==1e-6
   observation=dict(final_type='LayerNorm',scale_parameters=256,bias_parameters=256,epsilon=norm.eps,adamw_epsilon=cfg.training.adamw_epsilon,total_parameters=sum(p.numel() for p in model.parameters()),training_updates=0)
   assert observation['total_parameters']==28492416
   del model,data,cfg,norm;gc.collect()
   for case in plan['cases']:
    device=case['device'];dtype=getattr(torch,case['dtype']);width=case['width'];kind=case['kind'];rng=torch.get_rng_state().clone()
    m=FinalLayerNorm(width,kind).to(device=device,dtype=dtype);twin=FinalLayerNorm(width,KINDS[1] if kind==KINDS[0] else KINDS[0]).to(device=device,dtype=dtype)
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(m.bias,twin.bias)
    x=generated((case['batch'],9,width),.2).to(device=device,dtype=dtype).requires_grad_();assert torch.equal(m(x),twin(x))
    assert set(m.state_dict())==({'weight','bias'} if kind=='learned' else {'bias'}) and sum(p.numel() for p in m.parameters())==width*(2 if kind=='learned' else 1)
    with torch.no_grad():
     m.bias.copy_(generated((width,),1.3)*.1)
     if m.weight is not None:m.weight.copy_(generated((width,),.7))
    xr=x.detach().cpu().double().requires_grad_();w=m.weight.detach().cpu().double().requires_grad_() if m.weight is not None else torch.ones(width,dtype=torch.float64);b=m.bias.detach().cpu().double().requires_grad_()
    params=(x,m.weight,m.bias) if m.weight is not None else (x,m.bias);refs=(xr,w,b) if m.weight is not None else (xr,b)
    probe=generated(tuple(x.shape),.8).to(device=device,dtype=dtype);actual=m(x);expected=oracle(xr,w,b)
    ga=torch.autograd.grad((actual*probe).sum(),params,retain_graph=True);gb=torch.autograd.grad((expected*probe.cpu().double()).sum(),refs)
    tol=plan['normwise_tolerances'][case['dtype']];errors=[error(a,z)/max(1.,float(z.detach().abs().max())) for a,z in zip((actual,*ga),(expected,*gb))]
    assert max(errors)<=tol and all(torch.isfinite(a).all() for a in (actual,*ga))
    nx=x.detach().clone().requires_grad_();nw=m.weight.detach().clone().requires_grad_() if m.weight is not None else None;nb=m.bias.detach().clone().requires_grad_();ny=F.layer_norm(nx,(width,),nw,nb,1e-5)
    np=(nx,nw,nb) if nw is not None else (nx,nb);ng=torch.autograd.grad((ny*probe).sum(),np)
    assert torch.equal(actual,ny) and all(torch.equal(a,z) for a,z in zip(ga,ng))
    if kind=='learned':
     native=nn.LayerNorm(width,eps=1e-5).to(device=device,dtype=dtype);native.load_state_dict(m.state_dict());assert torch.equal(actual,native(x));del native
    assert error(ga[-1],probe.sum((0,1)))<=tol*max(1.,float(probe.sum((0,1)).abs().max()))
    future=torch.autograd.grad(actual[:,:5].square().sum(),x)[0][:,5:];assert torch.count_nonzero(future)==0
    changed=x.detach().clone();changed[:,5:]+=3.7;assert torch.equal(m(changed)[:,:5],actual[:,:5])
    fd=[]
    if dtype==torch.float64:
     for index,p in enumerate(params):
      with torch.no_grad():
       original=p.view(-1)[0].item();p.view(-1)[0]=original+1e-6;plus=(m(x)*probe).sum().item();p.view(-1)[0]=original-1e-6;minus=(m(x)*probe).sum().item();p.view(-1)[0]=original
      numerical=(plus-minus)/2e-6;analytic=ga[index].view(-1)[0].item();assert abs(numerical-analytic)<=1e-7+1e-5*abs(analytic);fd.append(abs(numerical-analytic))
    rows.append(dict(**case,forward_error=error(actual,expected),gradient_error=max(error(a,z) for a,z in zip(ga,gb)),max_normwise_error=max(errors),finite_difference_errors=fd,initial_function_exact=True,shared_bias_exact=True,rng_unchanged=True,native_forward_all_gradients_exact=True,exact_suffix_independence=True,exact_future_gradient_zero=True,trainable_parameters=sum(p.numel() for p in m.parameters())))
    del m,twin,x,xr,w,b,actual,expected,ga,gb,probe,changed,future,nx,nw,nb,ny,ng,params,refs,np
    if device=='mps':torch.mps.synchronize();torch.mps.empty_cache()
   for n,h in {**prior,**sources}.items():assert sha(R/n)==h
   v.artifacts(binding)
 except BaseException as exc:failure=repr(exc)
 result=dict(status='passed' if failure is None else 'failed',error=failure,cases=rows,incumbent_observation=observation,plan=ref(B/'component001-plan.json'),guard=ref(B/'component001-guard.json'),predecessor_sources_unchanged=len(prior),source_count=len(prior)+len(sources),full_model_integrated=False,scientific_training_launched=False,quality_adopted=False,no_retry=True,scope=plan['scope'])
 save('component001-result.json',result);print(json.dumps(dict(status=result['status'],cases=len(rows),error=failure)))
 if failure:raise RuntimeError(failure)
