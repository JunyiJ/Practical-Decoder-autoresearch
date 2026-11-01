"""Generated-tensor SwiGLU-product RMS numerical prerequisite; no corpus/training."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from src.models.mlp import SwiGLU
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from research.mla_score_quality_v1 import quality_execution as q
from research.block_norm_v1.fixture_dispatch import reference, read
B=Path(__file__).resolve().parent
ROOT=B.parents[2]
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent) as g:
 q.idle()
 plan=read(reference(B/'component001-plan.json'))
 for ref in plan['sources']:read(ref)
 assert not (B/'component001.json').exists() and not (B/'component001-guard.json').exists()
 with CpuPrerequisiteGuard(B/'component001-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
  rows=[]
  for dtype in (torch.float64,torch.float32):
   torch.manual_seed(90817)
   base=SwiGLU(256,672,0.,hidden_norm='none').to(dtype)
   rms=SwiGLU(256,672,0.,hidden_norm='rms').to(dtype)
   for name in ('linear_gate','linear_up','linear_down'):
    getattr(rms,name).load_state_dict(getattr(base,name).state_dict())
    assert torch.equal(getattr(rms,name).weight,getattr(base,name).weight)
   assert rms.hidden_norm.eps==1e-5 and torch.equal(rms.hidden_norm.weight,torch.ones(672,dtype=dtype))
   assert sum(p.numel() for p in rms.parameters())-sum(p.numel() for p in base.parameters())==672
   for kind,model in [('none',base),('rms',rms)]:
    for scale in (.01,1.,10.):
     x=(torch.randn(2,5,256,dtype=dtype)*scale).requires_grad_()
     actual=model(x)
     z=F.silu(x@model.linear_gate.weight.T)*(x@model.linear_up.weight.T)
     if kind=='rms':z=z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-5)*model.hidden_norm.weight
     expected=z@model.linear_down.weight.T
     atol,rtol=(1e-11,1e-10) if dtype==torch.float64 else (3e-5,1e-4)
     torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
     direction=torch.randn_like(actual)
     leaves=[x,*model.parameters()]
     ga=torch.autograd.grad((actual*direction).mean(),leaves)
     ge=torch.autograd.grad((expected*direction).mean(),leaves)
     for a,e in zip(ga,ge):
      assert torch.isfinite(a).all() and torch.isfinite(e).all()
      torch.testing.assert_close(a,e,atol=atol,rtol=rtol)
     altered=x.detach().clone();altered[:,3:]+=torch.randn_like(altered[:,3:])
     with torch.no_grad():torch.testing.assert_close(model(altered)[:,:3],actual.detach()[:,:3],atol=0,rtol=0)
     rows.append(dict(dtype=str(dtype),kind=kind,input_scale=scale,forward_max_abs=(actual-expected).abs().max().item(),all_gradient_max_abs=max((a-e).abs().max().item() for a,e in zip(ga,ge)),gradient_tensors=len(ga),prefix_unchanged=True))
  for ref in plan['sources']:read(ref)
  value=dict(status='passed',plan=reference(B/'component001-plan.json'),rows=rows,extra_parameters_per_block=672,expected_extra_parameters_four_layers=2688,shared_matrices_identical=True,epsilon=1e-5,gain_init=1.0,cpu_threads=torch.get_num_threads(),scope='CPU FP64/FP32 generated tensors only. No full-model integration, optimizer/resume, MPS, corpus, quality or deployment claim.',training_launched=False,holdout_scored=False)
 guard=read(reference(B/'component001-guard.json'));assert guard['status']=='finished' and guard['reason'] is None
 q.exclusive(B/'component001.json',dict(value,guard=reference(B/'component001-guard.json')))
 print(json.dumps(dict(status='passed',cases=len(rows),max_forward=max(r['forward_max_abs'] for r in rows),max_gradient=max(r['all_gradient_max_abs'] for r in rows))))
