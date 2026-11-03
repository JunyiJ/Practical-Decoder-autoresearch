"""Bounded CPU behavioral audit of the prepared memory-safe state reference."""
import ast,copy,hashlib,json
from pathlib import Path
import torch
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.adamw_epsilon_quality_v1.support import idle
R=Path.cwd();B=Path(__file__).resolve().parent;source=B/'model-resume002.py'
def ref(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def equal(a,b):
 if type(a) is not type(b):return False
 if isinstance(a,torch.Tensor):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
with session_lock(GLOBAL_LOCK.parent),CpuPrerequisiteGuard(B/'state-digest-oracle001-guard.json',R,timeout_seconds=60):
 idle();torch.set_num_threads(4);node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='digest_state');ns=dict(torch=torch,copy=copy,hashlib=hashlib)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns);digest=ns['digest_state'];rows=[]
 plan=dict(source=ref(source),runner=ref(Path(__file__).resolve()),scope='Finite generated CPU optimizer-state references only;no candidate full-model resume or memory-performance claim',dtypes=['float32','float64'],widths=[7,256],mutations=['one_tensor_value','tensor_shape','tensor_dtype','optimizer_rate','rng_byte','metadata_scalar'])
 with (B/'state-digest-oracle001-plan.json').open('x') as f:json.dump(plan,f,indent=2)
 rng=torch.get_rng_state().clone()
 for dtype in [torch.float32,torch.float64]:
  for width in [7,256]:
   p=torch.nn.Parameter(torch.linspace(.1,.9,width*width,dtype=dtype).reshape(width,width));opt=torch.optim.AdamW([p],lr=.00125,betas=(.9,.999),eps=1e-6)
   for step in range(3):p.grad=torch.sin(torch.arange(p.numel(),dtype=dtype).reshape_as(p)+step);opt.step()
   base=dict(model={'weight':p.detach().clone()},optimizer=copy.deepcopy(opt.state_dict()),ema={'200':p.detach().clone()*.75},rng=torch.get_rng_state().clone(),steps=3,history=[{'coefficient':1.,'threshold':1.}])
   same=copy.deepcopy(base);assert equal(base,same) and digest(base)==digest(same)
   for mutation in plan['mutations']:
    changed=copy.deepcopy(base)
    if mutation=='one_tensor_value':changed['model']['weight'][0,0]+=1e-3
    elif mutation=='tensor_shape':changed['model']['weight']=changed['model']['weight'].reshape(-1)
    elif mutation=='tensor_dtype':changed['model']['weight']=changed['model']['weight'].to(torch.float64 if dtype==torch.float32 else torch.float32)
    elif mutation=='optimizer_rate':changed['optimizer']['param_groups'][0]['lr']*=2
    elif mutation=='rng_byte':changed['rng'][0]^=1
    else:changed['steps']+=1
    assert not equal(base,changed) and digest(base)!=digest(changed)
    rows.append(dict(dtype=str(dtype),width=width,mutation=mutation,detected=True))
 assert torch.equal(rng,torch.get_rng_state()) and ref(source)==plan['source']
 result=dict(status='finite_state_identity_mutation_audit_passed',cases=len(rows),rows=rows,source=ref(source),runner=ref(Path(__file__).resolve()),clone_identity_exact=True,rng_unchanged=True,full_model_resume_executed=False,memory_saving_measured=False,scientific_training=False)
 with (B/'state-digest-oracle001.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(dict(status=result['status'],cases=len(rows))))
