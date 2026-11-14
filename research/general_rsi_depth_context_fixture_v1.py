"""Bounded CPU checks for the newly combined depth/context path, no corpus reads."""
import ast,datetime,fcntl,gc,hashlib,json,platform,shutil,traceback
from pathlib import Path
import torch
from research.general_rsi_depth_context_v1.model import Model
from research.general_rsi_depth_context_v1.contract import Spec,counts
from research.general_rsi_depth_context_v1.optimizer import build_optimizer
from research.general_rsi_cross_size_v1.model import DepthModel
from research.general_rsi_cross_size_v1.contract import Spec as DepthSpec
from research.general_rsi_cross_size_v1.optimizer import build_optimizer as depth_optimizer
from research.general_rsi_context_length_v1.model import ContextModel
from research.general_rsi_context_length_v1.contract import Spec as ContextSpec
from research.general_rsi_context_length_v1.optimizer import ContextOptimizer
from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
from research.gradient_clip_model_v1.clipping import clip_gradients
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/tokenizer-evidence/general-rsi-depth-context-fixture-001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,j):
 with p.open('x') as f:json.dump(j,f,indent=2);f.write('\n')
def read(p):return json.loads(p.read_text())
def digest(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def prepare():
 sources=dict(read(ROOT/'research/tokenizer-evidence/general-rsi-cross-size-quality-001/plan.json')['sources'])
 sources.update(read(ROOT/'research/tokenizer-evidence/general-rsi-context-length-quality-001/plan.json')['sources'])
 for p in [Path(__file__),* (ROOT/'research/general_rsi_depth_context_v1').glob('*.py'),ROOT/'src/research/cpu_prerequisite_guard.py']:sources[str(p.relative_to(ROOT))]=sha(p)
 assert all(sha(ROOT/p)==h for p,h in sources.items())
 p=dict(kind='native_depth_context_cpu_parity_v1',sources=sources,torch_version=torch.__version__,python_version=platform.python_version(),initializer=62111415010001,extra_group_seeds=[62111415010002,62111415010003,62111415010004],seed_audit='rg research/research_state JSON,JSONL,Python before allocation returned no matches for62111415010xxx',criteria='Exact initialization, corrected logits, training logits/loss/gradients, clipping, one optimizer update and child states against old depth128 at4/8/16 and old context256 at4. All six parameter inventories and distinct private block draws. Context buffer/identity guards; full256-token finite CPU forward at new depths8/16. No data access.',timeout_seconds=600,rss_limit_bytes=4*2**30,reserve_bytes=12*2**30,output_cap_bytes=8*2**20,whole_comparison_planning_cost='research/tokenizer-evidence/general-rsi-cross-size-attribution-001/context-transfer-planning-cost-v2.json',scientific_training=False,test_opened=False)
 OUT.mkdir();put(OUT/'plan.json',p)
 print('Prepared',sha(OUT/'plan.json'))
def make(p,depth,context,old=None):
 before=torch.random.get_rng_state().clone()
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(p['initializer'])
  if old=='depth':m=DepthModel(DepthSpec('native',depth,tuple(p['extra_group_seeds'])))
  elif old=='context':m=ContextModel(ContextSpec(context))
  else:m=Model(Spec(depth,context,tuple(p['extra_group_seeds'])))
 assert torch.equal(before,torch.random.get_rng_state())
 return m

def run(p):
 torch.set_num_threads(4)
 assert torch.__version__==p['torch_version'] and platform.python_version()==p['python_version']
 for f,h in p['sources'].items():assert sha(ROOT/f)==h,f
 # The optimizer source changes only its accepted model type/class and local contract.
 def body(path):
  tree=ast.parse(path.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef));cls.name='CanonicalOptimizer'
  for n in ast.walk(cls):
   if isinstance(n,ast.Name) and n.id in ('Model','DepthModel'):n.id='CanonicalModel'
  return ast.dump(cls,include_attributes=False)
 assert body(ROOT/'research/general_rsi_depth_context_v1/optimizer.py')==body(ROOT/'research/general_rsi_cross_size_v1/optimizer.py')
 x=torch.tensor([[0,1,42,256,999,50256,1000,17]],dtype=torch.int64);y=x.roll(-1,1)
 parity=[]
 for d,c,kind in [(4,128,'depth'),(8,128,'depth'),(16,128,'depth'),(4,256,'context')]:
  old=make(p,d,c,kind);new=make(p,d,c);oi=dict(old.ordered_parameters());ni=dict(new.ordered_parameters())
  assert list(oi)==list(ni) and all(torch.equal(oi[k],ni[k]) for k in oi)
  old.eval();new.eval()
  with torch.no_grad(),FP32RMSNormInference():a=old.primary_logits(x).values;b=new.primary_logits(x).values
  assert torch.equal(a,b)
  old.train();new.train();a=old.forward_train(x,y);b=new.forward_train(x,y)
  assert torch.equal(a.primary.values,b.primary.values) and torch.equal(a.loss,b.loss)
  a.loss.backward();b.loss.backward();assert all(torch.equal(oi[k].grad,ni[k].grad) for k in oi)
  op=depth_optimizer(old,.0025) if kind=='depth' else ContextOptimizer(old,.0025);np=build_optimizer(new,.0025)
  assert clip_gradients(oi.values(),1.)==clip_gradients(ni.values(),1.)
  op.step();np.step();assert all(torch.equal(oi[k],ni[k]) for k in oi)
  for oc,nc in zip(op._optimizers,np._optimizers):
   for og,ng in zip(oc.param_groups,nc.param_groups):
    assert {k:v for k,v in og.items() if k!='params'}=={k:v for k,v in ng.items() if k!='params'}
    os,ns=oc.state[og['params'][0]],nc.state[ng['params'][0]]
    assert os.keys()==ns.keys() and all(torch.equal(os[k],ns[k]) for k in os)
  parity.append(dict(depth=d,context=c,reference=kind,parameters=len(oi),loss=float(a.loss.detach()),all_parity_exact=True))
  del old,new,oi,ni,a,b,op,np,oc,nc,og,ng,os,ns;gc.collect()
 inventory=[];prefix={};shapes={}
 for d in (4,8,16):
  for c in (128,256):
   m=make(p,d,c);items=m.ordered_parameters();hashes={n:digest(v) for n,v in items}
   assert len(items)==counts(m.spec)[0] and sum(v.numel() for _,v in items)==counts(m.spec)[1]
   if c==128:
    assert all(hashes[k]==v for k,v in prefix.items());prefix=hashes;shapes[d]=hashes
   else:assert hashes==shapes[d]
   draws=[hashes[f'transformer.h.{i}.attn.dkv.weight'] for i in range(d)];assert len(set(draws))==d
   finite_long=None
   if d>4 and c==256:
    m.eval();xx=torch.arange(256,dtype=torch.int64)[None,:]
    with torch.no_grad(),FP32RMSNormInference():v=m.primary_logits(xx).values
    assert v.shape==(1,256,50257) and torch.isfinite(v).all();finite_long=True;del v
   inventory.append(dict(depth=d,context=c,parameter_tensors=len(items),parameters=sum(v.numel() for _,v in items),distinct_blocks=True,full256_finite=finite_long,parameter_sha256=hashes))
   del m,items;gc.collect()
 rejects=[]
 for d,c in [(12,128),(4,129),(True,128),(4,True)]:
  try:Spec(d,c,tuple(p['extra_group_seeds']))
  except ValueError:rejects.append([d,c])
  else:raise AssertionError('invalid shape accepted')
 m=make(p,4,128)
 try:m.primary_logits(torch.zeros((1,129),dtype=torch.int64))
 except ValueError:rejects.append('overlength input')
 else:raise AssertionError('overlength accepted')
 return dict(accepted=True,plan_sha256=sha(OUT/'plan.json'),parity=parity,inventory=inventory,negative_controls=rejects,optimizer_math_ast_equivalent=True,context_parameter_draws_equal=True,depth_prefix_draws_equal=True,caller_rng_unchanged=True,scientific_training=False,test_opened=False,scope='CPU combined-wrapper parity only; MPS/time-EMA restart/endpoint scorer/timing/memory remain unqualified.')

def main():
 import sys
 if sys.argv[1]=='prepare':prepare();return
 p=read(OUT/'plan.json');attempt=OUT/'attempt-01';attempt.mkdir()
 with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('r+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
  with CpuPrerequisiteGuard(attempt/'guard.json',ROOT,timeout_seconds=p['timeout_seconds'],rss_limit_bytes=p['rss_limit_bytes'],reserve_bytes=p['reserve_bytes']):
   try:r=run(p);put(attempt/'result.json',r);print(json.dumps(r['parity']))
   except BaseException as e:put(attempt/'failure.json',dict(exception=repr(e),traceback=traceback.format_exc(),plan_sha256=sha(OUT/'plan.json')));raise
if __name__=='__main__':main()
