"""Real MLA model/group parity, CPU/MPS update and beta2 checkpoint rejection."""
import copy,gc,hashlib,json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.adamw_beta2_model_v1.integration import identity,build_optimizer,load_model_weights
from research.adamw_beta2_model_v1.runtime.models.gpt import GPT
from research.adamw_beta2_model_v1.runtime.attention.cache import build_caches
from research.adamw_beta2_model_v1.runtime.utils.checkpoint import save_checkpoint
from research.adamw_beta2_model_v1.runtime.train.optimizers import POLICY
from research.gradient_clip_model_v1.clipping import clip_gradients
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.gradient_clip_quality_v2.quality_execution import idle
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
B=Path(__file__).resolve().parent;ROOT=B.parents[2]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def verify(plan):
 for p,h in plan['sources'].items():assert sha(ROOT/p)==h,p
 assert sha(ROOT/plan['config_path'])==plan['config_sha256']
def digest(model):
 h=hashlib.sha256()
 for n,p in model.state_dict().items():h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def groups(opt,model,normalize=False):
 names={id(p):n for n,p in model.named_parameters()};actual=[p for g in opt.param_groups for p in g['params']]
 assert len(actual)==len(names)==len({id(p) for p in actual})==88
 out=[]
 for child in opt.optimizers:
  rows=[]
  for g in child.param_groups:
   row={k:v for k,v in g.items() if k!='params'};row['names']=[names[id(p)] for p in g['params']]
   if normalize and 'betas' in row:row['betas']=(.9,.999)
   rows.append(row)
  out.append(dict(child=type(child).__name__,groups=rows))
 return out
def close(a,b,rtol=3e-4,atol=3e-5):
 a=a.detach().cpu();b=b.detach().cpu();torch.testing.assert_close(a,b,rtol=rtol,atol=atol);return float((a-b).abs().max())
def reject(fn):
 try:fn()
 except (ValueError,TypeError) as exc:return str(exc)
 raise AssertionError('Invalid optimizer identity accepted')
plan=json.loads((B/'model002-plan.json').read_text());verify(plan)
assert not any((B/n).exists() for n in ('model002.json','model002-guard.json','model002-failure.json'))
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
 idle()
 try:
  with StrictMpsGuard(B/'model002-guard.json',ROOT):
   base=OmegaConf.create(json.loads((ROOT/plan['config_path']).read_text()));base.training.adamw_beta2_policy=POLICY
   initial=None;rows=[];tokens=(torch.arange(18).reshape(2,9)%5+23).long();targets=tokens+1
   for beta2 in (.999,.95):
    cfg=copy.deepcopy(base);cfg.training.adamw_beta2=beta2;contract=identity(cfg)
    torch.manual_seed(91561);cpu=GPT(cfg.model);opt=build_optimizer(cpu,cfg);rng=torch.get_rng_state().clone();d=digest(cpu);gg=groups(opt,cpu)
    assert sum(p.numel() for p in cpu.parameters())==28492416
    with torch.no_grad():logits=cpu(tokens)[0].clone()
    if initial is None:
     initial=d;initial_rng=rng;initial_logits=logits;baseline=groups(opt,cpu,True)
     from research.readout_centering_model_v1.runtime.models.gpt import GPT as Incumbent
     from research.readout_centering_model_v1.integration import build_optimizer as incumbent_optimizer
     old=copy.deepcopy(cfg);del old.training.adamw_beta2;del old.training.adamw_beta2_policy
     torch.manual_seed(91561);incumbent=Incumbent(old.model);oldopt=incumbent_optimizer(incumbent,old)
     assert digest(incumbent)==d and torch.equal(rng,torch.get_rng_state()) and groups(oldopt,incumbent)==gg
     with torch.no_grad():assert torch.equal(incumbent(tokens)[0],logits)
     del incumbent,oldopt
    else:assert d==initial and torch.equal(rng,initial_rng) and torch.equal(logits,initial_logits) and groups(opt,cpu,True)==baseline
    gpu=copy.deepcopy(cpu).to('mps');gopt=build_optimizer(gpu,cfg);assert groups(gopt,gpu)==gg
    a,loss=cpu(tokens,targets);b,gloss=gpu(tokens.to('mps'),targets.to('mps'))
    row=dict(beta2=beta2,identity=contract,initial_state_sha256=d,logits_max_abs=close(a,b),loss_abs=close(loss,gloss,1e-5,1e-4))
    loss.backward();gloss.backward();row['gradient_max_abs']=max(close(p.grad,g.grad,3e-3,3e-5) for p,g in zip(cpu.parameters(),gpu.parameters()))
    for m in (cpu,gpu):clip_gradients(m.parameters(),1.)
    opt.step();gopt.step();torch.mps.synchronize();assert all(bool(torch.isfinite(p).all()) for m in (cpu,gpu) for p in m.parameters())
    row['cpu_mps_update_max_abs']=max(close(p,g,3e-3,3e-5) for p,g in zip(cpu.parameters(),gpu.parameters()))
    cpu.zero_grad(set_to_none=True);gpu.zero_grad(set_to_none=True)
    payload={};save_checkpoint(B/'not-written.pt',cpu,opt,step=1,cfg=cfg,payload_writer=lambda p,v:payload.update(v));load_model_weights(cpu,payload,cfg)
    other=copy.deepcopy(cfg);other.training.adamw_beta2=.95 if beta2==.999 else .999
    before=digest(cpu);row['foreign_config_rejected']=reject(lambda:load_model_weights(cpu,payload,other));assert digest(cpu)==before
    bad=copy.deepcopy(opt.state_dict());bad['optimizers'][1]['param_groups'][0]['betas']=(.9,other.training.adamw_beta2)
    states_before=[id(x.state) for x in opt.optimizers];row['foreign_saved_group_rejected']=reject(lambda:opt.load_state_dict(bad));assert states_before==[id(x.state) for x in opt.optimizers]
    with torch.no_grad():
     x=tokens.to('mps');full=gpu(x)[0];cache=build_caches(cfg,'mps',batch_size=2,dtype=torch.float32);parts=[];pos=0
     for count in (3,1,5):parts.append(gpu(x[:,pos:pos+count],cache=cache,start_pos=pos)[0]);pos+=count
     row['cache_max_abs']=close(torch.cat(parts,1),full)
     for cell in cache:cell.reset()
     row['reset_max_abs']=close(gpu(x,cache=cache,start_pos=0)[0],full)
     changed=x.clone();changed[:,4:]+=3;row['causality_max_abs']=close(gpu(changed)[0][:,:4],full[:,:4],0,0)
     row['cache_bytes']=sum(v.untyped_storage().nbytes() for cell in cache for v in cell.buffers());assert row['cache_bytes']==2*196608
    row['optimizer_groups']=gg;rows.append(row)
    del cpu,gpu,opt,gopt,payload,bad,a,b,loss,gloss,logits,full,cache,parts;gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
   invalid=[]
   for field,value in [('adamw_beta2',True),('adamw_beta2',.9),('adamw_beta2',float('nan')),('adamw_beta2_policy','implicit')]:
    cfg=copy.deepcopy(base);cfg.training.adamw_beta2=.999;cfg.training[field]=value;invalid.append(dict(field=field,value=repr(value),error=reject(lambda:identity(cfg))))
   verify(plan);result=dict(status='passed',rows=rows,invalid=invalid,shared_initialization_logits_rng_exact=True,incumbent_groups_exact=True,group_difference_only_beta2=True,parameter_count=28492416,ema_tensor_count=88,sources_unchanged=len(plan['sources']),scientific_training=False,checkpoint_io=False,scope='Real CPU/MPS one-update and in-memory raw payload; serialized repeat/resume/H200 still required. Optimizer beta2 does not alter cache geometry or forward semantics.')
  save(B/'model002.json',result);print(json.dumps(dict(status='passed',rows=[{k:v for k,v in r.items() if k!='optimizer_groups'} for r in rows])))
 except BaseException as exc:save(B/'model002-failure.json',dict(status='failed_closed',error=repr(exc),no_retry=True));raise
