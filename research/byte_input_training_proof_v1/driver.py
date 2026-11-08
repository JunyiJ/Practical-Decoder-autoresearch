"""Full-sized fresh real-train/val resume fixture, not a quality campaign."""
import copy,gc,hashlib,json
from pathlib import Path
import numpy as np
import torch
from research import byte_input_training_v2 as training
from research.byte_input_model_v3 import Spec,ByteInputModel,build_codebook,build_optimizer
from research.byte_input_proof_v1.transactions import tensor_record,snapshot
from .cases import BACKENDS,VARIANTS,HORIZON,BOUNDARY,REJECTIONS,LIVE_REJECTIONS,IO_REJECTIONS


def tree(x):
 if isinstance(x,torch.Tensor):return tensor_record(x,False)
 if isinstance(x,dict):return {k:tree(v)for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [tree(v)for v in x]
 return x

def copy_tree(x):
 if isinstance(x,dict):return {k:copy_tree(v)for k,v in x.items()}
 if isinstance(x,list):return [copy_tree(v)for v in x]
 if isinstance(x,tuple):return tuple(copy_tree(v)for v in x)
 return x

def cleanup():gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while x:=f.read(1048576):h.update(x)
 return h.hexdigest()

def semantic(c):
 params=dict(c.model.ordered_parameters());names={id(p):n for n,p in params.items()}
 return dict(parameters=tree(params),buffers=tree(dict(c.model.named_buffers())),
  children=[dict(kind=type(o).__name__,groups=[dict(name=names[id(g['params'][0])],options={k:v for k,v in g.items()if k!='params'})for g in o.param_groups],
   state={names[id(p)]:tree(v)for p,v in o.state.items()})for o in c.optimizer._optimizers],
  step=c.optimizer._steps,training=c.model.training,history=copy.deepcopy(c.history),
  averages=dict(spec=copy.deepcopy(c.averages.spec),step=c.averages.completed_steps,snapshots=c.averages.snapshots,values=tree(c.averages.averages)),
  samplers={n:tree(g.get_state())for n,g in c.view.generators.items()},
  cpu_rng=tree(torch.random.get_rng_state()),mps_rng=tree(torch.mps.get_rng_state()),identity=copy.deepcopy(c._identity))

def snap(c,probe):
 return dict(numerical=snapshot(c.model,c.optimizer,probe,probe),history=copy.deepcopy(c.history),
  averages=dict(id=id(c.averages),steps=c.averages.completed_steps,snapshots=c.averages.snapshots,
   spec=copy.deepcopy(c.averages.spec),betas=copy.deepcopy(c.averages.betas),
   tensors={h:{n:tensor_record(v)for n,v in panel.items()}for h,panel in c.averages.averages.items()}),
  samplers={n:dict(id=id(g),state=tree(g.get_state()))for n,g in c.view.generators.items()},
  data={n:tensor_record(v)for n,v in c.view.splits.items()},view_device=c.view.device,
  schedule=dict(spec=copy.deepcopy(c.schedule.spec),base=list(c.schedule.base_lrs),identities=c.schedule.identities),identity=copy.deepcopy(c._identity))

def construct(plan,book,variant,backend):
 spec=Spec(variant)if book is None else Spec(variant,plan['roles']['composer_initializer'],book.identity)
 old_mps=torch.mps.get_rng_state().clone()
 try:
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(plan['roles']['native_initializer']);model=ByteInputModel(spec,book)
 finally:torch.mps.set_rng_state(old_mps)
 model=model.to(backend);opt=build_optimizer(model,.00125)
 view=training.load_view(plan['roles'],backend)
 return training.Context(model,opt,view,HORIZON,.00125,plan['roles'],plan['binding'])

def expected_batch(view,split):
 data=view.splits[split];gen=torch.Generator(device='cpu').set_state(view.generators[split].get_state())
 if split=='train':
  starts=torch.randint(len(data)-512,(2,),generator=gen)
  x=torch.cat([data[int(i):int(i)+512]for i in starts]).reshape(8,128)
  y=torch.cat([data[int(i)+1:int(i)+513]for i in starts]).reshape(8,128)
 else:
  starts=torch.randint(len(data)-128,(8,),generator=gen)
  x=torch.stack([data[int(i):int(i)+128]for i in starts]);y=torch.stack([data[int(i)+1:int(i)+129]for i in starts])
 return x,y,gen.get_state()

def batch_record(x,y):return dict(shape=list(x.shape),inputs=x.detach().cpu().tolist(),labels=y.detach().cpu().tolist())
def validation_draw(c):
 x,y,state=expected_batch(c.view,'val');actual_x,actual_y=c.view.get_batch('val')
 assert torch.equal(actual_x.cpu(),x)and torch.equal(actual_y.cpu(),y)and torch.equal(c.view.generators['val'].get_state(),state)
 return dict(batch=batch_record(actual_x,actual_y),expected_batch=batch_record(x,y),sampler=tree(c.view.generators['val'].get_state()),expected_sampler=tree(state))

def observe_step(c,publish,label,shadow):
 expected_x,expected_y,next_rng=expected_batch(c.view,'train');observed=[];original=c.view.get_batch
 def observe(split='train'):
  x,y=original(split);observed.append(batch_record(x,y));return x,y
 c.view.get_batch=observe
 try:row=c.step()
 finally:del c.view.get_batch
 assert observed==[batch_record(expected_x,expected_y)] and torch.equal(c.view.generators['train'].get_state(),next_rng)
 step=c.optimizer._steps
 ratio=.1 if step==4 else 1.
 rates=[g['lr']for g in c.optimizer.param_groups]
 expected_rates=[.00125*(.5 if i==0 else 1.)*ratio for i,o in enumerate(c.optimizer._optimizers)for _ in o.param_groups]
 assert rates==expected_rates
 assert len(c.history)==step and row['gradient_tensors']==(88 if c.model.spec.variant=='native'else 105)
 if step>=2:
  alpha=1.-2.**(-1./200.)
  maximum=0.;passed=True
  for name,p in c.model.ordered_parameters():
   value=p.detach().cpu().numpy()
   if step==2:shadow[name]=value.astype('float64')
   else:shadow[name]+=alpha*(value.astype('float64')-shadow[name])
   actual=c.averages.averages['200'][name].detach().cpu().numpy().astype('float64')
   error=np.abs(actual-shadow[name]);maximum=max(maximum,float(error.max(initial=0.)))
   passed=passed and bool((error<=4e-6+4e-5*np.abs(shadow[name])).all())
  assert c.averages.snapshots==step-1
  publish(label+'-ema-reference.json',dict(step=step,passed=passed,max_abs=maximum,parameters=len(shadow),coordinates=sum(v.size for v in shadow.values()),atol=4e-6,rtol=4e-5))
  assert passed
 else:assert not c.averages.averages and c.averages.snapshots==0
 record=dict(batch=observed[0],expected_batch=batch_record(expected_x,expected_y),expected_sampler=tree(next_rng),history=row,rates=rates,state=semantic(c))
 publish(label+'.json',record);cleanup();return record

def mutate(name,payload):
 p=copy_tree(payload);ema=p['averages'];panel=ema['averages']['200'];small=min(panel,key=lambda n:panel[n].numel())
 if name=='extra_field':p['foreign']=True
 elif name=='kind':p['kind']='foreign'
 elif name=='identity_recipe':p['identity']['recipe']['max_iters']=20
 elif name=='identity_roles':p['identity']['roles']['train_sampler']=p['identity']['roles']['validation_sampler']
 elif name=='identity_data':p['identity']['data']={}
 elif name=='numerical_kind':p['numerical']['kind']='foreign'
 elif name=='history_missing':p['history']=p['history'][:-1]
 elif name=='history_negative_loss':p['history'][0]['ce_loss']=-1.
 elif name=='history_count':p['history'][0]['gradient_tensors']=1
 elif name=='history_names':p['history'][0]['active_parameter_names']=[]
 elif name=='history_coefficient':p['history'][0]['coefficient']=2.
 elif name=='scheduled_lr':
  for i,child in enumerate(p['numerical']['children']):
   for group in child['groups']:group['options']['lr']=.0025*(.5 if i==0 else 1.)
 elif name=='ema_step':ema['completed_steps']=1
 elif name=='ema_spec':ema['spec']['half_lives']=[100]
 elif name=='ema_snapshots':ema['snapshots']=2
 elif name=='ema_names':ema['parameter_names']=[]
 elif name=='ema_panel_missing':ema['averages']={}
 elif name=='ema_parameter_missing':del panel[small]
 elif name=='ema_shape':panel[small]=panel[small].reshape(*panel[small].shape,1).clone()
 elif name=='ema_dtype':panel[small]=panel[small].double()
 elif name=='ema_nonfinite':panel[small]=torch.full_like(panel[small],float('nan'))
 elif name=='ema_alias':panel[small]=p['numerical']['parameters'][small]
 elif name=='sampler_missing':del p['samplers']['train']
 elif name=='sampler_shape':p['samplers']['train']=torch.zeros(1,dtype=torch.uint8)
 elif name=='sampler_dtype':p['samplers']['train']=p['samplers']['train'].float()
 elif name=='sampler_swapped':p['samplers']['train']=p['samplers']['val'].clone()
 elif name=='sampler_alias':p['samplers']['train']=p['numerical']['cpu_rng']
 else:raise AssertionError(name)
 return p

def invalid(c,payload,probe,publish,prefix):
 for name in REJECTIONS:
  bad=mutate(name,payload);before=snap(c,probe);pb=tree(bad);error=None
  try:training.restore_pristine(c,bad)
  except (ValueError,RuntimeError,TypeError)as exc:error=repr(exc)
  after=snap(c,probe);pa=tree(bad)
  publish(prefix+'-reject-'+name+'.json',dict(error=error,before=before,after=after,payload_before=pb,payload_after=pa))
  assert error and before==after and pb==pa,name
 for name in LIVE_REJECTIONS:
  if name=='sampler_device':obj=c.view;key='device';old=obj.device;obj.device='cpu'if old=='mps'else'mps'
  elif name=='sampler_seed':obj=c.view;key='generators';old=obj.generators;obj.generators=dict(old,train=old['val'])
  elif name=='ema_coefficient':obj=c.averages;key='betas';old=obj.betas;obj.betas={'200':.5}
  elif name=='schedule_base':obj=c.schedule;key='base_lrs';old=obj.base_lrs;obj.base_lrs=tuple(v*2 for v in old)
  else:raise AssertionError(name)
  try:
   before=snap(c,probe);error=None
   try:training.restore_pristine(c,payload)
   except (ValueError,RuntimeError,TypeError)as exc:error=repr(exc)
   after=snap(c,probe);publish(prefix+'-live-'+name+'.json',dict(error=error,before=before,after=after));assert error and before==after,name
  finally:setattr(obj,key,old)

def draw():return dict(cpu=tree(torch.rand(257)),mps=tree(torch.rand(257,device='mps')))

def one(plan,archives,publish,book,variant,backend):
 prefix=backend+'-'+variant;probe=torch.zeros(1,dtype=torch.int64);c=construct(plan,book,variant,backend)
 before=snap(c,probe);zero=training.capture(c);assert before==snap(c,probe)
 empty=semantic(c);assert training.restore_pristine(c,zero)==0 and semantic(c)==empty
 publish(prefix+'-empty.json',dict(exact=True,state=empty));del zero,before,empty;cleanup()
 publish(prefix+'-validation-initial.json',validation_draw(c))
 shadow={}
 for step in (1,2):observe_step(c,publish,prefix+'-prefix'+str(step),shadow)
 start=semantic(c);before=snap(c,probe)
 receipt=training.write_new(c,(archives/(prefix+'-step2.pt')).resolve());assert receipt['size']<=(512 if variant=='native'else 320)*1024**2;publish(prefix+'-archive2.json',receipt)
 assert before==snap(c,probe)
 payload=training.read_bound(receipt);assert training.validate(c,payload)==BOUNDARY
 assert payload['history']==c.history and tree(payload['samplers'])=={n:tree(g.get_state())for n,g in c.view.generators.items()}
 for name in IO_REJECTIONS:
  r=dict(receipt);error=None;before=snap(c,probe);original_hash=sha(receipt['path'])
  if name=='wrong_hash':r['sha256']='0'*64
  elif name=='wrong_size':r['size']+=1
  elif name=='wrong_kind':r['kind']='foreign'
  try:
   if name=='overwrite':training.write_new(c,Path(receipt['path']))
   else:training.read_bound(r)
  except (ValueError,TypeError,RuntimeError)as exc:error=repr(exc)
  after=snap(c,probe);publish(prefix+'-io-'+name+'.json',dict(error=error,before=before,after=after,original_sha256=original_hash,after_sha256=sha(receipt['path'])));assert error and before==after and original_hash==sha(receipt['path'])
 expected_draw=draw();expected_val=validation_draw(c);expected={}
 for step in (3,4):
  expected[step]=observe_step(c,publish,prefix+'-control'+str(step),shadow)
  if step==3:
   values={n:p.detach().cpu().numpy()for n,p in c.model.ordered_parameters()}
   middle=archives/(prefix+'-parameters3.npz')
   with middle.open('xb') as f:
    np.savez(f,**values);f.flush();__import__('os').fsync(f.fileno())
   del values
   assert middle.stat().st_size<=(128 if variant=='native'else 80)*1024**2
   publish(prefix+'-middle.json',dict(path=str(middle),size=middle.stat().st_size,sha256=sha(middle)))
 final_receipt=training.write_new(c,(archives/(prefix+'-step4.pt')).resolve());assert final_receipt['size']<=(512 if variant=='native'else 320)*1024**2;publish(prefix+'-archive4.json',final_receipt)
 before=snap(c,probe);error=None
 try:c.step()
 except ValueError as exc:error=repr(exc)
 after=snap(c,probe);publish(prefix+'-horizon.json',dict(error=error,before=before,after=after));assert error and before==after
 del c,shadow,before,after;cleanup()
 c=construct(plan,book,variant,backend);invalid(c,payload,probe,publish,prefix)
 draw();assert training.restore_pristine(c,payload)==BOUNDARY
 restored=semantic(c);publish(prefix+'-restored-boundary.json',dict(expected=start,actual=restored));assert restored==start
 actual_draw=draw();actual_val=validation_draw(c)
 publish(prefix+'-continuation-draws.json',dict(expected_rng=expected_draw,actual_rng=actual_draw,expected_val=expected_val,actual_val=actual_val))
 assert actual_draw==expected_draw and actual_val==expected_val
 shadow={n:v.numpy().astype('float64')for n,v in payload['averages']['averages']['200'].items()}
 for step in (3,4):
  actual=observe_step(c,publish,prefix+'-resumed'+str(step),shadow)
  assert actual==expected[step], 'Exact sampled trajectory, history, rates, parameters, optimizer/EMA and RNG'
 before=snap(c,probe);error=None
 try:training.restore_pristine(c,payload)
 except ValueError as exc:error=repr(exc)
 after=snap(c,probe);publish(prefix+'-nonpristine.json',dict(error=error,before=before,after=after));assert error and before==after
 publish(prefix+'-complete.json',dict(backend=backend,variant=variant,shape=[8,128,50257],horizon=4,resume_boundary=2,updates_executed=6,
  rejected=len(REJECTIONS)+len(LIVE_REJECTIONS)+len(IO_REJECTIONS)+2,exact=True,ema_snapshots=3,parameter_count=len(c.model.ordered_parameters()),scope='Training-context implementation fixture, not scientific quality evaluation'))
 del c,payload,shadow,expected,start,restored,probe;cleanup()

def run(plan,archives,publish):
 assert torch.__version__==plan['runtime']['torch_version'] and torch.version.git_version==plan['runtime']['torch_git']
 assert torch.backends.mps.is_available();torch.set_num_threads(4)
 cpu=torch.random.get_rng_state().clone();mps=torch.mps.get_rng_state().clone()
 try:
  raw=[Path(pin['path']).read_bytes()for pin in plan['codebook_assets']]
  books={v:build_codebook(*raw,v,None if v=='raw_bytes'else plan['roles']['nonlexical_key'])for v in VARIANTS if v!='native'}
  for backend in BACKENDS:
   for variant in VARIANTS:one(plan,archives,publish,books.get(variant),variant,backend)
 finally:torch.random.set_rng_state(cpu);torch.mps.set_rng_state(mps)
 assert torch.equal(cpu,torch.random.get_rng_state())and torch.equal(mps,torch.mps.get_rng_state())
 publish('result.json',dict(accepted=True,cases=6,updates_executed=36,invalid_transactions=222,caller_rng_restored=True,
  exact_continuation=True,no_holdout_access=True,no_quality_claim=True,roles=plan['roles']))
