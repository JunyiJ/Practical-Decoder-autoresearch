"""One fixed1200-update slot through the already accepted numerical primitives."""
import copy,gc,hashlib,time
from pathlib import Path
import torch
from research import byte_input_training_v2 as training
from research import byte_input_inference_v1 as inference
from research.byte_input_cache_v1 import Session
from research.byte_input_model_v3 import Spec,ByteInputModel,build_codebook,build_optimizer

def clean():gc.collect();torch.mps.synchronize();torch.mps.empty_cache()

def diagnostic(c,step):
 c.check();train=c.view.generators['train'].get_state().clone()
 cpu=torch.random.get_rng_state().clone();mps=torch.mps.get_rng_state().clone()
 versions=[p._version for _,p in c.model.ordered_parameters()];mode=c.model.training
 c.model.eval();losses=[]
 try:
  with torch.no_grad():
   for _ in range(10):
    x,y=c.view.get_batch('val');result=c.model.forward_train(x,y)
    losses.append(float(result.loss));del result,x,y
 finally:c.model.train(mode)
 assert torch.equal(train,c.view.generators['train'].get_state())
 assert torch.equal(cpu,torch.random.get_rng_state()) and torch.equal(mps,torch.mps.get_rng_state())
 assert versions==[p._version for _,p in c.model.ordered_parameters()]
 c.check()
 return dict(step=step,targets_seen=step*1024,eval_iters=10,validation_ce=sum(losses)/10,
  individual_batch_losses=losses,train_sampler_unchanged=True,global_rng_unchanged=True,parameters_unchanged=True,
  diagnostic_only=True)

def run(plan,spec,archives,publish):
 torch.set_num_threads(4);cpu=torch.random.get_rng_state().clone();mps=torch.mps.get_rng_state().clone()
 variant=spec['variant'];lr=spec['learning_rate'];roles=plan['bundles'][spec['bundle']]
 assets=[Path(r['path']).read_bytes() for r in plan['codebook_assets']]
 try:
  book=None if variant=='native' else build_codebook(*assets,variant,roles['nonlexical_key'] if variant=='nonlexical_codes' else None)
  old_mps=torch.mps.get_rng_state().clone()
  try:
   with torch.random.fork_rng(devices=[]):
    torch.manual_seed(roles['native_initializer'])
    model=ByteInputModel(Spec(variant) if book is None else Spec(variant,roles['composer_initializer'],book.identity),book).to('mps')
  finally:torch.mps.set_rng_state(old_mps)
  view=training.load_view(roles,'mps')
  c=training.Context(model,build_optimizer(model,lr),view,1200,lr,roles,plan['binding']);del model
  identity=copy.deepcopy(c._identity);rows=[]
  for step in range(1,1201):
   torch.mps.synchronize();started=time.perf_counter();row=c.step();torch.mps.synchronize()
   row=dict(row,elapsed_seconds=time.perf_counter()-started,mps_driver_bytes=torch.mps.driver_allocated_memory(),mps_current_bytes=torch.mps.current_allocated_memory())
   assert row['step']==step and row['gradient_tensors']==(88 if variant=='native' else 105)
   publish('step'+str(step).zfill(4)+'.json',row);rows.append(row)
   if step in (1,301,601,901,1200):publish('diagnostic'+str(step).zfill(4)+'.json',diagnostic(c,step))
  assert c.optimizer._steps==c.averages.completed_steps==1200 and c.averages.snapshots==41
  assert all(g['lr']==lr*(.5 if i==0 else 1.)*.1 for i,o in enumerate(c.optimizer._optimizers) for g in o.param_groups)
  clean();parent=training.write_new(c,archives/'raw.pt')
  assert parent['size']<=(512 if variant=='native' else 320)*1024**2
  averaged=inference.export(c,parent);h200=inference.write_new(archives/'h200.pt',averaged)
  assert h200['size']<=(128 if variant=='native' else 80)*1024**2
  publish('artifacts.json',dict(parent=parent,h200=h200,identity=identity,completed_steps=1200,averaging_snapshots=41,
   parameters=len(c.model.ordered_parameters()),parameter_elements=sum(p.numel() for _,p in c.model.ordered_parameters())))
  del c,averaged;clean();model=inference.load(h200,identity,parent,book,'mps')
  started=time.perf_counter();score=inference.score(model,view,identity,h200);score_seconds=time.perf_counter()-started
  publish('validation.json',score)
  cache_rows=[];timings=[]
  for batch in (1,8):
   ids=view.train_data[:128].unsqueeze(0).repeat(batch,1).to('mps')
   with torch.no_grad():full=model.primary_logits(ids).values
   session=Session(model,batch);maximum=0.;gate=0.;hashes=[]
   chunks=[(0,32)]+[(i,i+1) for i in range(32,96)]+[(96,128)]
   for replay in (0,1):
    if replay:session.reset()
    for i,(lo,hi) in enumerate(chunks):
     actual=session.logits(ids[:,lo:hi],lo);target=full[:,lo:hi]
     torch.testing.assert_close(actual,target,atol=3e-5,rtol=3e-4,equal_nan=False)
     error=(actual-target).abs();maximum=max(maximum,float(error.max()));gate=max(gate,float((error/(3e-5+3e-4*target.abs())).max()))
     h=hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest()
     if replay:assert h==hashes[i]
     else:hashes.append(h)
     del actual,target,error
   cache_rows.append(dict(batch=batch,cache_bytes=session.storage_bytes,max_absolute_error=maximum,max_gate_ratio=gate,reset_exact=True))
   del full,session
   # Descriptive serving cost only. Construction/reset are outside timed sections.
   for repeat in range(7):
    session=Session(model,batch);torch.mps.synchronize();started=time.perf_counter()
    with torch.no_grad():output=inference._logits(model,ids)
    torch.mps.synchronize();full_seconds=time.perf_counter()-started;del output
    torch.mps.synchronize();started=time.perf_counter();output=session.logits(ids[:,:96],0)
    torch.mps.synchronize();prefill_seconds=time.perf_counter()-started;del output
    started=time.perf_counter()
    for pos in range(96,128):output=session.logits(ids[:,pos:pos+1],pos);del output
    torch.mps.synchronize();decode_seconds=time.perf_counter()-started
    timings.append(dict(batch=batch,repeat=repeat,warmup=repeat<2,full_seconds=full_seconds,prefill96_seconds=prefill_seconds,decode32_seconds=decode_seconds))
    del session
   del ids;clean()
  model.validate();publish('inference.json',dict(correctness=cache_rows,timings=timings,scope='One resident averaged model; two warmups and five measured repeats. Loading/cache allocation excluded. Validation and safety checks in each timed consumer remain included. Descriptive costs, no cost-only adoption.'))
  publish('result.json',dict(accepted=True,spec=spec,parent=parent,h200=h200,completed_steps=1200,targets_seen=1228800,
   validation_targets=score['report']['target']['targets'],validation_seconds=score_seconds,
   total_update_seconds=sum(r['elapsed_seconds'] for r in rows),max_update_seconds=max(r['elapsed_seconds'] for r in rows),
   mean_update_seconds=sum(r['elapsed_seconds'] for r in rows)/1200,max_mps_driver_bytes=max(r['mps_driver_bytes'] for r in rows),
   quality_selected=False))
  del model,view,book,score;clean()
 finally:torch.random.set_rng_state(cpu);torch.mps.set_rng_state(mps);clean()
