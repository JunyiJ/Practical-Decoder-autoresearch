"""Complete post-training work from original slot14 RAW; never run an update."""
import copy,gc,hashlib,time,json
from pathlib import Path
import torch
from research import byte_input_training_v2 as training
from research import byte_input_inference_v1 as inference
from research.byte_input_cache_v1 import Session
from research.byte_input_model_v3 import Spec,ByteInputModel,build_optimizer
from research.byte_input_quality_v1.driver import clean

def run(plan,spec,archives,publish,parent,original_out):
 assert spec==dict(slot=14,stage='confirmation',bundle='fresh1',variant='native',learning_rate=.00125)
 torch.set_num_threads(4);cpu=torch.random.get_rng_state().clone();mps=torch.mps.get_rng_state().clone()
 roles=plan['bundles'][spec['bundle']];lr=spec['learning_rate'];book=None;variant='native'
 try:
  old_mps=torch.mps.get_rng_state().clone()
  try:
   with torch.random.fork_rng(devices=[]):
    torch.manual_seed(roles['native_initializer']);model=ByteInputModel(Spec('native')).to('mps')
  finally:torch.mps.set_rng_state(old_mps)
  view=training.load_view(roles,'mps')
  c=training.Context(model,build_optimizer(model,lr),view,1200,lr,roles,plan['binding']);del model
  raw=training.read_bound(parent)
  assert training.restore_pristine(c,raw)==1200
  del raw
  assert c.optimizer._steps==c.averages.completed_steps==1200 and c.averages.snapshots==41
  assert parent['size']<=512*1024**2
  identity=copy.deepcopy(c._identity)
  rows=[json.loads((original_out/('step'+str(i).zfill(4)+'.json')).read_text()) for i in range(1,1201)]
  clean()
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
