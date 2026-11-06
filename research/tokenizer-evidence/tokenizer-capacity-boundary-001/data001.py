"""Real encoded data/private sampler proof; no optimizer updates/model scoring."""
import copy,json
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require,config as native_config
from research.body_allocation_model_v1.evidence import reference,environment
from research.body_allocation_worker_v1.data import load as original_load
from research.gpt2_prefix_data_v1.data import config,load,restore_samplers,architecture
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent

def reject(fn):
 try:fn()
 except (ValueError,RuntimeError):return
 raise AssertionError('Invalid state accepted')
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
with session_lock(GLOBAL_LOCK.parent):
 torch.set_num_threads(4);idle();env=environment();require(not(E/'data001.json').exists(),'New data/private sampler proof only')
 with CpuPrerequisiteGuard(E/'data001-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
  component=json.loads((E/'component002.json').read_text());require(component['status']=='passed_native_and8K_component_correctness','Actual corrective component proof required');pins=dict(component['source_bindings'])
  for p in (ROOT/'research/gpt2_prefix_data_v1').glob('*.py'):pins[str(p.relative_to(ROOT))]=reference(p)['sha256']
  require(len(pins)==984,'Exact additive source inventory required')
  for n,h in pins.items():require(reference(ROOT/n)['sha256']==h,'Proven source changed')
  before=torch.random.get_rng_state().clone();reports=[]
  for kind in ['native','prefix8192']:
   cfg=config(kind,seed=600422);a=load(cfg);b=load(cfg);require(a.data_identity==b.data_identity and set(a.splits)=={'train','val'} and not hasattr(a,'test_data'),'Exact private permitted data required')
   for split in ['train','val']:
    for _ in range(20):
     x,y=a.get_batch(split);u,v=b.get_batch(split);require(torch.equal(x,u) and torch.equal(y,v) and x.shape==y.shape==(8,128),'Fresh private sampler repeat differs')
   state={n:g.get_state().clone() for n,g in a.generators.items()};want=[a.get_batch(s) for s in ['train','val','train']];restore_samplers(b,state);got=[b.get_batch(s) for s in ['train','val','train']];require(all(torch.equal(x,u) and torch.equal(y,v) for (x,y),(u,v) in zip(want,got)),'Private sampler restore differs')
   snapshot={n:g.get_state().clone() for n,g in b.generators.items()}
   for bad in [{**state,'val':state['val'][:-1]},{'train':state['train']},{**state,'val':state['val'].to(torch.int64)}]:
    reject(lambda bad=bad:restore_samplers(b,bad));require(all(torch.equal(g.get_state(),snapshot[n]) for n,g in b.generators.items()),'Rejected sampler state mutated destination')
   reject(lambda:a.get_batch('test'));bad=copy.deepcopy(cfg);bad.prefix_model.effective_vocab_size=4096;reject(lambda:architecture(bad))
   if kind=='native':
    native=original_load(native_config('incumbent',steps=4,seed=600422));fresh=load(cfg);require(torch.equal(native.train_data,fresh.train_data) and torch.equal(native.val_data,fresh.val_data),'Native actual IDs differ')
    for split in ['train','val']:
     for _ in range(20):
      x,y=native.get_batch(split);u,v=fresh.get_batch(split);require(torch.equal(x,u) and torch.equal(y,v),'Native inherited sampler differs')
   reports.append(dict(kind=kind,data_identity=a.data_identity,sampler_repeat_draws_per_split=20,restore_draws=3,invalid_sampler_states_rejected=3,holdout_rejected=True,common_target_bytes=a.common_byte_plan['target_bytes'],genuine_new_config_without_old_model_masquerade=True))
  require(torch.equal(before,torch.random.get_rng_state()),'Data construction consumed global CPU RNG')
  result=dict(status='passed_real_data_and_private_sampler_contracts',source=reference(__file__),environment=env,source_bindings=pins,total_source_bindings=984,reports=reports,global_cpu_rng_unchanged=True,fixture_seed=600422,
   real_model_training=False,model_scoring=False,new_scientific_seed=False,holdout_encoded_or_scored=False,weight_files=0,
   next_action='Implement additive raw/H200 optimizer/EMA/clipping/CPU-MPS RNG transport with this explicit codec/model/data identity,then real4update fresh-repeat/serialized2+2resume proof before worker/resource/scientific gates. No numerical quality panel frozen.')
  save(E/'data001.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['source_bindings']}),flush=True)
