"""Distinct three-pilot H200 consumer cost; never a fresh-panel benchmark."""
import argparse,gc,json,os,shutil,statistics,time
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.readout_mixture_model_v1.model import GPT
from research.readout_mixture_resume_v1.checkpoint import load_export
from research.readout_mixture_resume_v1.data import model_spec
from research.readout_mixture_resume_v1.contract import runtime
from research.readout_mixture_quality_v1 import contract as C
from research.readout_mixture_quality_v1.validation import verify_artifact,cache_check
from research.readout_mixture_worker_v1.execution import idle
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock

E=Path(__file__).resolve().parent/'serving-probe001'
def publish(path,value):
 raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
 require(len(raw)<2*1024**2,'Bounded probe record required')
 with Path(path).open('xb') as stream:stream.write(raw)
def verify_sources(pins):
 for name,digest in pins.items():require(reference(ROOT/name)['sha256']==digest,'Pinned source changed: '+name)

def consume(row):
 began=time.perf_counter();worker=C.checked(row['worker']);cfg=OmegaConf.create(worker['config'])
 require(cfg.training.max_iters==20 and worker['step']==20,'Only original accepted twenty-pilot consumer inputs')
 start=time.perf_counter();verify_artifact(worker['h200']);initial_hash_seconds=time.perf_counter()-start
 with torch.random.fork_rng(devices=[]):
  start=time.perf_counter();model=GPT(model_spec(cfg),initializer_seed=cfg.initializer_seed).to('mps').eval();torch.mps.synchronize();construction_seconds=time.perf_counter()-start
  start=time.perf_counter();h=torch.load(worker['h200']['path'],map_location='cpu',weights_only=True);load_export(model,h,cfg);del h;torch.mps.synchronize();load_seconds=time.perf_counter()-start
  start=time.perf_counter();cache_checks=cache_check(model);torch.mps.synchronize();cache_check_seconds=time.perf_counter()-start
  measures=[];allocation_seconds=0.;protocol_start=time.perf_counter()
  with torch.no_grad():
   for batch in (1,8):
    start=time.perf_counter();x=(torch.arange(batch*128).reshape(batch,128)%400+1).to('mps');c=model.caches(batch);torch.mps.synchronize();allocation_seconds+=time.perf_counter()-start
    bytes_item=sum(t.numel()*t.element_size() for item in c.layers for t in item.buffers())//batch
    require(bytes_item==196608,'Exact native cache storage required')
    full_times=[];decode_times=[]
    for repeat in range(7):
     c.reset()
     torch.mps.synchronize();start=time.perf_counter();full=model(x).log_probs();torch.mps.synchronize();full_time=time.perf_counter()-start
     prefill=model(x[:,:96],cache=c).log_probs();del prefill;torch.mps.synchronize();start=time.perf_counter()
     for index in range(96,128):
      last=model(x[:,index:index+1],cache=c,start_pos=index).log_probs()
      if repeat==0:torch.testing.assert_close(full[:,index:index+1],last,rtol=3e-4,atol=3e-5)
     torch.mps.synchronize();decode_time=time.perf_counter()-start
     if repeat>=2:full_times.append(full_time);decode_times.append(decode_time)
     del full,last
    measures.append(dict(batch=batch,cache_bytes_per_item=bytes_item,full128_seconds=full_times,decode32_after_prefill96_seconds=decode_times,median_full128_seconds=statistics.median(full_times),median_decode32_seconds=statistics.median(decode_times)))
    del c,x
  protocol_seconds=time.perf_counter()-protocol_start
  del model;gc.collect();torch.mps.empty_cache()
 start=time.perf_counter();verify_artifact(worker['h200']);final_hash_seconds=time.perf_counter()-start
 return dict(arm=row['arm'],worker=row['worker'],h200=worker['h200'],measurements=measures,cache_reset_checks=cache_checks,wall_seconds=time.perf_counter()-began,
  overhead=dict(initial_hash_seconds=initial_hash_seconds,construction_seconds=construction_seconds,checkpoint_load_seconds=load_seconds,cache_check_seconds=cache_check_seconds,cache_and_input_allocation_seconds=allocation_seconds,protocol_wall_seconds=protocol_seconds,final_hash_seconds=final_hash_seconds))

def main(args):
 require(reference(E/'plan.json')['sha256']==args.plan_sha,'Exact checked probe plan required')
 plan=C.checked(reference(E/'plan.json'));require(plan['source']==reference(__file__) and plan['kind']=='three_pilot_readout_serving_consumer_cost_v1','Distinct pinned consumer plan required')
 require(not (E/'attempt.json').exists() and not (E/'STOP').exists(),'Original consumer command cannot retry')
 with session_lock(GLOBAL_LOCK.parent):
  torch.set_num_threads(4);idle();require(runtime()==plan['environment'],'Pinned strict environment required');verify_sources(plan['sources'])
  require(shutil.disk_usage(ROOT).free>=C.TARGET,'Complete remaining retained target required')
  review=C.checked(plan['pilot_review']);require(review['status']=='all_three_readout_twenty_update_pilots_independently_accepted' and [r['arm'] for r in review['rows']]==list(C.DESIGN['arms']),'Exactly three original accepted pilot inputs required')
  require([C.checked(r['worker'])['h200'] for r in review['rows']]==plan['inputs'],'Pilot input allowlist changed')
  C.checked(plan['source_review']);publish(E/'attempt.json',dict(pid=os.getpid(),plan=reference(E/'plan.json'),source=reference(__file__)))
  try:
   with StrictMpsGuard(E/'guard.json',ROOT):
    cpu_before=torch.get_rng_state().clone();mps_before=torch.mps.get_rng_state().clone();began=time.perf_counter()
    try:
     rows=[consume(row) for row in review['rows']]
     require(torch.equal(cpu_before,torch.get_rng_state()) and torch.equal(mps_before,torch.mps.get_rng_state()),'Read-only consumer changed global RNG')
    finally:torch.set_rng_state(cpu_before);torch.mps.set_rng_state(mps_before)
    require(torch.equal(cpu_before,torch.get_rng_state()) and torch.equal(mps_before,torch.mps.get_rng_state()),'Caller RNG preservation required')
    verify_sources(plan['sources']);require(len(rows)==3,'Distinct three-input scope required')
    for ref in plan['inputs']:verify_artifact(ref)
    require(sum(p.stat().st_size for p in E.rglob('*') if p.is_file())<1024**2,'Probe metadata must retain one MiB publication margin')
    publish(E/'result.json',dict(status='passed_distinct_three_pilot_serving_consumer_probe',rows=rows,elapsed_seconds=time.perf_counter()-began,plan=reference(E/'plan.json'),sources_unchanged=True,input_hashes_unchanged=True,caller_cpu_mps_rng_preserved=True,corpus_loaded=False,holdout_scored=False,optimizer_updates=0,weight_files_written=0,scientific_roles_used=False,fresh_serving_replaced=False,limitation='Pilot artifact consumer cost with generated IDs; serial local timings are descriptive, not stable bounds or quality/adoption evidence.'))
   publish(E/'STOP',dict(status='complete_distinct_three_pilot_serving_consumer_probe',result=reference(E/'result.json'),guard=reference(E/'guard.json'),no_retry=True,no_replay=True))
  except BaseException as exc:
   if not (E/'STOP').exists():publish(E/'STOP',dict(status='closed_failed',error=repr(exc),no_retry=True))
   raise
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--plan-sha',required=True);main(parser.parse_args())
