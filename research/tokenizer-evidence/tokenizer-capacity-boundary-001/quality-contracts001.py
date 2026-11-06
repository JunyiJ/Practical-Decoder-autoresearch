"""Actual pilot state/layout audit plus isolated full-controller forgeries;zero training."""
import contextlib,copy,gc,hashlib,io,json,math,sys
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.body_allocation_model_v1.evidence import reference,environment
from research.body_allocation_model_v1.optimizer import build_optimizer
from research.gpt2_prefix_model_v2.model import GPT
from research.gpt2_prefix_data_v1.data import load,config
from research.gpt2_prefix_resume_v1.checkpoint import validate_resume
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
from research.tokenizer_common_bytes_v1.metric import loss_summary
from research.gpt2_prefix_statistics_v1.statistics import review
from research.gpt2_prefix_quality_v1 import contract as C,execution as Q
from research.gpt2_prefix_quality_v1.training import run
from research.gpt2_prefix_quality_v1.validation import sampled_bytes
from research.gpt2_prefix_quality_v1.final_review import verify
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent;FIXTURE=E/'quality-contracts001-fixtures'
SEEDS=dict(screen=600450,evaluation=600454,fresh=[600451,600452,600453],bootstrap=600455)
def publish(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def overwrite(p,v):
 # Intentionally mutate ONLY this still-open synthetic source fixture.
 require(Path(p).is_relative_to(FIXTURE),'Original evidence mutation forbidden');Path(p).write_text(json.dumps(v,sort_keys=True,allow_nan=False)+'\n')
def reject(fn):
 try:fn()
 except (ValueError,RuntimeError,KeyError,TypeError,OSError):return
 raise AssertionError('Full scientific controller forgery accepted')
class CountStream:
 def __init__(self):self.bytes=0
 def write(self,value):n=memoryview(value).nbytes;self.bytes+=n;return n
 def tell(self):return self.bytes
 def flush(self):pass

def synthetic(plan,target,shift):
 lengths=plan['target_token_byte_lengths'];values=[];offset=0
 for size in lengths:
  values.append(math.fsum(1.3+shift+.03*math.sin(i/383)+.01*math.cos(i/991) for i in range(offset,offset+size)));offset+=size
 return dict(loss_summary(plan,values),target=copy.deepcopy(target),window_loss_sums=[math.fsum(values[i:i+128]) for i in range(0,len(values),128)],holdout_scored=False)

@contextlib.contextmanager
def fixture(name,pins,pilot):
 names=['B','RUN','PLAN','FREEZE','SELECTION','CONFIRM'];old={n:getattr(C,n) for n in names};base=FIXTURE/name;base.mkdir(parents=True)
 try:
  C.B=base/'evidence';C.B.mkdir();C.RUN=base/'run';C.RUN.mkdir();C.PLAN=C.B/'quality001-plan.json';C.FREEZE=C.B/'quality001-freeze.json';C.SELECTION=C.B/'quality001-selection.json';C.CONFIRM=C.B/'quality001-confirmation-plan.json'
  # Fixture-only path locators;ALL numerical source hashes remain actual originals.
  (C.B/'statistics-contracts001.json').write_bytes((E/'statistics-contracts001.json').read_bytes())
  gate=C.B/'quality-contracts001.json';publish(gate,dict(status='passed_full_quality_bind_state_sources_and_final_contracts',sources=pins,design=C.DESIGN,fixture_only=True))
  census=C.B/'quality-seed-census001/result.json';publish(census,dict(status='reserved_unused',seeds=SEEDS,fixture_only=True))
  prereqs={n:(reference(gate) if n=='quality-contracts001.json' else reference(census) if n=='quality-seed-census001/result.json' else reference(E/n)) for n in Q.PREREQUISITES}
  plan=dict(kind=C.SCREEN_KIND,root=str(C.RUN),runner=reference(Q.__file__),sources=pins,environment=environment(),design=C.DESIGN,seeds=SEEDS,specs=C.screen_specs(SEEDS),seconds=7200,compare_cpu=True,required_free=C.FLOOR,prerequisites=prereqs)
  publish(C.PLAN,plan);publish(C.FREEZE,dict(plan=reference(C.PLAN),design=C.DESIGN));yield plan
 finally:
  for n,v in old.items():setattr(C,n,v)

def resign_plan(plan):overwrite(C.PLAN,plan);overwrite(C.FREEZE,dict(plan=reference(C.PLAN),design=C.DESIGN))
def resign_row(slot,row):
 base=C.RUN/f'evidence/quality/{slot:02d}';overwrite(base/'accepted.json',row);overwrite(C.B/f'quality001-accepted-{slot:02d}.json',dict(accepted=reference(base/'accepted.json'),original_freeze=reference(C.FREEZE)))
def make_accepted(slot,plan,pilot):
 specs=plan['specs'] if slot<6 else C.fresh_specs(SEEDS,C.checked(reference(C.SELECTION))['choices']);spec=specs[slot if slot<6 else slot-6];arm=spec['arm'];proto=copy.deepcopy(next(r for r in pilot['rows'] if r['arm']==arm));w=C.checked(proto['worker']);base=C.RUN/f'evidence/quality/{slot:02d}';(base/'checkpoints').mkdir(parents=True)
 for key in ['parent','h200']:
  path=base/'checkpoints'/('parent.pt' if key=='parent' else 'h200.pt');path.write_bytes(('SYNTHETIC_LAYOUT_FIXTURE_ONLY_'+str(slot)+'_'+arm+'_'+key).encode());w[key]=dict(reference(path),size=path.stat().st_size)
 process=copy.deepcopy(C.checked(proto['process']));process.update(status='completed',returncode=0,timeout_seconds=7200,reserve_bytes=C.RESERVE,output_limit_bytes=spec['weight_cap']+10*1024**2,log_truncated=False,required_free_bytes=C.FLOOR)
 prior=0
 for i in range(slot):
  previous=C.checked(C.checked(C.checked(reference(C.B/f'quality001-accepted-{i:02d}.json'))['accepted'])['worker']);prior+=sum(previous[k]['size'] for k in ['parent','h200'])
 process['required_free_bytes']=max(C.RESERVE,C.FLOOR-prior)
 rq=dict(plan=reference(C.PLAN if slot<6 else C.CONFIRM),spec=spec,parent_pid=process['parent_pid'],slot=slot,stage='quality');publish(base/'request.json',rq);publish(base/'config.json',spec['config'])
 w.update(kind='gpt2_prefix_quality_worker_v1',config=spec['config'],request=reference(base/'request.json'),step=1200,targets=1228800,ema_snapshots=41,parameter_tensors=88,update_seconds=[.1]*1200,
  metrics=[dict(step=i+1,targets=(i+1)*1024,train_ce=10.,validation_ce=10.,lr_multiplier=1.,update_seconds=.1) for i in range(1200) if i%spec['config']['training']['eval_every']==0 or i==1199])
 publish(base/'worker.json',w);publish(base/'process/process.json',process)
 refs={}
 for k in ['worker_guard','parent_guard']:
  d=copy.deepcopy(C.checked(proto[k]));d['limits']['timeout_seconds']=7200;path=C.RUN/f'control/{"worker" if k=="worker_guard" else "parent"}-quality-{slot:02d}-guard.json';publish(path,d);refs[k]=reference(path)
 result=copy.deepcopy(proto['acceptance']);result.update(step=1200,targets=1228800,ema_snapshots=41,parameter_tensors=88,raw_target_byte_exposure=dict(sampled_target_tokens=1228800,per_update_target_bytes=[3000]*1200,sampled_target_bytes=3600000))
 result['binding']={k:dict(path=w[k]['path'],sha256=w[k]['sha256']) for k in ['parent','h200']};result['binding']['config']=reference(base/'config.json')
 row=dict(slot=slot,arm=arm,request=reference(base/'request.json'),worker=reference(base/'worker.json'),process=reference(base/'process/process.json'),acceptance=result,**refs)
 publish(base/'accepted.json',row);publish(C.B/f'quality001-accepted-{slot:02d}.json',dict(accepted=reference(base/'accepted.json'),original_freeze=reference(C.FREEZE)));return row

with session_lock(GLOBAL_LOCK.parent):
 torch.set_num_threads(4);idle();env=environment();require(not(E/'quality-contracts001-plan.json').exists() and not C.RUN.exists() and not C.PLAN.exists() and not C.FREEZE.exists(),'New source gate before any scientific attempt only')
 pins=C.sources();require(len(pins)==1004,'Exact current full runtime source count required');pilot=C.checked(reference(E/'pilot001-review.json'));require(pilot['all_original7200_projections_pass'],'Actual resource pilots required')
 publish(E/'quality-contracts001-plan.json',dict(source=reference(__file__),sources=pins,design=C.DESIGN,real_resource_review=reference(E/'pilot001-review.json'),fixture_seeds=SEEDS,synthetic_locator_fixtures_not_scientific_acceptance=True,training=False,model_scoring=False,holdout_encoded_or_scored=False))
 try:
  with StrictMpsGuard(E/'quality-contracts001-guard.json',ROOT),torch.random.fork_rng(devices=[]):
   # The numerical update loop is byte-identical except its finite horizon gate and record kind.
   previous=(ROOT/'research/gpt2_prefix_worker_v1/training.py').read_text();current=Path(Q.__file__).with_name('training.py').read_text();expected=previous.replace("require(cfg.training.max_iters in (4,20),'Only finite prerequisite horizons admitted')","require(cfg.training.max_iters==1200,'Only complete frozen scientific horizons admitted')").replace("kind='gpt2_prefix_worker_v1'","kind='gpt2_prefix_quality_worker_v1'");require(current==expected,'Numerical training equations changed')
   actual=[]
   for row in pilot['rows']:
    w=C.checked(row['worker']);cfg=OmegaConf.create(w['config']);raw=torch.load(w['parent']['path'],map_location='cpu',weights_only=True);h=torch.load(w['h200']['path'],map_location='cpu',weights_only=True)
    m=GPT(plain(cfg.prefix_model));o=build_optimizer(m,cfg);ema=TailWeightAverages(m,cfg.training);require(validate_resume(m,o,ema,raw,cfg)==20,'Actual pilot optimizer/EMA/RNG/clipping state invalid')
    for n,t in h['model_state'].items():require(torch.equal(t,raw['ema']['averages']['200'].get(n,raw['model_state'][n])),'Actual pilot H200 not RAW EMA')
    d=load(cfg,'cpu');exposure=sampled_bytes(d,cfg,raw['sampler_state']['train']);require(exposure['sampled_target_bytes']==({'native':67642,'prefix8192':58020}[row['arm']]),'Independent actual resource byte audit differs')
    for steps in [4,20]:reject(lambda steps=steps:run(config(row['arm'],steps=steps),None,None))
    stress=dict(raw,cfg=plain(C.spec(0,row['arm'],SEEDS['screen'],.00125,SEEDS['evaluation'])['config']),step=1200,clipping_history=[dict(raw['clipping_history'][-1],step=i) for i in range(1,1201)])
    a=CountStream();torch.save(stress,a);b=CountStream();torch.save(dict(h,cfg=stress['cfg'],step=1200,averaged_weights=dict(h['averaged_weights'],snapshots=41)),b)
    require(a.bytes+b.bytes<=C.CAPS[row['arm']]-8*1024**2,'Full-horizon serialization layout exceeds conservative cap margin')
    actual.append(dict(arm=row['arm'],actual_pilot_state_checked=True,full_history_layout_counted_bytes=a.bytes+b.bytes,cap=C.CAPS[row['arm']],fixture_values_not_full_optimizer_or_quality_state=True,sampled_byte_audit=exposure))
    del m,o,ema,raw,h,stress,d;gc.collect()
   checked_cases=[]
   for seedset in [dict(SEEDS,fresh=SEEDS['fresh'][:2]),dict(SEEDS,bootstrap=SEEDS['screen'])]:reject(lambda seedset=seedset:C.seeds_valid(seedset));checked_cases.append('invalid_seed_roles')
   for accepted,attempted in [([False]*12,[True]+[False]*11),([False,True]+[False]*10,[False,True]+[False]*10)]:reject(lambda accepted=accepted,attempted=attempted:C.prefix_order(accepted,attempted));checked_cases.append('partial_or_noncontiguous_prefix')
   reject(lambda:C.weight_credit({'old_pilot.pt':100},{ }));checked_cases.append('foreign_pilot_credit')
   with fixture('valid_zero',pins,pilot) as plan:require(Q.bind()==plan and Q.state(plan)[:2]==(0,0),'Actual source bind/empty state failed')
   with fixture('valid_one',pins,pilot) as plan:
    make_accepted(0,plan,pilot);require(Q.state(Q.bind())[0]==1,'Fixture positive accepted prefix failed')
   for name in ['source','environment','budget','design','stop','premature_selection','partial_slot','missing_anchor','request','kind','targets','weak_guard','ancestry','entry_budget','extra_weight','extra_tmp','foreign_seed']:
    with fixture('reject_'+name,pins,pilot) as plan:
     if name in ['source','environment','budget','design']:
      bad=copy.deepcopy(plan)
      if name=='source':bad['sources'].pop(next(iter(bad['sources'])))
      if name=='environment':bad['environment']['flags']['PYTORCH_ENABLE_MPS_FALLBACK']='1'
      if name=='budget':bad['required_free']=C.RESERVE
      if name=='design':bad['design']['mean_bpb_delta_threshold']=0.
      resign_plan(bad);reject(Q.bind)
     elif name=='stop':publish(C.RUN/'STOP',dict(closed=True));reject(Q.bind)
     elif name=='premature_selection':publish(C.SELECTION,dict(premature=True));reject(lambda:Q.state(plan))
     elif name=='partial_slot':(C.RUN/'evidence/quality/00').mkdir(parents=True);reject(lambda:Q.state(plan))
     else:
      row=make_accepted(0,plan,pilot);base=C.RUN/'evidence/quality/00'
      if name=='missing_anchor':(C.B/'quality001-accepted-00.json').rename(C.B/'retained-moved-fixture-anchor.json');reject(lambda:Q.state(plan))
      elif name=='request':(base/'request.json').rename(base/'retained-moved-fixture-request.json');reject(lambda:Q.state(plan))
      elif name in ['extra_weight','extra_tmp']:
       (base/'checkpoints'/('extra.pt' if name=='extra_weight' else 'extra.pt.tmp')).write_bytes(b'EXTRA_SYNTHETIC_ONLY');reject(lambda:Q.state(plan))
      elif name in ['kind','foreign_seed']:
       w=C.checked(row['worker'])
       if name=='kind':w['kind']='gpt2_prefix_worker_v1'
       else:
        badcfg=copy.deepcopy(w['config']);badcfg['training']['seed']+=999;rq=C.checked(row['request']);rq['spec']['config']=badcfg;overwrite(base/'request.json',rq);overwrite(base/'config.json',badcfg);w['config']=badcfg;w['request']=reference(base/'request.json');row['request']=w['request'];row['acceptance']['binding']['config']=reference(base/'config.json')
       overwrite(base/'worker.json',w);row['worker']=reference(base/'worker.json');resign_row(0,row);reject(lambda:Q.state(plan))
      elif name=='targets':row['acceptance']['targets']-=1;resign_row(0,row);reject(lambda:Q.state(plan))
      elif name=='weak_guard':g=C.checked(row['parent_guard']);g['limits']['reserve_bytes']=1;overwrite(Path(row['parent_guard']['path']),g);row['parent_guard']=reference(row['parent_guard']['path']);resign_row(0,row);reject(lambda:Q.state(plan))
      else:
       process=C.checked(row['process'])
       if name=='ancestry':process['parent_pid']+=999
       elif name=='entry_budget':process['required_free_bytes']=C.RESERVE
       overwrite(Path(row['process']['path']),process);row['process']=reference(row['process']['path']);resign_row(0,row);reject(lambda:Q.state(plan))
    checked_cases.append(name)
   with fixture('duplicate_weight_bytes',pins,pilot) as plan:
    first=make_accepted(0,plan,pilot);second=make_accepted(1,plan,pilot);a=C.checked(first['worker']);b=C.checked(second['worker']);Path(b['parent']['path']).write_bytes(Path(a['parent']['path']).read_bytes());b['parent']=dict(reference(b['parent']['path']),size=Path(b['parent']['path']).stat().st_size);overwrite(Path(second['worker']['path']),b);second['worker']=reference(second['worker']['path']);second['acceptance']['binding']['parent']={k:b['parent'][k] for k in ['path','sha256']};resign_row(1,second);reject(lambda:Q.state(plan));checked_cases.append('duplicate_weight_bytes')
   with fixture('six_and_immutable_selection',pins,pilot) as plan:
    for i in range(6):make_accepted(i,plan,pilot)
    require(Q.state(plan)[0]==6,'Complete screen state failed');Q.resolve_selection(plan);choice=Q.verify_selection(plan);require(choice['choices']=={'native':.000625,'prefix8192':.000625},'Immutable lower-LR tie failed');Q.verify_confirmation(plan)
    bad=C.checked(reference(C.SELECTION));bad['choices']['prefix8192']=.0025;overwrite(C.SELECTION,bad);reject(lambda:Q.verify_selection(plan));checked_cases.append('immutable_selection_forgery')
   templates={r['arm']:r['acceptance']['full_validation']['h200']['mps']['target'] for r in pilot['rows']};plans={a:load(config(a)).common_byte_plan for a in C.ARMS};panel=[]
   for i,seed in enumerate(SEEDS['fresh']):panel.append(dict(seed=seed,native=synthetic(plans['native'],templates['native'],.003*i),prefix8192=synthetic(plans['prefix8192'],templates['prefix8192'],-.03+.002*i)))
   primary=review(panel,SEEDS['bootstrap']);independent=verify(panel,SEEDS['fresh'],SEEDS['bootstrap'],primary);require(independent['quality_gates_pass'],'Independent nonconstant full-review fixture failed')
   bad=copy.deepcopy(primary);bad['intervals']['4096']['upper']+=.01;reject(lambda:verify(panel,SEEDS['fresh'],SEEDS['bootstrap'],bad));checked_cases.append('independent_review_forgery')
   require(C.sources()==pins and not C.RUN.exists() and not C.PLAN.exists() and not C.FREEZE.exists(),'Source changed or production attempt created')
  publish(E/'quality-contracts001.json',dict(status='passed_full_quality_bind_state_sources_and_final_contracts',source=reference(__file__),sources=pins,total_source_bindings=1004,design=C.DESIGN,environment=env,
   real_pilot_state_and_full_layout_checks=actual,rejected_cases=checked_cases,positive_actual_source_fixture_binding=True,fixture_path_locators_only=True,synthetic_tiny_payloads_NOT_numerical_or_quality_acceptance=True,independent_nonconstant_final_review_agreed=True,
   training=False,model_scoring=False,holdout_encoded_or_scored=False,scientific_seed_used=False,weight_files_written=0,tiny_synthetic_fixture_files_not_checkpoints=True,
   next_action='Run NEW unused scientific seed census;verify actual production roots/plans absent,all1004sources/proofs/strict environment/data/full capacity;prepare new freeze ONCE then run serial complete12trajectorypanel under existing authorization. No closed proof reruns.'))
  print(json.dumps(dict(status='passed_full_quality_bind_state_sources_and_final_contracts',rejected_cases=checked_cases,total_sources=1004,serialized_layouts=[dict(arm=r['arm'],counted_bytes=r['full_history_layout_counted_bytes']) for r in actual])),flush=True)
 except Exception as exc:publish(E/'STOP-quality-contracts001',dict(error=repr(exc),closed=True,no_retry=True));raise
