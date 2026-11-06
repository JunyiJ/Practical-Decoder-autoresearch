"""Actual pilot state/layout audit plus isolated full-controller forgeries;zero training."""
import contextlib,copy,gc,hashlib,io,json,math,sys,shutil
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_moe_resume_v1.contract import runtime as environment
from research.tokenizer_moe_model_v1.optimizer import build_optimizer
from research.tokenizer_moe_model_v1.model import GPT
from research.tokenizer_moe_resume_v1.data import load,config
from research.tokenizer_moe_resume_v1.checkpoint import validate_resume
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
from research.tokenizer_moe_quality_v1.statistics import review
from research.tokenizer_moe_quality_v1 import contract as C,execution as Q
from research.tokenizer_moe_quality_v1.training import run
from research.tokenizer_moe_quality_v1.final_review import verify
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.tokenizer_moe_worker_v1.execution import idle
E=Path(__file__).resolve().parent;FIXTURE=E/'quality-contracts001-fixtures'
SEEDS=dict(screen=600480,evaluation=600481,fresh=[600482,600483,600484],bootstrap=600485)
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

def synthetic(target,shift):
 values=[1.3+shift+.03*math.sin(i/383)+.01*math.cos(i/991) for i in range(34278)];total=math.fsum(values)
 return dict(target=copy.deepcopy(target),total_loss=total,ce=total/34278,bits_per_byte=total/math.log(2)/111536,
  window_loss_sums=[math.fsum(values[i:i+128]) for i in range(0,34278,128)],block_loss_sums=[math.fsum(values[i:i+2048]) for i in range(0,34278,2048)],loss_dtype='FP32_cross_entropy_then_CPU_FP64_aggregation',holdout_scored=False,aux_loss_included=False)

@contextlib.contextmanager
def fixture(name,pins,pilot):
 names=['B','RUN','PLAN','FREEZE','SELECTION','CONFIRM'];old={n:getattr(C,n) for n in names};base=FIXTURE/name;base.mkdir(parents=True)
 try:
  C.B=base/'evidence';C.B.mkdir();C.RUN=base/'run';C.RUN.mkdir();C.PLAN=C.B/'quality001-plan.json';C.FREEZE=C.B/'quality001-freeze.json';C.SELECTION=C.B/'quality001-selection.json';C.CONFIRM=C.B/'quality001-confirmation-plan.json'
  # Fixture-only path locators;ALL numerical source hashes remain actual originals.
  (C.B/'prerequisite-close001.json').write_bytes((E/'prerequisite-close001.json').read_bytes())
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
 specs=plan['specs'] if slot<9 else C.fresh_specs(SEEDS,C.checked(reference(C.SELECTION))['choices']);spec=specs[slot if slot<9 else slot-9];arm=spec['arm'];proto=copy.deepcopy(next(r for r in pilot['rows'] if r['arm']==arm));w=C.checked(proto['worker']);base=C.RUN/f'evidence/quality/{slot:02d}';(base/'checkpoints').mkdir(parents=True)
 for key in ['parent','h200']:
  path=base/'checkpoints'/('parent.pt' if key=='parent' else 'h200.pt');path.write_bytes(('SYNTHETIC_LAYOUT_FIXTURE_ONLY_'+str(slot)+'_'+arm+'_'+key).encode());w[key]=dict(reference(path),size=path.stat().st_size)
 process=copy.deepcopy(C.checked(proto['process']));process.update(status='completed',returncode=0,timeout_seconds=7200,reserve_bytes=C.RESERVE,output_limit_bytes=spec['weight_cap']+10*1024**2,log_truncated=False,required_free_bytes=C.FLOOR)
 prior=0
 for i in range(slot):
  previous=C.checked(C.checked(C.checked(reference(C.B/f'quality001-accepted-{i:02d}.json'))['accepted'])['worker']);prior+=sum(previous[k]['size'] for k in ['parent','h200'])
 process['required_free_bytes']=max(C.RESERVE,C.FLOOR-prior)
 rq=dict(plan=reference(C.PLAN if slot<9 else C.CONFIRM),spec=spec,parent_pid=process['parent_pid'],slot=slot,stage='quality');publish(base/'request.json',rq);publish(base/'config.json',spec['config'])
 w.update(kind='tokenizer_moe_quality_worker_v1',config=spec['config'],request=reference(base/'request.json'),step=1200,targets=1228800,ema_snapshots=41,parameter_tensors=(128 if arm=='moe' else 88),update_seconds=[.1]*1200,
  metrics=[dict(step=i+1,targets=(i+1)*1024,train_ce=10.,validation_ce=10.,lr_multiplier=1.,update_seconds=.1) for i in range(1200) if i%spec['config']['training']['eval_every']==0 or i==1199])
 publish(base/'worker.json',w);publish(base/'process/process.json',process)
 refs={}
 for k in ['worker_guard','parent_guard']:
  d=copy.deepcopy(C.checked(proto[k]));d['limits']['timeout_seconds']=7200;path=C.RUN/f'control/{"worker" if k=="worker_guard" else "parent"}-quality-{slot:02d}-guard.json';publish(path,d);refs[k]=reference(path)
 result=copy.deepcopy(proto['acceptance']);result.update(step=1200,targets=1228800,ema_snapshots=41,parameter_tensors=(128 if arm=='moe' else 88))
 result.update(parameter_gradient_steps={n:1200 for n in result['parameter_gradient_steps']},route_counts=([[614400]*4 for _ in range(4)] if arm=='moe' else None))
 w.update(parameter_gradient_steps=result['parameter_gradient_steps'],route_counts=result['route_counts']);overwrite(base/'worker.json',w)
 result['binding']={k:dict(path=w[k]['path'],sha256=w[k]['sha256']) for k in ['parent','h200']};result['binding']['config']=reference(base/'config.json')
 row=dict(slot=slot,arm=arm,request=reference(base/'request.json'),worker=reference(base/'worker.json'),process=reference(base/'process/process.json'),acceptance=result,**refs)
 publish(base/'accepted.json',row);publish(C.B/f'quality001-accepted-{slot:02d}.json',dict(accepted=reference(base/'accepted.json'),original_freeze=reference(C.FREEZE)));return row

with session_lock(GLOBAL_LOCK.parent):
 torch.set_num_threads(4);idle();env=environment();require(not(E/'quality-contracts001-plan.json').exists() and not C.RUN.exists() and not C.PLAN.exists() and not C.FREEZE.exists(),'New source gate before any scientific attempt only')
 require(shutil.disk_usage(ROOT).free>=8*1024**3+64*1024**2+1267904512+1458176+2*1024**3,'Complete metadata-only proof budget plus2GiB required')
 pins=C.sources();require(len(pins)==1035,'Exact current full runtime source count required');pilot=C.checked(reference(E/'pilot001-review.json'));require(pilot['status']=='all_three_moe_resource_pilots_independently_accepted','Actual resource pilots required')
 publish(E/'quality-contracts001-plan.json',dict(source=reference(__file__),sources=pins,design=C.DESIGN,real_resource_review=reference(E/'pilot001-review.json'),fixture_seeds=SEEDS,synthetic_locator_fixtures_not_scientific_acceptance=True,training=False,model_scoring=False,holdout_encoded_or_scored=False,capacity=dict(output_metadata_cap=64*1024**2,reserve=8*1024**3,external_headroom=1267904512,entry_headroom=1458176,free=shutil.disk_usage(ROOT).free),serialized_payloads_to_counting_stream_only=True))
 try:
  with StrictMpsGuard(E/'quality-contracts001-guard.json',ROOT),torch.random.fork_rng(devices=[]):
   # The numerical update loop is byte-identical except its finite horizon gate and record kind.
   previous=(ROOT/'research/tokenizer_moe_worker_v1/training.py').read_text();current=Path(Q.__file__).with_name('training.py').read_text()
   expected=previous.replace("cfg.training.max_iters in (4,20),'Only four/twenty-update prerequisites admitted'","cfg.training.max_iters==1200,'Only complete frozen scientific horizons admitted'").replace('start={4:2,20:13}[steps]','start=800').replace('interval=1,half_lives=[200]','interval=10,half_lives=[200]').replace('Exact prerequisite H200 tail required','Exact scientific H200 tail required').replace('Exact prerequisite cooldown required','Exact scientific cooldown required').replace("time.monotonic()-began<600,'Original prerequisite runtime exceeded'","time.monotonic()-began<7200,'Original scientific runtime exceeded'").replace("kind='tokenizer_moe_prerequisite_worker_v1'","kind='tokenizer_moe_quality_worker_v1'")
   require(current==expected,'Accepted numerical equations changed beyond declared full horizon/record gates')
   actual=[]
   for row in pilot['rows']:
    w=C.checked(row['worker']);cfg=OmegaConf.create(w['config']);raw=torch.load(w['parent']['path'],map_location='cpu',weights_only=True);h=torch.load(w['h200']['path'],map_location='cpu',weights_only=True)
    m=GPT(plain(cfg.moe_model));o=build_optimizer(m,cfg);ema=TailWeightAverages(m,cfg.training);require(validate_resume(m,o,ema,raw,cfg)==20,'Actual pilot optimizer/EMA/RNG/clipping state invalid')
    for n,t in h['model_state'].items():require(torch.equal(t,raw['ema']['averages']['200'].get(n,raw['model_state'][n])),'Actual pilot H200 not RAW EMA')
    for steps in [4,20]:reject(lambda steps=steps:run(config(row['arm'],steps=steps,seed=600480,eval_seed=600481),None,None))
    stress=dict(raw,cfg=plain(C.spec(0,row['arm'],SEEDS['screen'],.00125,SEEDS['evaluation'])['config']),step=1200,clipping_history=[dict(json.loads(json.dumps(raw['clipping_history'][-1])),step=i) for i in range(1,1201)])
    a=CountStream();torch.save(stress,a);b=CountStream();torch.save(dict(h,cfg=stress['cfg'],step=1200,averaged_weights=dict(h['averaged_weights'],snapshots=41)),b)
    require(a.bytes+b.bytes<=C.CAPS[row['arm']]-8*1024**2,'Full-horizon serialization layout exceeds conservative cap margin')
    actual.append(dict(arm=row['arm'],actual_pilot_state_checked=True,full_history_layout_counted_bytes=a.bytes+b.bytes,cap=C.CAPS[row['arm']],fixture_values_not_full_optimizer_or_quality_state=True))
    del m,o,ema,raw,h,stress;gc.collect()
   checked_cases=[]
   for seedset in [dict(SEEDS,fresh=SEEDS['fresh'][:2]),dict(SEEDS,bootstrap=SEEDS['screen'])]:reject(lambda seedset=seedset:C.seeds_valid(seedset));checked_cases.append('invalid_seed_roles')
   for accepted,attempted in [([False]*18,[True]+[False]*17),([False,True]+[False]*16,[False,True]+[False]*16)]:reject(lambda accepted=accepted,attempted=attempted:C.prefix_order(accepted,attempted));checked_cases.append('partial_or_noncontiguous_prefix')
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
      if name=='design':bad['design']['mean_ce_delta_threshold']=0.
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
       if name=='kind':w['kind']='tokenizer_moe_prerequisite_worker_v1'
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
   with fixture('nine_and_immutable_selection',pins,pilot) as plan:
    for i in range(9):make_accepted(i,plan,pilot)
    require(Q.state(plan)[0]==9,'Complete screen state failed');Q.resolve_selection(plan);choice=Q.verify_selection(plan);require(choice['choices']=={'dense':.000625,'wide':.000625,'moe':.000625},'Immutable lower-LR tie failed');Q.verify_confirmation(plan)
    bad=C.checked(reference(C.SELECTION));bad['choices']['moe']=.0025;overwrite(C.SELECTION,bad);reject(lambda:Q.verify_selection(plan));checked_cases.append('immutable_selection_forgery')
   with fixture('complete_eighteen_and_fresh_contracts',pins,pilot) as plan:
    for i in range(9):make_accepted(i,plan,pilot)
    Q.resolve_selection(plan);reject(Q.final)
    for i in range(9,18):make_accepted(i,plan,pilot)
    require(Q.state(Q.bind())[0]==18,'Complete18source/request/ancestry/weight inventory failed')
    resolved=Q.verify_confirmation(plan)
    require([v['arm'] for v in resolved['specs']]==['dense','wide','moe','wide','moe','dense','moe','dense','wide'],'Original three-way rotation changed')
    bad=copy.deepcopy(resolved);bad['specs'][0]['config']['training']['seed']+=99;overwrite(C.CONFIRM,bad);reject(lambda:Q.verify_confirmation(plan));checked_cases.extend(['partial_final','fresh_confirmation_forgery'])
   templates={r['arm']:r['acceptance']['full_validation']['h200']['mps']['target'] for r in pilot['rows']};panel=[]
   for i,seed in enumerate(SEEDS['fresh']):panel.append(dict(seed=seed,dense=synthetic(templates['dense'],.003*i),wide=synthetic(templates['wide'],-.01+.003*i),moe=synthetic(templates['moe'],-.06+.002*i)))
   primary=review(panel,SEEDS['bootstrap']);independent=verify(panel,SEEDS['fresh'],SEEDS['bootstrap'],primary);require(independent['quality_gates_pass'],'Independent nonconstant full-review fixture failed')
   bad=copy.deepcopy(primary);bad['contrasts']['wide']['intervals']['4096']['upper']+=.01;reject(lambda:verify(panel,SEEDS['fresh'],SEEDS['bootstrap'],bad));checked_cases.append('independent_review_forgery')
   # No partial/foreign/nonfinite panel or incomplete LR set is a quality result.
   from research.tokenizer_moe_quality_v1.statistics import select,report_contract
   for mode in ['partial','duplicate','foreign','nonfinite','target','aux']:
    bad=copy.deepcopy(panel)
    if mode=='partial':bad.pop()
    elif mode=='duplicate':bad[1]['seed']=bad[0]['seed']
    elif mode=='foreign':bad[0]['extra']=bad[0]['dense']
    elif mode=='nonfinite':bad[0]['moe']['ce']=float('nan')
    elif mode=='target':bad[0]['moe']['target']['targets']-=1
    else:bad[0]['moe']['aux_loss_included']=True
    reject(lambda bad=bad:review(bad,SEEDS['bootstrap']));checked_cases.append('statistics_'+mode)
   lrrows=[dict(status='accepted',arm=a,peak_lr=r,h200=synthetic(templates[a],0.)) for r in C.LRS for a in C.ARMS]
   require(select(lrrows)=={a:.000625 for a in C.ARMS},'Exact lower LR tie selection failed')
   best=copy.deepcopy(lrrows);best[4]['h200']=synthetic(templates['wide'],-.1);require(select(best)['wide']==.00125,'Per-arm complete minimum selection failed')
   reject(lambda:select(lrrows[:-1]));duplicate=copy.deepcopy(lrrows);duplicate[0]=duplicate[1];reject(lambda:select(duplicate));checked_cases.extend(['partial_screen','duplicate_screen'])
   total_metadata=sum(p.stat().st_size for p in FIXTURE.rglob('*') if p.is_file());require(total_metadata<=64*1024**2,'Bounded64MiB synthetic metadata cap exceeded')
   require(C.sources()==pins and not C.RUN.exists() and not C.PLAN.exists() and not C.FREEZE.exists(),'Source changed or production attempt created')
  publish(E/'quality-contracts001.json',dict(status='passed_full_quality_bind_state_sources_and_final_contracts',source=reference(__file__),sources=pins,total_source_bindings=1035,design=C.DESIGN,environment=env,
   real_pilot_state_and_full_layout_checks=actual,rejected_cases=checked_cases,positive_actual_source_fixture_binding=True,fixture_path_locators_only=True,synthetic_tiny_payloads_NOT_numerical_or_quality_acceptance=True,independent_nonconstant_final_review_agreed=True,
   training=False,model_scoring=False,holdout_encoded_or_scored=False,scientific_seed_used=False,weight_files_written=0,tiny_synthetic_fixture_files_not_checkpoints=True,
   next_action='Run NEW unused scientific seed census;verify actual production roots/plans absent,all1035sources/proofs/strict environment/data/full capacity;prepare new freeze ONCE then run serial complete18trajectorypanel under existing authorization. No closed proof reruns.'))
  print(json.dumps(dict(status='passed_full_quality_bind_state_sources_and_final_contracts',rejected_cases=checked_cases,total_sources=1035,serialized_layouts=[dict(arm=r['arm'],counted_bytes=r['full_history_layout_counted_bytes']) for r in actual])),flush=True)
 except Exception as exc:publish(E/'STOP-quality-contracts001',dict(error=repr(exc),closed=True,no_retry=True));raise
