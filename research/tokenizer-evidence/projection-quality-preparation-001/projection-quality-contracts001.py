"""Actual pilot state/layout audit plus isolated full-controller forgeries;zero training."""
import argparse,contextlib,copy,gc,hashlib,io,json,math,sys,shutil,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.body_allocation_model_v1.evidence import reference
from research.readout_projection_resume_v1.contract import runtime as environment
from research.readout_projection_model_v1.optimizer import build_optimizer
from research.readout_projection_model_v1.model import GPT
from research.readout_projection_resume_v1.data import config,model_spec,View
from research.readout_projection_resume_v1.checkpoint import FILE_CAPS,RAW,H200,validate_resume,identity as training_identity,model_state as validate_model_state,load_export
from research.block_norm_v1.runtime.train.weight_averaging import averaging_contract
from research.readout_projection_worker_v1.validation import verify_artifact
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
from research.readout_projection_quality_v1.statistics import review
from research.readout_projection_quality_v1 import contract as C,execution as Q
from research.readout_projection_quality_v1.training import run
from research.readout_projection_quality_v1.final_review import verify
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.readout_projection_worker_v1.execution import idle
E=Path(__file__).resolve().parent;FIXTURE=E/'projection-quality-contracts001-fixtures'
R=E/'projection-quality-contracts001'
ROLE_AUDIT=E/'projection-quality-seed-census001/result.json'
SEEDS=None
P=ROOT/'research/tokenizer-evidence/post-readout-boundary-001'
def publish(p,v):
 p=Path(p);require(p.is_relative_to(FIXTURE) or p.parent==R or p==E/'projection-quality-contracts001.json','Only NEW proof outputs permitted');p.parent.mkdir(parents=True,exist_ok=True)
 raw=(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();require(len(raw)<1024**2,'Proof record cap1MiB')
 require(sum(x.stat().st_size for x in E.rglob('*') if x.is_file())+len(raw)<128*1024**2,'Whole new sidecar cap128MiB')
 if p.is_relative_to(FIXTURE):require(sum(x.stat().st_size for x in FIXTURE.rglob('*') if x.is_file())+len(raw)<64*1024**2,'Proof fixture cap64MiB')
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def overwrite(p,v):
 # Intentionally mutate ONLY this still-open synthetic source fixture.
 require(Path(p).is_relative_to(FIXTURE),'Original evidence mutation forbidden');Path(p).write_text(json.dumps(v,sort_keys=True,allow_nan=False)+'\n')
def reject(fn):
 try:fn()
 except (ValueError,RuntimeError,KeyError,TypeError,OSError):return
 raise AssertionError('Full scientific controller forgery accepted')
class CountStream:
 def __init__(self,cap):self.bytes=0;self.cap=cap
 def write(self,value):
  n=memoryview(value).nbytes;require(self.bytes+n<=self.cap,'Full archive individual byte cap exceeded');self.bytes+=n;return n
 def tell(self):return self.bytes
 def flush(self):pass

def synthetic(target,shift):
 values=[1.7+shift+.021*math.sin(i/271+13*shift)+.013*math.cos(i/877+19*shift) for i in range(34278)];total=math.fsum(values)
 return dict(target=copy.deepcopy(target),total_loss=total,ce=total/34278,bits_per_byte=total/math.log(2)/111536,
  window_loss_sums=[math.fsum(values[i:i+128]) for i in range(0,34278,128)],block_loss_sums=[math.fsum(values[i:i+2048]) for i in range(0,34278,2048)],loss_dtype='FP32_own_distribution_NLL_then_CPU_FP64_aggregation',holdout_scored=False,aux_loss_included=False)

@contextlib.contextmanager
def fixture(name,pins,pilot):
 names=['B','RUN','PLAN','FREEZE','SELECTION','CONFIRM'];old={n:getattr(C,n) for n in names};base=FIXTURE/name;base.mkdir(parents=True)
 try:
  C.B=base/'evidence';C.B.mkdir();C.RUN=base/'run';C.RUN.mkdir();C.PLAN=C.B/'projection-quality001-plan.json';C.FREEZE=C.B/'projection-quality001-freeze.json';C.SELECTION=C.B/'projection-quality001-selection.json';C.CONFIRM=C.B/'projection-quality001-confirmation-plan.json'
  # Fixture-only path locators;ALL numerical source hashes remain actual originals.
  (C.B/'prerequisite-closure001.json').write_bytes((E/'prerequisite-closure001.json').read_bytes())
  publish(C.B/'quality-resource-preflight001.json',dict(status='passed_complete_projection_quality_resources',fixture_only=True))
  gate=C.B/'projection-quality-contracts001.json';publish(gate,dict(status='passed_full_quality_bind_state_sources_and_final_contracts',sources=pins,design=C.DESIGN,fixture_only=True))
  census=C.B/'projection-quality-seed-census001/result.json';publish(census,dict(status='reserved_unused',seeds=SEEDS,fixture_only=True))
  prereqs={n:(reference(gate) if n=='projection-quality-contracts001.json' else reference(census) if n=='projection-quality-seed-census001/result.json' else reference(C.B/n) if n in ['prerequisite-closure001.json','quality-resource-preflight001.json'] else reference(E/n)) for n in Q.PREREQUISITES}
  plan=dict(kind=C.SCREEN_KIND,root=str(C.RUN),runner=reference(Q.__file__),sources=pins,environment=environment(),design=C.DESIGN,seeds=SEEDS,specs=C.screen_specs(SEEDS),seconds=7200,compare_cpu=True,required_free=C.FLOOR,prerequisites=prereqs)
  publish(C.PLAN,plan);publish(C.FREEZE,dict(plan=reference(C.PLAN),design=C.DESIGN));yield plan
 finally:
  for n,v in old.items():setattr(C,n,v)

def resign_plan(plan):overwrite(C.PLAN,plan);overwrite(C.FREEZE,dict(plan=reference(C.PLAN),design=C.DESIGN))
def resign_row(slot,row):
 base=C.RUN/f'evidence/quality/{slot:02d}';overwrite(base/'accepted.json',row);overwrite(C.B/f'projection-quality001-accepted-{slot:02d}.json',dict(accepted=reference(base/'accepted.json'),original_freeze=reference(C.FREEZE)))
def make_accepted(slot,plan,pilot):
 specs=plan['specs'] if slot<9 else C.fresh_specs(SEEDS,C.checked(reference(C.SELECTION))['choices']);spec=specs[slot if slot<9 else slot-9];arm=spec['arm'];proto=copy.deepcopy(next(r for r in pilot['rows'] if r['arm']==arm));w=C.checked(proto['worker']);base=C.RUN/f'evidence/quality/{slot:02d}';(base/'checkpoints').mkdir(parents=True)
 for key in ['parent','h200']:
  path=base/'checkpoints'/('parent.pt' if key=='parent' else 'h200.pt');path.write_bytes(('NEW_PROJECTION_SYNTHETIC_META_ONLY_'+str(slot)+'_'+arm+'_'+key).encode());w[key]=dict(reference(path),size=path.stat().st_size)
 process=copy.deepcopy(C.checked(proto['process']));process.update(status='completed',returncode=0,timeout_seconds=7200,reserve_bytes=C.RESERVE,output_limit_bytes=spec['weight_cap']+10*1024**2,log_truncated=False,required_free_bytes=C.FLOOR)
 prior=0
 for i in range(slot):
  previous=C.checked(C.checked(C.checked(reference(C.B/f'projection-quality001-accepted-{i:02d}.json'))['accepted'])['worker']);prior+=sum(previous[k]['size'] for k in ['parent','h200'])
 process['required_free_bytes']=max(C.RESERVE,C.FLOOR-prior)
 rq=dict(plan=reference(C.PLAN if slot<9 else C.CONFIRM),spec=spec,parent_pid=process['parent_pid'],slot=slot,stage='quality');publish(base/'request.json',rq);publish(base/'config.json',spec['config'])
 w.update(kind='readout_projection_quality_worker_v1',config=spec['config'],request=reference(base/'request.json'),step=1200,targets=1228800,ema_snapshots=41,parameter_tensors=(88 if arm=='native' else 89),update_seconds=[.1]*1200,
  metrics=[dict(step=i+1,targets=(i+1)*1024,train_ce=10.,validation_ce=10.,lr_multiplier=1.,update_seconds=.1) for i in range(1200) if i%spec['config']['training']['eval_every']==0 or i==1199])
 publish(base/'worker.json',w);publish(base/'process/process.json',process)
 refs={}
 for k in ['worker_guard','parent_guard']:
  d=copy.deepcopy(C.checked(proto[k]));d['limits']['timeout_seconds']=7200;path=C.RUN/f'control/{"worker" if k=="worker_guard" else "parent"}-quality-{slot:02d}-guard.json';publish(path,d);refs[k]=reference(path)
 result=copy.deepcopy(proto['acceptance']);result.update(step=1200,targets=1228800,ema_snapshots=41,parameter_tensors=(88 if arm=='native' else 89))
 result.update(parameter_gradient_steps={n:1200 for n in result['parameter_gradient_steps']})
 w.update(parameter_gradient_steps=result['parameter_gradient_steps'],serialized_bytes=dict(raw=w['parent']['size'],h200=w['h200']['size'],pair=w['parent']['size']+w['h200']['size']));overwrite(base/'worker.json',w)
 result['binding']={k:dict(path=w[k]['path'],sha256=w[k]['sha256']) for k in ['parent','h200']};result['binding']['config']=reference(base/'config.json')
 row=dict(slot=slot,arm=arm,request=reference(base/'request.json'),worker=reference(base/'worker.json'),process=reference(base/'process/process.json'),acceptance=result,**refs)
 publish(base/'accepted.json',row);publish(C.B/f'projection-quality001-accepted-{slot:02d}.json',dict(accepted=reference(base/'accepted.json'),original_freeze=reference(C.FREEZE)));return row

def main(args):
 global SEEDS
 require(reference(R/'plan.json')['sha256']==args.plan_sha and {p.name for p in R.iterdir()}=={'plan.json'},'Exact pristine original proof plan required')
 entry=C.checked(reference(R/'plan.json'));ROLES=C.checked(reference(ROLE_AUDIT));SEEDS=ROLES['fixture_seeds']
 with session_lock(GLOBAL_LOCK.parent):
  torch.set_num_threads(4);idle();env=environment();require(not C.RUN.exists() and not C.PLAN.exists() and not C.FREEZE.exists() and not FIXTURE.exists() and not(E/'projection-quality-contracts001.json').exists(),'New source gate before any scientific attempt only')
  require(entry['environment']==env and entry['source']==reference(__file__) and entry['census']==reference(ROLE_AUDIT),'Pinned proof inputs differ')
  require(not ROLES['prior_matches'] and ROLES['status']=='reserved_unused' and ROLES['generated_fixture_roles_unused_before_execution'] and ROLES['scientific_roles_unused_before_execution'],'Actual sixteen unused roles required')
  require(shutil.disk_usage(ROOT).free>=C.TARGET,'Complete metadata-only proof AND original remaining retained target required')
  forecast=dict(rss_bytes=int(3.5*1024**3),mps_driver_bytes=512*1024**2,seconds=60)
  require(forecast['rss_bytes']<4*1024**3 and forecast['mps_driver_bytes']<6*1024**3 and forecast['seconds']<600,'Complete generated proof forecast must fit original resource guards')
  proof_pins=C.checked(reference(ROLE_AUDIT.parent/'plan.json'))['sources'];require(proof_pins==entry['sources'],'Exact outer proof pins required')
  for n,h in proof_pins.items():require(reference(ROOT/n)['sha256']==h,'Pinned proof source changed: '+n)
  require(len(proof_pins)==1164,'All runtime/proof/census bindings required')
  pins=C.sources();require(len(pins)==1161,'Exact current full runtime source count required');pilot=C.checked(reference(P/'projection-pilot001-review.json'));require(pilot['status']=='all_three_projection_twenty_update_pilots_independently_accepted','Actual resource pilots required')
  publish(R/'attempt.json',dict(pid=os.getpid(),plan=reference(R/'plan.json'),source=reference(__file__)))
  cpu_before=torch.get_rng_state().clone()
  try:
   with StrictMpsGuard(R/'mps-guard.json',ROOT),CpuPrerequisiteGuard(R/'cpu-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25),torch.random.fork_rng(devices=[]):
    # The numerical update loop is byte-identical except its finite horizon gate and record kind.
    previous=(ROOT/'research/readout_projection_worker_v1/training.py').read_text();current=Path(Q.__file__).with_name('training.py').read_text()
    expected=previous[previous.index('def run('):].replace("time.monotonic()-began<600,'Finite prerequisite time exceeded'","time.monotonic()-began<7200,'Finite scientific time exceeded'").replace("kind='readout_projection_prerequisite_worker_v1'","kind='readout_projection_quality_worker_v1'")
    require(current[current.index('def run('):]==expected,'Numerical loop differs beyond full guard/record kind')
    actual=[];schema_cases=[];global_mps=torch.mps.get_rng_state().clone()
    for row in pilot['rows']:
     w=C.checked(row['worker']);cfg=OmegaConf.create(w['config']);verify_artifact(w['parent']);verify_artifact(w['h200'])
     raw=torch.load(w['parent']['path'],map_location='cpu',weights_only=True);h=torch.load(w['h200']['path'],map_location='cpu',weights_only=True)
     m=GPT(model_spec(cfg),initializer_seed=cfg.initializer_seed);o=build_optimizer(m,cfg.training);ema=TailWeightAverages(m,cfg.training)
     view=View(splits={'train':None,'val':None},generators={n:torch.Generator(device='cpu').manual_seed(seed) for n,seed in [('train',cfg.training.seed),('val',cfg.training.eval_seed)]},batch_size=8,block_size=128,batch_layout='paired_blocks512',vocab_size=50257,data_identity=copy.deepcopy(raw['training_identity']['data']))
     require(validate_resume(m,o,ema,raw,cfg,view)==20,'Original complete pilot state must validate before any fixture mutation')
     for n,t in h['model_state'].items():require(torch.equal(t,raw['ema']['averages']['200'].get(n,raw['model_state'][n])),'Actual pilot H200 not exact RAW EMA')
     load_export(m,h,cfg)
     for steps in [4,20]:reject(lambda steps=steps:run(config(row['arm'],steps,SEEDS['screen'],SEEDS['screen'],SEEDS['evaluation']),None,None))
     fullcfg=copy.deepcopy(cfg);fullcfg.training.max_iters=1200;fullcfg.training.eval_every=300;fullcfg.training.eval_iters=1
     fullcfg.training.lr_schedule.update(total_steps=1200,start_step=800);fullcfg.training.weight_averaging.update(total_steps=1200,start_step=800,interval=10)
     del fullcfg.readout_projection_execution
     from research.body_allocation_model_v1.contract import fingerprint
     fullcfg.projection_quality_execution=dict(kind='projection_full_quality_execution_v1',sources_sha256=fingerprint(pins),target_contract_sha256=C.DESIGN['target_contract_sha256'])
     # Full schema and complete history layout, with pilot tensors and SYNTHETIC metadata.
     # Each step owns fresh name strings, avoiding optimistic pickle memoization.
     names=[n for n,_ in m.named_parameters()];history=[]
     for i in range(1,1201):
      hist=dict(raw['clipping_history'][-1],step=i,active_parameter_names=[''.join([n[:-1],n[-1]]) for n in names]);history.append(hist)
     require(len({id(r['active_parameter_names'][0]) for r in history})==1200,'Fresh perstep history strings required for conservative layout')
     state=copy.copy(raw['optimizer_state']);state.update(completed_steps=1200,gradient_steps={n:1200 for n in names});children=[]
     for child in state['optimizers']:
      c=copy.copy(child);c['param_groups']=[dict(g,lr=fullcfg.training.learning_rate*g['width_lr_scale']*.1) for g in child['param_groups']];c['state']={}
      for index,slot in child['state'].items():
       value=dict(slot)
       if 'step' in value:value['step']=torch.tensor(1200.,dtype=torch.float32)
       c['state'][index]=value
      children.append(c)
     state['optimizers']=children;spec=averaging_contract(fullcfg.training)
     stress=dict(raw,cfg=plain(fullcfg),step=1200,clipping_history=history,optimizer_state=state,ema=dict(raw['ema'],spec=spec,completed_steps=1200,snapshots=41),training_identity=training_identity(fullcfg,view))
     fullopt=build_optimizer(m,fullcfg.training);fullema=TailWeightAverages(m,fullcfg.training);require(validate_resume(m,fullopt,fullema,stress,fullcfg,view)==1200,'Complete full-horizon fixture schema must validate,not just its byte count')
     for mode in ('step','ema_count','exposure','clipping_tail','source_binding'):
      bad=dict(stress)
      if mode=='step':bad['step']=1199
      elif mode=='ema_count':bad['ema']=dict(stress['ema'],snapshots=40)
      elif mode=='exposure':bad['optimizer_state']=dict(stress['optimizer_state'],gradient_steps=dict(stress['optimizer_state']['gradient_steps'],**{names[0]:1199}))
      elif mode=='clipping_tail':bad['clipping_history']=history[:-1]+[dict(history[-1],ce_loss=float('nan'))]
      else:bad['training_identity']=dict(stress['training_identity'],source_fingerprint='0'*64)
      reject(lambda bad=bad:validate_resume(m,fullopt,fullema,bad,fullcfg,view));schema_cases.append('fullschema_'+row['arm']+'_'+mode)
     del bad
     exported=dict(h,cfg=plain(fullcfg),step=1200,averaged_weights=dict(h['averaged_weights'],spec=spec,snapshots=41))
     validate_model_state(m,exported['model_state']);load_export(m,exported,fullcfg);a=CountStream(FILE_CAPS[RAW]);torch.save(stress,a);b=CountStream(FILE_CAPS[H200]);torch.save(exported,b)
     require(a.bytes+b.bytes+4096<=C.CAPS[row['arm']]-8*1024**2,'Full-horizon pair layout must retain at least8MiB original cap margin')
     actual.append(dict(arm=row['arm'],actual_pilot_state_checked=True,full_fixture_schema_validated=True,raw_counted_bytes=a.bytes,h200_counted_bytes=b.bytes,full_history_layout_counted_bytes=a.bytes+b.bytes,cap=C.CAPS[row['arm']],margin_bytes=C.CAPS[row['arm']]-a.bytes-b.bytes,additional_ancestry_path_headroom_bytes=4096,fixture_values_NOT_trained_full_horizon=True))
     del m,o,ema,raw,h,stress,exported,fullopt,fullema,history,state,children,view;gc.collect();verify_artifact(w['parent']);verify_artifact(w['h200'])
    require(torch.equal(global_mps,torch.mps.get_rng_state()),'Metadata decoder proof changed global MPS RNG')
    checked_cases=list(schema_cases)
    for seedset in [dict(SEEDS,fresh=SEEDS['fresh'][:2]),dict(SEEDS,bootstrap=SEEDS['screen'])]:reject(lambda seedset=seedset:C.seeds_valid(seedset));checked_cases.append('invalid_seed_roles')
    for accepted,attempted in [([False]*18,[True]+[False]*17),([False,True]+[False]*16,[False,True]+[False]*16)]:reject(lambda accepted=accepted,attempted=attempted:C.prefix_order(accepted,attempted));checked_cases.append('partial_or_noncontiguous_prefix')
    reject(lambda:C.weight_credit({'old_pilot.pt':100},{ }));checked_cases.append('foreign_pilot_credit')
    with fixture('valid_zero',pins,pilot) as plan:
     require(Q.bind()==plan and Q.state(plan)[:2]==(0,0),'Actual source bind/empty state failed')
     previous_shutil=Q.shutil
     try:
      for delta,expected_pass in [(-1,False),(0,True)]:
       Q.shutil=SimpleNamespace(disk_usage=lambda _,delta=delta:SimpleNamespace(free=C.TARGET+delta))
       boundary=Q.remaining(plan);require(boundary['capacity_pass'] is expected_pass and boundary['cleanup_target']==C.TARGET,'Complete remaining plus2GiB boundary differs')
     finally:Q.shutil=previous_shutil
     checked_cases.append('remaining_extra2GiB_boundary')
    with fixture('valid_one',pins,pilot) as plan:
     make_accepted(0,plan,pilot);require(Q.state(Q.bind())[0]==1,'Fixture positive accepted prefix failed')
    for name in ['source','environment','budget','design','stop','premature_selection','partial_slot','missing_anchor','request','kind','targets','weak_guard','ancestry','entry_budget','serialized_count','extra_weight','extra_tmp','foreign_seed']:
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
       if name=='missing_anchor':(C.B/'projection-quality001-accepted-00.json').rename(C.B/'retained-moved-fixture-anchor.json');reject(lambda:Q.state(plan))
       elif name=='request':(base/'request.json').rename(base/'retained-moved-fixture-request.json');reject(lambda:Q.state(plan))
       elif name in ['extra_weight','extra_tmp']:
        (base/'checkpoints'/('extra.pt' if name=='extra_weight' else 'extra.pt.tmp')).write_bytes(b'EXTRA_SYNTHETIC_ONLY');reject(lambda:Q.state(plan))
       elif name in ['kind','foreign_seed','serialized_count']:
        w=C.checked(row['worker'])
        if name=='kind':w['kind']='readout_projection_prerequisite_worker_v1'
        elif name=='serialized_count':w['serialized_bytes']['pair']+=1
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
    for mode in ('parameter_count','ema_count','exposure','clipping_summary','target_policy'):
     with fixture('reject_new_'+mode,pins,pilot) as plan:
      row=make_accepted(0,plan,pilot);base=C.RUN/'evidence/quality/00';w=C.checked(row['worker'])
      if mode=='parameter_count':w['parameter_tensors']-=1;row['acceptance']['parameter_tensors']-=1
      elif mode=='ema_count':w['ema_snapshots']=40;row['acceptance']['ema_snapshots']=40
      elif mode=='exposure':
       first=next(iter(w['parameter_gradient_steps']));w['parameter_gradient_steps'][first]=1199;row['acceptance']['parameter_gradient_steps'][first]=1199
      elif mode=='clipping_summary':row['acceptance']['clipped_updates']+=1
      else:row['acceptance']['full_validation']['h200']['mps']['target']['policy']='foreign_projection_target'
      overwrite(base/'worker.json',w);row['worker']=reference(base/'worker.json');resign_row(0,row);reject(lambda:Q.state(plan));checked_cases.append('new_'+mode)
    with fixture('duplicate_weight_bytes',pins,pilot) as plan:
     first=make_accepted(0,plan,pilot);second=make_accepted(1,plan,pilot);a=C.checked(first['worker']);b=C.checked(second['worker']);Path(b['parent']['path']).write_bytes(Path(a['parent']['path']).read_bytes());b['parent']=dict(reference(b['parent']['path']),size=Path(b['parent']['path']).stat().st_size);overwrite(Path(second['worker']['path']),b);second['worker']=reference(second['worker']['path']);second['acceptance']['binding']['parent']={k:b['parent'][k] for k in ['path','sha256']};resign_row(1,second);reject(lambda:Q.state(plan));checked_cases.append('duplicate_weight_bytes')
    with fixture('nine_and_immutable_selection',pins,pilot) as plan:
     for i in range(9):make_accepted(i,plan,pilot)
     require(Q.state(plan)[0]==9,'Complete screen state failed');Q.resolve_selection(plan);choice=Q.verify_selection(plan);require(choice['choices']=={'native':.000625,'linear_projection1':.000625,'tanh_projection1':.000625},'Immutable lower-LR tie failed');Q.verify_confirmation(plan)
     bad=C.checked(reference(C.SELECTION));bad['choices']['tanh_projection1']=.0025;overwrite(C.SELECTION,bad);reject(lambda:Q.verify_selection(plan));checked_cases.append('immutable_selection_forgery')
    with fixture('complete_eighteen_and_fresh_contracts',pins,pilot) as plan:
     for i in range(9):make_accepted(i,plan,pilot)
     Q.resolve_selection(plan);reject(Q.final)
     for i in range(9,18):make_accepted(i,plan,pilot)
     require(Q.state(Q.bind())[0]==18,'Complete18source/request/ancestry/weight inventory failed')
     resolved=Q.verify_confirmation(plan)
     require([v['arm'] for v in resolved['specs']]==['native','linear_projection1','tanh_projection1','linear_projection1','tanh_projection1','native','tanh_projection1','native','linear_projection1'],'Original three-way rotation changed')
     bad=copy.deepcopy(resolved);bad['specs'][0]['config']['training']['seed']+=99;overwrite(C.CONFIRM,bad);reject(lambda:Q.verify_confirmation(plan));checked_cases.extend(['partial_final','fresh_confirmation_forgery'])
    templates={r['arm']:r['acceptance']['full_validation']['h200']['mps']['target'] for r in pilot['rows']};panel=[]
    for i,seed in enumerate(SEEDS['fresh']):panel.append(dict(seed=seed,native=synthetic(templates['native'],.003*i),linear_projection1=synthetic(templates['linear_projection1'],-.01+.003*i),tanh_projection1=synthetic(templates['tanh_projection1'],-.06+.002*i)))
    from research.readout_projection_quality_v1.statistics import groups
    for width in [2048,4096,8192]:
     for control in ['native','linear_projection1']:
      differences=[a-b for a,b in zip(groups(panel[0]['tanh_projection1'],width),groups(panel[0][control],width))];require(max(differences)-min(differences)>1e-5,'Bootstrap contrast must be nonconstant')
    primary=review(panel,SEEDS['bootstrap']);independent=verify(panel,SEEDS['fresh'],SEEDS['bootstrap'],primary);require(independent['quality_gates_pass'],'Independent nonconstant full-review fixture failed')
    bad=copy.deepcopy(primary);bad['contrasts']['linear_projection1']['intervals']['4096']['upper']+=.01;reject(lambda:verify(panel,SEEDS['fresh'],SEEDS['bootstrap'],bad));checked_cases.append('independent_review_forgery')
    # No partial/foreign/nonfinite panel or incomplete LR set is a quality result.
    from research.readout_projection_quality_v1.statistics import select,report_contract
    for mode in ['partial','duplicate','foreign','nonfinite','target','aux']:
     bad=copy.deepcopy(panel)
     if mode=='partial':bad.pop()
     elif mode=='duplicate':bad[1]['seed']=bad[0]['seed']
     elif mode=='foreign':bad[0]['extra']=bad[0]['native']
     elif mode=='nonfinite':bad[0]['tanh_projection1']['ce']=float('nan')
     elif mode=='target':bad[0]['tanh_projection1']['target']['targets']-=1
     else:bad[0]['tanh_projection1']['aux_loss_included']=True
     reject(lambda bad=bad:review(bad,SEEDS['bootstrap']));checked_cases.append('statistics_'+mode)
    lrrows=[dict(status='accepted',arm=a,peak_lr=r,h200=synthetic(templates[a],0.)) for r in C.LRS for a in C.ARMS]
    require(select(lrrows)=={a:.000625 for a in C.ARMS},'Exact lower LR tie selection failed')
    best=copy.deepcopy(lrrows);best[4]['h200']=synthetic(templates['linear_projection1'],-.1);require(select(best)['linear_projection1']==.00125,'Per-arm complete minimum selection failed')
    reject(lambda:select(lrrows[:-1]));duplicate=copy.deepcopy(lrrows);duplicate[0]=duplicate[1];reject(lambda:select(duplicate));checked_cases.extend(['partial_screen','duplicate_screen'])
    total_metadata=sum(p.stat().st_size for p in FIXTURE.rglob('*') if p.is_file());require(total_metadata<=64*1024**2,'Bounded64MiB synthetic metadata cap exceeded')
    require(all(reference(ROOT/n)['sha256']==h for n,h in proof_pins.items()),'Proof source changed')
    require(C.sources()==pins and not C.RUN.exists() and not C.PLAN.exists() and not C.FREEZE.exists(),'Source changed or production attempt created')
   require(torch.equal(cpu_before,torch.get_rng_state()),'Generated contract proof changed global CPU RNG')
   publish(E/'projection-quality-contracts001.json',dict(status='passed_full_quality_bind_state_sources_and_final_contracts',source=reference(__file__),sources=pins,total_source_bindings=1161,proof_source_bindings=1164,design=C.DESIGN,environment=env,
    real_pilot_state_and_full_layout_checks=actual,rejected_cases=checked_cases,positive_actual_source_fixture_binding=True,fixture_path_locators_only=True,synthetic_tiny_payloads_NOT_numerical_or_quality_acceptance=True,independent_nonconstant_final_review_agreed=True,
    training=False,model_scoring=False,holdout_encoded_or_scored=False,scientific_seed_used=False,weight_files_written=0,global_cpu_mps_rng_preserved=True,tiny_synthetic_fixture_files_not_checkpoints=True,
    next_action='Independent closed metadata review and actual distinctpilotserving/resource/admission review before scientificfreeze. Combinedcensusscientific10stillunused;generatedfixture6nowUSED. All production roots unattempted; no source/fixture replay.'))
   print(json.dumps(dict(status='passed_full_quality_bind_state_sources_and_final_contracts',rejected_cases=checked_cases,total_sources=1161,serialized_layouts=[dict(arm=r['arm'],counted_bytes=r['full_history_layout_counted_bytes']) for r in actual])),flush=True)
   publish(R/'STOP',dict(status='complete_new_projection_full_quality_contract_consumer',result=reference(E/'projection-quality-contracts001.json'),cpu_guard=reference(R/'cpu-guard.json'),mps_guard=reference(R/'mps-guard.json'),no_replay=True,no_retry=True))
  except BaseException as exc:
   if not(R/'STOP').exists():publish(R/'STOP',dict(status='closed_failed',error=repr(exc),no_retry=True))
   raise

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--plan-sha',required=True);main(parser.parse_args())
