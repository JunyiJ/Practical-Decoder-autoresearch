"""Frozen nine-screen/three-triple controller;every launch follows complete prefix/capacity."""
import argparse,gc,json,math,shutil
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_mla_gate_resume_v1.contract import runtime as environment
from omegaconf import OmegaConf
from research.tokenizer_mla_gate_resume_v1.data import architecture
from research.tokenizer_mla_gate_resume_v1.data import load
from .statistics import select,review,report_contract
from research.tokenizer_mla_gate_worker_v1.execution import idle
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_metadata import snapshot
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from . import contract as C
from .dispatch import launch
from .validation import verify_artifact
from .benchmark import measure
from .final_review import verify as independent_verify

PREREQUISITES=['quality-design001.json','continuation-completed003.json','gate-quality-contracts001.json','gate-quality-seed-census001/result.json']
def bind():
 require(not(C.RUN/'STOP').exists(),'Original terminal STOP forbids every scientific command')
 seal=C.checked(reference(C.FREEZE));plan=C.checked(seal['plan'])
 require(seal['plan']==reference(C.PLAN) and seal['design']==C.DESIGN and plan['kind']==C.SCREEN_KIND and plan['root']==str(C.RUN),'Exact original scientific freeze required')
 require(plan['sources']==C.sources() and plan['environment']==environment() and plan['design']==C.DESIGN and plan['specs']==C.screen_specs(plan['seeds']) and plan['runner']==reference(__file__),'Scientific source/config/environment differs')
 require(plan['seconds']==7200 and plan['compare_cpu'] is True and plan['required_free']==C.FLOOR,'Original scientific resource/acceptance gates differ')
 require(set(plan['prerequisites'])==set(PREREQUISITES),'Complete original prerequisites required')
 for ref in plan['prerequisites'].values():C.checked(ref)
 finite=C.checked(plan['prerequisites']['gate-quality-contracts001.json']);require(finite['status']=='passed_full_quality_bind_state_sources_and_final_contracts' and finite['sources']==plan['sources'] and finite['design']==C.DESIGN,'Original full-quality source gate differs')
 census=C.checked(plan['prerequisites']['gate-quality-seed-census001/result.json']);require(census['status']=='reserved_unused' and census['seeds']==plan['seeds'],'Original unused seed roles differ')
 return plan

def read_anchor(slot,plan):
 base=C.RUN/f'evidence/quality/{slot:02d}';anchor=C.checked(reference(C.B/f'gate-quality001-accepted-{slot:02d}.json'))
 require(anchor['accepted']==reference(base/'accepted.json') and anchor['original_freeze']==reference(C.FREEZE),'Original acceptance anchor differs')
 row=C.checked(anchor['accepted']);require(row['slot']==slot and row['worker']==reference(base/'worker.json') and row['request']==reference(base/'request.json'),'Exact worker/request/slot provenance required')
 expected=plan['specs'][slot] if slot<9 else C.fresh_specs(plan['seeds'],C.checked(reference(C.SELECTION))['choices'])[slot-9]
 rq=C.checked(row['request']);w=C.checked(row['worker']);require(rq['spec']==expected and rq['slot']==slot and rq['stage']=='quality' and rq['plan']==reference(C.PLAN if slot<9 else C.CONFIRM) and w['request']==row['request'] and row['arm']==expected['arm'],'Frozen request/config/arm differs')
 config=C.checked(reference(base/'config.json'));require(w['config']==config==expected['config'] and w['kind']=='tokenizer_mla_gate_quality_worker_v1' and w['step']==1200 and w['targets']==1228800 and w['ema_snapshots']==41 and w['parameter_tensors']==(92 if row['arm']=='sigmoid_elementwise_zero' else 88) and w['architecture']==architecture(OmegaConf.create(config)),'Original complete scientific endpoint required')
 require(len(w['update_seconds'])==1200 and [v['step'] for v in w['metrics']]==[i+1 for i in range(1200) if i%config['training']['eval_every']==0 or i==1199],'Complete original update/metric history required')
 process_doc=None;guard_docs={}
 for k,name in [('process','process/process.json'),('worker_guard',f'../../../control/worker-quality-{slot:02d}-guard.json'),('parent_guard',f'../../../control/parent-quality-{slot:02d}-guard.json')]:
  p=(base/name).resolve();require(row[k]==reference(p),'Original process/guard path differs');d=C.checked(row[k])
  if k=='process':
   require(d['status']=='completed' and d['returncode']==0 and d['timeout_seconds']==7200 and d['reserve_bytes']==C.RESERVE and d['output_limit_bytes']==expected['weight_cap']+10*1024**2 and not d['log_truncated'],'Original full process guards differ');process_doc=d
  else:
   require(d['status']=='finished' and d['reason'] is None and d['limits']==dict(mps_driver_limit_bytes=6*1024**3,reserve_bytes=C.RESERVE,rss_limit_bytes=4*1024**3,timeout_seconds=7200),'Original sampled resource guard failed/weakened');guard_docs[k]=d
 require(type(rq['parent_pid']) is int and rq['parent_pid']>0 and process_doc['parent_pid']==rq['parent_pid'] and guard_docs['parent_guard']['pid']==rq['parent_pid'] and guard_docs['worker_guard']['pid']==process_doc['pid'],'Actual parent/child process/guard ancestry differs')
 prior_bytes=0
 for prior in range(slot):
  old_anchor=C.checked(reference(C.B/f'gate-quality001-accepted-{prior:02d}.json'));old_row=C.checked(old_anchor['accepted']);old_worker=C.checked(old_row['worker']);prior_bytes+=sum(old_worker[k]['size'] for k in ['parent','h200'])
 require(process_doc['required_free_bytes']==max(C.RESERVE,C.FLOOR-prior_bytes),'Original complete remaining launch budget differs')
 result=row['acceptance'];require(result['status']=='accepted' and result['step']==1200 and result['targets']==1228800 and result['ema_snapshots']==41 and result['parameter_tensors']==w['parameter_tensors'] and result['architecture']==w['architecture'] and result['complete_raw_optimizer_ema_rng_sampler_clipping_h200_checked'] and result['holdout_scored'] is False,'Independent parent state acceptance incomplete')
 require(result['parameter_gradient_steps']==w['parameter_gradient_steps'] and result['clipped_updates']==w['clipped_updates'],'Independent exposure/routing/clipping acceptance differs')
 require(all(type(v) is int and 0<=v<=1200 for v in result['parameter_gradient_steps'].values()) and len(result['parameter_gradient_steps'])==w['parameter_tensors'],'Complete actual per-parameter update inventory required')
 require(sum(w[k]['size'] for k in ['parent','h200'])<=expected['weight_cap'],'Actual serialized pair exceeds frozen cap')
 require(all(v==1200 for v in result['parameter_gradient_steps'].values()),'Complete learned parameter update exposure required')
 expected_binding={'config':reference(base/'config.json')}
 for k,filename in [('parent','parent.pt'),('h200','h200.pt')]:
  require(w[k]['path']==str(base/'checkpoints'/filename),'Foreign/prerequisite tensor credit forbidden');verify_artifact(w[k]);expected_binding[k]=dict(path=w[k]['path'],sha256=w[k]['sha256'])
 require(result['binding']==expected_binding,'Parent artifact bindings differ')
 for name in ['raw','h200']:
  scores=result['full_validation'][name]
  value=scores['cpu_mps_ce_difference'];require(type(value) is float and math.isfinite(value) and 0<=value<=3e-5,'Full CPU/MPS native CE agreement missing')
  for device in ['cpu','mps']:report_contract(scores[device],row['arm'])
 for device in ['cpu','mps']:require([r['batch'] for r in result['cache'][device]]==[1,8] and all(r['cache_bytes_per_item']==196608 for r in result['cache'][device]),'Full/cache/reset checks missing')
 return row,w

def state(plan):
 accepted=[];attempted=[]
 for i in range(18):
  base=C.RUN/f'evidence/quality/{i:02d}';exists=(base/'accepted.json').exists();require(exists==(C.B/f'gate-quality001-accepted-{i:02d}.json').exists(),'Unanchored/missing original accepted endpoint')
  accepted.append(exists);attempted.append(base.exists())
 n=C.prefix_order(accepted,attempted);expected={};rows=[];hashes=set()
 if n<9:require(not C.SELECTION.exists() and not C.CONFIRM.exists(),'Premature screen selection/fresh resolution forbidden')
 require(not C.CONFIRM.exists() or C.SELECTION.exists(),'Fresh resolution requires immutable screen selection')
 if n>9:verify_selection(plan);verify_confirmation(plan)
 for i in range(n):
  row,w=read_anchor(i,plan);rows.append((row,w))
  for key in ['parent','h200']:
   relative=str(Path(w[key]['path']).relative_to(C.RUN));require(relative not in expected and w[key]['sha256'] not in hashes,'Duplicate byte-copy/path weight credit forbidden');hashes.add(w[key]['sha256']);expected[relative]=w[key]['size']
 actual={}
 if C.RUN.exists():actual={name:v['size'] for name,v in snapshot(C.RUN)['files'].items() if v['group'].endswith('/weights')}
 used=C.weight_credit(actual,expected);return n,used,rows

def remaining(plan):
 n,used,rows=state(plan);free=shutil.disk_usage(ROOT).free;required=max(C.RESERVE,C.FLOOR-used)
 return dict(prefix=n,free_bytes=free,actual_current_quality_retained_weight_bytes=used,required_free=required,wake_free=required+C.ENTRY,cleanup_target=required+2*1024**3,mandatory_deficit=max(0,required+C.ENTRY-free),capacity_pass=free>=required+C.ENTRY)

def selection_rows(plan):
 rows=[]
 for slot in range(9):
  row,w=read_anchor(slot,plan);rows.append(dict(slot=slot,status='accepted',arm=row['arm'],peak_lr=w['config']['training']['learning_rate'],h200=row['acceptance']['full_validation']['h200']['mps'],accepted=reference(C.RUN/f'evidence/quality/{slot:02d}/accepted.json')))
 return rows

def verify_selection(plan):
 chosen=C.checked(reference(C.SELECTION));rows=selection_rows(plan);require(chosen==dict(kind='immutable_per_arm_native_CE_LR_selection',original_freeze=reference(C.FREEZE),rows=rows,choices=select(rows)),'Original minimum/lowerLR selection changed');return chosen

def verify_confirmation(plan):
 selection=verify_selection(plan);expected=dict(kind=C.FRESH_KIND,root=str(C.RUN),sources=plan['sources'],environment=plan['environment'],design=C.DESIGN,seeds=plan['seeds'],specs=C.fresh_specs(plan['seeds'],selection['choices']),seconds=7200,compare_cpu=True,required_free=C.FLOOR,original_freeze=reference(C.FREEZE),selection=reference(C.SELECTION))
 result=C.checked(reference(C.CONFIRM));require(result==expected,'Original predeclared fresh-panel resolution differs');return result

def resolve_selection(plan):
 if not C.SELECTION.exists():
  rows=selection_rows(plan);C.publish(C.SELECTION,dict(kind='immutable_per_arm_native_CE_LR_selection',original_freeze=reference(C.FREEZE),rows=rows,choices=select(rows)))
 chosen=verify_selection(plan)
 if not C.CONFIRM.exists():C.publish(C.CONFIRM,dict(kind=C.FRESH_KIND,root=str(C.RUN),sources=plan['sources'],environment=plan['environment'],design=C.DESIGN,seeds=plan['seeds'],specs=C.fresh_specs(plan['seeds'],chosen['choices']),seconds=7200,compare_cpu=True,required_free=C.FLOOR,original_freeze=reference(C.FREEZE),selection=reference(C.SELECTION)))
 verify_confirmation(plan)

def prepare():
 require(not C.RUN.exists() and not C.PLAN.exists() and not C.FREEZE.exists(),'New unattempted finite scientific context only')
 pins=C.sources();prereqs={n:reference(C.B/n) for n in PREREQUISITES};values={n:C.checked(r) for n,r in prereqs.items()}
 require(values['continuation-completed003.json']['status']=='all_native_gate_worker_and_resource_prerequisites_closed_independently_reviewed' and values['gate-quality-contracts001.json']['status']=='passed_full_quality_bind_state_sources_and_final_contracts' and values['gate-quality-contracts001.json']['sources']==pins,'Actual correctness/resource/full-quality source contracts required')
 census=values['gate-quality-seed-census001/result.json'];require(census['status']=='reserved_unused','Current unused scientific seed census required');C.seeds_valid(census['seeds'])
 reserved={census['seeds']['screen'],census['seeds']['evaluation'],*census['seeds']['fresh'],census['seeds']['bootstrap']}
 for parent in (ROOT/'research_runs').iterdir():
  if parent.is_dir():
   for p in parent.glob('evidence/*/*/config.json'):require(json.loads(p.read_text()).get('training',{}).get('seed') not in reserved and json.loads(p.read_text()).get('training',{}).get('eval_seed') not in reserved,'Reserved scientific training/evaluation seed already used')
 observed=dict(free_bytes=shutil.disk_usage(ROOT).free,required_free=C.FLOOR,wake_free=C.FLOOR+C.ENTRY,cleanup_target=C.TARGET);require(observed['free_bytes']>=observed['wake_free'],'Complete all-output budget and measured entry margin required')
 plan=dict(kind=C.SCREEN_KIND,root=str(C.RUN),runner=reference(__file__),sources=pins,environment=environment(),design=C.DESIGN,seeds=census['seeds'],specs=C.screen_specs(census['seeds']),seconds=7200,compare_cpu=True,required_free=C.FLOOR,initial_capacity=observed,prerequisites=prereqs)
 C.RUN.mkdir();C.publish(C.PLAN,plan);C.publish(C.FREEZE,dict(plan=reference(C.PLAN),design=C.DESIGN,phase='Original nine-screen selection rule plus three fresh triples already declared;all weights retained.'))
 print(json.dumps(dict(status='scientific_comparison_frozen',freeze=reference(C.FREEZE),capacity=observed)),flush=True)

def advance(global_fd):
 plan=bind()
 with session_lock(C.RUN) as r:
  writer=MetadataWriter(C.RUN,(global_fd,r));observed=remaining(plan);n=observed['prefix']
  if n==18:return False
  if n==9:resolve_selection(plan)
  active=plan if n<9 else verify_confirmation(plan);spec=active['specs'][n if n<9 else n-9]
  if not observed['capacity_pass']:print(json.dumps(dict(status='capacity_wait_before_unattempted_slot',slot=n,capacity=observed)),flush=True);return False
  launch(writer,reference(C.PLAN if n<9 else C.CONFIRM),spec,observed['required_free'])
  C.publish(C.B/f'gate-quality001-accepted-{n:02d}.json',dict(accepted=reference(C.RUN/f'evidence/quality/{n:02d}/accepted.json'),original_freeze=reference(C.FREEZE)))
  return True

def final():
 plan=bind();n,used,rows=state(plan);require(n==18 and not(C.B/'gate-quality001-final-review.json').exists(),'Complete original unreviewed scientific panel required');verify_confirmation(plan);panel=[];fresh=[]
 for p,seed in enumerate(plan['seeds']['fresh']):
  pair=dict(seed=seed)
  for slot in [9+3*p,10+3*p,11+3*p]:
   row,w=rows[slot];require(w['config']['training']['seed']==seed and row['arm'] not in pair,'Original fresh seed/arm differs');pair[row['arm']]=row['acceptance']['full_validation']['h200']['mps'];fresh.append(row)
  require(set(pair)=={'seed',*C.ARMS},'Complete fresh triple required');panel.append(pair)
 with C.Guard(C.B/'gate-quality001-fresh-inference-guard.json',600):inference=measure(fresh);require(C.sources()==plan['sources'],'Frozen scientific source changed')
 C.publish(C.B/'gate-quality001-fresh-inference.json',dict(rows=inference,all_nine_fresh=True,selection=False,holdout_scored=False))
 primary=review(panel,plan['seeds']['bootstrap']);C.publish(C.B/'gate-quality001-primary-statistics.json',dict(original_freeze=reference(C.FREEZE),panel=panel,statistics=primary,adopted=False))
 with CpuPrerequisiteGuard(C.B/'gate-quality001-independent-review-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
  independent=independent_verify(panel,plan['seeds']['fresh'],plan['seeds']['bootstrap'],primary);require(C.sources()==plan['sources'],'Frozen sources changed before independent closure')
 C.publish(C.B/'gate-quality001-independent-final-review.json',dict(source=reference(Path(__file__).with_name('final_review.py')),primary=reference(C.B/'gate-quality001-primary-statistics.json'),result=independent))
 report=dict(status='complete_fresh_panel_reviewed',freeze=reference(C.FREEZE),selection=reference(C.SELECTION),confirmation=reference(C.CONFIRM),fresh_acceptances=[reference(C.RUN/f'evidence/quality/{i:02d}/accepted.json') for i in range(9,18)],statistics=primary,independent_review=reference(C.B/'gate-quality001-independent-final-review.json'),inference=reference(C.B/'gate-quality001-fresh-inference.json'),adopted=False,
  next_action='Separate evidence/adoption/checkpoint consumer review only if all strict quality gates pass;otherwise retain incumbent.Then continue evidence-qualified RSI.')
 C.publish(C.B/'gate-quality001-final-review.json',report);C.publish(C.RUN/'STOP',dict(status='complete_finite_scientific_panel',review=reference(C.B/'gate-quality001-final-review.json'),never_rerun_training_inference_review=True));print(json.dumps(dict(status=report['status'],statistics=primary)),flush=True)

def once(mode):
 with session_lock(GLOBAL_LOCK.parent) as g:
  idle();environment()
  try:
   if mode=='prepare':prepare();return False
   if mode=='advance':return advance(g)
   if mode=='final':
    with session_lock(C.RUN):final();return False
   if mode=='close-if-ready':
    with session_lock(C.RUN):
     if state(bind())[0]==18:final()
    return False
  except Exception as exc:
   if C.RUN.exists() and not(C.RUN/'STOP').exists():C.publish(C.RUN/'STOP',dict(error=repr(exc),mode=mode,no_retry=True))
   raise
if __name__=='__main__':
 torch.set_num_threads(4);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','advance','run-all','final']);args=p.parse_args()
 if args.mode=='run-all':
  while once('advance'):pass
  if not(C.RUN/'STOP').exists():once('close-if-ready')
 else:once(args.mode)
