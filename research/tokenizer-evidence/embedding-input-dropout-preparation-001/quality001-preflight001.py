"""Actual pre-freeze source/data/nonuse/capacity revalidation after human cleanup."""
import json,gc
from pathlib import Path
import torch
from research.embedding_input_dropout_quality_v1.execution import B,RUN,PLAN,FREEZE,remaining,sources,campaign_sources,quality_sources,environment,idle
from research.embedding_input_dropout_quality_v1.retention import publish
from research.embedding_input_dropout_resume_v1.contract import config
from research.embedding_input_dropout_worker_v1.data import load
from research.body_allocation_model_v1.evidence import reference
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
R=Path.cwd()
with session_lock(GLOBAL_LOCK.parent):
 torch.set_num_threads(4);idle();env=environment()
 with CpuPrerequisiteGuard(B/'quality001-preflight001-guard.json',R,timeout_seconds=120,poll_seconds=.25):
  assert not any(p.exists() for p in [RUN,PLAN,FREEZE]);finite=json.loads((B/'quality-contracts001.json').read_text())
  assert reference(finite['source']['path'])==finite['source']
  ws,ps,qs=sources(),campaign_sources(),quality_sources();assert (finite['worker_sources'],finite['pilot_sources'],finite['quality_sources'])==(ws,ps,qs)
  assert len({**ws,**ps,**qs})==972 and finite['pilot_dependencies_closed'] is True
  worker=json.loads((B/'worker002-review.json').read_text());pilot=json.loads((B/'pilot001-review.json').read_text())
  assert worker['status']=='two_generated_workers_independently_accepted' and pilot['status']=='two_resource_pilots_passed'
  prerequisite_refs=[]
  for name in ['worker001-plan.json','pilot001-plan.json']:
   original_plan=json.loads((B/name).read_text())
   for v in original_plan['prerequisites'].values():assert reference(v['path'])==v;prerequisite_refs.append(v)
  ready=json.loads((B/'worker002-ready001.json').read_text());assert reference(ready['import_proof']['path'])==ready['import_proof'];prerequisite_refs.append(ready['import_proof'])
  census=json.loads((B/'census001/result.json').read_text());assert census['status']=='reserved_unused';seeds=census['seeds'];count=0
  assert seeds['screen']==200090002 and seeds['fresh']==[200090011,200090012,200090013] and seeds['bootstrap']==200090004
  for root in (R/'research_runs').iterdir():
   if not root.is_dir():continue
   for p in root.glob('evidence/*/*/config.json'):
    count+=1;d=json.loads(p.read_text());assert d.get('training',{}).get('seed') not in [seeds['screen'],*seeds['fresh']]
  data=load(config(0.,steps=4,seed=600416),'cpu');identity=data.data_identity;del data;gc.collect()
  recoveries=[R/'research/tokenizer-evidence/human-old-checkpoint-retirement-002/completed.json',R/'research/tokenizer-evidence/human-old-checkpoint-retirement-003/completed.json',R/'research/tokenizer-evidence/storage-recovery-041/completed.json']
  assert json.loads(recoveries[0].read_text())['deleted_files']==404 and json.loads(recoveries[1].read_text())['deleted_files']==3 and json.loads(recoveries[2].read_text())['all_paths_and_readable_bytes_unchanged']
  cap=remaining();assert cap['capacity_pass'] and cap['free_bytes']>=cap['cleanup_target']
  result=dict(status='passed_actual_pre_freeze_revalidation',source=reference(__file__),environment=env,worker_sources=ws,pilot_sources=ps,quality_sources=qs,
   original_data=identity,actual_prerequisite_refs=prerequisite_refs,saved_numerical_configs_checked=count,screen_and_fresh_unused=True,bootstrap_unused=True,no_scientific_seed_consumed=True,
   fixture_private_sampler_seed=600416,no_model_or_training_or_checkpoint_loading=True,holdout_encoded_or_scored=False,
   actual_recovery=[reference(p) for p in recoveries],capacity=cap,quality_root_plan_freeze_absent=True)
  publish(B/'quality001-preflight001.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('worker_sources','pilot_sources','quality_sources','original_data')}),flush=True)
