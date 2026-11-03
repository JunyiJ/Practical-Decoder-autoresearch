"""Separate adoption audit after all original gates pass; no training or selection."""
import copy,json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
import torch
from research.adamw_epsilon_quality_v1 import campaign_contract as c,campaign_rules as rules,quality_execution as q
from research.block_norm_v1.fixture_dispatch import reference,read
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
B=q.EVIDENCE;SHA='a82295c46e8fbfef432d943eca5c26f59fd79a5b02235047ad0c1e15045bb1c4'
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent) as g,session_lock(q.ROOT) as lock:
 q.idle();assert not (q.ROOT/'STOP').exists()
 with CpuPrerequisiteGuard(B/'quality001-adopt001-guard.json',q.ROOT,timeout_seconds=120):
  writer=MetadataWriter(q.ROOT,(g,lock));plan=q.bind(writer,SHA);state=q.state(plan,SHA)
  finalref=reference(B/'quality001-final-review.json');final=read(finalref)
  assert final['status']=='reviewed_gate_passed_pending_adoption' and not final['quality_adopted'] and final['strict_quality']
  assert final['freeze']==dict(path=str(q.FREEZE),sha256=SHA)
  assert len(state['records'])==6 and final['accepted_anchors']==[reference(q.anchor_path(k)) for k in range(6)]
  assert rules.quality_gate(final['statistics'])['strict_quality'] and all(final['gate_checks'].values())
  assert len(final['inference_reviews'])==3 and not final['holdout_scored']
  for ref in final['inference_reviews']:
   row=read(ref);assert row['status']=='passed';guard=read(row['guard']);assert guard['status']=='finished' and guard['reason'] is None
  rows=state['receipts'];configs=[]
  for n,row in enumerate(rows):
   req=row['request'];spec=plan['fresh'][n];cfg=req['config'];assert req['trial']['spec']==spec
   assert cfg['training']['adamw_epsilon']==(1e-8 if spec['arm']=='eps8' else 1e-6)
   audit=row['artifact_acceptance']['audit'];assert audit['completed_steps']==1200 and audit['tokens_processed']==1228800 and audit['ema_snapshots']==41 and audit['ema_parameters']==88
   assert audit['optimizer_groups']==dict(beta1=.9,beta2=.999,epsilon=cfg['training']['adamw_epsilon'],adamw_groups=44,muon_groups=44)
   shared=copy.deepcopy(cfg)
   for k in ('adamw_epsilon','checkpoint_dir','metrics_file','summary_file'):del shared['training'][k]
   configs.append(shared)
  assert all(configs[k]==configs[k+1] for k in (0,2,4))
  # Fixed representative: first candidate in frozen order, never choose by loss.
  chosen=rows[1];assert chosen['request']['trial']['spec']['arm']=='eps6'
  recipe=dict(status='adopted_under_original_three_pair_gates',config=chosen['request']['config'],
    candidate='AdamW epsilon1e-6',previous='AdamW epsilon1e-8',scope=final['scope'],
    freeze=final['freeze'],source_inventory=plan['sources'],final_review=finalref,
    representative=dict(slot=1,reason='First candidate in frozen order; not loss-selected',files=chosen['request']['files']),
    accepted_anchors=final['accepted_anchors'],statistics=final['statistics'],gate_checks=final['gate_checks'],
    future_baseline='Use this exact learned-bias D256 MLA recipe with epsilon1e-6 through explicit additive source applicability. Historical baseline files and frozen guards stay immutable.',
    training_budget=dict(steps=1200,targets=1228800,lr=.00125,lr_search_per_arm=0),
    holdout_scored=False,population_seed_confidence_claimed=False)
  q.exclusive(B/'adopted-width256-mla-epsilon-recipe.json',recipe)
  current=q.state(plan,SHA);assert current['used_weights']==state['used_weights']
  value=dict(status='closed_adopted',at=datetime.now(timezone.utc).isoformat(),runner=reference(Path(__file__).resolve()),
    freeze=final['freeze'],final_review=finalref,adopted_recipe=reference(B/'adopted-width256-mla-epsilon-recipe.json'),
    statistics=final['statistics'],gate_checks=final['gate_checks'],accepted_anchors=final['accepted_anchors'],inference_reviews=final['inference_reviews'],
    accepted_endpoints=6,fresh_pairs=3,quality_adopted=True,strict_quality=True,selection=None,retained_quality_weight_bytes=current['used_weights'],
    sources_unchanged=len(plan['sources']),all_weights_unchanged=True,holdout_scored=False,training_launched=False,deletion_performed=False,
    free_bytes=shutil.disk_usage(q.ROOT).free,next_action='Review architecture and bounded HP, select one distinct prerequisite question using adopted epsilon1e-6 baseline outside frozen sources.')
  writer.publish_json('STOP',dict(status='completed_closed_adopted',final_review=finalref,adopted_recipe=value['adopted_recipe'],reason='Original strict gates and separate adoption audit passed. All quality training/inference/final commands closed; no extension or retry.'))
 value['guard']=reference(B/'quality001-adopt001-guard.json');assert read(value['guard'])['status']=='finished'
 q.exclusive(B/'quality001-closure.json',value);print(json.dumps({k:value[k] for k in ['status','retained_quality_weight_bytes','statistics','gate_checks']}))
