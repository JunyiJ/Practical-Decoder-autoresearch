"""Validate same-panel zero-update import and five-slot report routing; no training."""
import copy,json,gc
from pathlib import Path
import torch
from research.gradient_clip_quality_v2 import campaign_contract as c,quality_execution as q,artifact_acceptance as a
from research.gradient_clip_quality_v1 import campaign_contract as old
from research.block_norm_v1.fixture_dispatch import reference,read
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_metadata import snapshot,budgets
B=q.EVIDENCE;ROOT=c.ROOT/'research_runs/tokenizer-gradient-clip-recovery-integration002'
def reject(fn):
 try:fn()
 except ValueError:return
 raise AssertionError('Invalid recovery accepted')
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent) as g:
 q.idle();assert not ROOT.exists() and not q.ROOT.exists()
 prior=read(reference(B/'quality001-plan.json'));review=read(reference(B/'publication-recovery001/review.json'));assert review['status']=='readonly_payload_and_publication_recovery_checks_passed'
 sources=c.source_inventory();assert len(sources)==691 and all(sources[k]==v for k,v in prior['sources'].items())
 plan=dict(runner=reference(__file__),sources=sources,original_plan=reference(B/'quality001-plan.json'),recovery=reference(B/'publication-recovery001/review.json'),training=False)
 q.exclusive(B/'recovery-integration002-plan.json',plan)
 with q.QualityGuard(B/'recovery-integration002-guard.json',c.ROOT,600):
  checks=[]
  for spec in prior['fresh']:
   cfg=c.trial_config(q.ROOT/f"evidence/quality/{spec['slot']:02d}",spec,prior['seeds']);base=old.trial_config(c.ROOT/f"research_runs/tokenizer-gradient-clip-quality-v1-001/evidence/quality/{spec['slot']:02d}",spec,prior['seeds'])
   for x in (cfg,base):
    for k in ('checkpoint_dir','summary_file','metrics_file'):del x['training'][k]
   assert cfg==base;checks.append(spec['slot'])
  reject(lambda:q.run_once(None,0,'0'*64));reject(lambda:q.worker(None,0,'0'*64,[]))
  bad=copy.deepcopy(prior['seeds']);bad['fresh'][0]+=100
  reject(lambda:c.trial_config(q.ROOT/'evidence/quality/00',q.rules.fresh_specs(bad)[0],bad))
  request=read(review['worker'])['acceptance_request'];request=copy.deepcopy(request);request['sources']=sources
  validation=read(review['worker'])['validation'];accepted=a.accept(request,validation);cached=q.cache_audit(request)
  assert accepted['audit']==review['accepted_artifact_audit']
  worker=dict(kind=q.KIND,pid=123,parent_pid=122,request=reference(B/'recovery-integration002-plan.json'),acceptance_request=request,validation=validation,cache=cached,holdout_scored=False)
  receipt=dict(kind=q.KIND,slot=0,freeze=reference(B/'quality001-freeze.json'),request=request,artifact_acceptance=accepted,target_contract=validation['target'],cache=cached,process=reference(B/'publication-recovery001/review.json'),selection=None,cleanup=None,adopted=False)
  ROOT.mkdir()
  with session_lock(ROOT) as p:
   writer=MetadataWriter(ROOT,(g,p));prefix='evidence/quality/00/'
   for name,key in [('summary.json','summary'),('config.json','config'),('export.json','export')]:writer.publish_json(prefix+name,read(request['files'][key]))
   writer.publish_json(prefix+'resource-audit.json',worker);writer.publish_json(prefix+'resource-events.jsonl',receipt)
   snap=snapshot(ROOT);assert snap['checkpoint_logical_bytes']==0 and all(snap['groups'][k]<=v for k,v in budgets().items())
  original=read(read(reference(B/'publication-recovery001/review.json'))['plan'])
  for rel,ref in original['failed_root_files'].items():assert q.file_ref(Path(ref['path']),ref['path'].endswith('.pt'))==ref
  c.verify_sources(sources)
 result=dict(status='same_six_specs_zero_update_import_and_bounded_publication_passed',plan=reference(B/'recovery-integration002-plan.json'),guard=reference(B/'recovery-integration002-guard.json'),same_six_scientific_configs_except_output_paths=checks,slot00_training_forbidden=True,replacement_seed_rejected=True,artifact_audit=accepted['audit'],request=request,validation=validation,cache=cached,metadata=snap,numerical_updates=0,original_failed_artifacts_unchanged=True)
 q.exclusive(B/'recovery-integration002-review.json',result);print(json.dumps(dict(status=result['status'],sources=len(sources),tests=10)))
