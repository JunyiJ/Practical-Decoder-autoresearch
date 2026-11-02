"""Read-only failed-output audit and bounded publication regression; no training or retry."""
import json,copy,hashlib,gc
from pathlib import Path
import torch
from research.gradient_clip_quality_v1 import quality_execution as q,campaign_contract as c,artifact_acceptance as a
from research.gradient_clip_quality_v1.support import cache_audit
from research.block_norm_v1.fixture_dispatch import read,reference
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter,json_encoder
from src.research.tokenizer_wide_hybrid_metadata import budgets,snapshot,classify
B=q.EVIDENCE;OUT=B/'publication-recovery001';DISPOSABLE=c.ROOT/'research_runs/tokenizer-gradient-clip-publication-recovery001';OLD=q.ROOT

def inventory():
 return {str(p.relative_to(OLD)):q.file_ref(p,p.suffix=='.pt') for p in sorted(OLD.rglob('*')) if p.is_file()}
def length(value):return sum(len(x) for x in json_encoder().iterencode(value))+1

torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent) as g:
 q.idle();assert not OUT.exists() and not DISPOSABLE.exists();OUT.mkdir()
 plan=read(reference(B/'publication-recovery001-plan.json'));assert plan['runner']==reference(__file__);c.verify_sources(plan['sources'])
 before=inventory();assert before==plan['failed_root_files'];original=read(plan['original_plan']);stop=read(plan['stop'])
 assert stop['no_retry'] and stop['slot']==0 and not q.anchor_path(0).exists()
 with q.QualityGuard(OUT/'guard.json',c.ROOT,600):
  base=OLD/'evidence/quality/00';cfg=json.loads((base/'config.json').read_text());spec=original['fresh'][0]
  c.validate_trial_config(cfg,base,spec,original['seeds'])
  raw=next((base/'checkpoints').glob('final_*.pt'))
  files={k:q.file_ref(p,k in ('parent','h200')) for k,p in dict(config=base/'config.json',parent=raw,h200=base/'checkpoints/h200.pt',summary=base/'summary.json',metrics=base/'metrics.jsonl',export=base/'export.json').items()}
  request=dict(scope=a.TRIAL_SCOPE,sources=original['sources'],config=cfg,files=files,trial=dict(spec=spec,seeds=original['seeds']))
  binding={k:files[k] for k in ('config','parent','h200')}
  validation=q.v.evaluate(binding);gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
  accepted=a.accept(request,validation);cached=cache_audit(request)
  assert accepted['accepted'] and accepted['audit']['completed_steps']==1200 and accepted['audit']['clipping']['updates']==1200
  guard=read(reference(OLD/'control/worker-00-guard.json'));assert guard['status']=='fixture_exception' and guard['reason']=="ValueError('JSON payload exceeds metadata group cap')"
  assert guard['elapsed_seconds']<7200 and guard['observation']['peak_rss_bytes']<4*1024**3 and guard['observation']['peak_observed_mps_driver_bytes']<6*1024**3 and guard['observation']['free_bytes']>=8*1024**3
  worker=dict(kind=q.KIND,pid=guard['pid'],parent_pid=read(reference(base/'request.json'))['parent_pid'],request=reference(base/'request.json'),acceptance_request=request,validation=validation,cache=cached,holdout_scored=False)
  receipt=dict(kind=q.KIND,slot=0,freeze=plan['freeze'],request=request,artifact_acceptance=accepted,target_contract=validation['target'],cache=cached,process_failure=plan['stop'],selection=None,cleanup=None,adopted=False)
  observed=snapshot(OLD);old_group='evidence/quality/00/other';temporary=(base/'worker.json.tmp').stat().st_size+(base/'process/process.json.tmp').stat().st_size
  required_old=observed['groups'][old_group]-temporary+length(worker)
  assert required_old>budgets()[old_group]
  DISPOSABLE.mkdir()
  with session_lock(DISPOSABLE) as lock:
   writer=MetadataWriter(DISPOSABLE,(g,lock));prefix='evidence/quality/00/'
   # Copy only immutable non-weight metadata that was successfully published.
   for name in ['config.json','export.json','summary.json','request.json','process/launch.json']:
    writer.publish_json(prefix+name,json.loads((base/name).read_text()))
   for name in ['metrics.jsonl','process/worker.log']:
    stream=writer.open_append(prefix+name)
    try:stream.write((base/name).read_bytes())
    finally:stream.close()
   writer.publish_json(prefix+'resource-audit.json',worker)
   writer.publish_json(prefix+'resource-events.jsonl',receipt)
   writer.publish_json(prefix+'process/process.json',dict(status='publication_regression_only',original_failure=plan['stop'],training=False))
   writer.publish_json(prefix+'worker-guard.json',guard)
   writer.publish_json(prefix+'parent-guard.json',dict(status='readonly_regression_only',training=False))
   after=snapshot(DISPOSABLE)
   assert json.loads((DISPOSABLE/prefix/'resource-audit.json').read_text())==worker
   assert json.loads((DISPOSABLE/prefix/'resource-events.jsonl').read_text())==receipt
   assert all(after['groups'][k]<=v for k,v in budgets().items()) and after['checkpoint_logical_bytes']==0
   for name in ['resource-audit.json','resource-events.jsonl']:
    try:writer.publish_json(prefix+name,{})
    except FileExistsError:pass
    else:raise AssertionError('Existing metadata replaced')
  assert inventory()==before;c.verify_sources(plan['sources'])
  q.exclusive(OUT/'worker-reconstructed.json',worker);q.exclusive(OUT/'artifact-review.json',receipt)
  result=dict(status='readonly_payload_and_publication_recovery_checks_passed',plan=reference(B/'publication-recovery001-plan.json'),original_failure_unchanged=True,original_guard_failure_preserved=True,original_accepted_anchor_created=False,scientific_quality_accepted=False,numerical_updates=0,replacement_seed=False,
   accepted_artifact_audit=accepted['audit'],actual_pair_bytes=sum(Path(files[k]['path']).stat().st_size for k in ('parent','h200')),worker=reference(OUT/'worker-reconstructed.json'),artifact_review=reference(OUT/'artifact-review.json'),
   metadata=dict(old_group_cap=budgets()[old_group],actual_required_worker_publication_bytes=required_old,old_group_shortfall=required_old-budgets()[old_group],bounded_alternative=after,worker_report_group=classify(prefix+'resource-audit.json'),parent_receipt_group=classify(prefix+'resource-events.jsonl'),budget_changes=False),
   limitation='Original quality001 remains STOPPED and unaccepted. This independent read-only audit proves retained final raw/H200 integrity and bounded report routing; it does not reopen training or grant adoption. Any continuation needs a distinct frozen publication-recovery contract with zero-update import, original six specs/seeds/gates and every failed artifact preserved.')
 result['guard']=reference(OUT/'guard.json');q.exclusive(OUT/'review.json',result);print(json.dumps(dict(status=result['status'],metadata_shortfall=result['metadata']['old_group_shortfall'],audit=result['accepted_artifact_audit'])))
