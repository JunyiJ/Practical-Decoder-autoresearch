"""Freeze unchanged scientific panel and independently import slot00 with zero updates."""
import json,copy,shutil,datetime
from pathlib import Path
import torch
from research.gradient_clip_quality_v2 import campaign_contract as c,quality_execution as q,artifact_acceptance as a
from research.block_norm_v1.fixture_dispatch import read,reference
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_metadata import snapshot,budgets
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
B=q.EVIDENCE;torch.set_num_threads(4)
def root_unchanged():
 original=read(reference(B/'publication-recovery001-plan.json'))
 for ref in original['failed_root_files'].values():assert q.file_ref(Path(ref['path']),ref['path'].endswith('.pt'))==ref
 return original
with session_lock(GLOBAL_LOCK.parent) as g:
 q.idle();assert not q.ROOT.exists() and not q.FREEZE.exists()
 try:
  with CpuPrerequisiteGuard(B/'quality002-freeze-guard.json',c.ROOT,timeout_seconds=120):
   checked=read(reference(B/'recovery-integration002-review.json'));assert checked['status']=='same_six_specs_zero_update_import_and_bounded_publication_passed' and checked['numerical_updates']==0
   prereq=read(checked['plan']);sources=c.source_inventory();assert sources==prereq['sources'];original=root_unchanged();old=read(original['original_plan']);proof=read(reference(B/'publication-recovery001/review.json'))
   assert len(sources)==691 and old['fresh']==q.rules.fresh_specs(old['seeds']) and old['design']==q.rules.QUALITY_DESIGN
   assert c.environment()==old['environment'] and c.bind_data()==old['data']
   assert not (B/'quality001-final-review.json').exists() and not (B/'quality002-final-review.json').exists()
   original_config=Path(checked['request']['files']['config']['path'])
   allowed={str(original_config),str(c.ROOT/'research_runs/tokenizer-gradient-clip-publication-recovery001/evidence/quality/00/config.json'),str(c.ROOT/'research_runs/tokenizer-gradient-clip-recovery-integration002/evidence/quality/00/config.json')}
   used=[]
   for path in (c.ROOT/'research_runs').rglob('config.json'):
    cfg=json.loads(path.read_text());seed=cfg.get('training',{}).get('seed')
    if seed in old['seeds']['fresh']:
     assert str(path) in allowed and cfg==checked['request']['config'],'Unexpected use of reserved scientific seed'
     used.append(reference(path))
   assert len(used)==3
   q.exclusive(B/'quality002-seed-use.json',dict(original_census=old['prerequisites']['seeds'],same_seeds=old['seeds'],original_trained_slot=0,read_only_config_copies=used,remaining_fresh_seeds_unused=old['seeds']['fresh'][1:],bootstrap_unused=True,zero_repeat_updates=True,no_replacement_seeds=True))
   retained=proof['actual_pair_bytes'];free=shutil.disk_usage(c.ROOT).free;required=q.FULL_FREE-retained;assert free>=required
   active={k:v for k,v in budgets().items() if not k.startswith(('evidence/pilots/','evidence/quality/','evidence/.retention/')) or any(k.startswith(f'evidence/quality/{i:02d}/') for i in range(6))}
   # Original stopped metadata, both external evidence namespaces and the new active groups fit the unchanged256MiB allowance.
   assert sum(active.values())+16*1024**2+3*32*1024**2<=256*1024**2
   cap=dict(required_full_panel_bytes=q.FULL_FREE,already_retained_import_bytes=retained,required_remaining_bytes=required,free_bytes=free,reserve_bytes=8*1024**3,round_metadata_cap_bytes=256*1024**2,active_groups=active,original_stopped_metadata_allowance=16*1024**2,external_evidence_allowance=3*32*1024**2,weight_copies=0,guard_caps_changed=False)
   q.exclusive(B/'quality002-capacity.json',cap)
   plan=copy.deepcopy(old);plan.update(kind=q.KIND,root=str(q.ROOT),observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources,added_sources=sorted(set(sources)-set(read(old['pilot_plan'])['sources'])),observed_free_bytes=free,capacity_accounting=cap,builder=reference(__file__),final_sources={name:reference(B/name) for name in ['quality002-fresh-inference001.py','quality002-final-review001.py']},publication_recovery=dict(original_freeze=original['freeze'],original_STOP=original['stop'],proof=reference(B/'publication-recovery001/review.json'),integration=reference(B/'recovery-integration002-review.json'),import_slot=0,repeat_training_updates=0,remaining_slots=[1,2,3,4,5],scientific_design_unchanged=True,original_campaign_remains_closed=True))
   plan['prerequisites'].update(publication_recovery=reference(B/'publication-recovery001/review.json'),continuation_integration=reference(B/'recovery-integration002-review.json'),continued_seed_use=reference(B/'quality002-seed-use.json'),continued_capacity=reference(B/'quality002-capacity.json'))
   q.ROOT.mkdir()
   with session_lock(q.ROOT) as lock:
    writer=MetadataWriter(q.ROOT,(g,lock));path=writer.publish_json('evidence/meta/quality-plan.json',plan)
    with (B/'quality002-plan.json').open('xb') as f:f.write(path.read_bytes())
    q.exclusive(q.FREEZE,dict(kind=q.KIND,plan=reference(path),portable=reference(B/'quality002-plan.json'),builder=reference(__file__)))
    freeze=reference(q.FREEZE);assert q.bind(writer,freeze['sha256'])==plan
  with session_lock(q.ROOT) as lock:
   writer=MetadataWriter(q.ROOT,(g,lock));guard=B/'quality002-import00-guard.json'
   with q.QualityGuard(guard,c.ROOT,600):
    assert q.bind(writer,freeze['sha256'])==plan;root_unchanged()
    request=checked['request'];assert request['sources']==sources and request['trial']==dict(spec=plan['fresh'][0],seeds=plan['seeds'])
    accepted=a.accept(request,checked['validation']);cached=q.cache_audit(request);assert cached==checked['cache'] and accepted['audit']==checked['artifact_audit']
    receipt=dict(kind=q.KIND,slot=0,freeze=freeze,request=request,artifact_acceptance=accepted,target_contract=checked['validation']['target'],cache=cached,process=reference(B/'publication-recovery001/review.json'),selection=None,cleanup=None,adopted=False,publication_recovery=plan['publication_recovery'],training_updates_this_import=0,original_failure_preserved=True)
    path=writer.publish_json('evidence/quality/00/resource-events.jsonl',receipt)
    root_unchanged();c.verify_sources(sources)
   files={r['path']:r for r in original['failed_root_files'].values()}
   for r in [reference(path),reference(guard),reference(B/'publication-recovery001/review.json'),reference(B/'recovery-integration002-review.json')]:files[r['path']]=r
   q.exclusive(q.anchor_path(0),dict(kind=q.KIND,slot=0,previous=freeze,acceptance=reference(path),parent_guard=reference(guard),files=list(files.values()),import_only=True,repeat_training_updates=0,original_failed_anchor_absent=True))
   state=q.state(plan,freeze['sha256']);assert len(state['records'])==1 and state['used_weights']==retained and q.next_spec(plan,state)==plan['fresh'][1]
   q.exclusive(B/'quality002-preflight.json',dict(status='frozen_same_panel_with_zero_update_import',freeze=freeze,accepted_prefix=1,next_spec=q.next_spec(plan,state),imported_weights=retained,free_bytes=shutil.disk_usage(c.ROOT).free,required_remaining=q.FULL_FREE-retained,source_count=len(sources),original_campaign_closed=True,training_updates=0))
  print(json.dumps(dict(status='frozen_same_panel_with_zero_update_import',freeze=freeze,next_slot=1,free_bytes=shutil.disk_usage(c.ROOT).free)))
 except BaseException as exc:
  if q.ROOT.exists() and not (q.ROOT/'STOP').exists():q.exclusive(q.ROOT/'STOP',dict(error=repr(exc),no_retry=True,phase='freeze_import'))
  raise
