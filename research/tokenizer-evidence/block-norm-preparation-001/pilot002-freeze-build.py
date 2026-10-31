"""Freeze only the unstarted LN pilot, carrying independently audited RMS outputs."""
import copy,json,shutil,subprocess
from pathlib import Path
from datetime import datetime,timezone
import torch
from research.block_norm_v1 import pilot_continuation as p,campaign_contract as c,campaign_rules as rules
from research.block_norm_v1.fixture_dispatch import reference,read
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
b=p.EVIDENCE;torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent) as global_fd:
 idle()
 with CpuPrerequisiteGuard(b/'pilot002-freeze-guard.json',c.ROOT,timeout_seconds=120,poll_seconds=.25):
  assert not p.ROOT.exists() and not p.FREEZE.exists()
  prior=read(reference(b/'pilot001-plan.json'));sources=c.source_inventory();c.frozen_predecessors(sources)
  carried={k:reference(b/n) for k,n in dict(recovery='validation-recovery-001-rms-result.json',review='validation-recovery-001-review.json',guard='validation-recovery-001-guard.json',failure='pilot001-failure.json',recovery_plan='validation-recovery-plan-001.json').items()}
  seeds=prior['seeds'];specs=rules.pilot_specs(seeds);configs=[c.trial_config(p.ROOT/f'evidence/pilots/{slot:02d}',spec,seeds) for slot,spec in enumerate(specs)]
  free=shutil.disk_usage(c.ROOT).free;assert free>=p.FULL_FREE
  plan=dict(kind=p.KIND,status='frozen_pilots_only',root=str(p.ROOT),observed_at=datetime.now(timezone.utc).isoformat(),
   source_commit_before_freeze=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources=sources,environment=c.environment(),data=c.bind_data(),seeds=seeds,specs=specs,configs=configs,
   prerequisites={**prior['prerequisites'],'recovery_decision':reference(b/'validation-recovery-decision.json')},carried_rms=carried,
   builder_sha256=c.file_sha(Path(__file__).absolute()),limits=prior['limits'],allowed_slots=[1],
   capacity_observed_free_bytes=free,capacity_policy='Conservative unchanged full panel requirement15170221264B includes reserve8GiB; original failed RMS weights already occupy disk and are never credited as reclaimable. No cleanup assumed; remaining projected need bounded above by original full projection.',
   quality_plan_frozen=False,quality_dispatch_permitted=False,cleanup_permitted=False,
   acceptance=prior['acceptance'],failure_policy=prior['failure_policy'],retention='All failed RMS weights permanently protected. Keep LN pair until complete pilot audit and dependent prerequisites. No deletion in this freeze.',
   inference_scope=prior['inference_scope'],seed_census_note='Pilot374001 already used for carried RMS; same seed intentionally pairs LN. No quality screen/fresh training has occurred.',
   source_applicability='All391 recovery sources unchanged; additive pilot_continuation.py only. Original pilot_execution.py not invoked or mutated. Recovery is component acceptance, not reclassification of original failed worker.')
  p.verify_carried(plan)
  checks=[]
  for slot in (0,2,-1):
   try:p.run_once(None,slot,'0'*64)
   except ValueError:checks.append('reject_slot_'+str(slot))
   else:raise AssertionError('Disallowed slot accepted')
  for kind in ('seed','source','recovery_hash'):
   bad=copy.deepcopy(plan)
   if kind=='seed':bad['seeds']['pilot']=-1
   elif kind=='source':bad['sources']['research/block_norm_v1/validation.py']='0'*64
   else:bad['carried_rms']['recovery']['sha256']='0'*64
   try:p.verify_carried(bad)
   except ValueError:checks.append('reject_'+kind)
   else:raise AssertionError('Corrupt carried context accepted '+kind)
  p.ROOT.mkdir()
  with session_lock(p.ROOT) as campaign_fd:
   writer=MetadataWriter(p.ROOT,(global_fd,campaign_fd));path=writer.publish_json('evidence/meta/pilot-plan.json',plan)
   with (b/'pilot002-plan.json').open('xb') as f:f.write(path.read_bytes())
   p.exclusive(p.FREEZE,dict(kind=p.KIND,plan=reference(path),portable_plan=reference(b/'pilot002-plan.json'),builder=reference(Path(__file__).absolute()),quality_plan_frozen=False,cleanup_permitted=False))
   bound=p.bind(writer,reference(p.FREEZE)['sha256']);assert bound==plan
   p.exclusive(b/'pilot002-preflight.json',dict(status='actual_frozen_continuation_bound',sources=len(sources),allowed_slots=[1],negative_checks=checks,carried_original_process_reclassified=False,training_launched=False,freeze=reference(p.FREEZE)))
  with (b/'pilot002-source.py').open('xb') as f:f.write((c.ROOT/'research/block_norm_v1/pilot_continuation.py').read_bytes())
 print(json.dumps(dict(status='frozen_layernorm_only',freeze=reference(p.FREEZE),free_bytes=free,checks=checks)))
