"""Independent retirement verification after a terminal JSON-parser error.

No cleanup retry, restoration, training or weight copy. Preserve the closed
original receipt/exception and verify the incomplete failure file by its hash.
"""
import importlib.util,json,os,subprocess
from pathlib import Path
import torch
from research.final_ln_scale_quality_v2 import campaign_contract as c,quality_execution as q,artifact_acceptance as aa
from research.final_ln_scale_quality_v2.pilot_retirement import verify_payload_retirement,verify_retirement,DEST
from research.final_ln_scale_quality_v1 import quality_execution as oldq,artifact_acceptance as olda
from research.final_ln_scale_quality_v1 import campaign_rules as rules
from research.final_ln_scale_quality_v2.support import idle
from research.final_ln_scale_campaign_v2 import pilot
from research.block_norm_v1.fixture_dispatch import read,reference
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
s=c.ROOT/'research/tokenizer-evidence/mla-score-preparation-001/archive-workers.py'
spec=importlib.util.spec_from_file_location('hash_primitives',s);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent),session_lock(pilot.OLDROOT),session_lock(pilot.ROOT):
 idle();sources=c.source_inventory();assert len(sources)==912
 q.exclusive(DEST/'verification002-plan.json',dict(sources=sources,runner=reference(Path(__file__).resolve()),original_receipt=reference(DEST/'completed.json'),original_failed_guard=reference(DEST/'apply-guard.json'),scope='Independent terminal-evidence verification only; no deletion/training/restoration/retry. Original906 numerical contracts and all888 runtime sources unchanged.'))
 with CpuPrerequisiteGuard(DEST/'verification002-guard.json',c.ROOT,timeout_seconds=900):
  original=read(reference(DEST/'apply-guard.json'));assert original['status']=='fixture_exception' and 'JSONDecodeError' in original['reason']
  receipt=reference(DEST/'completed.json');done=verify_payload_retirement(receipt);manifest=read(done['manifest']);partial=pilot.OLDROOT/'evidence/pilots/00/accepted.json.tmp'
  # Confirm the preserved file really is a truncated original; do not repair it.
  partial_before=reference(partial)
  try:json.loads(partial.read_text())
  except json.JSONDecodeError:pass
  else:raise AssertionError('Expected immutable incomplete failure evidence')
  for evidence in manifest['eligibility']:assert a.ref(Path(evidence['path']))==evidence
  survivors=[row for shard in manifest['inventory_shards'] for row in a.bound(shard)]
  for row in survivors:assert a.record(Path(row['path']))==row
  assert len(survivors)==done['retained_files_verified']==46372
  # The copied acceptance implementation is byte-identical; only its relative
  # source-inventory provider changes to the additive912 bindings.
  assert Path(aa.__file__).read_bytes()==Path(olda.__file__).read_bytes()
  seeds=read(reference(pilot.B/'pilot002-plan.json'))['seeds'];configs=[]
  for spec in rules.fresh_specs(seeds):
   output=q.ROOT/f"evidence/quality/{spec['slot']:02d}";cfg=c.trial_config(output,spec,seeds)
   assert cfg==oldq.c.trial_config(output,spec,seeds);configs.append(cfg)
  assert q.FULL_FREE==oldq.FULL_FREE==12381585408 and q.ALLOWANCES==oldq.ALLOWANCES
  assert q.ROOT==oldq.ROOT and not q.ROOT.exists() and not q.FREEZE.exists()
  for bad in (0,7201):
   try:q.QualityGuard(Path('/tmp/unexecuted-scale-v2-guard'),c.ROOT,bad)
   except ValueError:pass
   else:raise AssertionError('Relaxed deadline accepted')
  assert reference(partial)==partial_before;c.verify_sources(sources)
  value=dict(status='passed',sources=sources,completed_retirement=receipt,original_failed_guard=reference(DEST/'apply-guard.json'),runner=reference(Path(__file__).resolve()),retained_files_verified=len(survivors),original_partial_failure_bytes_unchanged=True,partial_failure=partial_before,all_four_tensors_absent=all(not Path(row['path']).exists() for row in manifest['delete_files']),source_count=912,predecessors_preserved=906,acceptance_implementation_byte_identical=True,six_trial_configs_identical=True,resource_guards_unchanged=True,quality_training_launched=False,training_updates=0,checkpoint_copies=0,deletion_performed=False,original_cleanup_closed=True)
 q.exclusive(DEST/'verification002.json',value);verify_retirement(receipt)
 print(json.dumps(dict(status='passed',retained_files_verified=len(survivors),sources=912,original_terminal_exception_preserved=True)),flush=True)
