"""Final finite quality freeze after actual pilots, inference and integration."""
import json,shutil,subprocess
from pathlib import Path
from datetime import datetime,timezone
import torch
from research.block_norm_v1 import quality_execution as q,campaign_contract as c,campaign_rules as rules
from research.block_norm_v1.fixture_dispatch import reference,read
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
b=q.EVIDENCE;torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent) as g:
 idle()
 with CpuPrerequisiteGuard(b/'quality001-freeze-guard.json',c.ROOT,timeout_seconds=120,poll_seconds=.25):
  assert not q.ROOT.exists() and not q.FREEZE.exists()
  integration=read(reference(b/'quality-integration-002-review.json'));assert integration['status']=='passed' and len(integration['tests'])==9 and integration['protected_pilot_roots_unchanged']
  checked=read(integration['plan']);sources=c.source_inventory();assert sources==checked['sources']
  inference=read(reference(b/'pilot-pair-inference-001-review.json'));assert inference['status']=='passed' and inference['original_pilot_roots_unchanged']
  prior=read(reference(b/'pilot002-plan.json'));seeds=prior['seeds'];c.frozen_predecessors(sources)
  capacity=read(reference(b/'quality-remaining-capacity-001.json'));assert capacity['required_free_bytes']==q.FULL_FREE and capacity['trajectory_allowances']==q.ALLOWANCES
  free=shutil.disk_usage(c.ROOT).free;assert free>=q.FULL_FREE
  plan=dict(kind=q.KIND,status='frozen_quality_only',root=str(q.ROOT),observed_at=datetime.now(timezone.utc).isoformat(),source_commit_before_freeze=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources=sources,added_sources=checked['added_sources'],environment=c.environment(),data=c.bind_data(),seeds=seeds,screens=rules.screen_specs(seeds),
   limits=dict(c.LIMITS),allowances=q.ALLOWANCES,required_free_bytes=q.FULL_FREE,observed_free_bytes=free,pilot_plan=reference(b/'pilot002-plan.json'),protected_pilot_inventories=[reference(b/'pilot001-failure.json'),reference(b/'pilot002-result.json')],
   prerequisites={k:reference(b/n) for k,n in dict(integration='quality-integration-002-review.json',inference='pilot-pair-inference-001-review.json',rms_recovery='validation-recovery-001-review.json',layer_pilot='pilot002-result.json',capacity='quality-remaining-capacity-001.json',seeds='scientific-seed-census-001/result.json').items()},
   finite_contract=dict(screens=6,confirmations=6,steps_per_trial=1200,targets_per_trial=1228800,learning_rates=list(rules.RATES),selection='Within each arm minimum full-validation H200 CE; lower LR tie-break. Save both choices after all six independently accepted screens.',confirmation='Three fresh paired seeds in frozen alternating arm order using saved per-arm rates. No extension/retry/seed replacement.',statistics=dict(blocks=list(rules.BLOCKS),draws=rules.DRAWS,quantile=rules.GATE_QUANTILE,gate='LayerNorm minus RMS mean below -0.02, every seed below0, each simultaneous conditional text upper bound below0; no cost adoption route.',adoption_automatic=False)),
   capacity_accounting='Both pilot pairs already consume observed free space and remain retained; failed RMS permanently protected. At each quality entry require max(8GiB,FULL_FREE-verified currently retained quality weight bytes). Same allowance margin; no credit for unperformed cleanup. Initial FULL_FREE includes256MiB metadata and8GiBreserve; exact screen deletion mandatory before confirmations.',
   retention='After all6screens independently audited again and LR selection persisted, permit only four losing raw/H200 pairs by exact size/SHA/path manifest with fsynced intent/deletion journal and complete protected evidence verification under shared/campaign locks. Keep two selected screens and all six fresh pairs. Partial cleanup blocks dispatch forever; no retry. No pilot deletion.',
   provenance='Every child request bound to frozen plan and previous external acceptance. Child clean exit, guard, full validation/cache and independent parent recomputation precede external anchor. Unaccepted attempt cannot retry. No old campaign reopened.',
   holdout='No test loss, test training batch, or test-based decision. Unchanged inherited training loader materializes test; read-only evaluators encode train/validation only.',
   future_inference='Measure all three final trained confirmation pairs after quality completion with existing frozen cached contract; descriptive only.',quality_adopted=False,builder=reference(Path(__file__).absolute()))
  q.ROOT.mkdir()
  with session_lock(q.ROOT) as p:
   writer=MetadataWriter(q.ROOT,(g,p));path=writer.publish_json('evidence/meta/quality-plan.json',plan)
   with (b/'quality001-plan.json').open('xb') as f:f.write(path.read_bytes())
   q.exclusive(q.FREEZE,dict(kind=q.KIND,plan=reference(path),portable=reference(b/'quality001-plan.json'),builder=reference(Path(__file__).absolute())))
   freeze_sha=reference(q.FREEZE)['sha256'];assert q.bind(writer,freeze_sha)==plan
   current=q.state(plan,freeze_sha);assert not current['records'] and q.next_spec(plan,current)==rules.screen_specs(seeds)[0]
   q.exclusive(b/'quality001-preflight.json',dict(status='frozen_quality_entry_bound',freeze=reference(q.FREEZE),accepted_prefix=0,next_spec=q.next_spec(plan,current),sources=len(sources),free_bytes=free,training_launched=False))
 print(json.dumps(dict(status='finite_quality_frozen',freeze=reference(q.FREEZE),free_bytes=free)))
