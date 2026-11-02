"""Freeze exactly three fresh pairs at common LR0.00125, without LR screening."""
import json,shutil,subprocess
from pathlib import Path
from datetime import datetime,timezone
import torch
from research.readout_centering_quality_v1 import quality_execution as q,campaign_contract as c,campaign_rules as rules
from research.readout_centering_quality_v1.pilot_retirement import verify_retirement,DEST
from research.readout_centering_worker_v1 import inference
from research.block_norm_v1.fixture_dispatch import reference,read
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_metadata import budgets
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
B=q.EVIDENCE;torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent) as g:
 q.idle()
 with CpuPrerequisiteGuard(B/'quality001-freeze-guard.json',c.ROOT,timeout_seconds=120):
  assert not q.ROOT.exists() and not q.FREEZE.exists()
  integration=read(reference(B/'quality-integration001-review.json'));assert integration['status']=='passed' and integration['tests'] and all(t['outcome']=='passed' for t in integration['tests']) and integration['protected_pilot_roots_unchanged']
  checked=read(integration['plan']);sources=c.source_inventory();assert sources==checked['sources'];prior=read(reference(B/'pilot001-plan.json'));seeds=prior['seeds'];c.frozen_predecessors(sources)
  reserved=set(seeds['fresh']);examined=[]
  for path in (c.ROOT/'research_runs').rglob('config.json'):
   cfg=json.loads(path.read_text());used=cfg.get('training',{}).get('seed');assert used not in reserved,'Reserved fresh seed already used: '+str(path)
   examined.append(reference(path))
  q.exclusive(B/'quality001-seed-use.json',dict(status='reserved_fresh_seeds_unused',reserved=sorted(reserved),original_census=reference(B/'scientific-seed-census-001/result.json'),config_files=examined,bootstrap_seed=seeds['bootstrap'],scope='Original census plus all research_runs/config.json training seeds; no scientific quality/bootstrap invocation. Synthetic statistical tests use separate non-scientific seeds.'))
  retired=reference(DEST/'completed.json');verify_retirement(retired)
  active={k:v for k,v in budgets().items() if not k.startswith(('evidence/pilots/','evidence/quality/','evidence/.retention/')) or any(k.startswith(f'evidence/quality/{i:02d}/') for i in range(6))}
  assert sum(active.values())+32*1024**2<=256*1024**2
  free=shutil.disk_usage(c.ROOT).free;assert free>=q.FULL_FREE,'Complete retained-output requirement does not fit'
  capacity=dict(required_free_bytes=q.FULL_FREE,free_bytes=free,margin_bytes=free-q.FULL_FREE,raw_cap_bytes=448*1024**2,h200_cap_bytes=112*1024**2,weight_cap_bytes=560*1024**2,
   trajectory_output_cap_bytes=576*1024**2,round_metadata_cap_bytes=256*1024**2,external_metadata_cap_bytes=32*1024**2,active_campaign_group_cap_bytes=sum(active.values()),active_campaign_groups=active,reserve_bytes=8*1024**3,
   retained_peak_quality_pairs=6,pilot_retirement=retired,serialization='Raw plus H200 including checkpoint temporary publication count within each560MiB aggregate allowance; same-inode publication duplicates names only. Metadata temporary overlap included. No full-payload extra copy.',
   retention='Both successful pilots retired after dependent acceptance/inference completed. All six fresh trajectories remain protected; no LR screens, selection or fresh retirement. Failure stops without retry.')
  q.exclusive(B/'quality001-capacity.json',capacity)
  plan=dict(kind=q.KIND,status='frozen_quality_only',root=str(q.ROOT),observed_at=datetime.now(timezone.utc).isoformat(),source_commit_before_freeze=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
   sources=sources,added_sources=checked['added_sources'],environment=c.environment(),data=c.bind_data(),seeds=seeds,fresh=rules.fresh_specs(seeds),design=rules.QUALITY_DESIGN,limits=dict(c.LIMITS),allowances=q.ALLOWANCES,required_free_bytes=q.FULL_FREE,observed_free_bytes=free,pilot_plan=reference(B/'pilot001-plan.json'),pilot_retirement=retired,
   prerequisites={n:reference(B/p) for n,p in dict(integration='quality-integration001-review.json',pilots='pilot001-final003-review.json',inference='pilot001-review.json',capacity='quality001-capacity.json',seeds='scientific-seed-census-001/result.json',retention='retention-boundary002.json',seed_use='quality001-seed-use.json').items()},
   finite_contract=dict(screens=0,confirmations=6,steps_per_trial=1200,targets_per_trial=1228800,learning_rate=.00125,lr_search_budget_per_arm=0,selection=None,rescue=False,confirmation='Three reserved fresh pairs in alternating arm order; same peak LR. No extension, replacement seed, retry or interim stopping.',statistics=dict(blocks=list(rules.BLOCKS),draws=rules.DRAWS,quantile=rules.GATE_QUANTILE,gate='RMS-with-bias minus LayerNorm: mean<-0.02, each fresh paired delta<0, all simultaneous conditional-text upper bounds<0. Separate final review; no cost-only route or seed-population/holdout inference.',adoption_automatic=False)),
   capacity_accounting=capacity,retention='Completed exact pilot retirement bound. Every fresh raw/H200 pair and historical selected/fresh evidence remains protected. No quality cleanup command.',
   provenance='Actual child argv/request/previous acceptance, clean process reaping, guarded worker output, independent parent optimizer/EMA/export/full-validation/cache re-audit before every anchor. Existing/unaccepted attempts cannot retry. Frozen sources/data/guards.',
   holdout='No test scoring, training test batch or test-based decisions. Inherited training loader materializes test tokens; evaluator/inference use train/validation-only view.',future_inference=dict(contract=inference.CONTRACT,pair_count=3,scope='All three accepted fresh raw/H200 pairs; descriptive timing, no cost-only adoption.'),quality_adopted=False,builder=reference(Path(__file__).resolve()),
   final_sources={name:reference(B/name) for name in ['quality001-fresh-inference001.py','quality001-final-review001.py']})
  q.ROOT.mkdir()
  with session_lock(q.ROOT) as p:
   writer=MetadataWriter(q.ROOT,(g,p));path=writer.publish_json('evidence/meta/quality-plan.json',plan)
   with (B/'quality001-plan.json').open('xb') as f:f.write(path.read_bytes())
   q.exclusive(q.FREEZE,dict(kind=q.KIND,plan=reference(path),portable=reference(B/'quality001-plan.json'),builder=reference(Path(__file__).resolve())))
   h=reference(q.FREEZE)['sha256'];assert q.bind(writer,h)==plan
   current=q.state(plan,h);assert not current['records'] and q.next_spec(plan,current)==rules.fresh_specs(seeds)[0]
   q.exclusive(B/'quality001-preflight.json',dict(status='frozen_quality_entry_bound',freeze=reference(q.FREEZE),accepted_prefix=0,next_spec=q.next_spec(plan,current),sources=len(sources),free_bytes=free,training_launched=False))
 print(json.dumps(dict(status='finite_quality_frozen',freeze=reference(q.FREEZE),free_bytes=free,required_free_bytes=q.FULL_FREE)),flush=True)
