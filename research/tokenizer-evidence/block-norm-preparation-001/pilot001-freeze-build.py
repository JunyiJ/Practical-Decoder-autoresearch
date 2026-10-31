"""Freeze exactly two actual-corpus20step pilots, before any pilot training."""
import json,shutil,subprocess
from pathlib import Path
from datetime import datetime,timezone
import torch
from research.block_norm_v1 import pilot_execution as p,campaign_contract as c,campaign_rules as rules
from research.block_norm_v1.fixture_dispatch import reference,read
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
b=p.EVIDENCE;torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent) as global_fd:
 idle()
 with CpuPrerequisiteGuard(b/'pilot001-freeze-guard.json',c.ROOT,timeout_seconds=120,poll_seconds=.25):
  assert not p.ROOT.exists() and not p.FREEZE.exists()
  prereq={name:reference(b/path) for name,path in {
   'artifact_regression':'pilot-acceptance-001-review.json','inference':'inference-measurement-001-review.json',
   'transport':'fixture-dispatch-002-review.json','seed_census':'scientific-seed-census-001/result.json',
   'projection':'shape-storage-review.json'}.items()}
  for name,count in [('artifact_regression',64),('inference',21),('transport',22)]:
   value=read(prereq[name]);assert value['status']=='passed' and len(value['tests'])==count and value['frozen_sources_and_closed_campaigns_unchanged']
  census=read(prereq['seed_census']);assert census['status']=='fresh_seeds_reserved' and not census['matches'];seeds=census['seeds']
  sources=c.source_inventory();c.frozen_predecessors(sources)
  regression=json.loads((b/'pilot-acceptance-plan-001.json').read_text());assert regression['requests']['rms_incumbent']['sources']==sources
  assert all(c.file_sha(c.ROOT/path)==sha for path,sha in regression['current_runtime_sha256'].items())
  free=shutil.disk_usage(c.ROOT).free;assert free>=p.FULL_FREE
  specs=rules.pilot_specs(seeds);configs=[c.trial_config(p.ROOT/f'evidence/pilots/{slot:02d}',spec,seeds) for slot,spec in enumerate(specs)]
  plan=dict(kind=p.KIND,status='frozen_pilots_only',root=str(p.ROOT),observed_at=datetime.now(timezone.utc).isoformat(),
   source_commit_before_freeze=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources=sources,environment=c.environment(),data=c.bind_data(),seeds=seeds,specs=specs,configs=configs,prerequisites=prereq,
   builder_sha256=c.file_sha(Path(__file__).absolute()),
   limits=dict(cpu_threads=4,seconds=600,rss_bytes=4*1024**3,driver_bytes=6*1024**3,reserve_bytes=8*1024**3,full_required_free_bytes=p.FULL_FREE,checkpoint_allowances=list(p.ALLOWANCES)),
   capacity_observed_free_bytes=free,quality_plan_frozen=False,quality_dispatch_permitted=False,cleanup_permitted=False,
   acceptance='Exactly20updates/max_iters; raw/H200 lineage and all learned EMA tensors; optimizer/RNG resume preflight; exact scheduled metrics0/19; parent independent fullvalidation; raw/H200 batch1/8 full96/cache32+64/reset exact replay; clean guarded child exit and external acceptance anchor.',
   failure_policy='Any existing request, partial output, failed guard, missing completion or missing external acceptance anchor blocks progress. Preserve original failed sources/artifacts; never retry a pilot slot.',
   retention='Keep both pilot raw/H200 pairs until complete pilot audit and dependent quality prerequisites; no deletion authority in this freeze.',
   inference_scope='Only cache correctness/storage required for pilots; descriptive paired timing prerequisite already passed on retained v2 fixtures. Fresh trained-pair timings remain a later requirement.',
   seed_census_note='The new test uses reserved seed numbers only for arithmetic config construction, without training. New configs are prospective uses of this reservation; no prior recorded training used these values.')
  p.ROOT.mkdir()
  with session_lock(p.ROOT) as campaign_fd:
   writer=MetadataWriter(p.ROOT,(global_fd,campaign_fd));path=writer.publish_json('evidence/meta/pilot-plan.json',plan)
  raw=path.read_bytes()
  with (b/'pilot001-plan.json').open('xb') as f:f.write(raw)
  p.exclusive(p.FREEZE,dict(kind=p.KIND,plan=reference(path),portable_plan=reference(b/'pilot001-plan.json'),builder=reference(Path(__file__).absolute()),quality_plan_frozen=False,cleanup_permitted=False))
  assert p.bind(MetadataWriter(p.ROOT,(global_fd,os_fd)),reference(p.FREEZE)['sha256']) if False else True
 print(json.dumps(dict(status='frozen_two_pilots_only',freeze=reference(p.FREEZE),free_bytes=free,seeds=seeds)))
