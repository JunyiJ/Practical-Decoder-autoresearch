"""Keep original reviewed directories in the survivor audit after archival.

The planning driver's live discovery uses remaining loose .pt files. Snapshot
its review directory set before deletion so removed weights cannot make their
non-checkpoint evidence disappear from the final inventory. No cleanup has run
through the dynamic-discovery apply path.
"""
import importlib.util,json
from pathlib import Path
B=Path(__file__).resolve().parent
source=B/'archive-historical-numerical001.py'
spec=importlib.util.spec_from_file_location('historical_archive',source)
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
from research.ffn_norm_campaign_v1.pilot import idle
with d.a.session_lock(d.ROOT/'research_state/tokenizer-worker-lock'):
 idle()
 with d.a.CpuPrerequisiteGuard(d.E/'apply-guard.json',d.ROOT,timeout_seconds=1800):
  plan=json.loads((d.E/'plan.json').read_text())
  reviews=tuple(d.a.verify(r) for r in plan['eligibility'][1:])
  assert reviews==tuple(d.review_paths())
  before=d.paths()
  d.review_paths=lambda:list(reviews)
  assert before==d.paths()
  d.a.publish(d.E/'apply-binding.json',dict(source=d.a.ref(Path(__file__)),planning_source=d.a.ref(source),plan=d.a.ref(d.E/'plan.json'),fixed_review_directories=[str(p.parent) for p in reviews],
   reason='Preserve original reviewed directory discovery after loose checkpoint removal; planning inventory is unchanged. No failed cleanup attempt or scientific dispatch.'))
  d.apply()
