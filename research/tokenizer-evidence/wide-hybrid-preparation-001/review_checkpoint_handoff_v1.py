"""Archive handoff fixture and verify closed002 evidence remains unchanged."""
from pathlib import Path
import sys,json,base64
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_driver import session_lock
from src.research.protocol import file_digest
E=Path(__file__).resolve().parent;R=ROOT/'research_runs/tokenizer-metadata-contention-handoff-001'
with session_lock(ROOT/'research_state/tokenizer-worker-lock'),CpuPrerequisiteGuard(E/'checkpoint-handoff-v1-review-guard.json',ROOT,timeout_seconds=600):
 result=json.loads((R/'result.json').read_text());assert result['status']=='completed'
 for n in ('parent-guard.json','child-guard.json'):assert json.loads((R/n).read_text())['status']=='finished'
 files={str(p.relative_to(R)):dict(size=p.stat().st_size,sha256=file_digest(p)) for p in R.rglob('*') if p.is_file()}
 assert sum(v['size'] for v in files.values())<64*1024**2
 meta={n:base64.b64encode((R/n).read_bytes()).decode() for n in files if not n.endswith('.pt')}
 failed=json.loads((E/'campaign002-lock-failure-v2-review.json').read_text());closed=ROOT/'research_runs/tokenizer-wide-hybrid-002'
 assert all((closed/n).stat().st_size==v['size'] and file_digest(closed/n)==v['sha256'] for n,v in failed['files'].items())
 sources=json.loads((E/'checkpoint-handoff-v1-test-source.json').read_text());assert all(file_digest(ROOT/n)==h for n,h in sources.items())
 guard=json.loads((E/'checkpoint-handoff-v1-guard.json').read_text());assert guard['status']=='finished'
 testlog=(E/'checkpoint-handoff-v1-tests.log').read_text();assert '134 passed' in testlog
 plan=json.loads((E/'metadata-contention-handoff-v1-plan.json').read_text());oldplan=json.loads((E/'metadata-contention-baseline-v1-plan.json').read_text())
 changes=[n for n in plan['sources'] if plan['sources'][n]!=oldplan['sources'][n]]
 assert changes==['src/research/tokenizer_wide_hybrid_checkpoint_writer.py']
 report=dict(status='handoff_mitigation_verified_scoped_no_training_authority',tests_passed=134,test_source_hashes=sources,
  test_log=testlog,test_guard=guard,fixture_plan_sha256=file_digest(E/'metadata-contention-handoff-v1-plan.json'),
  files=files,nonweight_bytes_base64=meta,parent_trace=json.loads((R/'parent-trace.json').read_text()),child_trace=json.loads((R/'child-trace.json').read_text()),
  baseline_review_sha256=file_digest(E/'metadata-contention-baseline-v1-review.json'),
  production_files_changed=changes,campaign002_all_files_unchanged=True,campaign002_file_hashes=failed['files'],
  mechanism='After each accumulated MiB, yield20ms outside the namespace lock. Lock timeout remains2s; no lock/inode/cap/resource check removed. Synthetic byte hashes and torch serialization tests pass.',
  review_preflight_note='Initial system-python review command failed importing omegaconf before entering any guard or changing files; review then uses repository venv. Fixtures were not rerun.',
  limitations=['Original failure was not reproduced by baseline; tested scheduling mitigation is not proof of eliminated failure.',
  'One baseline and one changed-source56MiB fixture with800metadata files only; original failed campaign had250files and761MiB partial.',
  'No new numerical/training/quality/inference result or general timing claim. Larger checkpoints incur intentional scheduling overhead.',
  'Completed002 prefix remains evidence under original source; new applicability review and finite plan precede training.'],
  next_action='Review applicability of scheduling-only change to preserved numerical/input evidence; audit storage/metadata including protected002 weights and partial; freeze fresh contract before training. Never reopen002 or retry its slot5.')
 with (E/'checkpoint-handoff-v1-review.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps({k:report[k] for k in ('status','tests_passed','production_files_changed','campaign002_all_files_unchanged')}))
