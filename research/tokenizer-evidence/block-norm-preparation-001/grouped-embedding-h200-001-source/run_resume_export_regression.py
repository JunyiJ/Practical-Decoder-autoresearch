"""One guarded CPU-only resume export regression prerequisite with immutable attempt/source evidence."""
import argparse,contextlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
import pytest,torch
from src.research.protocol import ROOT,digest,file_digest
from src.research.runner import source_fingerprint
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock
from src.research.tokenizer_wide_hybrid_metadata import snapshot
parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True);args=parser.parse_args()
assert len(args.attempt)==3 and args.attempt.isdigit()
code=ROOT/'research/block_norm_v1';base=ROOT/'research/tokenizer-evidence/block-norm-preparation-001'
stem='resume-export-regression-'+args.attempt
paths={k:base/(stem+suffix) for k,suffix in [('guard','-guard.json'),('log','.log'),('review','-review.json')]}
saved=base/(stem+'-source');assert not saved.exists() and not any(p.exists() for p in paths.values())
saved.mkdir()
for path in sorted(code.rglob('*.py')):
    dest=saved/path.relative_to(code);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
pins={str(p.relative_to(ROOT)):file_digest(p) for p in code.rglob('*.py')}
class Results:
    def __init__(self):self.tests=[]
    def pytest_runtest_logreport(self,report):
        if report.when=='call' or report.failed:self.tests.append(dict(nodeid=report.nodeid,when=report.when,outcome=report.outcome,numerical_observations=dict(report.user_properties)))
results=Results();error=None;exit_code=1;unchanged=False
manifest_path=base/'resume-replay-delta-001.json'
manifest=json.loads(manifest_path.read_text());manifest_sha=file_digest(manifest_path)
map_sha=file_digest(base/'runtime-dependency-map.json')
try:
    with session_lock(GLOBAL_LOCK.parent):
        idle()
        with CpuPrerequisiteGuard(paths['guard'],ROOT,timeout_seconds=120):
            torch.set_num_threads(4);original=source_fingerprint()
            assert file_digest(base/'runtime-baseline-copy-manifest.json')==manifest['baseline_manifest_sha256']
            assert all(file_digest(ROOT/name)==sha for name,sha in manifest['current_runtime_sha256'].items())
            roots=[ROOT/'research_runs'/name for name in ('tokenizer-activation-001','tokenizer-activation-inference-recovery-001','tokenizer-activation-inference-recovery-002')]
            before={str(p):snapshot(p) for p in roots}
            metadata={str(root/name):file_digest(root/name) for root in roots for name,entry in before[str(root)]['files'].items() if not entry['group'].endswith('/weights')}
            plan=json.loads((roots[-1]/'evidence/meta/recovery-plan.json').read_text())
            assert digest(original)==plan['original_source_sha256']
            assert all(file_digest(ROOT/name)==sha for name,sha in plan['external_source'].items())
            with paths['log'].open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                exit_code=pytest.main(['-q','-p','no:cacheprovider',str(code/'test_resume_export_regression.py')],plugins=[results])
            assert source_fingerprint()==original
            assert all(file_digest(ROOT/name)==sha for name,sha in plan['external_source'].items())
            assert all(snapshot(Path(path))==seen for path,seen in before.items())
            assert all(file_digest(Path(path))==sha for path,sha in metadata.items())
            assert all(file_digest(ROOT/path)==sha for path,sha in pins.items())
            assert file_digest(manifest_path)==manifest_sha and file_digest(base/'runtime-dependency-map.json')==map_sha
            idle();unchanged=True
except Exception as exc:error=repr(exc)
value=dict(status='passed' if error is None and exit_code==0 else 'failed',error=error,pytest_exit_code=int(exit_code),tests=results.tests,
    observed_at=datetime.now(timezone.utc).isoformat(),code_sha256=pins,
    source_snapshot={str(p.relative_to(ROOT)):file_digest(p) for p in saved.rglob('*.py')},
    baseline_manifest_sha256=manifest_sha,dependency_map_sha256=map_sha,
    guard_sha256=file_digest(paths['guard']) if paths['guard'].exists() else None,
    log_sha256=file_digest(paths['log']) if paths['log'].exists() else None,
    frozen_sources_and_closed_campaigns_unchanged=unchanged,
    scope='Read-only exact H200 rederivation from retained synthetic transport fixtures, rejection of their incomplete resume state, and legacy low-level resume behavior. Fixture hashes verified before/after; no new binary files or updates.',
    limitations='Read-only synthetic regression only, not tokenizer-bound trainer integration, gradient oracle, candidate cache numerical/MPS or quality evidence. Full-panel capacity gate remains unsatisfied.',
    training_launched=False,checkpoints_loaded=True,trained_checkpoints_loaded=False,holdout_scored=False,deletion_performed=False)
with paths['review'].open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps(dict(status=value['status'],error=error,tests=len(results.tests),failures=[t['nodeid'] for t in results.tests if t['outcome']=='failed'])))
raise SystemExit(0 if value['status']=='passed' else 1)
