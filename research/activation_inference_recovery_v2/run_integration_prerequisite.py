"""Guarded disposable integration checks; retain each attempt and source snapshot."""
import argparse,contextlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
import pytest
from src.research.protocol import ROOT,digest,file_digest
from src.research.runner import source_fingerprint
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock
from src.research.tokenizer_wide_hybrid_metadata import snapshot
from research.activation_inference_recovery_v1.source_binding import SOURCE_SHA256,ORIGINAL_ROOT
from research.activation_inference_recovery_v1.context import RECOVERY_ROOT,external_source

parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('execution','final'),required=True);parser.add_argument('--attempt',required=True)
args=parser.parse_args();assert len(args.attempt)==3 and args.attempt.isdigit()
base=ROOT/'research/tokenizer-evidence/activation-preparation-001';code=ROOT/'research/activation_inference_recovery_v2'
stem=f'inference-recovery-v2-{args.stage}-prerequisite-{args.attempt}'
paths={k:base/(stem+suffix) for k,suffix in [('guard','-guard.json'),('log','.log'),('review','-review.json')]}
saved=base/(stem+'-source')
assert not saved.exists() and not any(p.exists() for p in paths.values()),'No repeat/overwrite'
saved.mkdir()
for path in sorted(code.glob('*.py')):shutil.copy2(path,saved/path.name)
pins={str(path.relative_to(ROOT)):file_digest(path) for path in sorted(code.glob('*.py'))}
class Results:
    def __init__(self):self.tests=[]
    def pytest_runtest_logreport(self,report):
        if report.when=='call' or report.failed:self.tests.append(dict(nodeid=report.nodeid,when=report.when,outcome=report.outcome))
results=Results();error=None;exit_code=1;unchanged=False
try:
    with session_lock(GLOBAL_LOCK.parent):
        idle()
        with CpuPrerequisiteGuard(paths['guard'],ROOT,timeout_seconds=180):
            assert digest(source_fingerprint())==SOURCE_SHA256;v1=external_source()
            before={str(root):snapshot(root) for root in (ORIGINAL_ROOT,RECOVERY_ROOT)}
            metadata={str(root/name):file_digest(root/name) for root in (ORIGINAL_ROOT,RECOVERY_ROOT)
                      for name,entry in before[str(root)]['files'].items() if not entry['group'].endswith('/weights')}
            with paths['log'].open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                exit_code=pytest.main(['-q','-p','no:cacheprovider',str(code/('test_'+args.stage+'.py'))],plugins=[results])
            assert all(snapshot(Path(root))==value for root,value in before.items())
            assert all(file_digest(Path(path))==sha for path,sha in metadata.items())
            assert external_source()==v1 and digest(source_fingerprint())==SOURCE_SHA256
            assert all(file_digest(ROOT/path)==sha for path,sha in pins.items())
            idle();unchanged=True
except Exception as exc:error=repr(exc)
value=dict(status='passed' if error is None and exit_code==0 else 'failed',error=error,pytest_exit_code=int(exit_code),tests=results.tests,
    observed_at=datetime.now(timezone.utc).isoformat(),code_sha256=pins,
    source_snapshot={str(path.relative_to(ROOT)):file_digest(path) for path in sorted(saved.glob('*.py'))},
    guard_sha256=file_digest(paths['guard']) if paths['guard'].exists() else None,
    log_sha256=file_digest(paths['log']) if paths['log'].exists() else None,
    original_and_v1_campaigns_unchanged=unchanged,
    scope='Disposable serial entry/acceptance/final tests with explicit GPU/model/source-import/process doubles. Execution also uses four real sequential bounded pipe children through production later-child entry, prior-prefix and review_outputs with real monitor updates during a doubled source audit; no model production in those children.',
    limitations='No launch freeze or actual inference. Source applicability/capacity and exact finite plan still required. Frozen previous attempts remain immutable.',
    training_launched=False,inference_launched=False,holdout_scored=False,dispatch_authorized=False)
with paths['review'].open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps(dict(status=value['status'],error=error,tests=len(results.tests))))
raise SystemExit(0 if value['status']=='passed' else 1)
