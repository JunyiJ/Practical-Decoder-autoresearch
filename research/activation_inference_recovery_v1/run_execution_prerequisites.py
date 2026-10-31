"""One guarded CPU run of external execution/final tests, no research inference."""
import contextlib
from datetime import datetime,timezone
import json
import pytest
from src.research.protocol import ROOT,digest,file_digest
from src.research.runner import source_fingerprint
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock
from src.research.tokenizer_wide_hybrid_metadata import snapshot
from research.activation_inference_recovery_v1.source_binding import ORIGINAL_ROOT,SOURCE_SHA256
base=ROOT/'research/tokenizer-evidence/activation-preparation-001';code=ROOT/'research/activation_inference_recovery_v1'
stem='inference-recovery-execution-prerequisite-002'
paths=dict(guard=base/(stem+'-guard.json'),log=base/(stem+'.log'),
 execution=base/(stem+'-review.json'),final=base/'inference-recovery-final-prerequisite-002-review.json')
for path in paths.values():
    if path.exists():raise FileExistsError('Existing prerequisite evidence; no overwrite')
class Results:
    def __init__(self):self.tests=[]
    def pytest_runtest_logreport(self,report):
        if report.when=='call' or report.failed:self.tests.append(dict(nodeid=report.nodeid,when=report.when,outcome=report.outcome))
results=Results();error=None;exit_code=1
try:
    with session_lock(GLOBAL_LOCK.parent):
        idle()
        with CpuPrerequisiteGuard(paths['guard'],ROOT,timeout_seconds=180):
            assert digest(source_fingerprint())==SOURCE_SHA256
            before=snapshot(ORIGINAL_ROOT)
            metadata={name:file_digest(ORIGINAL_ROOT/name) for name,entry in before['files'].items() if not entry['group'].endswith('/weights')}
            with paths['log'].open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                exit_code=pytest.main(['-q','-p','no:cacheprovider',str(code/'test_execution.py'),str(code/'test_final.py')],plugins=[results])
            assert snapshot(ORIGINAL_ROOT)==before and digest(source_fingerprint())==SOURCE_SHA256
            assert all(file_digest(ORIGINAL_ROOT/name)==sha for name,sha in metadata.items())
            idle()
except Exception as exc:error=repr(exc)
for role in ('execution','final'):
    tests=[t for t in results.tests if '/test_'+role+'.py::' in t['nodeid']]
    value=dict(status='passed' if error is None and exit_code==0 and tests else 'failed',error=error,
        pytest_exit_code=int(exit_code),tests=tests,observed_at=datetime.now(timezone.utc).isoformat(),
        code_sha256={str(p.relative_to(ROOT)):file_digest(p) for p in [code/n for n in
            ('context.py','worker.py','controller.py','final.py','test_execution.py','test_final.py','run_execution_prerequisites.py')]},
        guard_sha256=file_digest(paths['guard']) if paths['guard'].exists() else None,
        log_sha256=file_digest(paths['log']) if paths['log'].exists() else None,
        original_source_sha256=SOURCE_SHA256,original_campaign_inventory_unchanged=error is None,
        original_metadata_files_verified=len(metadata) if error is None else None,
        scope='Real disposable bounded metadata and phase guards; explicit original-audit/model/GPU/process doubles. Tests connect separate source/output roots, worker entry and parent acceptance, three serial pairs, immutable final receipt and no-retry failures. Original OS supervisor concurrency was independently tested in census prerequisite; no research models/corpus/checkpoints loaded.',
        limitations='Does not freeze a plan or measure inference. Real source/import/freeze applicability and current capacity review required before actual new-root dispatch.',
        inference_launched=False,training_launched=False,holdout_scored=False,dispatch_authorized=False)
    with paths[role].open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps(dict(pytest_exit_code=int(exit_code),error=error,tests=len(results.tests))))
raise SystemExit(0 if error is None and exit_code==0 else 1)
