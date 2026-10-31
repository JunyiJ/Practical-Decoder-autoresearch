"""One guarded contract test plus actual read-only original evidence import."""
import contextlib
from datetime import datetime,timezone
import json
from pathlib import Path
import pytest

from src.research.protocol import ROOT,digest,file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock,MetadataWriter,Session,reference
from src.research.tokenizer_wide_hybrid_metadata import snapshot
from research.activation_inference_recovery_v1.source_binding import collect_source_import,ORIGINAL_ROOT

base=ROOT/'research/tokenizer-evidence/activation-preparation-001'
code=ROOT/'research/activation_inference_recovery_v1'
stem='inference-recovery-binding-prerequisite-001'
paths={name:base/(stem+suffix) for name,suffix in [('guard','-guard.json'),('log','.log'),('review','-review.json'),('import','-import.json')]}
for path in paths.values():
    if path.exists():raise FileExistsError('Existing evidence attempt; no overwrite')
class Results:
    def __init__(self):self.tests=[]
    def pytest_runtest_logreport(self,report):
        if report.when=='call' or report.failed:self.tests.append(dict(nodeid=report.nodeid,when=report.when,outcome=report.outcome))
results=Results();imported=None;error=None;exit_code=1
try:
    with session_lock(GLOBAL_LOCK.parent) as g,session_lock(ORIGINAL_ROOT) as c:
        idle()
        with CpuPrerequisiteGuard(paths['guard'],ROOT,timeout_seconds=180):
            before=snapshot(ORIGINAL_ROOT)
            with paths['log'].open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                exit_code=pytest.main(['-q','-p','no:cacheprovider',str(code/'test_source_binding.py')],plugins=[results])
            if exit_code!=0:raise RuntimeError('Binding contract tests failed')
            ledger=Session(MetadataWriter(ORIGINAL_ROOT,(g,c)),reference(ORIGINAL_ROOT,'evidence/meta/activation-plan.json'))
            imported=collect_source_import(ledger)
            assert snapshot(ORIGINAL_ROOT)==before,'Original campaign changed'
            with paths['import'].open('x') as stream:json.dump(imported,stream,indent=2);stream.write('\n')
            idle()
except Exception as exc:
    error=repr(exc)
value=dict(status='passed' if error is None else 'failed',pytest_exit_code=int(exit_code),error=error,
    tests=results.tests,observed_at=datetime.now(timezone.utc).isoformat(),
    code_sha256={str(code/name):file_digest(code/name) for name in ('source_binding.py','test_source_binding.py','run_binding_prerequisite.py')},
    guard=reference(ROOT,str(paths['guard'].relative_to(ROOT))) if paths['guard'].exists() else None,
    log=reference(ROOT,str(paths['log'].relative_to(ROOT))) if paths['log'].exists() else None,
    imported=reference(ROOT,str(paths['import'].relative_to(ROOT))) if paths['import'].exists() else None,
    import_digest=digest(imported) if imported is not None else None,
    original_files=len(imported['files']) if imported else None,
    protected_checkpoint_files=len(imported['protected_checkpoints']) if imported else None,
    protected_checkpoint_bytes=sum(ref['size'] for ref in imported['protected_checkpoints'].values()) if imported else None,
    scope='Contract tests use explicit provenance doubles and disposable JSON. Actual integration independently replays original freeze, source/runtime, fourteen dispatches and acceptances, both retention boundaries, failure anchor and all sixteen checkpoint hashes; imports three loader-facing fresh-pair bindings. No model/corpus load, numerical scoring, training, inference or cleanup.',
    limitations='Preparation only: new-root controller/worker, independent acceptance/final reporting and exact external freeze still required. Original failed campaign remains closed to all entries.',
    inference_launched=False,training_launched=False,holdout_scored=False,dispatch_authorized=False)
with paths['review'].open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps({k:value[k] for k in ('status','error','original_files','protected_checkpoint_files','protected_checkpoint_bytes')}))
raise SystemExit(0 if error is None else 1)
