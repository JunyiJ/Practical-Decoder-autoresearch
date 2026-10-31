"""One guarded real-process metadata prerequisite, neither campaign is resumed."""
import contextlib
from datetime import datetime,timezone
import json
from pathlib import Path
import pytest

from src.research.protocol import ROOT,digest,file_digest
from src.research.runner import source_fingerprint
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock
from src.research.tokenizer_wide_hybrid_metadata import snapshot
from research.activation_inference_recovery_v1.source_binding import SOURCE_SHA256,ORIGINAL_ROOT
from research.activation_inference_recovery_v1.context import RECOVERY_ROOT,external_source

base=ROOT/'research/tokenizer-evidence/activation-preparation-001';code=ROOT/'research/activation_inference_recovery_v2'
stem='inference-recovery-v2-scoped-inventory-prerequisite-001'
paths={k:base/(stem+suffix) for k,suffix in [('guard','-guard.json'),('log','.log'),('review','-review.json')]}
for path in paths.values():
    if path.exists():raise FileExistsError('Existing evidence attempt; no overwrite')
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
            v1=external_source()
            before={str(root):snapshot(root) for root in (ORIGINAL_ROOT,RECOVERY_ROOT)}
            metadata={str(root/name):file_digest(root/name) for root in (ORIGINAL_ROOT,RECOVERY_ROOT)
                      for name,entry in before[str(root)]['files'].items() if not entry['group'].endswith('/weights')}
            with paths['log'].open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                exit_code=pytest.main(['-q','-p','no:cacheprovider',str(code/'test_inventory.py')],plugins=[results])
            assert all(snapshot(Path(root))==value for root,value in before.items())
            assert all(file_digest(Path(path))==sha for path,sha in metadata.items())
            assert external_source()==v1 and digest(source_fingerprint())==SOURCE_SHA256
            idle()
except Exception as exc:error=repr(exc)
value=dict(status='passed' if error is None and exit_code==0 else 'failed',error=error,pytest_exit_code=int(exit_code),
    tests=results.tests,observed_at=datetime.now(timezone.utc).isoformat(),
    code_sha256={str(path.relative_to(ROOT)):file_digest(path) for path in sorted(code.glob('*.py'))},
    guard_sha256=file_digest(paths['guard']) if paths['guard'].exists() else None,
    log_sha256=file_digest(paths['log']) if paths['log'].exists() else None,
    original_source_sha256=SOURCE_SHA256,original_and_v1_campaigns_unchanged=error is None,
    frozen_v1_source_sha256=digest(v1) if error is None else None,
    noncheckpoint_files_verified=len(metadata) if error is None else None,
    scope='Disposable metadata only. Ten real bounded pipe children, sequentially, cover prior3611/current3612 and prior3612/current3613 with ordinary periodic parent monitor replacement plus explicit handshake. Old whole-root equality rejects valid overlap while per-slot hashes remain equal. Same-size prior-byte, prior-control, extra-prior-file and unknown-control injections are rejected. Full global quota/path/alias/STOP/failure/reserve checks remain on every scoped read.',
    limitations='Component prerequisite, not a new controller/worker freeze or scientific inference. Remaining-pair import/controller/final integration still must use this helper and pass actual later-entry concurrency coverage. Both failed roots stay closed; do not repeat valid3611 timing.',
    inference_launched=False,training_launched=False,holdout_scored=False,dispatch_authorized=False)
with paths['review'].open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps(dict(status=value['status'],error=error,pytest_exit_code=int(exit_code),tests=len(results.tests))))
raise SystemExit(0 if value['status']=='passed' else 1)
