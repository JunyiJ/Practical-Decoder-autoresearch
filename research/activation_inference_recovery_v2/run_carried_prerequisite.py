"""Guarded carry tests and actual read-only first-pair import; no timing repeat."""
import contextlib
from datetime import datetime,timezone
import json
from pathlib import Path
import pytest
from src.research.protocol import ROOT,digest,file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock,MetadataWriter,Session,reference
from src.research.tokenizer_wide_hybrid_metadata import snapshot
from research.activation_inference_recovery_v1.source_binding import ORIGINAL_ROOT
from research.activation_inference_recovery_v1.context import RECOVERY_ROOT,IMPORT_SHA256
from research.activation_inference_recovery_v2.carried_pair import collect_carried_pair

base=ROOT/'research/tokenizer-evidence/activation-preparation-001';code=ROOT/'research/activation_inference_recovery_v2'
stem='inference-recovery-v2-carried-pair-prerequisite-001'
paths={k:base/(stem+suffix) for k,suffix in [('guard','-guard.json'),('log','.log'),('review','-review.json'),('import','-import.json')]}
for path in paths.values():
    if path.exists():raise FileExistsError('Existing prerequisite evidence; no overwrite')
class Results:
    def __init__(self):self.tests=[]
    def pytest_runtest_logreport(self,report):
        if report.when=='call' or report.failed:self.tests.append(dict(nodeid=report.nodeid,when=report.when,outcome=report.outcome))
results=Results();error=None;exit_code=1;carried=None
try:
    with session_lock(GLOBAL_LOCK.parent) as g,session_lock(ORIGINAL_ROOT) as s:
        idle()
        with CpuPrerequisiteGuard(paths['guard'],ROOT,timeout_seconds=180):
            before={str(root):snapshot(root) for root in (ORIGINAL_ROOT,RECOVERY_ROOT)}
            with paths['log'].open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                exit_code=pytest.main(['-q','-p','no:cacheprovider',str(code/'test_carried_pair.py')],plugins=[results])
            if exit_code!=0:raise RuntimeError('Carried-pair prerequisite tests failed')
            source=Session(MetadataWriter(ORIGINAL_ROOT,(g,s)),reference(ORIGINAL_ROOT,'evidence/meta/activation-plan.json'))
            imported_path=base/'inference-recovery-binding-prerequisite-001-import.json'
            assert file_digest(imported_path)==IMPORT_SHA256
            imported=json.loads(imported_path.read_text());carried=collect_carried_pair(g,source,imported)
            assert all(snapshot(Path(root))==seen for root,seen in before.items())
            with paths['import'].open('x') as stream:json.dump(carried,stream,indent=2);stream.write('\n')
            idle()
except Exception as exc:error=repr(exc)
value=dict(status='passed' if error is None else 'failed',error=error,pytest_exit_code=int(exit_code),tests=results.tests,
    observed_at=datetime.now(timezone.utc).isoformat(),
    code_sha256={str((code/name).relative_to(ROOT)):file_digest(code/name) for name in ('carried_pair.py','test_carried_pair.py','run_carried_prerequisite.py')},
    outputs={name:reference(ROOT,str(path.relative_to(ROOT))) for name,path in paths.items() if name!='review' and path.exists()},
    carried_import_sha256=digest(carried) if carried else None,carried_files=len(carried['files']) if carried else None,
    scope='Disposable first-pair transport/provenance tests use explicit v1 GPU/model/process doubles. Actual import independently replays original accepted3611 process/request/guards/model lineage/timing arithmetic, pins all27files of closed recovery001 including later failure, and verifies original training/source/16protected checkpoints. No real inference repeated, no corpus/checkpoint deserialization, no old entry called.',
    limitations='Component prerequisite only; new version must integrate carried credit with only3612/3613, use scoped inventories, test real later-child entry concurrency and freeze exact source/evidence before dispatch.',
    inference_launched=False,repeat_first_pair_allowed=False,training_launched=False,holdout_scored=False,dispatch_authorized=False)
with paths['review'].open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps({k:value[k] for k in ('status','error','carried_files','carried_import_sha256')}))
raise SystemExit(0 if error is None else 1)
