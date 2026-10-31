"""Guarded original-endpoint arithmetic, not a final campaign entry."""
import contextlib
from datetime import datetime,timezone
import json
import pytest
from src.research.protocol import ROOT,file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock,MetadataWriter,Session,reference
from src.research.tokenizer_wide_hybrid_metadata import snapshot
from research.activation_inference_recovery_v1.source_binding import ORIGINAL_ROOT
from research.activation_inference_recovery_v1.quality import analyze_imported_quality
base=ROOT/'research/tokenizer-evidence/activation-preparation-001';code=ROOT/'research/activation_inference_recovery_v1'
stem='inference-recovery-quality-prerequisite-001'
paths={name:base/(stem+suffix) for name,suffix in [('guard','-guard.json'),('log','.log'),('review','-review.json'),('analysis','-analysis.json')]}
for p in paths.values():
    if p.exists():raise FileExistsError('Existing evidence attempt; no overwrite')
class Results:
    def __init__(self):self.tests=[]
    def pytest_runtest_logreport(self,report):
        if report.when=='call' or report.failed:self.tests.append(dict(nodeid=report.nodeid,when=report.when,outcome=report.outcome))
results=Results();error=None;analysis=None;exit_code=1
input_path=base/'inference-recovery-binding-prerequisite-001-import.json'
input_ref=reference(ROOT,str(input_path.relative_to(ROOT)))
try:
    with session_lock(GLOBAL_LOCK.parent) as g,session_lock(ORIGINAL_ROOT) as c:
        idle()
        with CpuPrerequisiteGuard(paths['guard'],ROOT,timeout_seconds=180):
            before=snapshot(ORIGINAL_ROOT)
            with paths['log'].open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                exit_code=pytest.main(['-q','-p','no:cacheprovider',str(code/'test_quality.py')],plugins=[results])
            if exit_code!=0:raise RuntimeError('Quality import tests failed')
            ledger=Session(MetadataWriter(ORIGINAL_ROOT,(g,c)),reference(ORIGINAL_ROOT,'evidence/meta/activation-plan.json'))
            analysis=analyze_imported_quality(ledger,json.loads(input_path.read_text()))
            assert reference(ROOT,input_ref['path'])==input_ref and snapshot(ORIGINAL_ROOT)==before
            with paths['analysis'].open('x') as stream:json.dump(analysis,stream,indent=2);stream.write('\n')
            idle()
except Exception as exc:error=repr(exc)
value=dict(status='passed' if error is None else 'failed',error=error,pytest_exit_code=int(exit_code),tests=results.tests,
    observed_at=datetime.now(timezone.utc).isoformat(),source_import=input_ref,
    code_sha256={str(code/name):file_digest(code/name) for name in ('quality.py','test_quality.py','run_quality_prerequisite.py')},
    outputs={name:reference(ROOT,str(p.relative_to(ROOT))) for name,p in paths.items() if name!='review' and p.exists()},
    scope='Independent original-file provenance before/after the original predeclared statistical engine on accepted fresh metadata. No weights loaded, new validation scoring, training, inference, cleanup or terminal campaign entry.',
    formal_final_completed=False,holdout_scored=False,inference_launched=False)
with paths['review'].open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps(dict(status=value['status'],error=error,statistics=analysis['analysis']['statistics'] if analysis else None)))
raise SystemExit(0 if error is None else 1)
