"""One guarded disposable CPU metadata prerequisite; never a campaign entry."""
import contextlib
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pytest
from src.research.protocol import ROOT, digest, file_digest
from src.research.runner import source_fingerprint
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, idle, session_lock
from src.research.tokenizer_wide_hybrid_metadata import snapshot

base = ROOT/'research/tokenizer-evidence/activation-preparation-001'
code = ROOT/'research/activation_inference_recovery_v1'
root = ROOT/'research_runs/tokenizer-activation-001'
stem = 'inference-recovery-census-prerequisite-001'
expected_source = '7a4c02946a31820a7b1ff36b8cc4be42659a4fc4a62424cbc4ea5f32d6cbc487'
log = base/(stem+'.log')
guard = base/(stem+'-guard.json')
report = base/(stem+'-review.json')
for path in (log, guard, report):
    if path.exists():
        raise FileExistsError('Prerequisite evidence already exists; do not overwrite')

class Results:
    def __init__(self): self.tests = []
    def pytest_runtest_logreport(self, report):
        if report.when == 'call' or report.failed:
            self.tests.append(dict(nodeid=report.nodeid,when=report.when,outcome=report.outcome))

results = Results()
with session_lock(GLOBAL_LOCK.parent):
    idle()
    with CpuPrerequisiteGuard(guard, ROOT, timeout_seconds=180):
        before_source = digest(source_fingerprint())
        assert before_source == expected_source
        before = snapshot(root)
        metadata_pins = {name:file_digest(root/name) for name, entry in before['files'].items()
                         if not entry['group'].endswith('/weights')}
        with log.open('x') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            result = pytest.main(['-q','-p','no:cacheprovider',str(code/'test_census.py')], plugins=[results])
        assert snapshot(root) == before, 'Original campaign inventory changed'
        assert all(file_digest(root/name) == sha for name,sha in metadata_pins.items())
        assert digest(source_fingerprint()) == before_source
        idle()
value = dict(status='passed' if result == 0 else 'failed',pytest_exit_code=int(result),tests=results.tests,
 observed_at=datetime.now(timezone.utc).isoformat(),original_source_sha256=before_source,
 original_metadata_files_verified=len(metadata_pins),original_inventory_unchanged=True,
 code_sha256={str(path.relative_to(ROOT)):file_digest(path) for path in sorted(code.glob('*.py'))},
 guard_sha256=file_digest(guard),log_sha256=file_digest(log),
 scope='Disposable metadata only; one bounded metadata subprocess; actual flock, atomic writer and original process supervisor. No models, corpus, checkpoints or inference measurements.',
 lock_audit='Original process_v2 observe, preflight and failure snapshots already hold writer._serialized(). Worker activation_inference.budget lacks it. Writer publication, append and capacity snapshots already hold the same metadata lock. Shared inherited global/campaign descriptors alone do not provide parent-child exclusion.',
 limitations='Prerequisite only. New inference adapter, exact original-evidence binding, independent acceptance/final reporting, dedicated-root source contract and freeze remain required. No new campaign or scientific measurement authorized by this report.',
 original_failed_entry_retried=False,training_launched=False,inference_launched=False,holdout_scored=False)
with report.open('x') as stream:
    json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps({k:value[k] for k in ('status','pytest_exit_code','original_source_sha256','original_metadata_files_verified')}))
raise SystemExit(result)
