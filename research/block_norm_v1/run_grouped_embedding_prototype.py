"""One guarded strict MPS numerical prerequisite with immutable attempt/source evidence."""
import argparse,contextlib,json,shutil,os
from datetime import datetime,timezone
from pathlib import Path
import pytest,torch
from src.research.protocol import ROOT,digest,file_digest
from src.research.runner import source_fingerprint
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock
from src.research.tokenizer_wide_hybrid_metadata import snapshot
parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True);args=parser.parse_args()
assert len(args.attempt)==3 and args.attempt.isdigit()
code=ROOT/'research/block_norm_v1';base=ROOT/'research/tokenizer-evidence/block-norm-preparation-001'
stem='grouped-embedding-prototype-'+args.attempt
paths={k:base/(stem+suffix) for k,suffix in [('guard','-guard.json'),('log','.log'),('review','-review.json')]}
trace=base/(stem+'-events.jsonl');assert not trace.exists()
os.environ['BLOCK_NORM_TRACE_PATH']=str(trace)
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
manifest_path=base/('grouped-embedding-prototype-plan-'+args.attempt+'.json')
manifest=json.loads(manifest_path.read_text());manifest_sha=file_digest(manifest_path)
map_sha=file_digest(base/'runtime-dependency-map.json')
try:
    with session_lock(GLOBAL_LOCK.parent):
        idle()
        with StrictMpsGuard(paths['guard'],ROOT):
            torch.set_num_threads(4);original=source_fingerprint()
            assert file_digest(base/'runtime-baseline-copy-manifest.json')==manifest['baseline_manifest_sha256']
            assert all(file_digest(ROOT/name)==sha for name,sha in manifest['current_runtime_sha256'].items())
            assert all(file_digest(ROOT/name)==sha for name,sha in manifest['prerequisite_source_sha256'].items())
            assert all(file_digest(ROOT/name)==sha for name,sha in manifest['generated_fixture_sha256'].items())
            roots=[ROOT/'research_runs'/name for name in ('tokenizer-activation-001','tokenizer-activation-inference-recovery-001','tokenizer-activation-inference-recovery-002')]
            before={str(p):snapshot(p) for p in roots}
            metadata={str(root/name):file_digest(root/name) for root in roots for name,entry in before[str(root)]['files'].items() if not entry['group'].endswith('/weights')}
            plan=json.loads((roots[-1]/'evidence/meta/recovery-plan.json').read_text())
            assert digest(original)==plan['original_source_sha256']
            assert all(file_digest(ROOT/name)==sha for name,sha in plan['external_source'].items())
            with paths['log'].open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                exit_code=pytest.main(['-q','-x','-p','no:cacheprovider',str(code/'test_grouped_embedding_prototype.py')],plugins=[results])
            torch.mps.synchronize()
            assert source_fingerprint()==original
            assert all(file_digest(ROOT/name)==sha for name,sha in plan['external_source'].items())
            assert all(snapshot(Path(path))==seen for path,seen in before.items())
            assert all(file_digest(Path(path))==sha for path,sha in metadata.items())
            assert all(file_digest(ROOT/path)==sha for path,sha in pins.items())
            assert file_digest(manifest_path)==manifest_sha and file_digest(base/'runtime-dependency-map.json')==map_sha
            assert all(file_digest(ROOT/name)==sha for name,sha in manifest['generated_fixture_sha256'].items())
            idle();unchanged=True
except Exception as exc:error=repr(exc)
value=dict(status='passed' if error is None and exit_code==0 else 'failed',error=error,pytest_exit_code=int(exit_code),tests=results.tests,
    observed_at=datetime.now(timezone.utc).isoformat(),code_sha256=pins,
    source_snapshot={str(p.relative_to(ROOT)):file_digest(p) for p in saved.rglob('*.py')},
    source_delta_sha256=manifest_sha,baseline_manifest_sha256=manifest['baseline_manifest_sha256'],dependency_map_sha256=map_sha,
    guard_sha256=file_digest(paths['guard']) if paths['guard'].exists() else None,
    log_sha256=file_digest(paths['log']) if paths['log'].exists() else None,
    frozen_sources_and_closed_campaigns_unchanged=unchanged,
    trace_sha256=file_digest(trace) if trace.exists() else None,
    trace_path=str(trace.relative_to(ROOT)),numerical_acceptance_claimed=False,
    scope='Standalone all-MPS grouped embedding backward prototype: native forward, unique-token FP32 membership matmul and unique-index copy. Eleven input/upstream cases,88backwards with four repeats each deterministic-off/on, FP64 analytic oracle, ten unsupported-contract rejections. No runtime adoption.',
    limitations='Standalone first-order dense fixed50257x256 embedding only, at most1024tokens. Not full-model gradient/cache/resume or resource pilot. Timing observations include checks and synchronization, not a throughput benchmark. No quality/holdout/optimizer updates.',
    backend_environment={k:os.environ.get(k) for k in ['PYTORCH_ENABLE_MPS_FALLBACK','PYTORCH_MPS_FAST_MATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
    quality_training_launched=False,synthetic_optimizer_steps=False,in_memory_checkpoint_loaded=False,persisted_checkpoints_loaded=False,trained_checkpoints_loaded=False,holdout_scored=False,deletion_performed=False)
with paths['review'].open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps(dict(status=value['status'],error=error,tests=len(results.tests),failures=[t['nodeid'] for t in results.tests if t['outcome']=='failed'])))
raise SystemExit(0 if value['status']=='passed' else 1)
