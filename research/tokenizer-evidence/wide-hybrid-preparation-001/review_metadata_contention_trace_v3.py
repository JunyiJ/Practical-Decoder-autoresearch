"""Independent read-only review; never loads weights or runs the diagnostic."""
import base64,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.protocol import file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_driver import session_lock
HERE=Path(__file__).resolve().parent
RUN=ROOT/'research_runs/tokenizer-metadata-contention-trace-v3-001'

def inventory(root):
    assert not any(p.is_symlink() for p in root.rglob('*'))
    return {str(p.relative_to(root)):dict(size=p.stat().st_size,sha256=file_digest(p)) for p in root.rglob('*') if p.is_file()}

with session_lock(ROOT/'research_state/tokenizer-worker-lock'),session_lock(RUN/'campaign'),CpuPrerequisiteGuard(HERE/'metadata-contention-trace-v3-review-guard.json',ROOT,timeout_seconds=600):
    plan=json.loads((HERE/'metadata-contention-trace-v3-plan.json').read_text())
    assert all(file_digest(ROOT/n)==h for n,h in plan['sources'].items())
    assert file_digest(HERE/'run_metadata_contention_trace_v3.py')==plan['script_sha256']
    result=json.loads((RUN/'result.json').read_text())
    assert result['status'] in ('completed','failed_closed') and result['retry_allowed'] is False
    assert result['training'] is False and result['source_unchanged'] is True
    assert result['plan_sha256']==file_digest(HERE/'metadata-contention-trace-v3-plan.json')
    files=inventory(RUN)
    assert sum(v['size'] for v in files.values())<plan['output_cap_bytes']
    for n,v in result['files'].items():assert files[n]==v
    traces={who:json.loads((RUN/(who+'-trace.json')).read_text()) for who in ('parent','child')}
    guards={who:json.loads((RUN/(who+'-guard.json')).read_text()) for who in ('parent','child')}
    for who,g in guards.items():
        assert g['status'] in ('finished','fixture_exception')
        assert g['observation']['peak_rss_bytes']<plan['rss_limit_bytes']
        assert g['observation']['free_bytes']>=plan['reserve_bytes']
        assert g['elapsed_seconds']<(plan['wall_seconds'] if who=='parent' else plan['child_wall_seconds'])
        assert files[who+'-trace.json']['size']<=plan['external_json_cap_bytes']
    if result['status']=='completed':
        assert result['error'] is None and result['process']
        assert all(g['status']=='finished' and g['reason'] is None for g in guards.values())
        assert all(t['complete'] and not t['failures'] and not t['dropped_records'] and not t['dropped_failures'] for t in traces.values())
        assert not traces['child']['resource_errors'] and len(traces['child']['samples'])>=1
        receipt=json.loads((RUN/'campaign/evidence/quality/23/result.json').read_text())['receipt']
        checkpoint=RUN/'campaign/evidence/quality/23/checkpoints/fixture.pt'
        assert Path(receipt['path'])==checkpoint
        assert files[str(checkpoint.relative_to(RUN))]==dict(size=receipt['size'],sha256=receipt['sha256'])
        assert plan['tensor_payload_bytes']<receipt['size']<=plan['checkpoint_allowance_bytes']
        assert not list(RUN.rglob('*.tmp'))
    prior=json.loads((HERE/'metadata-contention-trace-v2-review.json').read_text())
    unchanged={}
    for name,expected in [('tokenizer-wide-hybrid-003',prior['closed003_inventory']),('tokenizer-metadata-contention-trace-v2-001',prior['files'])]:
        observed=inventory(ROOT/'research_runs'/name)
        assert observed==expected
        unchanged[name]=len(observed)
    summaries={}
    for who,t in traces.items():
        locks=[x for x in t['records'] if x['kind']=='lock']
        writes=[x for x in t['records'] if x.get('operation')=='checkpoint._write']
        summaries[who]=dict(records=len(t['records']),lock_scopes=len(locks),max_wait_seconds=t['max_wait'],max_hold_seconds=t['max_hold'],
            complete=t['complete'],dropped_records=t['dropped_records'],dropped_failures=t['dropped_failures'],failures=t['failures'],
            operations=t['operations'],checkpoint_writes=len(writes),checkpoint_bytes=sum(x['bytes'] for x in writes),
            small_checkpoint_writes=sum(x['bytes']<1024**2 for x in writes),
            per_thread_max_wait={n:max(x['wait'] for x in locks if x['thread']==n) for n in sorted({x['thread'] for x in locks})})
    archive={n:base64.b64encode((RUN/n).read_bytes()).decode() for n in files if '/checkpoints/' not in n}
    assert all(file_digest(RUN/n)==v['sha256'] for n,v in files.items())
    report=dict(status='completed_synthetic_comparison_reviewed' if result['status']=='completed' else 'closed_failed_comparison_reviewed',created_unix=time.time(),
        reviewer_sha256=file_digest(Path(__file__)),plan_sha256=result['plan_sha256'],result=result,guards=guards,summary=summaries,
        resource_samples=len(traces['child']['samples']),resource_errors=traces['child']['resource_errors'],files=files,
        noncheckpoint_bytes_base64=archive,protected_weights={n:v for n,v in files.items() if '/checkpoints/' in n},
        closed_evidence_files_verified_unchanged=unchanged,training=False,holdout_consumed=False,cleanup_performed=False,retry_allowed=False,
        limitations=['One instrumented synthetic comparison; handoff and recorder overhead both changed. Does not prove original uninstrumented failure cause or future reliability.',
          'Flat synthetic file topology, CPU zero tensors and no preceding MPS training differ from the failed campaign.',
          'Guards sample resource usage; individual namespace operations are not preempted. Existing lock timeout remains2s.',
          'No training authorized by this outcome alone; source applicability, new finite evidence contract and full-panel storage remain prerequisites.'])
    data=(json.dumps(report,indent=2)+'\n').encode();assert len(data)<16*1024**2
    with (HERE/'metadata-contention-trace-v3-review.json').open('xb') as f:f.write(data)
    print(json.dumps(dict(status=report['status'],summary={k:{n:v for n,v in s.items() if n!='operations'} for k,s in summaries.items()},unchanged=unchanged)),flush=True)
