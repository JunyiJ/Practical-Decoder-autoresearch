"""Read-only preservation and timing review of the single closed trace-v2 attempt."""
import base64, collections, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.protocol import file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_driver import session_lock
HERE=Path(__file__).resolve().parent
RUN=ROOT/'research_runs/tokenizer-metadata-contention-trace-v2-001'
OLD=ROOT/'research_runs/tokenizer-wide-hybrid-003'

with session_lock(ROOT/'research_state/tokenizer-worker-lock'),session_lock(RUN/'campaign'),CpuPrerequisiteGuard(HERE/'metadata-contention-trace-v2-review-guard.json',ROOT,timeout_seconds=600):
    plan=json.loads((HERE/'metadata-contention-trace-v2-plan.json').read_text())
    assert all(file_digest(ROOT/n)==h for n,h in plan['sources'].items())
    assert file_digest(HERE/'run_metadata_contention_trace_v2.py')==plan['script_sha256']
    result=json.loads((RUN/'result.json').read_text())
    assert result['status']=='failed_closed' and result['retry_allowed'] is False
    assert 'Metadata producer lock timeout' in result['error']
    assert result['plan_sha256']==file_digest(HERE/'metadata-contention-trace-v2-plan.json')
    assert not any(p.is_symlink() for p in RUN.rglob('*'))
    files={str(p.relative_to(RUN)):dict(size=p.stat().st_size,sha256=file_digest(p)) for p in RUN.rglob('*') if p.is_file()}
    assert sum(v['size'] for v in files.values())<plan['output_cap_bytes']
    for n,v in result['files'].items():assert files[n]==v
    parent=json.loads((RUN/'parent-trace.json').read_text())
    child=json.loads((RUN/'child-trace.json').read_text())
    guards={n:json.loads((RUN/n).read_text()) for n in ('parent-guard.json','child-guard.json')}
    for g in guards.values():
        assert g['status']=='fixture_exception'
        assert g['observation']['peak_rss_bytes']<plan['rss_limit_bytes']
        assert g['observation']['free_bytes']>plan['reserve_bytes']
    failure=next(x for x in parent['failures'] if 'Metadata producer lock timeout' in x['error'])
    begin=failure['start_monotonic'];end=begin+failure['wait']
    ops=[x for x in child['records'] if x['kind']=='operation' and begin<=x['start_monotonic']<=end]
    writes=[x for x in ops if x['operation']=='checkpoint._write']
    locks=[x for x in child['records'] if x['kind']=='lock' and x['hold'] is not None]
    covered=0.;intersect=[]
    for x in locks:
        acquired=x['start_monotonic']+x['wait'];released=acquired+x['hold']
        overlap=max(0.,min(end,released)-max(begin,acquired))
        if overlap:covered+=overlap;intersect.append(x)
    prior=json.loads((HERE/'campaign003-lock-failure-v2-review.json').read_text())
    oldfiles={str(p.relative_to(OLD)):dict(size=p.stat().st_size,sha256=file_digest(p)) for p in OLD.rglob('*') if p.is_file()}
    assert oldfiles==prior['files']
    archive={n:base64.b64encode((RUN/n).read_bytes()).decode() for n in files if '/checkpoints/' not in n}
    summary=dict(parent_wait_seconds=failure['wait'],parent_failure_record_index=parent['records'].index(failure),
        parent_trace_records=len(parent['records']),parent_trace_limit=plan['trace_records_per_process'],
        child_trace_records=len(child['records']),child_resource_samples=len(child['samples']),
        window_start_monotonic=begin,window_end_monotonic=end,checkpoint_writes=len(writes),
        checkpoint_bytes=sum(x['bytes'] for x in writes),write_size_counts=dict(collections.Counter(x['bytes'] for x in writes)),
        intersecting_child_lock_scopes=len(intersect),child_hold_coverage_seconds=covered,
        child_hold_fraction=covered/failure['wait'],max_intersecting_child_hold_seconds=max(x['hold'] for x in intersect),
        max_child_operation_starting_in_window_seconds=max(x['duration'] for x in ops),
        parent_max_hold_seconds=parent['max_hold'],child_max_hold_seconds=child['max_hold'])
    report=dict(status='closed_diagnostic_reviewed_with_trace_cap_limitation',created_unix=time.time(),
        plan_sha256=result['plan_sha256'],reviewer_sha256=file_digest(Path(__file__)),result=result,
        guards=guards,summary=summary,files=files,noncheckpoint_bytes_base64=archive,
        protected_partials={n:v for n,v in files.items() if '/checkpoints/' in n},
        closed003_files_verified_unchanged=len(oldfiles),closed003_inventory=oldfiles,
        findings=[
          'The original two-second parent lock timeout occurred before the instrumentation count cap was reached.',
          'During this wait, the child repeatedly reacquired short lock scopes for small torch.save writes after its last MiB-sized write; no two-second hold appears in the saved child trace.',
          'The byte-only handoff does not schedule a gap until another cumulative MiB. Thousands of small metadata/census operations can consume the deadline before that threshold.',
          'The trace supports starvation during this synthetic small-write tail; it does not prove the uninstrumented campaign003 failure had exactly this mechanism.',
          'Parent tracing reached16384 records during later failure publication/shutdown, leaving a partial process control JSON; its final fsync wrapper raised Trace count cap. This is a failed diagnostic with a bounded, incomplete tail, not a passing test.',
          'Child termination interrupted torch.save and produced a secondary unexpected-position exception; the synthetic checkpoint remains partial and is not loaded.'
        ],next_work='Design a byte-independent bounded handoff for small callbacks without extending the two-second timeout. Improve bounded diagnostic operation aggregation so per-JSON-piece events cannot exhaust the trace. Any changed-source comparison requires a new frozen one-attempt contract and fresh output root; no existing attempt is rerun. Training remains blocked by storage and missing validated fix.',
        training=False,holdout_consumed=False,cleanup_performed=False,retry_allowed=False)
    assert all(file_digest(RUN/n)==v['sha256'] for n,v in files.items())
    data=(json.dumps(report,indent=2)+'\n').encode();assert len(data)<16*1024**2
    with (HERE/'metadata-contention-trace-v2-review.json').open('xb') as f:f.write(data)
    print(json.dumps(summary),flush=True)
