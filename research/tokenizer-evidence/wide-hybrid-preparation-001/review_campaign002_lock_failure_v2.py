"""Archive the closed slot5 producer-lock failure without loading/retrying weights."""
import base64,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.protocol import file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_driver import session_lock
HERE=Path(__file__).resolve().parent;RUN=ROOT/'research_runs/tokenizer-wide-hybrid-002'
with session_lock(ROOT/'research_state/tokenizer-worker-lock'),session_lock(RUN),CpuPrerequisiteGuard(HERE/'campaign002-lock-failure-v2-guard.json',ROOT,timeout_seconds=600):
    terminal=json.loads((HERE/'campaign002-lr-05-v2-terminal.json').read_text())
    process=json.loads((RUN/'control/process-quality-05.json').read_text())
    assert terminal['status']=='failed_closed' and terminal['retry_allowed'] is False
    assert process['status']=='failed' and process['returncode']==-15 and process['error']==terminal['error']
    assert 'Metadata producer lock timeout' in process['error']
    assert not (RUN/'evidence/meta/quality-dispatch-05.json').exists()
    assert not (RUN/'evidence/quality/06').exists()
    files={str(p.relative_to(RUN)):dict(size=p.stat().st_size,sha256=file_digest(p)) for p in RUN.rglob('*') if p.is_file()}
    assert not any(p.is_symlink() for p in RUN.rglob('*'))
    weights={n:v for n,v in files.items() if '/checkpoints/' in n}
    partials={n:v for n,v in weights.items() if n.endswith('.tmp')}
    assert len(partials)==1 and next(iter(partials)).startswith('evidence/quality/05/checkpoints/')
    assert len(weights)==11
    # Preserve every accepted raw/H200 pair and independently compare against
    # the last completed prefix archive, without promoting any failed output.
    prior=json.loads((HERE/'campaign002-lr-prefix-05-v2-review.json').read_text())
    for n,v in prior['files'].items():assert files[n]==v
    metadata={n:base64.b64encode((RUN/n).read_bytes()).decode() for n in files if n not in weights}
    sources={n:dict(sha256=file_digest(ROOT/n),bytes_base64=base64.b64encode((ROOT/n).read_bytes()).decode()) for n in (
      'src/research/tokenizer_wide_hybrid_metadata_writer.py','src/research/tokenizer_wide_hybrid_checkpoint_writer.py',
      'src/research/tokenizer_wide_hybrid_process_v2.py','src/research/tokenizer_wide_hybrid_training_outputs.py')}
    metrics=[json.loads(x) for x in (RUN/'evidence/quality/05/metrics.jsonl').read_text().splitlines()]
    events=[json.loads(x) for x in (RUN/'evidence/quality/05/resource-events.jsonl').read_text().splitlines()]
    samples=[x for x in events if x['event']=='sample']
    assert metrics[-1]['step']==1199
    assert max(x['peak_rss_bytes'] for x in samples)<4*1024**3
    assert max(x['mps_driver_bytes'] for x in samples)<6*1024**3
    assert min(x['free_bytes'] for x in samples)>8*1024**3
    missing=[n for n in ('training-summary.json','training-outputs.json','result.json','resource-audit.json','tensor-audit.json','worker-outputs.json') if not (RUN/'evidence/quality/05'/n).exists()]
    report=dict(status='campaign002_closed_after_slot5_metadata_lock_timeout',created_unix=time.time(),
       plan_sha256=prior['plan_sha256'],terminal=terminal,process=process,
       accepted_prefix_unchanged=True,accepted_cells=prior['records'],protected_weights=weights,failed_partials=partials,
       missing_slot5_outputs=missing,last_metric=metrics[-1],last_resource_sample=samples[-1],
       peak_observed_rss_bytes=max(x['peak_rss_bytes'] for x in samples),peak_observed_mps_bytes=max(x['mps_driver_bytes'] for x in samples),
       files=files,noncheckpoint_bytes_base64=metadata,source_evidence=sources,
       findings=['All1200 updates logged; interruption occurred before complete raw checkpoint publication.',
       'Parent metadata lock acquisition timed out after its frozen2-second acquisition deadline; child was terminated/reaped.',
       'Checkpoint writer releases the shared lock after each1MiB chunk but immediately competes again; flock nonblocking polling gives no fairness guarantee.',
       'Evidence is consistent with starvation or a slow critical section. No lock-acquisition trace distinguishes those mechanisms.',
       'No completed slot5 raw/H200 pair, full-validation result, tensor/resource audit or parent acceptance exists; sampled metrics are not selection evidence.'],
       next_work='Design a separate bounded producer/supervisor contention fixture and versioned fix preserving all resource/storage/integrity gates. Establish new source/evidence contract and finite plan before further training; do not reopen002 or retry slot5.',
       retries_allowed=False,training_authorized_by_report=False,holdout_consumed=False,cleanup_performed=False)
    assert all(file_digest(RUN/n)==v['sha256'] for n,v in files.items())
    with (HERE/'campaign002-lock-failure-v2-review.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:report[k] for k in ('status','accepted_prefix_unchanged','failed_partials','missing_slot5_outputs','peak_observed_rss_bytes','peak_observed_mps_bytes')}),flush=True)
