"""Read-only source applicability and retention audit; no launch/load/delete."""
import datetime,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.tokenizer_wide_hybrid_plan_v2 import source_fingerprint,runtime_identity,PROJECTION
from src.research.tokenizer_wide_hybrid_retention import storage_envelope
from src.research.protocol import digest,file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_driver import session_lock
E=Path(__file__).resolve().parent

def inventory(root):
    assert not any(p.is_symlink() for p in root.rglob('*'))
    return {str(p.relative_to(root)):dict(size=p.stat().st_size,sha256=file_digest(p)) for p in root.rglob('*') if p.is_file()}

with session_lock(ROOT/'research_state/tokenizer-worker-lock'),CpuPrerequisiteGuard(E/'checkpoint-time-handoff-applicability-v2-guard.json',ROOT,timeout_seconds=600):
    original=json.loads((E/'finite-campaign-v2-frozen/meta/plan.json').read_text())
    prior=json.loads((E/'checkpoint-handoff-source-applicability-v1-review.json').read_text())
    campaign=json.loads((E/'campaign003-frozen/meta/plan.json').read_text())
    current=source_fingerprint();runtime=runtime_identity()
    assert runtime==original['runtime']==campaign['runtime'] and digest(runtime)==prior['runtime_sha256']
    cp='src/research/tokenizer_wide_hybrid_checkpoint_writer.py'
    changes={n:dict(before=original['source'].get(n),after=current.get(n)) for n in set(current)|set(original['source']) if current.get(n)!=original['source'].get(n)}
    assert set(changes)==set(prior['changed_files'])
    for n,v in prior['changed_files'].items():
        if n!=cp:assert changes[n]==v
    new_changes={n:dict(before=campaign['source'].get(n),after=current.get(n)) for n in set(current)|set(campaign['source']) if current.get(n)!=campaign['source'].get(n)}
    assert set(new_changes)=={cp}
    tests=json.loads((E/'checkpoint-time-handoff-v2-component-review.json').read_text())
    assert all(file_digest(ROOT/n)==h for n,h in tests['source_hashes'].items())
    assert tests['new_source_sha256']==current[cp]
    for n,v in tests['tests'].items():
        assert file_digest(E/(n+'-tests.log'))==v['log_sha256']
        assert json.loads((E/(n+'-guard.json')).read_text())==v['guard']
    transport=json.loads((E/'metadata-contention-trace-v3-review.json').read_text())
    assert transport['status']=='completed_synthetic_comparison_reviewed'
    assert json.loads((E/'metadata-contention-trace-v3-review-guard.json').read_text())['status']=='finished'
    traceplan=json.loads((E/'metadata-contention-trace-v3-plan.json').read_text())
    assert traceplan['sources']==current and file_digest(E/'metadata-contention-trace-v3-plan.json')==transport['plan_sha256']
    original_reports={};original_artifacts={}
    for n,h in prior['original_report_hashes'].items():
        p=E/n;value=json.loads(p.read_text())
        assert file_digest(p)==h
        assert value['source_sha256']==digest(original['source']) and value['runtime_sha256']==digest(runtime)
        for k,s in value['artifacts'].items():
            artifact=p.parent.parent/k;assert file_digest(artifact)==s
            original_artifacts[str(artifact.relative_to(ROOT))]=s
        original_reports[n]=h
    numeric=json.loads((E/'rank-numerical-v2-reviewed-001/meta/numerical-trace.json').read_text())
    for n,h in numeric['original_files'].items():assert file_digest(Path(n))==h
    # Original executed numerical/source receipts are verified, never regenerated.
    categories={};checkpoints={};closed={}
    def add(path,size,sha,category):
        path=Path(path);relative=str(path.relative_to(ROOT))
        assert relative not in checkpoints and not path.is_symlink()
        assert path.stat().st_size==size and file_digest(path)==sha
        checkpoints[relative]=dict(size=size,sha256=sha,retention_reason=category)
        categories.setdefault(category,dict(files=0,bytes=0))
        categories[category]['files']+=1;categories[category]['bytes']+=size
    manifest_path=ROOT/'research/tokenizer-evidence/checkpoint-cleanup-002/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    assert all(not (ROOT/n).exists() for n in manifest['deleted_paths'])
    for v in manifest['protected_checkpoints']:
        add(ROOT/v['path'],v['bytes'],v['sha256'],'selected_and_fresh_pairs_retained_by_cleanup002')
    for name,reviewname in [('tokenizer-wide-hybrid-002','campaign002-lock-failure-v2-review.json'),('tokenizer-wide-hybrid-003','campaign003-lock-failure-v2-review.json'),('tokenizer-metadata-contention-trace-v2-001','metadata-contention-trace-v2-review.json'),('tokenizer-metadata-contention-trace-v3-001','metadata-contention-trace-v3-review.json')]:
        root=ROOT/'research_runs'/name;review=json.loads((E/reviewname).read_text());files=inventory(root)
        assert files==review['files'];closed[name]=dict(files=len(files),inventory_sha256=digest(files))
        category='incomplete_campaign_pairs_and_failed_partial' if name.startswith('tokenizer-wide-hybrid') else 'preserved_synthetic_diagnostic'
        for n,v in files.items():
            if '/checkpoints/' in n:add(root/n,v['size'],v['sha256'],category)
    for n,h in numeric['original_files'].items():
        p=Path(n)
        if p.suffix=='.pt':add(p,p.stat().st_size,h,'protected_completed_numerical_prerequisite')
    for name,reviewname in [('tokenizer-metadata-contention-baseline-001','metadata-contention-baseline-v1-review.json'),('tokenizer-metadata-contention-handoff-001','metadata-contention-handoff-v1-review.json')]:
        # The original review schema embeds the outcome inventory in result.
        root=ROOT/'research_runs'/name;result=json.loads((root/'result.json').read_text())
        for p in root.rglob('*.pt'):
            key=str(p.relative_to(root));entry=result['files'][key]
            add(p,entry['size'],entry['sha256'],'preserved_synthetic_diagnostic')
    observed={str(p.relative_to(ROOT)) for d in (ROOT/'research_runs').glob('tokenizer-*') if d.is_dir() for p in d.rglob('*') if p.is_file() and (p.name.endswith('.pt') or p.name.endswith('.pt.tmp'))}
    assert observed==set(checkpoints),dict(unclassified=sorted(observed-set(checkpoints)),missing=sorted(set(checkpoints)-observed))
    required=storage_envelope(json.loads(PROJECTION.read_text()))['required_free_bytes']+128*1024
    free=shutil.disk_usage(ROOT).free
    assert source_fingerprint()==current and runtime_identity()==runtime
    report=dict(kind='wide_hybrid_source_applicability_review_v4',status='prerequisites_applicable_storage_blocked',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        before_source_sha256=digest(original['source']),after_source_sha256=digest(current),runtime_sha256=digest(runtime),changed_files=changes,changes_since_campaign003=new_changes,
        unchanged_original_source_files=len(current)-len(changes),unchanged_campaign003_source_files=len(current)-len(new_changes),
        original_report_hashes=original_reports,original_artifacts=original_artifacts,numerical_original_files_verified=len(numeric['original_files']),
        prior_applicability_sha256=file_digest(E/'checkpoint-handoff-source-applicability-v1-review.json'),component_review_sha256=file_digest(E/'checkpoint-time-handoff-v2-component-review.json'),
        synthetic_comparison_review_sha256=file_digest(E/'metadata-contention-trace-v3-review.json'),closed_evidence=closed,
        retention=dict(scope='All tokenizer-* research_runs .pt/.pt.tmp files; no unrelated or historical character data inspected/deleted',checkpoints=checkpoints,categories=categories,
            total_files=len(checkpoints),total_bytes=sum(v['size'] for v in checkpoints.values()),eligible_deletion_files=[],eligible_deletion_bytes=0,
            cleanup002_manifest_sha256=file_digest(manifest_path),prior_deleted_files_still_absent=len(manifest['deleted_paths']),
            policy='Completed prior cleanup retains selected/fresh pairs. Failed partials and incomplete screens have no valid selection permitting deletion. Numerical and synthetic prerequisite evidence remains preserved; successful synthetic saves are not resource-pilot or losing-screen weights.'),
        storage=dict(required_free_bytes=required,observed_free_bytes=free,shortfall_bytes=max(0,required-free),reserve_bytes=8*1024**3),
        review=['Only new production change since003 is elapsed-time checkpoint yielding, including preflight/empty calls and flush; model/optimizer/EMA/RNG/data math is unchanged.',
            'Completed numerical procedure uses original checkpoint helpers, not the bounded writer. Exact-resume/EMA evidence applies to unchanged math; no numerical rerun.',
            'Input encoding has no checkpoint scheduling dependency. Original report and artifact bytes verified, no reencoding or holdout access.',
            '140 regression tests and three recorder tests passed; new instrumented actual-scale synthetic comparison completed. This does not establish future MPS training reliability.',
            'New campaign requires explicit carry-forward provenance, fresh pilot/process/retention evidence and finite full-panel storage freeze. No old partial panel is pooled, selected or promoted.',
            'Existing full-panel storage guard is not reduced. No eligible tokenizer weight deletion identified; external storage or additional user-managed local free space is required for this unchanged design.'],
        numerical_rerun_required=False,input_reencoding_required=False,original_evidence_modified=False,new_campaign_freeze_required=True,
        training=False,cleanup_performed=False,closed_campaigns_reopened=False,holdout_consumed=False,
        next_action='Prepare lossless prospective source-applicability transport and metadata sizing without creating/dispatching a campaign. Full-panel launch remains blocked until verified capacity reaches the unchanged bound. Report actionable shortfall once; do not repeatedly poll unchanged disk or retry closed fixtures.')
    with (E/'checkpoint-time-handoff-source-applicability-v2-review.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:report[k] for k in ('status','after_source_sha256','unchanged_original_source_files','unchanged_campaign003_source_files','storage')}),flush=True)
