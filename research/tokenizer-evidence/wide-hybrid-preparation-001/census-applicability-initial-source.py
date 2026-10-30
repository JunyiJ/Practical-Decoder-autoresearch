"""Archive-aware source applicability and capacity audit; no training or cleanup."""
import base64,datetime,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.protocol import digest,file_digest
from src.research.runner import source_fingerprint
from src.research.tokenizer_study import runtime_identity
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_driver import session_lock
from src.research.tokenizer_wide_hybrid_retention import storage_envelope
from src.research.tokenizer_wide_hybrid_plan_v2 import PROJECTION
E=Path(__file__).resolve().parent;A=E.parent/'checkpoint-compaction-001'

with session_lock(ROOT/'research_state/tokenizer-worker-lock'),session_lock(ROOT/'research_runs/tokenizer-wide-hybrid-004'),CpuPrerequisiteGuard(E/'census-source-applicability-v2-guard.json',ROOT,timeout_seconds=600):
    original=json.loads((E/'finite-campaign-v2-frozen/meta/plan.json').read_text())
    prior=json.loads((E/'checkpoint-time-handoff-source-applicability-v2-review.json').read_text())
    component=json.loads((E/'confirmation-census-v2-review.json').read_text())
    source=source_fingerprint();runtime=runtime_identity()
    assert digest(source)==component['after_source_sha256'] and runtime==original['runtime']
    assert digest(runtime)==prior['runtime_sha256']==component['runtime_sha256']
    for n,h in component['source_and_test_hashes'].items():assert file_digest(ROOT/n)==h
    for phase,v in component['tests'].items():
        assert file_digest(E/f'confirmation-census-v2-{phase}-tests.log')==v['log_sha256']
        assert json.loads((E/f'confirmation-census-v2-{phase}-guard.json').read_text())==v['guard']
    changes={n:dict(before=original['source'].get(n),after=source.get(n)) for n in set(source)|set(original['source']) if source.get(n)!=original['source'].get(n)}
    assert set(changes)==set(prior['changed_files'])|set(component['changed_files'])
    for n,v in prior['changed_files'].items():assert changes[n]==v
    for n,v in component['changed_files'].items():assert changes[n]==v
    archive_review=json.loads((A/'logical-evidence-review.json').read_text())
    for n,v in archive_review['archives'].items():
        p=ROOT/n;assert not p.is_symlink() and p.stat().st_size==v['size'] and file_digest(p)==v['sha256']
    for name,key in [('plan.json','archive_manifest_sha256'),('completed.json','completion_receipt_sha256'),('journal.jsonl','journal_sha256')]:assert file_digest(A/name)==archive_review[key]
    completed=json.loads((A/'completed.json').read_text());archived={}
    assert completed['status']=='completed' and completed['no_checkpoint_bytes_lost']
    for receipt in completed['archives']:
        pinned=json.loads((A/(receipt['group']+'-archive.json')).read_text())
        assert pinned['verified_before_removal'] and pinned['members']==receipt['members']
        assert pinned['archive_sha256']==receipt['archive_sha256']
        archived.update(receipt['members'])
    assert archived==archive_review['archived_members']
    for n in archived:assert not (ROOT/n).exists()
    checked={}
    def logical(p,sha,size=None):
        n=str(p.relative_to(ROOT))
        if n in archived:
            assert archived[n]['sha256']==sha and (size is None or archived[n]['size']==size)
        else:
            if n not in checked:
                assert p.is_file() and not p.is_symlink()
                checked[n]=dict(size=p.stat().st_size,sha256=file_digest(p))
            assert checked[n]['sha256']==sha and (size is None or checked[n]['size']==size)
    reports={};artifacts={}
    for n,h in prior['original_report_hashes'].items():
        p=E/n;assert file_digest(p)==h;v=json.loads(p.read_text())
        assert v['source_sha256']==digest(original['source']) and v['runtime_sha256']==digest(runtime)
        reports[n]=h
        for k,s in v['artifacts'].items():
            q=p.parent.parent/k;logical(q,s);artifacts[str(q.relative_to(ROOT))]=s
    numeric=json.loads((E/'rank-numerical-v2-reviewed-001/meta/numerical-trace.json').read_text())
    for n,h in numeric['original_files'].items():logical(Path(n),h)
    closed={}
    for name,reviewname in [('tokenizer-wide-hybrid-002','campaign002-lock-failure-v2-review.json'),('tokenizer-wide-hybrid-003','campaign003-lock-failure-v2-review.json'),('tokenizer-wide-hybrid-004','campaign004-fresh16-failure-review-v2.json')]:
        report=json.loads((E/reviewname).read_text());root=ROOT/'research_runs'/name
        for n,v in report['files'].items():logical(root/n,v['sha256'],v['size'])
        if 'noncheckpoint_bytes_base64' in report:
            for n,b in report['noncheckpoint_bytes_base64'].items():assert (root/n).read_bytes()==base64.b64decode(b)
        closed[name]=dict(files=len(report['files']),review_sha256=file_digest(E/reviewname),inventory_sha256=digest(report['files']))
    old=json.loads((A/'plan.json').read_text());weights=dict(old['protected'])
    latest=json.loads((E/'campaign004-fresh16-failure-review-v2.json').read_text())
    for n,v in latest['files'].items():
        if n.endswith(('.pt','.pt.tmp')):weights['research_runs/tokenizer-wide-hybrid-004/'+n]=v
    for n,v in weights.items():logical(ROOT/n,v['sha256'],v['size'])
    observed={str(p.relative_to(ROOT)) for root in (ROOT/'research_runs').glob('tokenizer-*') if root.is_dir() for p in root.rglob('*') if p.is_file() and p.name.endswith(('.pt','.pt.tmp'))}
    assert observed==set(weights)
    eligible={};eligibility={}
    for name,reviewname in [('tokenizer-metadata-contention-baseline-001','metadata-contention-baseline-v1-review.json'),('tokenizer-metadata-contention-handoff-001','metadata-contention-handoff-v1-review.json'),('tokenizer-metadata-contention-trace-v3-001','metadata-contention-trace-v3-review.json')]:
        root=ROOT/'research_runs'/name;result=json.loads((root/'result.json').read_text())
        assert result['status']=='completed' and result['error'] is None and result['training'] is False
        candidates={n:v for n,v in weights.items() if n.startswith('research_runs/'+name+'/')}
        assert len(candidates)==1
        for n,v in candidates.items():
            assert (ROOT/n).name=='fixture.pt' and result['files'][str((ROOT/n).relative_to(root))]==v
            eligible[n]=v
        eligibility[name]=dict(result_sha256=file_digest(root/'result.json'),review_sha256=file_digest(E/reviewname),reason='Completed successful synthetic numerical/transport prerequisite; no model selection, fresh pair or failed checkpoint. Lossless archive only, never discard bytes.')
    required=storage_envelope(json.loads(PROJECTION.read_text()))['required_free_bytes']+128*1024
    free=shutil.disk_usage(ROOT).free;eligible_bytes=sum(v['size'] for v in eligible.values())
    assert source_fingerprint()==source and runtime_identity()==runtime
    report=dict(kind='wide_hybrid_source_applicability_review_v5',status='prerequisites_applicable_storage_blocked',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),before_source_sha256=digest(original['source']),after_source_sha256=digest(source),runtime_sha256=digest(runtime),changed_files=changes,changes_since_campaign004=component['changed_files'],original_report_hashes=reports,original_artifacts=artifacts,numerical_original_files_verified=len(numeric['original_files']),prior_applicability_sha256=file_digest(E/'checkpoint-time-handoff-source-applicability-v2-review.json'),component_review_sha256=file_digest(E/'confirmation-census-v2-review.json'),archive_review_sha256=file_digest(A/'logical-evidence-review.json'),archives=archive_review['archives'],archived_members_verified=len(archived),closed_evidence=closed,
        retention=dict(checkpoints=weights,eligible_lossless_archive=eligible,eligibility_evidence=eligibility,eligible_logical_bytes=eligible_bytes,remaining_protected_loose_files=len(weights)-len(eligible),eligible_irreversible_deletion=[],qualification='The previous compaction excluded these successful diagnostic weights for that operation. Current authorization permits lossless numerical-prerequisite archival; selected/fresh/failure weights remain loose. Archives stay protected and require manifest-aware future audits.'),
        storage=dict(required_free_bytes=required,observed_free_bytes=free,shortfall_bytes=max(0,required-free),eligible_logical_bytes=eligible_bytes,minimum_shortfall_even_if_eligible_archive_were_zero_bytes=max(0,required-free-eligible_bytes),reserve_bytes=8*1024**3),
        review=['Only new source changes since004 reuse the existing stable identity comparison in three fresh-startup modules. Model/optimizer/EMA/RNG/data math and frozen request semantics are unchanged.',
                'Original executed numerical/input reports, artifacts and logical file identities verified without reexecution; archived members bound through unchanged fully verified archives.',
                'Original completed numerical semantics remain applicable; this is explicit source applicability, not new numerical execution.',
                'All selected/fresh/failure weights remain protected; exact successful synthetic diagnostic weights may be losslessly archived under a new manifest.',
                'A future full-panel plan still requires prospective metadata and actual complete capacity gates; no old process or cleanup history may be reused as a new execution.'],
        numerical_rerun_required=False,input_reencoding_required=False,original_evidence_modified=False,new_campaign_freeze_required=True,training=False,cleanup_performed=False,closed_campaigns_reopened=False,holdout_consumed=False)
    with (E/'census-source-applicability-v2-review.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:report[k] for k in ('status','after_source_sha256','storage')}),flush=True)
