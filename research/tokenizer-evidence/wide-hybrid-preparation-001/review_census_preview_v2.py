"""Check lossless preview transport without creating or dispatching a campaign."""
import base64,gzip,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.protocol import digest,file_digest
from src.research.runner import source_fingerprint
from src.research.tokenizer_study import runtime_identity
from src.research.tokenizer_wide_hybrid_retention_audit import numerical_review
from src.research.tokenizer_wide_hybrid_timing import token_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_driver import session_lock
E=Path(__file__).resolve().parent;OUT=E/'census-campaign005-preview'
with session_lock(ROOT/'research_state/tokenizer-worker-lock'),CpuPrerequisiteGuard(E/'census-preview-transport-review-v2-guard.json',ROOT,timeout_seconds=180):
    preview=json.loads((E/'census-campaign005-preview-review.json').read_text())
    guard=json.loads((E/'census-campaign005-preview-guard.json').read_text())
    assert guard['status']=='finished' and guard['reason'] is None
    assert preview['status']=='preview_only_not_frozen' and not preview['campaign_created'] and not Path(preview['campaign']).exists()
    plan=json.loads((OUT/'meta/plan.json').read_text());assert file_digest(OUT/'meta/plan.json')==preview['plan_sha256']
    source=source_fingerprint();runtime=runtime_identity();assert digest(source)==preview['source_sha256'] and digest(runtime)==preview['runtime_sha256']
    trace=json.loads(gzip.decompress((OUT/'meta/source-applicability.json.gz').read_bytes()))
    assert trace['not_a_new_numerical_or_input_run'] and trace['archived_originals_preserved']
    for n,v in trace['originals'].items():
        assert (ROOT/n).read_bytes()==base64.b64decode(v['bytes_base64']) and file_digest(ROOT/n)==v['sha256']
    context=dict(source_sha256=digest(source),runtime_sha256=digest(runtime),corpus_sha256=plan['panel_design']['corpus_sha256']) if 'corpus_sha256' in plan['panel_design'] else None
    numerical=json.loads((OUT/'meta/numerical.json').read_text())
    context=dict(source_sha256=digest(source),runtime_sha256=digest(runtime),corpus_sha256=numerical['corpus_sha256'])
    bound=numerical_review(OUT,plan['numerical_ref'],context)
    original=json.loads((E/'training-input-v2-002/meta/training-input.json').read_text())
    value=json.loads((OUT/'meta/training-input.json').read_text())
    assert file_digest(OUT/plan['input_ref']['path'])==plan['input_ref']['sha256']
    assert value['source_sha256']==context['source_sha256'] and value['runtime_sha256']==context['runtime_sha256'] and value['corpus_sha256']==context['corpus_sha256']
    assert value['status']=='passed' and value['holdout_consumed'] is False and value['split']=='train' and value['token_count']==96
    assert value['training_token_sha256']==original['training_token_sha256'] and value['tokenizer_sha256']==original['tokenizer_sha256']
    for n,h in value['artifacts'].items():assert file_digest(OUT/n)==h
    ids=json.loads((OUT/value['input_ids_ref']['path']).read_text());assert ids['split']=='train' and len(ids['ids'])==96 and token_digest(ids['ids'])==value['training_token_sha256']
    assert source_fingerprint()==source and runtime_identity()==runtime and not Path(preview['campaign']).exists()
    report=dict(status='prospective_transport_verified_not_frozen',preview_sha256=file_digest(E/'census-campaign005-preview-review.json'),plan_sha256=preview['plan_sha256'],source_sha256=digest(source),runtime_sha256=digest(runtime),originals_lossless_verified=len(trace['originals']),numerical_artifacts_bound=bound,training_ids_unchanged=True,metadata_fits=preview['metadata_fits'],metadata_group=preview['groups']['evidence/meta'],capacity_fits=preview['capacity_fits'],shortfall_bytes=preview['shortfall_bytes'],campaign_created=False,plan_frozen=False,training=False,holdout_consumed=False,numerical_reexecuted=False,limitation='Verifies lossless prerequisite transport and prospective byte ledger only. Full bind_campaign cannot run at this preview location, and actual complete capacity still blocks fresh campaign publication.')
    with (E/'census-preview-transport-review-v2.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:report[k] for k in ('status','metadata_fits','capacity_fits','shortfall_bytes')}),flush=True)
