"""Prepare lossless prerequisite transport only: no campaign plan or dispatch."""
import base64,gzip,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.protocol import digest,file_digest
from src.research.tokenizer_wide_hybrid_plan_v2 import source_fingerprint,runtime_identity
from src.research.tokenizer_wide_hybrid_retention_audit import numerical_review
from src.research.tokenizer_wide_hybrid_metadata_writer import json_encoder
from src.research.tokenizer_wide_hybrid_timing import token_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_driver import session_lock
E=Path(__file__).resolve().parent
OUT=E/'time-handoff-prerequisite-transport-v2'
def encoded(value):return (json_encoder().encode(value)+'\n').encode('ascii')
def sha(value):return hashlib.sha256(value).hexdigest()

with session_lock(ROOT/'research_state/tokenizer-worker-lock'),CpuPrerequisiteGuard(E/'time-handoff-prerequisite-transport-v2-guard.json',ROOT,timeout_seconds=180):
    assert not OUT.exists()
    review_path=E/'checkpoint-time-handoff-source-applicability-v2-review.json'
    review=json.loads(review_path.read_text());source=source_fingerprint();runtime=runtime_identity()
    assert digest(source)==review['after_source_sha256'] and digest(runtime)==review['runtime_sha256']
    numeric=E/'rank-numerical-v2-reviewed-001/meta/numerical.json'
    inputs=E/'training-input-v2-002/meta/training-input.json'
    originals={str(p.relative_to(ROOT)):dict(sha256=file_digest(p),bytes_base64=base64.b64encode(p.read_bytes()).decode()) for p in (review_path,numeric,inputs)}
    trace=encoded(dict(kind='explicit_source_applicability_transport_v2',executed_source_sha256=review['before_source_sha256'],applicable_source_sha256=digest(source),originals=originals,not_a_new_numerical_or_input_run=True))
    packed=gzip.compress(trace,compresslevel=9,mtime=0)
    assert gzip.decompress(packed)==trace
    payloads={'source-applicability.json.gz':packed}
    app_ref=dict(path='meta/source-applicability.json.gz',sha256=sha(packed))
    for name,original in [('numerical.json',numeric),('training-input.json',inputs)]:
        value=json.loads(original.read_bytes())
        assert value['source_sha256']==review['before_source_sha256']
        for n,h in value['artifacts'].items():
            artifact=original.parent.parent/n;assert file_digest(artifact)==h
            relative=Path(n);assert relative.parent==Path('meta')
            raw=artifact.read_bytes()
            if relative.suffix=='.json' and name=='numerical.json':
                compressed=gzip.compress(raw,compresslevel=9,mtime=0);new=n+'.gz'
                payloads[Path(new).name]=compressed
                value['artifacts']={new:sha(compressed)}
                value.setdefault('artifact_encodings',{})[new]=dict(compression='gzip',decoded_sha256=sha(raw),decoded_bytes=len(raw))
                assert gzip.decompress(compressed)==raw
            else:
                if relative.name in payloads:assert payloads[relative.name]==raw
                payloads[relative.name]=raw
        value['source_sha256']=digest(source)
        value['artifacts'][app_ref['path']]=app_ref['sha256']
        value['source_applicability']=dict(kind='reviewed_carry_forward_not_rerun',executed_source_sha256=review['before_source_sha256'],original_report_sha256=file_digest(original),review=app_ref)
        value.setdefault('artifact_encodings',{})[app_ref['path']]=dict(compression='gzip',decoded_sha256=sha(trace),decoded_bytes=len(trace))
        payloads[name]=encoded(value)
    assert sum(map(len,payloads.values()))<2*1024**2
    meta=OUT/'meta';meta.mkdir(parents=True)
    for n,b in payloads.items():
        with (meta/n).open('xb') as f:f.write(b)
    context=dict(source_sha256=digest(source),runtime_sha256=digest(runtime),corpus_sha256=json.loads(numeric.read_text())['corpus_sha256'])
    numeric_ref=dict(path='meta/numerical.json',sha256=sha(payloads['numerical.json']))
    bound=numerical_review(OUT,numeric_ref,context)
    value=json.loads(payloads['training-input.json']);original_input=json.loads(inputs.read_text())
    assert value['source_sha256']==context['source_sha256'] and value['runtime_sha256']==context['runtime_sha256']
    assert value['corpus_sha256']==context['corpus_sha256'] and value['kind']=='wide_hybrid_training_input_v2' and value['status']=='passed'
    assert value['holdout_consumed'] is False and value['split']=='train' and value['token_count']==96
    assert value['tokenizer_sha256']==original_input['tokenizer_sha256'] and value['training_token_sha256']==original_input['training_token_sha256']
    for n,h in value['artifacts'].items():assert file_digest(OUT/n)==h
    ids=json.loads((OUT/value['input_ids_ref']['path']).read_text())
    assert ids['split']=='train' and len(ids['ids'])==96 and token_digest(ids['ids'])==value['training_token_sha256']
    for n,v in originals.items():assert (ROOT/n).read_bytes()==base64.b64decode(v['bytes_base64']) and file_digest(ROOT/n)==v['sha256']
    for n,b in payloads.items():assert (meta/n).read_bytes()==b
    assert source_fingerprint()==source and runtime_identity()==runtime
    report=dict(status='lossless_prerequisite_transport_prepared_not_campaign_frozen',source_sha256=digest(source),executed_source_sha256=review['before_source_sha256'],runtime_sha256=digest(runtime),
        output=str(OUT.relative_to(ROOT)),files={n:dict(size=len(b),sha256=sha(b)) for n,b in payloads.items()},total_bytes=sum(map(len,payloads.values())),
        original_reports_unchanged=True,lossless_originals_verified=True,numerical_artifacts_bound=bound,training_ids_unchanged=True,
        current_source_applicability_sha256=file_digest(review_path),script_sha256=file_digest(Path(__file__)),campaign_created=False,plan_frozen=False,training=False,
        limitation='Transport checks preserve original executed-source identity; no numerical reexecution. Complete campaign metadata high-water and capacity gate remain separate before any freeze or dispatch.')
    with (E/'time-handoff-prerequisite-transport-v2-review.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=report['status'],files=len(payloads),total_bytes=report['total_bytes'])),flush=True)
