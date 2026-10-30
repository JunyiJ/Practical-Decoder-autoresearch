"""Fresh campaign004 after elapsed-time handoff and verified archival, with explicit source applicability.

Reuse only reviewed numerical/input evidence, not prior scientific trajectories,
process receipts or cleanup history. Preserve original byte streams losslessly.
"""
import base64,copy,gzip,hashlib,importlib.util,json,re,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research import tokenizer_wide_hybrid_plan_v2 as p
from src.research.protocol import digest,file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_metadata_writer import json_encoder
from src.research.tokenizer_wide_hybrid_metadata import budgets,classify,MAX_FILES
HERE=Path(__file__).resolve().parent
CAMPAIGN=ROOT/'research_runs/tokenizer-wide-hybrid-004'

def encoded(v):return (json_encoder().encode(v)+'\n').encode('ascii')
def sha(v):return hashlib.sha256(v).hexdigest()

def prepare():
    assert not CAMPAIGN.exists(), 'Fresh campaign only; no retry or overwrite'
    archive=HERE/'finite-campaign-v2-frozen/meta'
    original_plan=json.loads((archive/'plan.json').read_text())
    prior=json.loads((HERE/'finite-campaign-freeze-v2-review.json').read_text())
    assert file_digest(archive/'plan.json')==prior['plan_sha256']
    application_path=HERE/'checkpoint-time-handoff-source-applicability-v2-review.json'
    application=json.loads(application_path.read_text())
    source,runtime=p.source_fingerprint(),p.runtime_identity()
    assert digest(source)==application['after_source_sha256']
    assert digest(original_plan['source'])==application['before_source_sha256']
    assert digest(runtime)==application['runtime_sha256'] and runtime==original_plan['runtime']
    actual_changes={n:dict(before=original_plan['source'].get(n),after=source.get(n))
                    for n in set(source)|set(original_plan['source']) if source.get(n)!=original_plan['source'].get(n)}
    assert actual_changes==application['changed_files']
    assert set(actual_changes)=={'src/research/tokenizer_wide_hybrid_quality_readiness_v2.py',
                                'src/research/tokenizer_wide_hybrid_quality_entry_v2.py',
                                'src/research/tokenizer_wide_hybrid_checkpoint_writer.py'}
    component=json.loads((HERE/'checkpoint-time-handoff-v2-component-review.json').read_text())
    for n,s in component['source_hashes'].items():assert file_digest(ROOT/n)==s
    for n,v in component['tests'].items():
        assert file_digest(HERE/(n+'-tests.log'))==v['log_sha256']
        assert json.loads((HERE/(n+'-guard.json')).read_text())==v['guard']
    archival_path=HERE.parent/'checkpoint-compaction-001/logical-evidence-review.json'
    archival=json.loads(archival_path.read_text())
    assert archival['status']=='logical_original_evidence_preserved_in_verified_archives'
    assert json.loads(archival_path.with_name('logical-evidence-review-guard.json').read_text())['status']=='finished'
    for n,v in archival['archives'].items():
        assert (ROOT/n).stat().st_size==v['size'] and file_digest(ROOT/n)==v['sha256']
    for n,v in archival['archived_members'].items():assert not (ROOT/n).exists()
    completion=archival_path.with_name('completed.json')
    assert file_digest(completion)==archival['completion_receipt_sha256']
    assert file_digest(archival_path.with_name('journal.jsonl'))==archival['journal_sha256']
    assert file_digest(archival_path.with_name('plan.json'))==archival['archive_manifest_sha256']
    originals={str(f):file_digest(f) for f in archive.iterdir()}
    assert all(prior['original_files_unchanged'] for _ in [0])
    payloads={f.name:f.read_bytes() for f in archive.iterdir() if f.name!='plan.json'}
    numerical_original=HERE/'rank-numerical-v2-reviewed-001/meta/numerical.json'
    input_original=HERE/'training-input-v2-002/meta/training-input.json'
    decoded=gzip.decompress(payloads['numerical-trace.json.gz'])
    assert sha(decoded)==sha((numerical_original.parent/'numerical-trace.json').read_bytes())
    for f in (numerical_original,input_original):
        value=json.loads(f.read_bytes());assert value['source_sha256']==application['before_source_sha256']
        assert value['runtime_sha256']==application['runtime_sha256']
        assert all(file_digest(f.parent.parent/n)==s for n,s in value['artifacts'].items())
        originals[str(f)]=file_digest(f)
    # The original reports and applicability review are losslessly embedded in
    # one small hash-bound trace, avoiding repeated extra frozen-file maps.
    carried={str(f.relative_to(ROOT)):dict(sha256=file_digest(f),bytes_base64=base64.b64encode(f.read_bytes()).decode())
             for f in (application_path,numerical_original,input_original,archival_path)}
    trace=encoded(dict(kind='explicit_source_applicability_transport_v2',
        executed_source_sha256=application['before_source_sha256'],applicable_source_sha256=digest(source),
        originals=carried,not_a_new_numerical_or_input_run=True,archived_originals_preserved=True))
    packed=gzip.compress(trace,compresslevel=9,mtime=0);assert gzip.decompress(packed)==trace
    payloads['source-applicability.json.gz']=packed
    applicability=dict(path='meta/source-applicability.json.gz',sha256=sha(packed))
    for name,original in [('numerical.json',numerical_original),('training-input.json',input_original)]:
        report=json.loads(payloads[name]);report['source_sha256']=digest(source)
        report['artifacts'][applicability['path']]=applicability['sha256']
        report['source_applicability']=dict(kind='reviewed_carry_forward_not_rerun',
            executed_source_sha256=application['before_source_sha256'],original_report_sha256=file_digest(original),
            review=applicability)
        report.setdefault('artifact_encodings',{})[applicability['path']]=dict(compression='gzip',decoded_sha256=sha(trace),decoded_bytes=len(trace))
        payloads[name]=encoded(report)
    plan=copy.deepcopy(original_plan);plan.update(output=str(CAMPAIGN),source=source,runtime=runtime,
        requests=p.request_identity(p.request_options(CAMPAIGN)))
    external=[p.PROTOCOL,p.PROJECTION,p.RECIPE,Path(p.request_options(CAMPAIGN)['pilots'][0]['config']['data']['path']),
              HERE/'review_final_campaign_v2.py',Path(__file__).resolve()]
    bindings={str(f.resolve()):file_digest(f) for f in external}
    bindings.update({str(CAMPAIGN/'evidence/meta'/n):sha(b) for n,b in payloads.items()})
    plan['files']=[dict(path=n,sha256=s) for n,s in sorted(bindings.items())]
    for key,name in [('numerical_ref','numerical.json'),('input_ref','training-input.json')]:
        plan[key]=dict(path='meta/'+name,sha256=sha(payloads[name]))
    payloads['plan.json']=encoded(plan);plan_sha=sha(payloads['plan.json'])
    bindings[str(CAMPAIGN/'evidence/meta/plan.json')]=plan_sha
    # Reuse the immutable completed topology only for byte accounting. No
    # transformed synthetic document is ever published into this campaign.
    manifest=json.loads((HERE/'inference-acceptance-reusable-completed-prefix.json').read_text());fixture=Path(manifest['root'])
    assert {str(f.relative_to(fixture)):file_digest(f) for f in fixture.rglob('*') if f.is_file()}==manifest['files']
    spec=importlib.util.spec_from_file_location('size_audit',HERE/'audit_full_metadata_v2.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    initial={'meta/'+n:dict(size=len(b),sha256=sha(b)) for n,b in payloads.items()}
    removed={'meta/source.txt','meta/corpus.txt','meta/numerical.trace','meta/numerical-trace.json'}
    controllers={f'inference/{s}/controller.json':dict(size=99999,sha256='0'*64) for s in (3511,3512,3513)}
    def transform(v):
        if isinstance(v,str):
            v=v.replace(str(fixture),str(CAMPAIGN))
            v=re.sub(r'final_(\d{6})_\d+\.pt',r'final_\1_9999999999999999999.pt',v)
            return v.replace('/raw.pt','/final_001200_9999999999999999999.pt')
        if isinstance(v,dict):
            if 'frozen_files' in v:v['frozen_files']=dict(bindings)
            if 'meta/plan.json' in v and isinstance(v['meta/plan.json'],dict):
                for n in removed:v.pop(n,None)
                v.update(copy.deepcopy(initial));v.update(controllers)
            return {transform(k):transform(x) for k,x in v.items()}
        if isinstance(v,list):return [transform(x) for x in v]
        return v
    ledger={n:s for n,s in prior['ledger'].items() if not n.startswith('evidence/meta/')}
    for f in (fixture/'evidence/meta').glob('*.json'):
        if 'meta/'+f.name in initial:continue
        value=transform(json.loads(f.read_text()))
        ledger['evidence/meta/'+f.name]=audit.size(value)+audit.padding(value)
    ledger.update({'evidence/'+n:v['size'] for n,v in initial.items()})
    groups={g:0 for g in budgets()}
    for n,s in ledger.items():groups[classify(n)]+=s
    groups['control']+=max(s for n,s in ledger.items() if classify(n)=='control')
    assert all(groups[g]<=limit for g,limit in budgets().items()), {g:(groups[g],limit) for g,limit in budgets().items() if groups[g]>limit}
    count=prior['conservative_file_count']+1;assert count<MAX_FILES
    required=p.storage_envelope(json.loads(p.PROJECTION.read_text()))['required_free_bytes']+128*1024
    assert shutil.disk_usage(ROOT).free>=required
    assert p.source_fingerprint()==source and p.runtime_identity()==runtime
    assert all(file_digest(Path(n))==s for n,s in originals.items())
    # Fresh publication begins only after all prerequisite and size reviews.
    meta=CAMPAIGN/'evidence/meta';meta.mkdir(parents=True)
    for n,b in payloads.items():
        with (meta/n).open('xb') as f:f.write(b)
    bound=p.bind_campaign(meta/'plan.json',plan_sha)
    assert bound['numerical_review_bound'] and bound['training_input_bound'] and not bound['dispatch_authorized']
    assert gzip.decompress((meta/'source-applicability.json.gz').read_bytes())==trace
    assert all(file_digest(Path(n))==s for n,s in originals.items())
    result=dict(kind='wide_hybrid_replacement_finite_freeze_v2',status='frozen_pending_new_resource_pilots',
        campaign=str(CAMPAIGN),plan_sha256=plan_sha,source_sha256=digest(source),runtime_sha256=digest(runtime),
        original_source_sha256=application['before_source_sha256'],applicability_ref=applicability,
        applicability_review_sha256=file_digest(application_path),script_sha256=file_digest(Path(__file__)),
        original_files=originals,original_files_unchanged=True,lossless_carry_forward_verified=True,
        numerical_or_input_rerun=False,old_process_or_cleanup_receipts_reused=False,
        ledger=ledger,groups={g:dict(bound_bytes=s,cap_bytes=budgets()[g],remaining=budgets()[g]-s) for g,s in groups.items()},
        conservative_file_count=count,required_free_bytes=required,observed_free_bytes=shutil.disk_usage(ROOT).free,
        pilot_requests=3,quality_trajectories=24,conditional_quality_options=108,
        closed_campaigns_reopened=False,archived_evidence_preserved=True,
        archival_review_sha256=file_digest(archival_path),
        prior_quality_results_not_pooled=True,screen_seed_reused_not_independent=True,training_launched=False,quality_launch_authorized=False,
        qualification='Original numerical/input executions retain their original source identity; explicit control-census/checkpoint-scheduling applicability supports the new frozen source. New full panel uses unchanged grid/seeds; prior incomplete screening results are not pooled or selected. Metadata envelope retains previous fixed-schema padding and runtime producer enforcement.')
    out=HERE/'campaign004-freeze-v2-review.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    saved=HERE/'campaign004-frozen';saved.mkdir();shutil.copytree(meta,saved/'meta')
    return result

if __name__=='__main__':
    from src.research.tokenizer_wide_hybrid_driver import session_lock
    with session_lock(ROOT/'research_state/tokenizer-worker-lock'),CpuPrerequisiteGuard(HERE/'campaign004-freeze-v2-guard.json',ROOT,timeout_seconds=180):result=prepare()
    print(json.dumps({k:result[k] for k in ('status','campaign','plan_sha256','conservative_file_count')}))
