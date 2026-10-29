"""Fresh-only finite pilot/campaign identity freeze; never launches a worker.

Lossless numerical trace transport preserves every original byte outside this
new campaign. Full source/runtime, actual input, corrected size accounting and
the final decision procedure are bound before any resource pilot is possible.
"""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research import tokenizer_wide_hybrid_plan_v2 as p
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.protocol import digest,file_digest
from src.research.tokenizer_wide_hybrid_metadata import budgets
from src.research.tokenizer_wide_hybrid_metadata_writer import json_encoder

HERE=Path(__file__).resolve().parent
CAMPAIGN=ROOT/'research_runs/tokenizer-wide-hybrid-001'
IDENTITY=Path('/private/tmp/wide-hybrid-real-prerequisite-bind-001')


def encoded(value):return (json_encoder().encode(value)+'\n').encode('ascii')


def prepare():
    assert not CAMPAIGN.exists(), 'Fresh campaign only; no overwrite or retry'
    old_review=json.loads((HERE/'full-metadata-v2-review.json').read_text())
    for name in ('full-metadata-v2-guard.json','cpu-final-decision-v2-guard.json',
                 'cpu-final-decision-wrapper-v2-guard.json','rank-numerical-v2-review-guard.json'):
        assert json.loads((HERE/name).read_text())['status']=='finished'
    tests=(HERE/'cpu-final-decision-v2-tests.log').read_text()+(HERE/'cpu-final-decision-wrapper-v2-tests.log').read_text()
    assert '14 passed' in tests and '3 passed' in tests
    hashes=json.loads((HERE/'final-decision-wrapper-v2-source-hashes.json').read_text())
    assert all(file_digest(ROOT/n)==sha for n,sha in hashes.items())
    source,runtime=p.source_fingerprint(),p.runtime_identity()
    assert digest(source)==old_review['source_sha256'] and digest(runtime)==old_review['runtime_sha256']
    identity=json.loads((HERE/'actual-prerequisite-binding-v2-review.json').read_text())
    assert file_digest(IDENTITY/'evidence/meta/plan.json')==identity['plan_sha256']
    originals=dict(identity['original_files'])
    assert all(file_digest(Path(n))==sha for n,sha in originals.items())
    payloads={path.name:path.read_bytes() for path in (IDENTITY/'evidence/meta').iterdir() if path.name!='plan.json'}
    trace=payloads.pop('numerical-trace.json')
    compressed=gzip.compress(trace,compresslevel=9,mtime=0)
    assert gzip.decompress(compressed)==trace
    import hashlib
    compressed_sha=hashlib.sha256(compressed).hexdigest()
    numeric=json.loads(payloads['numerical.json'])
    assert numeric['artifacts']=={'meta/numerical-trace.json':hashlib.sha256(trace).hexdigest()}
    numeric['artifacts']={'meta/numerical-trace.json.gz':compressed_sha}
    numeric['artifact_encodings']={'meta/numerical-trace.json.gz':dict(compression='gzip',
        decoded_sha256=hashlib.sha256(trace).hexdigest(),decoded_bytes=len(trace))}
    payloads['numerical.json']=encoded(numeric)
    payloads['numerical-trace.json.gz']=compressed
    old_manifest=json.loads((HERE/'inference-acceptance-reusable-completed-prefix.json').read_text())
    fixture=Path(old_manifest['root'])
    assert {str(f.relative_to(fixture)):file_digest(f) for f in fixture.rglob('*') if f.is_file()}==old_manifest['files']

    def relocate(value):
        if isinstance(value,str):
            value=value.replace(str(IDENTITY),str(CAMPAIGN)).replace(str(fixture),str(CAMPAIGN))
            value=re.sub(r'final_(\d{6})_\d+\.pt',r'final_\1_9999999999999999999.pt',value)
            # Both pilot20 and quality1200 step fields have six digits.
            return value.replace('/raw.pt','/final_001200_9999999999999999999.pt')
        if isinstance(value,dict):return {relocate(k):relocate(v) for k,v in value.items()}
        if isinstance(value,list):return [relocate(v) for v in value]
        return value

    plan=relocate(json.loads((IDENTITY/'evidence/meta/plan.json').read_text()))
    plan['requests']=p.request_identity(p.request_options(CAMPAIGN))
    external=[p.PROTOCOL,p.PROJECTION,p.RECIPE,Path(p.request_options(CAMPAIGN)['pilots'][0]['config']['data']['path']),
              HERE/'review_final_campaign_v2.py',Path(__file__).resolve()]
    frozen={str(f.resolve()):file_digest(f) for f in external}
    frozen.update({str(CAMPAIGN/'evidence/meta'/n):hashlib.sha256(b).hexdigest() for n,b in payloads.items()})
    plan['files']=[dict(path=n,sha256=sha) for n,sha in sorted(frozen.items())]
    plan['numerical_ref']['sha256']=hashlib.sha256(payloads['numerical.json']).hexdigest()
    payloads['plan.json']=encoded(plan)
    plan_sha=hashlib.sha256(payloads['plan.json']).hexdigest()
    frozen[str(CAMPAIGN/'evidence/meta/plan.json')]=plan_sha
    initial={'meta/'+n:dict(size=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in payloads.items()}
    spec=importlib.util.spec_from_file_location('metadata_envelope',HERE/'audit_full_metadata_v2.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    assert file_digest(HERE/'audit_full_metadata_v2.py')==old_review['script_sha256']
    removed={'meta/source.txt','meta/corpus.txt','meta/numerical.trace','meta/numerical-trace.json'}
    controllers={f'inference/{s}/controller.json':dict(size=99999,sha256='0'*64) for s in (3511,3512,3513)}
    counts=dict(frozen_maps=0,full_inventories=0)
    def update(value):
        if isinstance(value,dict):
            if 'frozen_files' in value:
                value['frozen_files']=dict(frozen);counts['frozen_maps']+=1
            if 'meta/plan.json' in value and isinstance(value['meta/plan.json'],dict):
                assert set(value['meta/plan.json'])=={'size','sha256'}
                for n in removed:value.pop(n,None)
                value.update(copy.deepcopy(initial));value.update(controllers);counts['full_inventories']+=1
            return {k:update(v) if k!='frozen_files' else v for k,v in value.items()}
        if isinstance(value,list):return [update(v) for v in value]
        return value
    ledger=dict(old_review['ledger'])
    for name in list(ledger):
        if name.startswith('evidence/meta/'):del ledger[name]
    for path in (fixture/'evidence/meta').glob('*.json'):
        if 'meta/'+path.name in initial:continue
        value=update(relocate(json.loads(path.read_text())))
        ledger['evidence/meta/'+path.name]=audit.size(value)+audit.padding(value)
    for name,row in initial.items():ledger['evidence/'+name]=row['size']
    # Correct raw filename growth in every non-meta receipt too. The initial
    # ledger already covered timestamp growth for fixtures using final_*.pt.
    for name in list(ledger):
        path=fixture/name
        if not name.startswith('evidence/meta/') and path.is_file() and name.endswith('.json'):
            ledger[name]+=path.read_text().count('/raw.pt')*29
    groups={g:0 for g in budgets()}
    for name,count in ledger.items():groups[audit.classify(name)]+=count
    control_overlap=max(v for n,v in ledger.items() if audit.classify(n)=='control')
    groups['control']+=control_overlap
    assert all(groups[g]<=cap for g,cap in budgets().items()), 'Corrected metadata envelope does not fit'
    assert old_review['conservative_file_count']<audit.MAX_FILES
    external_final_review=128*1024
    required=p.storage_envelope(json.loads(p.PROJECTION.read_text()))['required_free_bytes']+external_final_review
    free=shutil.disk_usage(ROOT).free
    assert free>=required
    assert p.source_fingerprint()==source and p.runtime_identity()==runtime
    assert all(file_digest(Path(n))==sha for n,sha in originals.items())
    # All calculation/verification above precedes the one-way fresh freeze.
    CAMPAIGN.mkdir();meta=CAMPAIGN/'evidence/meta';meta.mkdir(parents=True)
    for name,raw in payloads.items():
        with (meta/name).open('xb') as stream:stream.write(raw)
    bound=p.bind_campaign(meta/'plan.json',plan_sha)
    assert bound['numerical_review_bound'] and bound['training_input_bound']
    assert bound['dispatch_authorized'] is False and bound['measured_pilots_verified'] is False
    assert gzip.decompress((meta/'numerical-trace.json.gz').read_bytes())==trace
    assert all(file_digest(Path(n))==sha for n,sha in originals.items())
    report=dict(kind='wide_hybrid_finite_campaign_freeze_review_v2',status='frozen_pending_measured_resource_pilots',
        campaign=str(CAMPAIGN),plan_sha256=plan_sha,plan_bytes=len(payloads['plan.json']),
        script_sha256=file_digest(Path(__file__)),final_procedure_sha256=file_digest(HERE/'review_final_campaign_v2.py'),
        source_sha256=digest(source),runtime_sha256=digest(runtime),source_runtime_unchanged=True,
        original_files=originals,original_files_unchanged=True,trace_original_bytes=len(trace),
        trace_compressed_bytes=len(compressed),trace_decoded_sha256=hashlib.sha256(trace).hexdigest(),
        trace_compressed_sha256=compressed_sha,lossless_roundtrip_verified=True,
        numerical_ref=plan['numerical_ref'],input_ref=plan['input_ref'],
        supersedes_size_claim=old_review['script_sha256'],
        corrections=['All552 raw.pt references in meta now reserve production timestamp names.',
            'Both preparation and final decision sources are additional frozen files.',
            'Numerical trace is lossless gzip transport; original report/trace and all checkpoints remain untouched.'],
        groups={g:dict(bound_bytes=n,cap_bytes=budgets()[g],remaining=budgets()[g]-n) for g,n in groups.items()},
        meta_occurrences=counts,ledger=ledger,conservative_file_count=old_review['conservative_file_count'],
        final_review_external_allowance_bytes=external_final_review,required_free_bytes=required,observed_free_bytes=free,
        tests=dict(new_distinct_tests=17,training_workers=0,full_fixture_rebuilds=0),
        pilot_requests=3,conditional_quality_options=108,quality_trajectories=24,
        measured_pilots_verified=False,quality_launch_authorized=False,training_launched=False,
        size_scope='Fixed successful producer schemas with finite numeric widths, exact real configs, five/two metrics, '
            'full resource samples and worst token text. Per-non-meta-document4KiB covers remaining fixed strings '
            'and extra absolute bindings; actual producers and watchdogs still enforce all caps.',
        next_action='Inspect processes; acquire both original worker locks; run frozen pilots serially through '
            'run_pilot_once. Preserve failures. Only audited pilot boundary and quality readiness may authorize quality.')
    with (HERE/'finite-campaign-freeze-v2-review.json').open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    return report


if __name__=='__main__':
    with CpuPrerequisiteGuard(HERE/'finite-campaign-freeze-v2-guard.json',ROOT,timeout_seconds=180):
        report=prepare()
    print(json.dumps({k:report[k] for k in ('status','campaign','plan_sha256','trace_compressed_bytes','required_free_bytes')}))
