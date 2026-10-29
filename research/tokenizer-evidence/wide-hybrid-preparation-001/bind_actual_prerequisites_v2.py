"""One fresh, real-input identity-gate check; no worker or campaign launch.

The disposable identity plan exercises the production frozen-identity reader.
It does not establish future full-panel metadata fit or measured pilot costs.
"""
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research import tokenizer_wide_hybrid_plan_v2 as p
from src.research.protocol import digest,file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_metadata_writer import json_encoder

HERE=Path(__file__).resolve().parent


def run():
    root=Path('/private/tmp/wide-hybrid-real-prerequisite-bind-001')
    root.mkdir(exist_ok=False);meta=root/'evidence/meta';meta.mkdir(parents=True)
    numerical=HERE/'rank-numerical-v2-reviewed-001'
    inputs=HERE/'training-input-v2-002'
    assert json.loads((HERE/'rank-numerical-v2-review-guard.json').read_text())['status']=='finished'
    assert json.loads((inputs/'guard.json').read_text())['status']=='finished'
    originals={}
    for package in (numerical,inputs):
        for path in (package/'meta').iterdir():
            target=meta/path.name;assert not target.exists()
            originals[str(path)]=file_digest(path);shutil.copyfile(path,target)
            assert file_digest(target)==originals[str(path)]
    shutil.copyfile(p.PROJECTION,meta/'projection.json')
    source,runtime=p.source_fingerprint(),p.runtime_identity();options=p.request_options(root)
    inp=json.loads((meta/'training-input.json').read_text())
    tokenizer=options['pilots'][0]['config']['model']['output_bias_prior']['tokenizer']
    token_sha=inp['training_token_sha256']
    files=[p.PROTOCOL,p.PROJECTION,p.RECIPE,Path(options['pilots'][0]['config']['data']['path']),*sorted(meta.iterdir())]
    plan=dict(kind=p.KIND,status='frozen',output=str(root),source=source,runtime=runtime,
        protocol_sha256=file_digest(p.PROTOCOL),projection_sha256=file_digest(p.PROJECTION),
        panel_design=p.design(),layout=p.layout_contract(),
        storage_contract=p.storage_envelope(json.loads(p.PROJECTION.read_text())),
        pilot_contract=dict(p.PILOT_V1,version=2),quality_contract=p.RESOURCE_LIMITS,
        cache_contract=p.CACHE_CONTRACT,requests=p.request_identity(options),tokenizer=tokenizer,
        training_token_sha256=token_sha,
        inference_contract=dict(p.SERVING_V1,version=2,tokenizer_sha256=digest(tokenizer),token_sha256=token_sha),
        numerical_ref=dict(path='meta/numerical.json',sha256=file_digest(meta/'numerical.json')),
        input_ref=dict(path='meta/training-input.json',sha256=file_digest(meta/'training-input.json')),
        files=[dict(path=str(path.resolve()),sha256=file_digest(path)) for path in files])
    plan_path=meta/'plan.json'
    with plan_path.open('x') as stream:stream.write(json_encoder().encode(plan)+'\n')
    bound=p.bind_campaign(plan_path,file_digest(plan_path))
    assert bound['numerical_review_bound'] and bound['training_input_bound']
    assert all(bound[k] is False for k in ('current_storage_fit_verified','measured_pilots_verified',
                                         'producer_enforcement_verified','dispatch_authorized'))
    assert all(file_digest(Path(n))==s for n,s in originals.items())
    assert p.source_fingerprint()==source and p.runtime_identity()==runtime
    report=dict(kind='wide_hybrid_actual_prerequisite_binding_review_v2',
        status='real_numerical_and_input_identity_bound_no_dispatch',
        root=str(root),plan_sha256=file_digest(plan_path),script_sha256=file_digest(Path(__file__)),
        numerical_ref=plan['numerical_ref'],input_ref=plan['input_ref'],original_files=originals,
        plan_bytes=plan_path.stat().st_size,initial_metadata_bytes=bound['metadata_bytes'],
        frozen_file_count=len(bound['files']),pilot_requests=len(bound['requests']['pilots']),
        conditional_quality_requests=sum(map(len,bound['requests']['quality'].values())),
        request_identity=plan['requests'],source_sha256=digest(source),runtime_sha256=digest(runtime),
        production_gate_unmodified=True,mocks_used=False,model_loaded=False,
        current_storage_fit_verified=False,full_panel_metadata_fit_proven=False,
        measured_pilots_verified=False,dispatch_authorized=False,
        qualifications=['This frozen identity is only for the disposable check directory; it is not the pilot/quality campaign freeze.',
            'Complete future metadata/inventory/transient growth and actual pilot costs remain unresolved.',
            'No quality request, training worker, model or holdout evaluation was created.'])
    with (HERE/'actual-prerequisite-binding-v2-review.json').open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({k:report[k] for k in ('status','plan_bytes','initial_metadata_bytes','frozen_file_count')}))


if __name__=='__main__':
    with CpuPrerequisiteGuard(HERE/'actual-prerequisite-binding-v2-guard.json',ROOT,timeout_seconds=180):run()
