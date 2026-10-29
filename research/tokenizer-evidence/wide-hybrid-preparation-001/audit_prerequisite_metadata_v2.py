"""Read-only measured census plus explicitly incomplete real-field projection.

Never creates a campaign, alters retained bytes, or certifies scientific input.
The numerical artifact count/size sensitivity is accounting, not passing data.
"""
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.protocol import digest,file_digest
from src.research import tokenizer_wide_hybrid_plan_v2 as plan
from src.research.tokenizer_wide_hybrid_metadata import snapshot,budgets,MAX_FILES
from src.research.tokenizer_wide_hybrid_metadata_writer import json_encoder

EVIDENCE=Path(__file__).resolve().parent


def size(value):
    return len((json_encoder().encode(value)+'\n').encode('ascii'))


def audit():
    manifest_path=EVIDENCE/'inference-acceptance-reusable-completed-prefix.json'
    manifest=json.loads(manifest_path.read_text());root=Path(manifest['root'])
    def verify():
        assert {str(p.relative_to(root)):file_digest(p) for p in root.rglob('*')
                if p.is_file()}==manifest['files']
    verify();census=snapshot(root)
    source,runtime=plan.source_fingerprint(),plan.runtime_identity()
    package=EVIDENCE/'training-input-v2-002'
    inp=json.loads((package/'meta/training-input.json').read_text())
    assert inp['source_sha256']==digest(source) and inp['runtime_sha256']==digest(runtime)
    assert json.loads((package/'guard.json').read_text())['status']=='finished'
    actual_inputs={str(p.relative_to(package)):file_digest(p) for p in (package/'meta').iterdir()}
    assert all(actual_inputs[name]==sha for name,sha in inp['artifacts'].items())
    options=plan.request_options(root)
    tokenizer=options['pilots'][0]['config']['model']['output_bias_prior']['tokenizer']
    assert digest(tokenizer)==inp['tokenizer_sha256']
    # Use the longer retained-fixture root for path-size sensitivity. This is
    # not an output path choice or a generated frozen plan.
    known={str(p.resolve()):file_digest(p) for p in (plan.PROTOCOL,plan.PROJECTION,plan.RECIPE,
                  Path(options['pilots'][0]['config']['data']['path']))}
    known.update({str(root/'evidence'/name):sha for name,sha in actual_inputs.items()})
    draft=dict(kind=plan.KIND,status='INCOMPLETE_ACCOUNTING_ONLY',output=str(root),
        source=source,runtime=runtime,protocol_sha256=file_digest(plan.PROTOCOL),
        projection_sha256=file_digest(plan.PROJECTION),panel_design=plan.design(),
        layout=plan.layout_contract(),storage_contract=plan.storage_envelope(json.loads(plan.PROJECTION.read_text())),
        pilot_contract=dict(plan.PILOT_V1,version=2),quality_contract=plan.RESOURCE_LIMITS,
        cache_contract=plan.CACHE_CONTRACT,requests=plan.request_identity(options),tokenizer=tokenizer,
        training_token_sha256=inp['training_token_sha256'],
        inference_contract=dict(plan.SERVING_V1,version=2,tokenizer_sha256=digest(tokenizer),
                                token_sha256=inp['training_token_sha256']),
        input_ref=dict(path='meta/training-input.json',sha256=actual_inputs['meta/training-input.json']),
        numerical_ref=dict(path='meta/numerical.json',sha256='0'*64),
        files=[dict(path=n,sha256=s) for n,s in sorted(known.items())])
    # Only a fixed-width hash placeholder is used for the absent numerical
    # review. No numerical content, success status or launch authority is made.
    known[str(root/'evidence/meta/numerical.json')]='0'*64
    draft['files'].append(dict(path=str(root/'evidence/meta/numerical.json'),sha256='0'*64))
    known[str(root/'evidence/meta/plan.json')]='0'*64
    docs={str(p.relative_to(root)):json.loads(p.read_text()) for p in (root/'evidence/meta').rglob('*.json')}
    occurrences=[]
    def substitute(value,bindings,path=''):
        if isinstance(value,dict):
            for key,item in list(value.items()):
                if key=='frozen_files':
                    occurrences.append(path+'/'+key)
                    value[key]=dict(bindings)
                else:substitute(item,bindings,path+'/'+key)
        elif isinstance(value,list):
            for i,item in enumerate(value):substitute(item,bindings,path+'/'+str(i))
    def projected(extra):
        bindings=dict(known)
        for i in range(extra):bindings[str(root/f'evidence/meta/numerical-case-{i:02d}.json')]='0'*64
        total=census['groups']['evidence/meta']
        for name,original in docs.items():
            new=copy.deepcopy(original)
            if name=='evidence/meta/plan.json':
                new=copy.deepcopy(draft)
                new['files'] += [dict(path=n,sha256=s) for n,s in bindings.items()
                    if n not in known]
            else:substitute(new,bindings,name)
            total+=size(new)-(root/name).stat().st_size
        total+=sum((package/name).stat().st_size for name in actual_inputs)
        return total
    base=projected(0);unique_occurrences=sorted(set(occurrences))
    sensitivities=[]
    for count in (1,9,18):
        estimate=projected(count)
        sensitivities.append(dict(additional_numerical_artifact_files=count,
            projected_meta_bytes_excluding_new_numerical_payloads=estimate,
            remaining_for_payloads_and_unmodeled_growth=budgets()['evidence/meta']-estimate))
    control_sizes=sorted((row['size'] for name,row in census['files'].items()
                          if name.startswith('control/')),reverse=True)
    verify()
    assert plan.source_fingerprint()==source and plan.runtime_identity()==runtime
    return dict(kind='wide_hybrid_prerequisite_metadata_accounting_v2',
        status='incomplete_projection_no_fit_or_launch_claim',
        script_sha256=file_digest(Path(__file__)),manifest_sha256=file_digest(manifest_path),
        retained_prefix=str(root),retained_files_verified=len(manifest['files']),
        retained_checkpoint_files=sum(n.endswith('.pt') for n in manifest['files']),
        actual_input_artifacts=actual_inputs,actual_input_bytes=sum((package/n).stat().st_size for n in actual_inputs),
        source_sha256=digest(source),runtime_sha256=digest(runtime),
        current_metadata_bytes=census['metadata_bytes'],current_file_count=len(census['files']),file_cap=MAX_FILES,
        current_meta_bytes=census['groups']['evidence/meta'],meta_cap=budgets()['evidence/meta'],
        known_draft_plan_bytes=size(draft),known_frozen_binding_count=len(known),
        frozen_binding_map_occurrences=len(unique_occurrences),
        known_field_projection_meta_bytes=base,
        remaining_for_numerical_and_unmodeled_growth=budgets()['evidence/meta']-base,
        numerical_artifact_count_sensitivity=sensitivities,
        current_control_bytes=census['groups']['control'],control_cap=budgets()['control'],
        two_largest_existing_control_replacement_overlap_bytes=sum(control_sizes[:2]),
        unresolved=['Actual nine-case numerical review, trace sizes and transitive references',
            'Propagation through original inventory rows and any non-frozen_files nested receipts',
            'Actual resource-audit and metrics growth at full time/step budgets',
            'Final controller receipts and aggregate quality/cost result under real bindings',
            'Complete per-group/file-count/transient high-water proof after actual prerequisites'],
        original_bytes_unchanged=True,plan_frozen=False,training_launched=False,
        note='Known field projection retains synthetic history; it is not a semantically valid replay or a full upper bound.')


if __name__=='__main__':
    with CpuPrerequisiteGuard(EVIDENCE/'prerequisite-metadata-v2-guard.json',ROOT,timeout_seconds=180):
        report=audit()
        with (EVIDENCE/'prerequisite-metadata-v2-review.json').open('x') as stream:
            json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({k:report[k] for k in ('known_field_projection_meta_bytes',
        'remaining_for_numerical_and_unmodeled_growth','frozen_binding_map_occurrences',
        'numerical_artifact_count_sensitivity')}))
