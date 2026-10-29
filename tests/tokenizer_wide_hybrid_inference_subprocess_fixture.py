"""Strict tiny /tmp-only source/model mocks for real CLI subprocess tests.

This module is never imported by production. The test's temporary sitecustomize
installs it in an otherwise real child. Locks, argv/PIDs/groups, metadata writers,
worker/parent CPU guards, timing arithmetic and transport acceptance stay real.
Full quality replay, loading and MPS are synthetic here, tested separately.
"""
import copy
import json
import os
from pathlib import Path

from src.research import tokenizer_wide_hybrid_inference_entry_v2 as entry
from src.research import tokenizer_wide_hybrid_inference_outputs_v2 as outputs
from src.research import tokenizer_wide_hybrid_inference_preparation_v2 as prep
from src.research import tokenizer_wide_hybrid_plan_v2 as plan_module
from src.research import tokenizer_wide_hybrid_cleanup as cleanup
from src.research.tokenizer_wide_hybrid_inference import PROJECTION


def install(campaign,set_attribute=setattr):
    root=Path(campaign)
    assert root==root.resolve() and root.is_relative_to(Path('/tmp').resolve())
    assert not any(p.is_symlink() for p in root.rglob('*'))
    evidence=root/'evidence';plan=json.loads((evidence/'meta/plan.json').read_text())
    assert json.loads((evidence/'meta/source.json').read_text())==dict(synthetic_inference_only=True)
    assert sum(p.stat().st_size for p in evidence.rglob('*.pt'))<1024**2
    requests={};triples={seed:{} for seed in outputs.SEEDS}
    for slot,spec in enumerate(outputs.confirmation_specs(json.loads((evidence/prep.REQUEST).read_text())['selection']),15):
        trial=evidence/f'quality/{slot:02d}';rq=json.loads((trial/'request.json').read_text())
        assert rq['config']==dict(synthetic_family=spec['family'],synthetic_seed=spec['seed'])
        requests[str(slot)]=[rq]
        triples[spec['seed']][spec['family']]=dict(trial=str(trial),request=rq,
            request_sha256=outputs.file_digest(trial/'request.json'),result_sha256=outputs.file_digest(trial/'result.json'),
            training_contract=dict(tokenizer=plan['tokenizer']))
    frozen=dict(files={str(evidence/'meta/plan.json'):outputs.file_digest(evidence/'meta/plan.json')},requests=dict(quality=requests))
    def campaign_binding(path,sha,*,live_writer=None):
        assert live_writer is None or live_writer.root==root
        assert path==evidence/'meta/plan.json' and sha==frozen['files'][str(path)]
        assert outputs.file_digest(path)==sha
        return copy.deepcopy(frozen)
    def full_binding(writer,**kw):
        assert writer.root==root
        request=json.loads((evidence/prep.REQUEST).read_text())
        assert all(kw[k]==request[k] for k in ('plan_sha256','completion_ref','projection_ref','rank_completion_ref'))
        inventory=prep.gate.history.acceptance._inventory(evidence)
        controls=prep.gate.history.acceptance.readiness._controls(writer,'none')
        files={str(evidence/n):v['sha256'] for n,v in inventory.items()}
        files.update({str(root/n):v['sha256'] for n,v in controls.items()})
        return dict(plan_sha256=request['plan_sha256'],completion_ref=request['completion_ref'],contract=request['contract'],
            selection=request['selection'],tokenizer=plan['tokenizer'],token_sha256=request['token_sha256'],triples=copy.deepcopy(triples),
            records_sha256=request['records_sha256'],checkpoint_pairs_sha256=request['checkpoint_pairs_sha256'],
            retention_sha256=request['retention_sha256'],inventory_sha256=request['quality_inventory_sha256'],
            controls_sha256=request['quality_controls_sha256'],files=files)
    rows=json.loads(PROJECTION.read_text())['records']
    def load(bindings,seed,selection,tokenizer,token_sha):
        if os.environ.get('TOKENIZER_INFERENCE_TEST_FAIL_SEED')==str(seed):raise RuntimeError('Synthetic model loading failure')
        metadata={}
        for family,b in bindings.items():
            result=json.loads((Path(b['trial'])/'result.json').read_text());spec=b['request']['spec']
            row=next(r for r in rows if r['width']==512 and r['family']==family and r['bottleneck_divisor']==spec['divisor'])
            metadata[family]=dict(checkpoint=result['h200']['checkpoint'],sha256=result['h200']['checkpoint_sha256'],
                seed=seed,family=family,divisor=spec['divisor'],lr=spec['lr'],config_sha256=outputs.digest(b['request']['config']),
                tokenizer_sha256=outputs.digest(tokenizer),parameters=row['parameters'],parameter_storage_bytes=row['parameter_bytes'])
        assert outputs.token_digest(list(range(96)))==token_sha
        return dict(models={},configs={},token_ids=list(range(96)),metadata=metadata)
    set_attribute(cleanup,'GLOBAL_LOCK',root.parent/'global.lock')
    set_attribute(plan_module,'bind_campaign',campaign_binding);set_attribute(entry,'bind_campaign',campaign_binding)
    set_attribute(prep.gate,'_bind',full_binding);set_attribute(prep.gate,'_contract',lambda config:dict(tokenizer=plan['tokenizer']))
    set_attribute(outputs,'load_selected_exports',load)
    set_attribute(outputs,'start_memory_guard',lambda:None)
    set_attribute(outputs,'mps_callbacks',lambda *a:(lambda f,b:dict(passed=True,max_absolute_error=0.),
        lambda f,b:dict(prefill_seconds=.1,decode_seconds=.2,state_bytes=100*b,
            observed_mps_current_bytes=1000,observed_mps_driver_bytes=2000)))
