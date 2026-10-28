"""Compose pilot producers inside one resource interval; deliberately no CLI.

The future controller must perform the idle complete-campaign storage/audit gate
BEFORE creating this slot's request/process files, and enforce bounded process
capture around this body. This function does not replace that transition gate or
authorize dispatch, cleanup or continuation. All real numerical prerequisites
are nevertheless independently required by bind_campaign before training.
"""
import gc
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..train.train import run_training, _make_loader, _training_contract
from ..train.weight_averaging import export_averaged_checkpoint
from .protocol import digest, file_digest
from .tokenizer_wide_hybrid_plan_v2 import bind_campaign, PROJECTION
from .tokenizer_wide_hybrid_pilot_config import specs
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_retention import allocations
from .tokenizer_wide_hybrid_retention_receipts import _document, _path, _sha
from .tokenizer_wide_hybrid_scores import _json, _require
from .tokenizer_wide_hybrid_timing import token_digest
from .tokenizer_wide_hybrid_training_outputs import TrainingOutputs
from .tokenizer_wide_hybrid_evaluation_v2 import evaluate_and_publish
from .tokenizer_wide_hybrid_pilot_cache_v2 import review_and_publish
from .tokenizer_wide_hybrid_campaign_resources import CampaignPilotResourceRecorder
from .tokenizer_wide_hybrid_retention_audit import reject_output_failures

KIND='wide_hybrid_pilot_execution_v2'


def _inputs(request,tokenizer,token_sha):
    cfg=OmegaConf.create(request['config'])
    loader=_make_loader(cfg,'cpu')
    try:
        contract=_training_contract(cfg,'mps',loader)
        _require(digest(contract.get('tokenizer'))==digest(tokenizer)
                 and token_digest(loader.train_data[:96].tolist())==token_sha,
                 'Actual tokenizer/training inputs differ from frozen campaign')
        return contract
    finally:
        del loader
        gc.collect()


def execute_pilot_outputs(writer,slot,plan_sha256,request_sha256,*,projection_ref):
    """Fresh guarded body only. Parent storage/process/stage gates remain required."""
    _require(isinstance(writer,MetadataWriter) and type(slot) is int and 0<=slot<3,
             'Exact bounded pilot slot required')
    _sha(plan_sha256);_sha(request_sha256)
    evidence=writer.root/'evidence';stem=f'pilots/{slot:02d}';trial=_path(evidence,stem)
    plan_path=evidence/'meta/plan.json'
    bound=bind_campaign(plan_path,plan_sha256)
    plan=_json(plan_path,plan_sha256)
    request=bound['requests']['pilots'][slot]
    _require(digest(request['spec'])==digest(specs()[slot])
             and digest(_json(trial/'request.json',request_sha256))==digest(request),
             'Pilot execution request/order differs')
    _require(isinstance(projection_ref,dict) and set(projection_ref)=={'path','sha256'}
             and type(projection_ref['path']) is str and projection_ref['path'].startswith('meta/'),
             'Frozen campaign projection reference required')
    projection=_document(evidence,projection_ref)
    _require(bound['files'].get(str(_path(evidence,projection_ref['path'])))==projection_ref['sha256']
             and bound['files'].get(str(PROJECTION.resolve()))==projection_ref['sha256'],
             'Execution projection must be the bound frozen projection copy')
    allowance=allocations(projection)[request['spec']['family'],2]
    forbidden=('worker-attempt.json','worker-outputs.json','worker-failure.json',
               'resource-events.jsonl','resource-audit.json','training-attempt.json',
               'training-outputs.json','evaluation-attempt.json','result.json','audit-attempt.json','audit.json')
    with writer._serialized():
        reject_output_failures(trial)
        _require(not any((trial/(n+s)).exists() for n in forbidden for s in ('','.tmp')),
                 'Existing pilot producer attempt cannot be retried')
        _require(not (trial/'checkpoints').exists() or not any((trial/'checkpoints').iterdir()),
                 'Existing checkpoint output cannot be reused')
    attempt=dict(kind=KIND,status='started',slot=slot,plan_sha256=plan_sha256,
                 request_sha256=request_sha256,projection_ref=projection_ref,
                 checkpoint_allowance_bytes=allowance,dispatch_authorized=False)
    marker=writer.publish_json('evidence/'+stem+'/worker-attempt.json',attempt)
    marker_sha=file_digest(marker)
    refs={}
    def recheck():
        _require(not (writer.root/'STOP').exists(),'Campaign STOP prevents pilot progress')
        _require(file_digest(marker)==marker_sha and file_digest(trial/'request.json')==request_sha256,
                 'Pilot attempt/request changed')
        _require(all(file_digest(Path(p))==sha for p,sha in bound['files'].items()),'Frozen pilot evidence changed')
        for ref in refs.values():_document(evidence,ref)
        reject_output_failures(trial)
    try:
        with CampaignPilotResourceRecorder(writer,slot,plan_sha256,request_sha256) as resources:
            torch.set_num_threads(4)
            contract=_inputs(request,plan['tokenizer'],plan['training_token_sha256'])
            recheck()
            cfg=OmegaConf.create(request['config'])
            with TrainingOutputs(writer,'pilots',slot,allowance,request['config']) as outputs:
                training=run_training(cfg,output_sink=outputs)
                _require(training['completed_steps']==request['spec']['steps'],'Incomplete pilot training budget')
                recheck()
                gc.collect();torch.mps.empty_cache()
                raw=Path(training['final_checkpoint'])
                export_averaged_checkpoint(raw,raw.parent/'h200.pt',200,payload_writer=outputs.checkpoint)
            refs['training_outputs']=dict(path=stem+'/training-outputs.json',
                                          sha256=file_digest(trial/'training-outputs.json'))
            recheck()
            refs['result']=evaluate_and_publish(writer,'pilots',slot,request,request_sha256,
                refs['training_outputs']['sha256'],contract,plan['training_token_sha256'],started=resources.started)
            recheck()
            # Preserve the exact successful attempt before cache/resource completion.
            ep=trial/'evaluation-attempt.json'
            evaluation=_json(ep,file_digest(ep))
            _require(evaluation.get('kind')=='wide_hybrid_evaluation_v2'
                     and evaluation.get('status')=='started'
                     and evaluation.get('request_sha256')==request_sha256
                     and evaluation.get('training_outputs_sha256')==refs['training_outputs']['sha256']
                     and evaluation.get('training_contract_sha256')==digest(contract)
                     and evaluation.get('training_token_sha256')==plan['training_token_sha256'],
                     'Successful evaluation attempt provenance differs')
            refs['evaluation_attempt']=dict(path=stem+'/evaluation-attempt.json',sha256=file_digest(ep))
            refs['cache']=review_and_publish(writer,slot,dict(trial=str(trial),request=request,
                request_sha256=request_sha256,result_sha256=refs['result']['sha256'],training_contract=contract),
                plan['tokenizer'],plan['training_token_sha256'],projection_ref=projection_ref)
            recheck()
            resources.mark_success(refs['result']['sha256'],refs['cache']['sha256'])
        refs['resources']=dict(path=stem+'/resource-audit.json',sha256=file_digest(trial/'resource-audit.json'))
        recheck()
        # Source/runtime and the full frozen prerequisites are checked again only
        # after the resource thread has closed; no concurrent census races.
        again=bind_campaign(plan_path,plan_sha256)
        _require(digest(again['files'])==digest(bound['files']),'Frozen pilot provenance changed')
        recheck()
        report=dict(kind=KIND,status='pilot_outputs_complete_pending_independent_stage_review',
            slot=slot,plan_sha256=plan_sha256,request_sha256=request_sha256,
            attempt=dict(path=stem+'/worker-attempt.json',sha256=marker_sha),artifacts=refs,
            adopted=False,dispatch_authorized=False,stage_audits_bound=False)
        path=writer.publish_json('evidence/'+stem+'/worker-outputs.json',report)
        return dict(path=stem+'/worker-outputs.json',sha256=file_digest(path))
    except BaseException as exc:
        try:writer.publish_json('evidence/'+stem+'/worker-failure.json',dict(kind=KIND,status='failed',
            slot=slot,request_sha256=request_sha256,error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as error:exc.add_note('Pilot failure publication failed: '+repr(error)[:1000])
        raise
