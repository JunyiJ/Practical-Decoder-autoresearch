"""Bounded fresh H200 pair loading/timing and independent artifact acceptance.

Call only through the activation inference entry: this module does not certify
process exit or grant next-seed credit. No training, checkpoint writes or holdout.
"""
import gc
import os
import shutil
import time

import torch
from omegaconf import OmegaConf

from ..models.gpt import GPT
from ..models.tokenizer_activation import activation_contract
from ..models.tokenizer_architecture import validate_architecture_model
from ..utils.checkpoint import read_checkpoint, load_model_weights, checkpoint_loader
from ..attention.cache import build_caches
from .protocol import digest
from .memory_guard import start_memory_guard
from .tokenizer_activation_session import require, reference
from .tokenizer_activation_worker import input_identity, input_contract
from .tokenizer_activation_audit import PARAMETERS
from .tokenizer_activation_rules import FAMILIES,FRESH_SEEDS,confirmation_specs
from .tokenizer_activation_timing import CONTRACT,slot,measure_pair,review_result
from .tokenizer_wide_hybrid_timing import token_digest,allocated_bytes
from .tokenizer_wide_hybrid_measurements import check_kernel,measure_kernel
from .tokenizer_wide_hybrid_metadata import snapshot


def budget(ledger):
    seen=snapshot(ledger.root)
    size=sum(v['size'] for k,v in seen['files'].items() if k.startswith('evidence/inference/'))
    require(size+16384<=CONTRACT['panel_limit_bytes'],'Inference panel output cap exceeded')
    require(shutil.disk_usage(ledger.root).free>=CONTRACT['reserve_bytes'],'Inference reserve crossed')
    require(not (ledger.root/'STOP').exists() and not (ledger.root/'STOP').is_symlink(),'STOP blocks inference')
    return seen


def quality_binding(ledger,seed):
    from .tokenizer_activation_entry import bind,prefix
    from .tokenizer_activation_retention import verify_boundaries
    slot(seed);plan=bind(ledger,capacity=True);state=ledger.inspect()
    require(state['next']['kind']=='review_final' and len(state['records'])==12,'Complete accepted quality panel required')
    dispatches=prefix(ledger,state);verify_boundaries(ledger,state)
    bindings={}
    for i,spec in enumerate(confirmation_specs(state['selection']),6):
        if spec['seed']!=seed:continue
        review_ref=state['reviews'][i+2];review=ledger.document(review_ref)
        request=ledger.document(review['request'])
        require(request['spec']==spec and review['request']['path']==f'evidence/quality/{i:02d}/request.json',
                'Exact fresh selected pair required')
        for audit in review['audits'].values():
            require(ledger.document(audit).get('reviewer')=='activation_independent_v1','Independent quality review missing')
        for ref in review['weights'].values():ledger.verify(ref)
        bindings[spec['family']]=dict(request=review['request'],acceptance=review_ref,weights=review['weights'])
    require(set(bindings)==set(FAMILIES),'Both fresh families required')
    budget(ledger)
    return dict(plan=ledger.plan,seed=seed,selection=state['selection'],quality_reviews=state['reviews'],
        dispatches=dispatches,inputs=plan['inputs'],bindings=bindings,contract=CONTRACT)


def strict_mps():
    require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='1' and torch.backends.mps.is_available(),
            'Strict MPS required, fallback forbidden')


def validate_model(model,cfg,family):
    contract=activation_contract(cfg.model)
    require(contract is not None and contract['gate_kind']==family and not model.training,'Exact eval activation model required')
    require(digest(OmegaConf.to_container(OmegaConf.create(model.cfg),resolve=True))==
            digest(OmegaConf.to_container(cfg.model,resolve=True)),'Model/cache config differs')
    validate_architecture_model(model)
    require(sum(p.numel() for p in model.parameters())==PARAMETERS,'Parameter geometry differs')
    require(all(p.device.type=='mps' and p.dtype==torch.float32 for p in model.parameters()),'FP32 MPS parameters required')
    require(all(b.device.type=='mps' and (not b.is_floating_point() or b.dtype==torch.float32) for b in model.buffers()),
            'MPS buffers with FP32 floating values required')


def expected_models(ledger,binding):
    expected={}
    for family,refs in binding['bindings'].items():
        rq=ledger.document(refs['request']);ref=refs['weights']['h200']
        expected[family]=dict(checkpoint=ref,seed=binding['seed'],family=family,lr=rq['spec']['lr'],
            config_sha256=digest(rq['config']),tokenizer_sha256=digest(binding['inputs']['tokenizer']),
            parameters=PARAMETERS,parameter_storage_bytes=PARAMETERS*4)
    return expected


def check_export(payload,rq,refs,inputs,root):
    require(digest(payload['cfg'])==digest(rq['config']) and
            digest(payload.get('tokenizer_contract'))==digest(inputs['tokenizer']), 'Loaded export contract differs')
    require('optimizer_state' not in payload and 'training_state' not in payload
            and payload.get('averaged_weights',{}).get('purpose')=='inference_only_not_optimizer_resume'
            and payload['averaged_weights'].get('half_life')==200
            and payload.get('parent_checkpoint')==dict(path=str(root/refs['weights']['raw']['path']),
                sha256=refs['weights']['raw']['sha256']), 'Inference-only H200 lineage required')
    require(bool(payload['model_state']) and all(isinstance(v,torch.Tensor) and v.device.type=='cpu'
        and bool(torch.isfinite(v).all()) for v in payload['model_state'].values()), 'Finite CPU export values required')
    require(payload.get('step')==1200 and payload.get('model_math_version')==2
            and payload.get('parameter_layout_version')==2,'Fresh export step/layout differs')


def load_pair(ledger,binding,recheck):
    strict_mps();models={};configs={};metadata={};fixed=None
    expected=expected_models(ledger,binding)
    for family in FAMILIES:
        recheck();refs=binding['bindings'][family];rq=ledger.document(refs['request'])
        input_contract(rq,binding['inputs'])
        ref=refs['weights']['h200'];path=ledger.verify(ref);payload=read_checkpoint(path,map_location='cpu')
        ledger.verify(ref)
        check_export(payload,rq,refs,binding['inputs'],ledger.root)
        cfg=OmegaConf.create(rq['config']);loader=checkpoint_loader(cfg,payload,cfg.data.path,'cpu')
        require(digest(input_identity(loader))==digest(binding['inputs']),'Loaded inference input identity differs')
        ids=loader.train_data[:96].tolist()
        require(token_digest(ids)==binding['inputs']['training96_sha256'] and (fixed is None or fixed==ids),
                'Fixed paired training IDs differ');fixed=ids
        model=GPT(cfg.model).to('mps').eval();load_model_weights(model,payload);torch.mps.synchronize()
        validate_model(model,cfg,family)
        metadata[family]=dict(expected[family],parameters=sum(p.numel() for p in model.parameters()),
                             parameter_storage_bytes=allocated_bytes(model.parameters()))
        require(metadata[family]==expected[family],'Actual parameter storage differs')
        models[family]=model;configs[family]=cfg
        del payload,loader;gc.collect();recheck()
    return dict(models=models,configs=configs,metadata=metadata,token_ids=fixed)


def callbacks(loaded,sha,recheck):
    strict_mps()
    require(set(loaded['models'])==set(FAMILIES) and set(loaded['configs'])==set(FAMILIES),'Complete model pair required')
    require(token_digest(loaded['token_ids'])==sha,'Timing tokens changed')
    for family in FAMILIES:validate_model(loaded['models'][family],loaded['configs'][family],family)
    fixed=torch.tensor(loaded['token_ids'],device='mps',dtype=torch.int64)
    inputs={b:fixed.unsqueeze(0).repeat(b,1) for b in (1,8)}
    def args(family,batch):
        require(family in FAMILIES and type(batch) is int and batch in inputs,'Finite family/batch required')
        require(not loaded['models'][family].training,'Model left eval mode')
        return loaded['models'][family],inputs[batch],lambda:build_caches(loaded['configs'][family],'mps',batch_size=batch,dtype=torch.float32)
    def check(family,batch):
        recheck();result=check_kernel(*args(family,batch));recheck();return result
    def measure(family,batch):
        # Frequent budget/identity checks stay outside synchronized timed regions.
        recheck();result=measure_kernel(*args(family,batch),synchronize=torch.mps.synchronize,
            clock=time.perf_counter,memory=lambda:(torch.mps.current_allocated_memory(),torch.mps.driver_allocated_memory()))
        recheck();return result
    return check,measure


def produce(ledger,request_ref,binding,recheck):
    from .tokenizer_activation_entry import PhaseGuard,guard_ref,verify_guard
    seed=binding['seed'];physical=slot(seed);stem=f'evidence/inference/{physical}'
    trial=ledger.root/stem;loaded=None
    require({p.name for p in trial.iterdir()}<={'request.json','process'},'Existing inference attempt cannot retry')
    try:
        with PhaseGuard(ledger.writer,'inference',physical,'worker'):
            recheck();strict_mps();start_memory_guard();torch.set_num_threads(4)
            ledger.writer.publish_json(stem+'/attempt.json',dict(request=request_ref,pid=os.getpid(),parent_pid=os.getppid(),seed=seed))
            loaded=load_pair(ledger,binding,recheck)
            check,measure=callbacks(loaded,binding['inputs']['training96_sha256'],lambda:budget(ledger))
            result=measure_pair(seed,binding['selection'],loaded['token_ids'],binding['inputs']['training96_sha256'],check=check,measure=measure)
            review_result(result,seed,binding['selection'],binding['inputs']['training96_sha256']);recheck()
            for name,value in [('models',loaded['metadata']),('result',result)]:ledger.writer.publish_json(stem+'/'+name+'.json',value)
            ledger.writer.publish_json(stem+'/worker-outputs.json',dict(kind='activation_inference_outputs_v1',request=request_ref,
                seed=seed,pid=os.getpid(),parent_pid=os.getppid(),status='complete_pending_parent_review',
                models=reference(ledger.root,stem+'/models.json'),result=reference(ledger.root,stem+'/result.json')))
            recheck()
        verify_guard(ledger,guard_ref(ledger,'inference',physical,'worker'),os.getpid())
        return reference(ledger.root,stem+'/worker-outputs.json')
    except BaseException as exc:
        try:ledger.writer.publish_json(stem+'/worker-failure.json',dict(error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as error:exc.add_note('Inference failure publication: '+repr(error)[:1000])
        raise
    finally:
        del loaded;gc.collect()
        if torch.backends.mps.is_available():torch.mps.empty_cache()


def review_outputs(ledger,request_ref,binding,child_pid,parent_pid):
    """Read back measured values and rebind their original accepted fresh pair."""
    from .tokenizer_activation_entry import guard_ref,verify_guard
    seed=binding['seed'];physical=slot(seed);stem=f'evidence/inference/{physical}'
    before=budget(ledger)
    require(not any(n.startswith('evidence/inference/') and (n.endswith('.tmp') or 'failure' in n) for n in before['files']),
            'Failed/partial inference output')
    refs={name:reference(ledger.root,stem+'/'+name+'.json') for name in ('request','attempt','models','result','worker-outputs')}
    require(refs['request']==request_ref,'Inference request changed')
    require(ledger.document(refs['attempt'])==dict(request=request_ref,pid=child_pid,parent_pid=parent_pid,seed=seed),'Worker identity differs')
    expected=dict(kind='activation_inference_outputs_v1',request=request_ref,seed=seed,pid=child_pid,parent_pid=parent_pid,
        status='complete_pending_parent_review',models=refs['models'],result=refs['result'])
    require(ledger.document(refs['worker-outputs'])==expected,'Worker completion differs')
    require(ledger.document(refs['models'])==expected_models(ledger,binding),'Loaded model provenance differs')
    review=review_result(ledger.document(refs['result']),seed,binding['selection'],binding['inputs']['training96_sha256'])
    verify_guard(ledger,guard_ref(ledger,'inference',physical,'worker'),child_pid)
    # Rebuilding this binding replays all accepted quality, retention and
    # dispatch evidence and checks every surviving checkpoint hash.
    require(quality_binding(ledger,seed)==binding,'Completed quality history changed during inference')
    require(budget(ledger)==before and all(reference(ledger.root,r['path'])==r for r in refs.values()),'Inference changed during review')
    return dict(kind='activation_inference_review_v1',status='reviewed_pending_transport',request=request_ref,
        binding_sha256=digest(binding),review=review,files=list(refs.values()))
