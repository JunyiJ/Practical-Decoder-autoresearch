"""Bounded validation/result producer for an already completed v2 trajectory.

No CLI, launch gate or training. Caller must bind frozen campaign identity,
remaining storage, numerical/input evidence and resource guards before entry.
"""
import gc
import math
import os
from pathlib import Path
import time

import torch
from omegaconf import OmegaConf

from ..models.gpt import GPT
from ..utils.checkpoint import read_checkpoint, checkpoint_loader, load_model_weights
from ..utils.generate import generate
from .coverage_eval import score_validation
from .protocol import digest, file_digest
from .tokenizer_wide_hybrid_driver import CORPUS_SHA256
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_plan_v2 import request_options
from .tokenizer_wide_hybrid_retention_receipts import _sha
from .tokenizer_wide_hybrid_scores import _json, _require, _score
from .tokenizer_wide_hybrid_measurements import validate_mps_model
from .tokenizer_wide_hybrid_tensors import _finite_tree
from .tokenizer_wide_hybrid_timing import token_digest

KIND='wide_hybrid_evaluation_v2'
SAMPLE=dict(prompt='ROMEO:\n',seed=12345,temperature=.8,top_k=40,new_tokens=96)


def _endpoint(kind,path,sha,request,contract,token_sha,raw_ref,recheck):
    """One endpoint owns its loader/payload/model; release before the next load."""
    payload=loader=model=ids=None
    try:
        recheck()
        payload=read_checkpoint(path,map_location='cpu')
        recheck()
        cfg=OmegaConf.create(request['config'])
        _require(digest(payload.get('cfg'))==digest(request['config'])
                 and type(payload.get('step')) is int and payload['step']==cfg.training.max_iters
                 and payload.get('parameter_layout_version')==2,'Evaluation checkpoint config/step/layout differs')
        if kind=='raw':
            _require(digest(payload.get('training_state',{}).get('contract'))==digest(contract)
                     and 'averaged_weights' not in payload,'Raw evaluation contract differs')
        else:
            _require(digest(payload.get('tokenizer_contract'))==digest(contract['tokenizer'])
                     and 'optimizer_state' not in payload and 'training_state' not in payload
                     and payload.get('averaged_weights',{}).get('half_life')==200
                     and payload.get('parent_checkpoint')==raw_ref,
                     'H200 evaluation must use inference-only export')
        _require(isinstance(payload.get('model_state'),dict) and bool(payload['model_state']),
                 'Missing model state')
        _finite_tree(payload['model_state'])
        loader=checkpoint_loader(cfg,payload,cfg.data.path,'cpu')
        _require(digest(loader.tokenizer_contract)==digest(contract['tokenizer'])
                 and token_digest(loader.train_data[:96].tolist())==token_sha,
                 'Rebuilt tokenizer/training inputs differ')
        # No test_data access or test scoring. Coverage is checked before model allocation.
        _require(loader.val_data.ndim==1 and len(loader.val_data)==34279,'Validation target coverage differs')
        target_bytes=sum(len(loader.enc.decode_single_token_bytes(int(i))) for i in loader.val_data[1:])
        _require(target_bytes==111536,'Validation byte coverage differs')
        model=GPT(cfg.model).to('mps').eval()
        load_model_weights(model,payload)
        del payload;payload=None
        validate_mps_model(model,cfg,request['spec']['family'])
        score=score_validation(model,loader.val_data,128,8,2048)
        score.update(target_bytes=target_bytes,bits_per_byte=sum(score['block_loss_sums'])/target_bytes/math.log(2),
                     perplexity=math.exp(score['mean_ce']),checkpoint=str(path),checkpoint_sha256=sha,
                     parameters=sum(p.numel() for p in model.parameters()))
        _score(score,score['parameters'])
        if kind=='h200':
            torch.manual_seed(SAMPLE['seed'])
            ids=torch.tensor([loader.encode(SAMPLE['prompt'])],device='mps')
            score['sample']=loader.decode(generate(model,ids,SAMPLE['new_tokens'],
                temperature=SAMPLE['temperature'],top_k=SAMPLE['top_k'])[0])
            score['sample_settings']=dict(SAMPLE)
        torch.mps.synchronize()
        recheck()
        return score
    finally:
        del ids,model,payload,loader
        gc.collect()
        torch.mps.empty_cache()


def evaluate_and_publish(writer,stage,slot,request,request_sha256,training_outputs_sha256,
                         training_contract,training_token_sha256,*,started):
    """Return bounded result reference; no tensor/cache/resource audit success implied."""
    _require(isinstance(writer,MetadataWriter) and stage in ('pilots','quality')
             and type(slot) is int and 0<=slot<(3 if stage=='pilots' else 24),'Declared trajectory slot required')
    _require(type(started) in (float,int) and math.isfinite(started) and 0<=started<=time.monotonic(),
             'Original trajectory monotonic start required')
    _require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='1' and torch.backends.mps.is_available(),
             'Strict MPS required')
    _sha(request_sha256);_sha(training_outputs_sha256);_sha(training_token_sha256)
    options=request_options(writer.root)
    candidates=[options['pilots'][slot]] if stage=='pilots' else options['quality'][str(slot)]
    _require(any(digest(request)==digest(r) for r in candidates),'Request differs from declared slot recipes')
    cfg=request['config'];stem=f'evidence/{stage}/{slot:02d}';trial=writer.root/stem
    _require(digest(training_contract.get('model'))==digest(cfg['model'])
             and digest(training_contract.get('weight_averaging'))==digest(cfg['training']['weight_averaging'])
             and isinstance(training_contract.get('tokenizer'),dict) and bool(training_contract['tokenizer']),
             'Expected training contract differs')
    request_path=trial/'request.json';outputs_path=trial/'training-outputs.json'
    _require(digest(_json(request_path,request_sha256))==digest(request),'Saved request differs')
    outputs=_json(outputs_path,training_outputs_sha256)
    _require(outputs.get('kind')=='wide_hybrid_training_outputs_v2' and outputs.get('status')=='completed'
             and outputs.get('config_sha256')==digest(cfg),'Training/export outputs incomplete')
    summary_path=trial/'training-summary.json';summary_sha=file_digest(summary_path)
    summary=_json(summary_path,summary_sha)
    _require(type(summary.get('completed_steps')) is int and summary['completed_steps']==cfg['training']['max_iters']
             and summary.get('final_checkpoint')==outputs['raw']['path'],'Training summary endpoint differs')
    bound={str(request_path):request_sha256,str(outputs_path):training_outputs_sha256,
           str(summary_path):summary_sha,str(Path(cfg['data']['path']).resolve()):CORPUS_SHA256}
    for kind in ('raw','h200'):
        row=outputs[kind];path=Path(row['path'])
        _require(path==path.resolve() and path.parent==trial/'checkpoints'
                 and path.is_file() and type(row['size']) is int and path.stat().st_size==row['size']
                 and (kind!='h200' or path.name=='h200.pt'),'Checkpoint output location/size differs')
        bound[str(path)]=_sha(row['sha256'])
    _require(outputs['raw']['path']!=outputs['h200']['path'],'Distinct endpoint pair required')
    forbidden=['evaluation-attempt.json','evaluation-failure.json','result.json']
    def recheck():
        with writer._serialized():
            _require(not (writer.root/'STOP').exists(),'Campaign STOP prevents evaluation')
            snapshot(writer.root)
            _require(not any((trial/('training-failure.json'+suffix)).exists() for suffix in ('','.tmp')),
                     'Training failure evidence prevents evaluation')
            _require(all(file_digest(Path(p))==sha for p,sha in bound.items()),'Evaluation inputs changed')
    recheck()
    with writer._serialized():
        _require(not any((trial/(name+suffix)).exists() for name in forbidden for suffix in ('','.tmp')),
                 'Existing evaluation attempt cannot be retried')
    attempt=writer.publish_json(stem+'/evaluation-attempt.json',dict(kind=KIND,status='started',
        request_sha256=request_sha256,training_outputs_sha256=training_outputs_sha256,
        training_contract_sha256=digest(training_contract),training_token_sha256=training_token_sha256,
        files=bound.copy(),sample_settings=SAMPLE))
    bound[str(attempt)]=file_digest(attempt)
    try:
        evaluations={}
        for kind in ('raw','h200'):
            row=outputs[kind]
            evaluations[kind]=_endpoint(kind,Path(row['path']),row['sha256'],request,
                                        training_contract,training_token_sha256,
                                        {k:outputs['raw'][k] for k in ('path','sha256')},recheck)
        recheck()
        result=dict(**request['spec'],status='completed',**evaluations,training=summary,
            wall_seconds=time.monotonic()-started,holdout_consumed=False,request_sha256=request_sha256)
        path=writer.publish_json(stem+'/result.json',result)
        recheck()
        return dict(path=str(path.relative_to(writer.root/'evidence')),sha256=file_digest(path))
    except BaseException as exc:
        try:writer.publish_json(stem+'/evaluation-failure.json',dict(kind=KIND,status='failed',
            request_sha256=request_sha256,error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as error:exc.add_note('Evaluation failure publication failed: '+repr(error)[:1000])
        raise
