"""Bounded activation worker body and independent parent acceptance adapter.

No CLI/dispatcher. The executable entry must enforce the full frozen plan,
resource projection, source/runtime/input pins, inherited locks and idle/serial
process gates. Its recheck callback is trusted controller code, never user input.
No synthetic fixture verdict is sufficient to enter a research worker.
"""
import gc
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..models.gpt import GPT
from ..models.tokenizer_architecture import validate_architecture_model
from ..attention.cache import build_caches
from ..train.train import run_training, _make_loader, _training_contract
from ..train.weight_averaging import export_averaged_checkpoint
from ..utils.checkpoint import read_checkpoint, load_model_weights, checkpoint_loader
from .coverage_eval import score_validation
from .memory_guard import start_memory_guard
from .protocol import digest, file_digest
from .tokenizer_activation_session import KIND, require, reference
from .tokenizer_activation_audit import (PARAMETERS, RESOURCE_KIND, profile, check_measurements,
                                         check_resources, review_pair)
from .tokenizer_wide_hybrid_campaign_resources import _CampaignRecorder
from .tokenizer_wide_hybrid_resources import QualityResourceRecorder
from .tokenizer_wide_hybrid_training_outputs import TrainingOutputs
from .tokenizer_wide_hybrid_measurements import check_kernel, RTOL, ATOL
from .tokenizer_wide_hybrid_timing import token_digest, allocated_bytes
from .tokenizer_wide_hybrid_driver import CORPUS_SHA256

ALLOWANCE = 788953140
CACHE_BYTES = 196608  # four layers *128 context *(64 latent+32 rotary)*4 bytes


class ActivationResources(_CampaignRecorder, QualityResourceRecorder):
    KIND = RESOURCE_KIND

    def __init__(self, writer, stage, slot, plan_sha, request_sha):
        require(stage in ('pilots', 'quality'), 'Activation stage required')
        self.STAGE, self.SLOTS = stage, 2 if stage == 'pilots' else 12
        self.LIMITS = profile(stage == 'pilots')
        super().__init__(writer, slot, plan_sha, request_sha)

    @property
    def sample_cap(self):
        return 603 if self.STAGE == 'pilots' else 7203


def input_identity(loader):
    require(loader.val_data.ndim == 1 and len(loader.val_data) == 34279, 'Validation coverage differs')
    return dict(tokenizer=loader.tokenizer_contract, training96_sha256=token_digest(loader.train_data[:96].tolist()),
                validation_sha256=hashlib.sha256(loader.val_data.cpu().numpy().astype('<i8').tobytes()).hexdigest(),
                validation_targets=34278, validation_target_bytes=sum(
                    len(loader.enc.decode_single_token_bytes(int(i))) for i in loader.val_data[1:]))


def input_contract(request, expected_inputs):
    cfg = OmegaConf.create(request['config'])
    require(file_digest(Path(cfg.data.path)) == CORPUS_SHA256, 'Corpus changed')
    loader = _make_loader(cfg, 'cpu')
    require(digest(input_identity(loader)) == digest(expected_inputs)
            and expected_inputs['validation_target_bytes'] == 111536, 'Frozen tokenizer/input identity differs')
    return _training_contract(cfg, 'mps', loader)


def endpoint(role, row, request, contract, expected_inputs, recheck):
    """Evaluate one endpoint; pilot cache checks also use only training token IDs."""
    payload = loader = model = None
    try:
        recheck(); path = Path(row['path']); require(file_digest(path) == row['sha256'], 'Checkpoint changed')
        payload = read_checkpoint(path, map_location='cpu'); cfg = OmegaConf.create(request['config'])
        require(digest(payload['cfg']) == digest(request['config']), 'Checkpoint config differs')
        loader = checkpoint_loader(cfg, payload, cfg.data.path, 'cpu')
        require(digest(input_identity(loader)) == digest(expected_inputs), 'Endpoint input identity differs')
        model = GPT(cfg.model).to('mps').eval(); load_model_weights(model, payload)
        del payload; payload = None
        validate_architecture_model(model)
        require(sum(p.numel() for p in model.parameters()) == PARAMETERS
                and all(p.device.type == 'mps' and p.dtype == torch.float32 for p in model.parameters()),
                'Strict FP32 MPS activation model required')
        score = score_validation(model, loader.val_data, 128, 8, 2048)
        score.update(target_bytes=111536, bits_per_byte=sum(score['block_loss_sums'])/111536/math.log(2),
                     perplexity=math.exp(score['mean_ce']), checkpoint=str(path), checkpoint_sha256=row['sha256'],
                     parameters=PARAMETERS)
        cache = []
        if request['spec']['stage'] == 'pilot':
            for batch in (1, 8):
                ids = loader.train_data[:96].to('mps').unsqueeze(0).repeat(batch, 1)
                caches = build_caches(cfg, 'mps', batch_size=batch, dtype=torch.float32)
                checked = check_kernel(model, ids, lambda: caches)
                size = allocated_bytes(b for c in caches for b in c.buffers())
                require(size == batch*CACHE_BYTES, 'Actual cache allocation differs')
                cache.append(dict(batch=batch, state_bytes=size, **checked))
                del ids, caches
        torch.mps.synchronize()
        require(file_digest(path) == row['sha256'], 'Checkpoint changed during evaluation')
        recheck(); return score, cache
    finally:
        del payload, loader, model; gc.collect(); torch.mps.empty_cache()


def _pending(ledger, request_ref):
    state = ledger.inspect(); pending = state['pending']
    require(not state['failed'] and pending is not None and pending['ref'] == request_ref,
            'Exactly matching pending ledger request required')
    return pending


def produce(ledger, request_ref, expected_inputs, recheck):
    """One fresh producer under caller's complete freeze/process/storage gate."""
    recheck()
    require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '1' and torch.backends.mps.is_available(),
            'Strict MPS required; CPU fallback forbidden')
    pending = _pending(ledger, request_ref); request = pending['request']; stem = pending['stem']
    trial = ledger.root/stem; writer = ledger.writer; stage, slot = Path(stem).parts[1:]
    require({p.name for p in trial.iterdir()} <= {'request.json', 'process'}, 'Existing producer attempt cannot be retried')
    writer.publish_json(stem+'/worker-attempt.json', dict(kind='activation_worker_v1', request=request_ref, status='started'))
    try:
        start_memory_guard()
        with ActivationResources(writer, stage, int(slot), ledger.plan['sha256'], request_ref['sha256']) as resources:
            torch.set_num_threads(4)
            contract = input_contract(request, expected_inputs); recheck()
            with TrainingOutputs(writer, stage, int(slot), ALLOWANCE, request['config']) as sink:
                training = run_training(OmegaConf.create(request['config']), output_sink=sink)
                require(training['completed_steps'] == request['spec']['steps'], 'Incomplete training budget')
                recheck(); gc.collect(); torch.mps.empty_cache()
                raw = Path(training['final_checkpoint'])
                export_averaged_checkpoint(raw, raw.parent/'h200.pt', 200, payload_writer=sink.checkpoint)
            outputs = json.loads((trial/'training-outputs.json').read_text())
            measured = dict(training_contract=contract, cache={})
            for role in ('raw', 'h200'):
                measured[role], measured['cache'][role] = endpoint(role, outputs[role], request, contract, expected_inputs, recheck)
            # Audit actual saved tensors inside the same trial resource interval.
            tensor_values, _ = review_pair(ledger.root, request, outputs, contract)
            writer.publish_json(stem+'/producer-tensors.json', tensor_values)
            metrics = [json.loads(line) for line in (trial/'metrics.jsonl').read_text().splitlines()]
            check_measurements(request, measured, outputs, training, metrics)
            writer.publish_json(stem+'/measurements.json', measured)
            row = dict(request['spec'], status='completed')
            if stage == 'quality': row['h200_ce'] = measured['h200']['mean_ce']
            result = dict(kind=KIND, request=request_ref, record=row, holdout_consumed=False,
                          status='completed', request_sha256=request_ref['sha256'],
                          measurements=reference(ledger.root, stem+'/measurements.json'))
            writer.publish_json(stem+'/result.json', result)
            recheck(); resources.mark_success(file_digest(trial/'result.json'))
        recheck()
        writer.publish_json(stem+'/worker-outputs.json', dict(kind='activation_worker_v1',
            status='complete_pending_parent_review', request=request_ref,
            result=reference(ledger.root, stem+'/result.json'), dispatch_authorized=False))
        return reference(ledger.root, stem+'/worker-outputs.json')
    except BaseException as exc:
        try: writer.publish_json(stem+'/worker-failure.json', dict(kind='activation_worker_v1', request=request_ref,
            status='failed', error=repr(exc)[:1000], retry_allowed=False))
        except BaseException as error: exc.add_note('Failure publication: '+repr(error)[:1000])
        raise


def accept_outputs(ledger, request_ref, expected_inputs, recheck):
    """Independent read-back; caller must first prove child exit and no live descendants."""
    recheck(); pending = _pending(ledger, request_ref); stem = pending['stem']; request = pending['request']
    trial = ledger.root/stem; pilot = request['spec']['stage'] == 'pilot'
    require(not any(p.name.endswith('.tmp') or 'failure' in p.name for p in trial.rglob('*')), 'Failed/partial worker')
    refs = {name: reference(ledger.root, stem+'/'+name) for name in (
        'request.json','result.json','measurements.json','training-outputs.json','training-summary.json',
        'metrics.jsonl','resource-audit.json','resource-events.jsonl','worker-attempt.json','worker-outputs.json','producer-tensors.json')}
    docs = {name: ledger.document(ref) for name, ref in refs.items() if name.endswith('.json')}
    result = docs['result.json']; outputs = docs['training-outputs.json']; measured = docs['measurements.json']
    require(result.get('request') == request_ref and result.get('measurements') == refs['measurements.json']
            and result.get('status') == 'completed' and result.get('request_sha256') == request_ref['sha256']
            and result.get('holdout_consumed') is False, 'Result binding differs')
    require(docs['worker-outputs.json'] == dict(kind='activation_worker_v1', status='complete_pending_parent_review',
            request=request_ref, result=refs['result.json'], dispatch_authorized=False), 'Worker completion differs')
    contract = input_contract(request, expected_inputs)
    require(digest(measured['training_contract']) == digest(contract), 'Rebuilt training contract differs')
    metrics = [json.loads(line) for line in (trial/'metrics.jsonl').read_text().splitlines()]
    training_check = check_measurements(request, measured, outputs, docs['training-summary.json'], metrics)
    tensors, weights = review_pair(ledger.root, request, outputs, contract)
    require(tensors == docs['producer-tensors.json'], 'Independent tensor review differs')
    expected_row = dict(request['spec'], status='completed')
    if not pilot: expected_row['h200_ce'] = measured['h200']['mean_ce']
    require(result['record'] == expected_row, 'Selected score differs from measured H200')
    require(set(measured['cache']) == {'raw', 'h200'}, 'Both cache endpoint records required')
    for rows in measured['cache'].values():
        require(isinstance(rows, list) and len(rows) == (2 if pilot else 0), 'Pilot cache coverage differs')
        if pilot:
            for batch, row in zip((1, 8), rows):
                require(row.get('batch') == batch and row.get('state_bytes') == CACHE_BYTES*batch
                        and row.get('passed') is True and type(row.get('max_absolute_error')) in (int, float)
                        and math.isfinite(row['max_absolute_error']) and row['max_absolute_error'] >= 0,
                        'Pilot cache report invalid')
    report = docs['resource-audit.json']; events = [json.loads(line) for line in (trial/'resource-events.jsonl').read_text().splitlines()]
    require(report['events']['sha256'] == refs['resource-events.jsonl']['sha256'], 'Resource event hash differs')
    resources = check_resources(report, events, ledger.plan['sha256'], request_ref['sha256'], refs['result.json']['sha256'], pilot)
    recheck()
    require(all(reference(ledger.root, ref['path']) == ref for ref in refs.values()), 'Producer evidence changed during review')
    details = dict(training=training_check, evaluation=dict(target_count=34278, target_bytes=111536,
                  h200_ce=measured['h200']['mean_ce']), tensors=dict(values=tensors, cache=measured['cache']), resources=resources)
    audits = {}
    for name, detail in details.items():
        path = stem+'/'+name+'-audit.json'
        ledger.writer.publish_json(path, dict(status='passed', request=request_ref, result=refs['result.json'], weights=weights,
                                              detail=detail, reviewer='activation_independent_v1'))
        audits[name] = reference(ledger.root, path)
    review = dict(kind='activation_acceptance_v1', status='accepted', request=request_ref, result=refs['result.json'],
                  audits=audits, weights=weights, outputs=[reference(ledger.root, str(p.relative_to(ledger.root)))
                  for p in sorted(trial.rglob('*')) if p.is_file()])
    ledger.writer.publish_json(stem+'/acceptance.json', review)
    ref = reference(ledger.root, stem+'/acceptance.json'); recheck(); ledger.record('accept', ref)
    return ref
