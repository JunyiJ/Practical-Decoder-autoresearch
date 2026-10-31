"""Activation-specific independent artifact checks, never launch or adoption.

Expected request and training contract must be supplied by the frozen entry,
not inferred from results. Caller owns locks, wall/RSS guards and immutable pins.
"""
import gc
import json
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..models.gpt import GPT
from ..train.optimizers import build_optimizer
from ..train.lr_schedule import LearningRateSchedule, multiplier, schedule_contract
from .protocol import digest
from .tokenizer_activation_config import config
from .tokenizer_activation_session import require, reference
from .tokenizer_wide_hybrid_scores import _shapes, _score
from .tokenizer_wide_hybrid_tensors import audit_values
from .tokenizer_wide_hybrid_resources import _verify_events
from .tokenizer_wide_hybrid_retention_audit import RESOURCE_LIMITS, PILOT_RESOURCE_LIMITS

PARAMETERS = 28492416
RESOURCE_KIND = 'activation_trial_resources_v1'


def profile(pilot):
    return dict(PILOT_RESOURCE_LIMITS if pilot else RESOURCE_LIMITS)


def check_payloads(cfg, contract, raw, averaged, raw_ref):
    """Full architecture geometry, optimizer structure, finite values and EMA equality."""
    cfg = OmegaConf.create(cfg); steps = cfg.training.max_iters
    with torch.device('meta'):
        model = GPT(cfg.model)
        optimizer = build_optimizer(model, cfg.training)
    LearningRateSchedule(optimizer, cfg.training).apply(steps-1)
    expected_opt = optimizer.state_dict()
    named = dict(model.named_parameters()); expected = model.state_dict()
    require(sum(p.numel() for p in named.values()) == PARAMETERS, 'Activation parameter count differs')
    snapshots = 1 + (steps-cfg.training.weight_averaging.start_step)//cfg.training.weight_averaging.interval
    cfg_plain = OmegaConf.to_container(cfg, resolve=True)
    for role, payload in [('raw', raw), ('h200', averaged)]:
        require(digest(payload.get('cfg')) == digest(cfg_plain) and type(payload.get('step')) is int
                and payload['step'] == steps and payload.get('model_math_version') == 2
                and payload.get('parameter_layout_version') == 2, 'Checkpoint identity/version/step differs')
        _shapes(payload['model_state'], expected)
        if role == 'raw':
            require('averaged_weights' not in payload and 'optimizer_state' in payload,
                    'Raw endpoint must be online and resumable')
            require(digest(payload['training_state'].get('contract')) == digest(contract), 'Training/tokenizer contract differs')
            ema = payload['training_state']['weight_averaging_state']
            require(type(ema.get('version')) is int and ema['version'] == 1
                    and set(ema['averages']) == {'200'}, 'EMA version/panel differs')
            _shapes(ema['averages']['200'], named)
        else:
            require('optimizer_state' not in payload and 'training_state' not in payload
                    and payload.get('parent_checkpoint') == raw_ref
                    and digest(payload.get('tokenizer_contract')) == digest(contract['tokenizer']),
                    'H200 must be inference-only with exact raw/tokenizer lineage')
            ema = payload['averaged_weights']
            require(type(ema.get('half_life')) is int and ema['half_life'] == 200
                    and ema.get('purpose') == 'inference_only_not_optimizer_resume', 'H200 purpose/half-life differs')
        require(digest(ema['spec']) == digest(cfg_plain['training']['weight_averaging'])
                and type(ema['completed_steps']) is int and ema['completed_steps'] == steps
                and type(ema['snapshots']) is int and ema['snapshots'] == snapshots
                and ema['parameter_names'] == list(named), 'EMA horizon/count/name order differs')
    saved = raw['optimizer_state']
    require(set(saved) == set(expected_opt) and type(saved['version']) is int and saved['version'] == 1
            and saved['independent_decay'] == expected_opt['independent_decay']
            and len(saved['optimizers']) == len(expected_opt['optimizers']), 'Optimizer composition differs')
    for actual, template, child in zip(saved['optimizers'], expected_opt['optimizers'], optimizer.optimizers):
        require(set(actual) == {'state', 'param_groups'}
                and digest(actual['param_groups']) == digest(template['param_groups']), 'Optimizer groups/scheduled rates differ')
        ids = [i for group in template['param_groups'] for i in group['params']]
        require(set(actual['state']) == set(ids) and all(type(i) is int for i in actual['state']),
                'Incomplete or extra optimizer parameter states')
        for group in template['param_groups']:
            for i, name in zip(group['params'], group['parameter_names']):
                values = actual['state'][i]
                muon = isinstance(child, torch.optim.Muon)
                keys = {'momentum_buffer'} if muon else {'step', 'exp_avg', 'exp_avg_sq'}
                require(set(values) == keys, 'Optimizer tensor keys differ')
                for key in keys - {'step'}: _shapes({name: values[key]}, {name: named[name]})
                if not muon:
                    step = values['step']
                    require(isinstance(step, torch.Tensor) and step.shape == torch.Size([])
                            and step.dtype == torch.float32 and step.device.type == 'cpu'
                            and float(step) == steps, 'Optimizer update counter differs')
    result = audit_values(raw, averaged)
    result.update(optimizer_structure_audited=True, parameter_count=PARAMETERS,
                  optimizer_update_count=steps, snapshots=snapshots)
    # Historical helper's explicit nonclaims remain: no resume or EMA replay here.
    return result


def check_measurements(request, measurements, outputs, summary, metrics):
    cfg = request['config']; t = cfg['training']; steps = request['spec']['steps']
    require(digest(cfg) == digest(config(Path(t['checkpoint_dir']).parent, request['spec'])),
            'Request differs from finite activation recipe')
    require(outputs.get('status') == 'completed' and outputs.get('config_sha256') == digest(cfg)
            and summary.get('device') == 'mps' and summary.get('stop_reason') == 'max_iters'
            and type(summary.get('completed_steps')) is int and summary['completed_steps'] == steps
            and summary.get('max_iters') == steps and summary.get('best_checkpoint') is None
            and summary.get('final_checkpoint') == outputs['raw']['path'], 'Incomplete training endpoint')
    require(summary.get('metrics') == metrics and metrics and summary.get('final_metric') == metrics[-1],
            'Metrics/summary agreement differs')
    expected_steps = [i for i in range(steps) if i % t['eval_every'] == 0 or i == steps-1]
    require([r.get('step') for r in metrics] == expected_steps, 'Metric coverage differs')
    schedule = schedule_contract(OmegaConf.create(t))
    for row in metrics:
        i = row['step']; ratio = multiplier(schedule, i)
        require(row.get('train_aux_loss') is None and row.get('val_aux_loss') is None,
                'Activation comparison requires CE-only objective')
        for a, b in [('train_loss', 'train_ce_loss'), ('train_total_loss', 'train_ce_loss'), ('val_loss', 'val_ce_loss')]:
            require(type(row.get(a)) in (int, float) and math.isfinite(row[a]) and row[a] >= 0
                    and row[a] == row.get(b), 'Invalid CE training metric')
        require(row.get('tokens_processed') == (i+1)*8*128
                and row.get('lr_multiplier') == ratio and row.get('base_learning_rate') == t['learning_rate']
                and row.get('scheduled_learning_rate') == t['learning_rate']*ratio,
                'Metric schedule or token budget differs')
    require(set(measurements) == {'raw', 'h200', 'training_contract', 'cache'}, 'Measurement panel differs')
    for role in ('raw', 'h200'):
        score = measurements[role]; _score(score, PARAMETERS)
        require(score['checkpoint'] == outputs[role]['path'] and score['checkpoint_sha256'] == outputs[role]['sha256'],
                'Evaluation/producer checkpoint identity differs')
    avg = summary.get('weight_averaging', {})
    expected_snapshots = 8 if request['spec']['stage'] == 'pilot' else 41
    require(avg == dict(spec=t['weight_averaging'], completed_steps=steps, snapshots=expected_snapshots),
            'Summary EMA differs')
    return dict(metric_rows=len(metrics), steps=steps, snapshots=expected_snapshots,
                target_count=34278, target_bytes=111536, h200_ce=measurements['h200']['mean_ce'])


def check_resources(report, events, plan_sha, request_sha, result_sha, pilot):
    _verify_events(report, events, kind=RESOURCE_KIND, limits=profile(pilot), sample_cap=603 if pilot else 7203)
    require(report['plan_sha256'] == plan_sha and report['request_sha256'] == request_sha
            and report['result_sha256'] == result_sha, 'Resource trace provenance differs')
    return dict(samples=len(report['samples']), peak_rss_bytes=max(s['peak_rss_bytes'] for s in report['samples']),
                peak_mps_driver_bytes=max(s['mps_driver_bytes'] for s in report['samples']))


def review_pair(root, request, outputs, contract):
    """Map one immutable local pair and independently verify all tensor values."""
    refs = {}
    for role in ('raw', 'h200'):
        path = Path(outputs[role]['path'])
        require(path.is_absolute() and path == path.resolve() and path.parent == Path(request['config']['training']['checkpoint_dir']),
                'Pair escaped requested slot')
        ref = reference(root, str(path.relative_to(root)))
        require(ref['size'] <= 2*1024**3 and ref['size'] == outputs[role]['size']
                and ref['sha256'] == outputs[role]['sha256'], 'Pair changed')
        refs[role] = ref
    require(refs['raw']['path'] != refs['h200']['path'], 'Distinct endpoints required')
    raw = avg = None
    try:
        raw = torch.load(root/refs['raw']['path'], map_location='cpu', weights_only=True, mmap=True)
        avg = torch.load(root/refs['h200']['path'], map_location='cpu', weights_only=True, mmap=True)
        result = check_payloads(request['config'], contract, raw, avg,
                               {k: outputs['raw'][k] for k in ('path', 'sha256')})
    finally:
        del raw, avg; gc.collect()
    require(all(reference(root, ref['path']) == ref for ref in refs.values()), 'Pair changed during review')
    return result, refs
