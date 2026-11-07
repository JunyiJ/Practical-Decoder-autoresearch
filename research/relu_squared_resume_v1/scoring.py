"""Own matched-matrix FFN native-target NLL and exact coverage/byte aggregation."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tiktoken
import torch
from research.body_allocation_model_v1.contract import require
from research.final_ln_scale_model_v1.runtime.models.output_bias import digest
from research.relu_squared_model_v1.model import GPT
from .data import architecture, load

POLICY = 'relu_squared_native_distribution_validation_v1'
ARITHMETIC = 'FP32_own_distribution_NLL_then_CPU_FP64_aggregation'
PAIR_ATOL = 3e-5
PAIR_MAX_BYTES = 512*1024


def windows(targets):
    require(type(targets) is int and 0 < targets <= 34278, 'Bounded complete target count required')
    return [(start, min(128, targets-start)) for start in range(0, targets, 128)]


def target_contract(model, data, cfg):
    require(type(model) is GPT and model.signature == architecture(cfg), 'Genuine own FFN model/config required')
    model.validate(); require(not model.training, 'Evaluation model required')
    expected = load(cfg, 'cpu'); ids = data.val_data
    require(data.data_identity == expected.data_identity and data.tokenizer_contract == expected.tokenizer_contract
            and set(data.splits) == {'train', 'val'}, 'Original train/validation identity required')
    require(type(ids) is torch.Tensor and ids.device.type == 'cpu' and ids.dtype == torch.int64 and ids.shape == (34279,), 'Complete physical validation IDs required')
    values = ids.tolist()
    require(digest(values) == expected.data_identity['validation_ids_sha256']
            and digest(values[1:]) == expected.data_identity['validation_target_ids_sha256'], 'Physical target ordering differs')
    enc = tiktoken.get_encoding('gpt2'); n = 34278
    result = dict(policy=POLICY, validation_tokens=n+1, targets=n, ids_sha256=digest(values), target_ids_sha256=digest(values[1:]),
        target_bytes=sum(len(enc.decode_single_token_bytes(i)) for i in values[1:]), window_counts=[v for _, v in windows(n)],
        block_counts=[min(2048, n-i) for i in range(0, n, 2048)], tokenizer=copy.deepcopy(data.tokenizer_contract),
        context=128, first_target_offset=1, tail_retained=True, holdout_scored=False)
    require(result['target_bytes'] == 111536, 'Original complete native target bytes required')
    return result


def window_losses(model, ids):
    """Bounded numerical core; public score below binds actual physical targets."""
    require(type(model) is GPT and not model.training, 'Own evaluation distribution model required'); model.validate()
    require(type(ids) is torch.Tensor and ids.device.type == 'cpu' and ids.dtype == torch.int64 and ids.ndim == 1
            and 2 <= len(ids) <= 34279 and bool(((ids >= 0) & (ids < 50257)).all()), 'Bounded CPU native IDs required')
    device = next(model.parameters()).device; result = []
    with torch.no_grad():
        for start, count in windows(len(ids)-1):
            x = ids[start:start+count].unsqueeze(0).to(device)
            y = ids[start+1:start+count+1].unsqueeze(0).to(device)
            v = model.loss(x, y, reduction='none')
            require(v.dtype == torch.float32 and v.shape == (count,) and bool(torch.isfinite(v).all()) and bool((v >= 0).all()), 'Complete finite nonnegative own NLL required')
            result.append(v.cpu().to(torch.float64))
    return result


def aggregate(losses, target):
    require(type(losses) is list and [v.numel() for v in losses] == target['window_counts'], 'Exact target/window coverage required')
    require(all(type(v) is torch.Tensor and v.device.type == 'cpu' and v.dtype == torch.float64 and v.ndim == 1
                and bool(torch.isfinite(v).all()) and bool((v >= 0).all()) for v in losses), 'CPU FP64 finite nonnegative losses required')
    values = torch.cat(losses); n = target['targets']; total = float(values.sum())
    require(values.numel() == n and type(target['target_bytes']) is int and target['target_bytes'] > 0, 'Complete target/byte denominator required')
    return dict(target=target, window_loss_sums=[float(v.sum()) for v in losses],
        block_loss_sums=[float(values[i:i+2048].sum()) for i in range(0, n, 2048)], total_loss=total, ce=total/n,
        bits_per_byte=total/math.log(2)/target['target_bytes'], loss_dtype=ARITHMETIC, holdout_scored=False, aux_loss_included=False)


def score(model, data, cfg):
    target = target_contract(model, data, cfg)
    result = aggregate(window_losses(model, data.val_data), target)
    validate_report(result, target)
    return result


def validate_report(report, target):
    require(type(report) is dict and set(report) == {'target', 'window_loss_sums', 'block_loss_sums', 'total_loss', 'ce', 'bits_per_byte', 'loss_dtype', 'holdout_scored', 'aux_loss_included'}, 'Exact own score report required')
    require(report['target'] == target and target['policy'] == POLICY and target['targets'] == 34278
            and target['target_bytes'] == 111536 and target['window_counts'] == [v for _, v in windows(34278)]
            and target['block_counts'] == [min(2048, 34278-i) for i in range(0, 34278, 2048)]
            and target['context'] == 128 and target['first_target_offset'] == 1 and target['tail_retained'] is True
            and target['holdout_scored'] is False and report['holdout_scored'] is False and report['aux_loss_included'] is False, 'Complete native CE-only target required')
    require(report['loss_dtype'] == ARITHMETIC and type(report['total_loss']) is float and math.isfinite(report['total_loss']) and report['total_loss'] >= 0, 'Exact finite score arithmetic required')
    for name, counts in [('window_loss_sums', target['window_counts']), ('block_loss_sums', target['block_counts'])]:
        values = report[name]
        require(type(values) is list and len(values) == len(counts) and all(type(v) is float and math.isfinite(v) and v >= 0 for v in values), 'Complete finite aggregation required')
        require(abs(math.fsum(values)-report['total_loss']) < 1e-8, 'Conserved target loss required')
    require(type(report['ce']) is float and report['ce'] == report['total_loss']/34278, 'CE denominator differs')
    require(type(report['bits_per_byte']) is float and report['bits_per_byte'] == report['total_loss']/math.log(2)/111536, 'Native byte denominator differs')


def publish_pair_then_require(path, cpu_report, mps_report, provenance):
    """Own-policy paired diagnostics, after caller validates BOTH physical reports.

    The frozen projection helper is policy-specific and cannot accept this model.
    This NEW consumer requires its own gate; it never admits a checkpoint/scorer.
    """
    path = Path(path)
    require(path.suffix == '.json' and not path.is_symlink(), 'New ordinary JSON destination required')
    for report in (cpu_report, mps_report):
        require(type(report) is dict and type(report.get('ce')) is float and math.isfinite(report['ce'])
                and report['ce'] >= 0 and type(report.get('target')) is dict,
                'Already caller-validated finite physical reports required')
    require(type(provenance) is dict and bool(provenance), 'Explicit caller source/state/runtime provenance required')
    cpu, mps = copy.deepcopy(cpu_report), copy.deepcopy(mps_report)
    difference = abs(cpu['ce']-mps['ce'])
    targets_equal = cpu['target'] == mps['target']
    coverage = all(t.get('policy') == POLICY and t.get('targets') == 34278 and t.get('target_bytes') == 111536
                   and t.get('context') == 128 and t.get('first_target_offset') == 1
                   and t.get('tail_retained') is True and t.get('holdout_scored') is False
                   for t in (cpu['target'], mps['target']))
    result = dict(kind='relu_squared_validation_pair_record_v1', cpu=cpu, mps=mps,
                  provenance=copy.deepcopy(provenance), cpu_mps_ce_difference=difference,
                  absolute_ce_tolerance=PAIR_ATOL, ce_pass=difference <= PAIR_ATOL,
                  target_equal=targets_equal, canonical_coverage_pass=coverage,
                  accepted=difference <= PAIR_ATOL and targets_equal and coverage,
                  physical_scorer_or_checkpoint_acceptance=False)
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    require(len(raw) <= PAIR_MAX_BYTES, 'Own paired report metadata cap exceeded')
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    require(result['accepted'], 'CPU/MPS comparison rejected; both reports saved SHA256='+hashlib.sha256(raw).hexdigest())
    return result
