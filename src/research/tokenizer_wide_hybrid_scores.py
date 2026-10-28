"""Read-only draft score adapter with meta-device checkpoint inspection.

No value/finiteness audit, optimizer replay, evaluation, dispatcher or adoption.
Callers must bind expected inputs to the frozen plan and audited runtime/source.
Only use on locally produced research checkpoints (torch pickle format).
"""
import json
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..models.gpt import GPT
from .protocol import digest, file_digest
from .tokenizer_wide_hybrid_config import rank_config
from . import tokenizer_wide_hybrid_rules as rules


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path, expected_hash):
    _require(path.is_file() and path.stat().st_size <= 4*1024**2, 'Missing/oversized metadata')
    _require(file_digest(path) == expected_hash, 'Metadata hash mismatch')
    value = json.loads(path.read_text())
    _require(file_digest(path) == expected_hash, 'Metadata changed while reading')
    return value


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _score(score, parameters):
    counts = [min(2048, 34278-i) for i in range(0, 34278, 2048)]
    fixed = dict(context=128, eval_batch_size=8, block_tokens=2048, target_count=34278,
                 target_bytes=111536, window_count=math.ceil(34278/128), parameters=parameters,
                 block_target_counts=counts)
    _require(all(digest(score.get(k)) == digest(v) for k, v in fixed.items()),
             'Saved evaluation coverage/parameter count differs')
    sums = score.get('block_loss_sums', [])
    _require(len(sums) == len(counts) and all(_finite(x) for x in sums), 'Invalid loss sums')
    total = sum(sums)
    ce = total/34278
    _require(math.isfinite(total) and ce < 700, 'Invalid total loss')
    for key, expected in dict(mean_ce=ce, bits_per_byte=total/111536/math.log(2),
                              perplexity=math.exp(ce)).items():
        _require(_finite(score.get(key)) and math.isclose(score[key], expected, rel_tol=1e-12, abs_tol=1e-12),
                 'Saved score arithmetic mismatch: '+key)


def _shapes(panel, expected):
    _require(set(panel) == set(expected), 'Checkpoint tensor names differ')
    _require(all(isinstance(panel[k], torch.Tensor) and panel[k].shape == expected[k].shape
                 and panel[k].dtype == expected[k].dtype for k in expected),
             'Checkpoint tensor shape/dtype differs')


def review_trial(trial, expected_request, request_sha256, result_sha256, training_contract):
    """Return selection row and aligned H200 blocks only after metadata review.

    Expected request/config/contract and file hashes must come from independent
    frozen provenance, never be inferred from the result being reviewed.
    """
    trial = Path(trial).resolve()
    request_path, result_path = trial/'request.json', trial/'result.json'
    request = _json(request_path, request_sha256)
    _require(digest(request) == digest(expected_request), 'Request differs from expected plan')
    spec = request['spec']
    # Full stage completion/order and saved selection are enforced by the controller.
    _require(set(spec) == {'stage', 'family', 'seed', 'lr', 'divisor', 'width', 'steps'}, 'Unexpected spec fields')
    _require(spec['stage'] in ('lr', 'rank', 'confirm') and spec['family'] in rules.FAMILIES
             and spec['lr'] in rules.RATES and type(spec['seed']) is int
             and spec['seed'] in (rules.SEEDS if spec['stage'] == 'confirm' else (3501,))
             and (spec['stage'] != 'lr' or spec['divisor'] == 4)
             and (spec['stage'] != 'rank' or spec['divisor'] in (2, 8))
             and type(spec['width']) is int and spec['width'] == 512
             and type(spec['steps']) is int and spec['steps'] == 1200, 'Undeclared quality spec')
    cfg = rank_config(trial, spec['family'], 512, spec['seed'], spec['lr'], 1200, 'mps',
                      bottleneck_divisor=spec['divisor'])
    _require(digest(request['config']) == digest(cfg), 'Requested config differs from explicit recipe')
    _require(Path(request['result_path']).resolve() == result_path, 'Wrong result path')
    _require(training_contract.get('tokenizer') is not None
             and digest(training_contract.get('model')) == digest(cfg['model'])
             and digest(training_contract.get('weight_averaging')) == digest(cfg['training']['weight_averaging']),
             'Missing or inconsistent expected training contract')
    result = _json(result_path, result_sha256)
    _require(all(digest(result.get(k)) == digest(v) for k, v in spec.items())
             and result.get('status') == 'completed' and result.get('holdout_consumed') is False
             and result.get('request_sha256') == request_sha256, 'Result identity/status mismatch')
    _require(type(result['training']['completed_steps']) is int and result['training']['completed_steps'] == 1200
             and type(result['training']['weight_averaging']['snapshots']) is int
             and result['training']['weight_averaging']['snapshots'] == 41, 'Incomplete training/EMA')
    paths = {k: Path(result[k]['checkpoint']).resolve() for k in ('raw', 'h200')}
    _require(paths['raw'] != paths['h200'] and not paths['raw'].samefile(paths['h200']), 'Raw/export must be distinct')
    _require(Path(result['training']['final_checkpoint']).resolve() == paths['raw'], 'Wrong raw endpoint')
    with torch.device('meta'):
        model = GPT(OmegaConf.create(cfg['model']))
    expected_state = model.state_dict()
    named = dict(model.named_parameters())
    parameters = sum(p.numel() for p in named.values())
    for kind in ('raw', 'h200'):
        path, score = paths[kind], result[kind]
        _require(path.is_relative_to((trial/'checkpoints').resolve()) and path.is_file()
                 and 0 < path.stat().st_size <= 2*1024**3, 'Checkpoint location/size invalid')
        _require(file_digest(path) == score['checkpoint_sha256'], 'Checkpoint hash mismatch')
        _score(score, parameters)
        payload = torch.load(path, map_location='meta', weights_only=False)
        _require(digest(payload['cfg']) == digest(cfg) and type(payload['step']) is int
                 and payload['step'] == 1200 and payload['model_math_version'] == 2
                 and payload['parameter_layout_version'] == 2, 'Checkpoint config/version/step mismatch')
        _shapes(payload['model_state'], expected_state)
        if kind == 'raw':
            _require('averaged_weights' not in payload and 'optimizer_state' in payload,
                     'Raw checkpoint is not an online checkpoint')
            _require(digest(payload['training_state']['contract']) == digest(training_contract),
                     'Saved training/tokenizer contract differs')
            ema = payload['training_state']['weight_averaging_state']
            _require(set(ema['averages']) == {'200'}, 'Undeclared EMA panel')
            _shapes(ema['averages']['200'], named)
        else:
            _require('optimizer_state' not in payload and 'training_state' not in payload,
                     'Export must be inference-only')
            _require(digest(payload['tokenizer_contract']) == digest(training_contract['tokenizer'])
                     and payload['parent_checkpoint']['sha256'] == result['raw']['checkpoint_sha256']
                     and Path(payload['parent_checkpoint']['path']).resolve() == paths['raw'],
                     'Export tokenizer/parent lineage differs')
            ema = payload['averaged_weights']
            _require(type(ema['half_life']) is int and ema['half_life'] == 200
                     and ema['purpose'] == 'inference_only_not_optimizer_resume', 'Wrong export purpose/half-life')
        _require(digest(ema['spec']) == digest(cfg['training']['weight_averaging'])
                 and type(ema['completed_steps']) is int and ema['completed_steps'] == 1200
                 and type(ema['snapshots']) is int and ema['snapshots'] == 41
                 and ema['parameter_names'] == list(named), 'EMA metadata/name order differs')
        del payload, ema
        _require(file_digest(path) == score['checkpoint_sha256'], 'Checkpoint changed during review')
    # Recheck the complete pair, request and result after all reads.
    _require(file_digest(request_path) == request_sha256 and file_digest(result_path) == result_sha256
             and all(file_digest(paths[k]) == result[k]['checkpoint_sha256'] for k in paths),
             'Trial artifacts changed during review')
    return dict(selection_row=dict(spec, status='completed', ce=result['h200']['mean_ce']),
                h200_loss_sums=result['h200']['block_loss_sums'],
                target_counts=result['h200']['block_target_counts'],
                request_sha256=request_sha256, result_sha256=result_sha256,
                checkpoint_hashes={k: result[k]['checkpoint_sha256'] for k in paths},
                metadata_reviewed=True, tensor_values_audited=False, adopted=False)
