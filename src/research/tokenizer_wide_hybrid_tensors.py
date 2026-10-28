"""Read-only tensor/export audit; caller owns locks, timeout and RSS guards.

No training, optimizer replay, numerical cache check, freeze or dispatcher.
Use only locally produced checkpoint pairs whose metadata is independently bound.
"""
import gc
import math
from pathlib import Path

import torch

from .protocol import file_digest
from .tokenizer_wide_hybrid_scores import _json, _require, review_pilot, review_trial

CHUNK_ELEMENTS = 262144


def _chunks(tensor):
    _require(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu'
             and tensor.layout == torch.strided and tensor.is_contiguous()
             and not tensor.is_quantized, 'Expected contiguous ordinary CPU tensors')
    flat = tensor.view(-1)
    for start in range(0, flat.numel(), CHUNK_ELEMENTS):
        yield flat[start:start+CHUNK_ELEMENTS]


def _finite_tree(value):
    """Return tensor count; temporary finiteness masks are bounded by one chunk."""
    if isinstance(value, torch.Tensor):
        for chunk in _chunks(value):
            _require(bool(torch.isfinite(chunk).all()), 'Nonfinite checkpoint tensor')
        return 1
    if isinstance(value, dict):
        return sum(_finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_finite_tree(v) for v in value)
    _require(value is None or type(value) in (str, int, bool)
             or type(value) is float and math.isfinite(value), 'Invalid optimizer scalar/container')
    return 0


def _equal(actual, expected):
    _require(isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
             and actual.shape == expected.shape and actual.dtype == expected.dtype,
             'Export tensor geometry differs')
    # Both empty and nonempty tensors must pass device/layout validation.
    left, right = _chunks(actual), _chunks(expected)
    for a, b in zip(left, right):
        _require(torch.equal(a, b), 'Export differs from stored EMA or persistent buffer')


def audit_values(raw, exported):
    """Check actual values, not the correctness of the training EMA recurrence.

    Exact equality proves export agrees with stored EMA. Independent numerical
    continuation/EMA prerequisites must establish that the stored EMA is correct.
    """
    model = raw['model_state']
    state = raw['training_state']['weight_averaging_state']
    names = state['parameter_names']
    _require(isinstance(names, list) and names and all(type(n) is str for n in names)
             and len(set(names)) == len(names) and set(names) <= set(model), 'Invalid parameter names')
    _require(set(state['averages']) == {'200'}, 'Undeclared EMA panel')
    ema = state['averages']['200']
    _require(set(ema) == set(names) and set(exported['model_state']) == set(model),
             'Incomplete EMA/export tensor set')
    _require(exported['averaged_weights']['parameter_names'] == names,
             'Export parameter order differs')
    _require('optimizer_state' in raw and 'averaged_weights' not in raw
             and 'optimizer_state' not in exported and 'training_state' not in exported,
             'Online/export roles differ')
    panels = dict(raw_model=model, stored_ema=ema, optimizer=raw['optimizer_state'],
                  exported_model=exported['model_state'])
    counts = {kind: _finite_tree(panel) for kind, panel in panels.items()}
    _require(counts['optimizer'] > 0, 'Missing optimizer tensors')
    for name, value in model.items():
        _require(isinstance(value, torch.Tensor), 'Model state must contain tensors')
        if name in ema:
            _require(isinstance(ema[name], torch.Tensor) and ema[name].shape == value.shape
                     and ema[name].dtype == value.dtype, 'Stored EMA geometry differs')
        _equal(exported['model_state'][name], ema[name] if name in ema else value)
    return dict(finite_model_ema_optimizer_export_values=True,
                exact_export_matches_stored_ema=True, persistent_buffers_unchanged=True,
                tensor_counts=counts, parameters_compared=len(names),
                buffers_compared=len(model)-len(names),
                optimizer_structure_audited=False, optimizer_resume_audited=False,
                ema_recurrence_audited=False, adopted=False)


def review_values(trial, expected_request, request_sha256, result_sha256,
                  training_contract, *, kind):
    """Bind metadata first, map one raw/export pair, then recheck all hashes.

    CPU mmap avoids explicit file-sized byte buffers and extra model construction.
    It does not guarantee a measured peak: caller must retain the RSS watchdog.
    """
    _require(kind in ('pilot', 'quality'), 'Explicit pilot/quality audit kind required')
    trial = Path(trial).resolve()
    reviewer = review_pilot if kind == 'pilot' else review_trial
    receipt = reviewer(trial, expected_request, request_sha256, result_sha256, training_contract)
    result = _json(trial/'result.json', result_sha256)
    paths = {k: Path(result[k]['checkpoint']).resolve() for k in ('raw', 'h200')}
    hashes = {str(trial/'request.json'): request_sha256, str(trial/'result.json'): result_sha256}
    for name, path in paths.items():
        _require(path.is_relative_to((trial/'checkpoints').resolve()) and path.is_file()
                 and 0 < path.stat().st_size <= 2*1024**3, 'Invalid checkpoint location/size')
        hashes[str(path)] = receipt['checkpoint_hashes'][name]
    _require(paths['raw'] != paths['h200'] and not paths['raw'].samefile(paths['h200']),
             'Distinct raw/export files required')
    def recheck():
        _require(all(file_digest(Path(p)) == sha for p, sha in hashes.items()), 'Audit inputs changed')
    recheck()
    raw = exported = None
    try:
        raw = torch.load(paths['raw'], map_location='cpu', weights_only=True, mmap=True)
        exported = torch.load(paths['h200'], map_location='cpu', weights_only=True, mmap=True)
        values = audit_values(raw, exported)
        recheck()
    finally:
        del raw, exported
        gc.collect()
    return dict(kind=kind, status='export_values_reviewed_pending_numerical_and_resource_audits',
                files=hashes, metadata=receipt, values=values, adopted=False)
