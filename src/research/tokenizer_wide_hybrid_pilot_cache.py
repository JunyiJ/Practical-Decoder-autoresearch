"""Prospective serial raw/H200 cache audit; caller supplies guards and locks."""
import gc
import json
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..attention.cache import build_caches
from ..models.gpt import GPT
from ..utils.checkpoint import checkpoint_loader, load_model_weights
from .protocol import digest, file_digest
from .tokenizer_wide_hybrid_driver import CORPUS_SHA256
from .tokenizer_wide_hybrid_pilot_config import config, specs, PROJECTION
from .tokenizer_wide_hybrid_scores import _json, _require
from .tokenizer_wide_hybrid_tensors import review_values, _finite_tree
from .tokenizer_wide_hybrid_measurements import validate_mps_model, check_kernel, RTOL, ATOL
from .tokenizer_wide_hybrid_timing import allocated_bytes, token_digest

CACHE_CONTRACT = dict(context=128, batches=[1,8], tokens=96, prefill_tokens=32,
                      decode_tokens=64, reset_replay=True, rtol=RTOL, atol=ATOL,
                      device='mps', dtype='float32', endpoints=['raw','h200'])


def _projection(family):
    rows = [r for r in json.loads(PROJECTION.read_text())['records']
            if r['width'] == 512 and r['family'] == family and r['bottleneck_divisor'] == 2]
    _require(len(rows) == 1, 'Missing or duplicate pilot geometry projection')
    return rows[0]


def check_model(model, cfg, family, token_ids, expected_sha256):
    validate_mps_model(model, cfg, family)
    _require(token_digest(token_ids) == expected_sha256, 'Training cache input changed')
    projection = _projection(family)
    parameters = sum(p.numel() for p in model.parameters())
    storage = allocated_bytes(model.parameters())
    _require(parameters == projection['parameters'] and storage == projection['parameter_bytes'],
             'Loaded parameter geometry differs from pilot projection')
    fixed = torch.tensor(token_ids, device='mps', dtype=torch.int64)
    rows = []
    for batch in CACHE_CONTRACT['batches']:
        ids = fixed.unsqueeze(0).repeat(batch, 1)
        caches = build_caches(cfg, 'mps', batch_size=batch, dtype=torch.float32)
        result = check_kernel(model, ids, lambda: caches)
        torch.mps.synchronize()
        state = allocated_bytes(b for c in caches for b in c.buffers())
        _require(state == batch*projection['cache_bytes_batch1'], 'Actual cache allocation differs from projection')
        rows.append(dict(batch=batch, **result, state_bytes=state,
            observed_mps_current_bytes=torch.mps.current_allocated_memory(),
            observed_mps_driver_bytes=torch.mps.driver_allocated_memory()))
        del ids, caches
    return dict(parameters=parameters, parameter_storage_bytes=storage, checks=rows)


def review_caches(binding, expected_tokenizer, expected_token_sha256):
    """Review tensors, then load one checkpoint at a time for full/cache/reset.

    This is not a worker or launch gate. Frozen provenance, nine-case numerical
    prerequisites, full-panel fit, locks and RSS/driver/time guards belong to the
    caller. Returned allocator samples are observations, not deployment peaks.
    """
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1' or not torch.backends.mps.is_available():
        raise RuntimeError('Strict MPS required')
    _require(set(binding) == {'trial','request','request_sha256','result_sha256','training_contract'},
             'Incomplete pilot binding')
    trial = Path(binding['trial']).resolve()
    request = binding['request']; spec = request['spec']; family = spec['family']
    _require(any(digest(spec) == digest(s) for s in specs()), 'Undeclared pilot spec')
    cfg_dict = config(trial, family)
    _require(digest(request['config']) == digest(cfg_dict)
             and Path(request['result_path']).resolve() == trial/'result.json', 'Pilot config/path differs')
    _require(isinstance(expected_tokenizer,dict) and bool(expected_tokenizer)
             and digest(binding['training_contract'].get('tokenizer')) == digest(expected_tokenizer),
             'Missing or inconsistent tokenizer binding')
    corpus = Path(cfg_dict['data']['path']).resolve()
    _require(file_digest(corpus) == CORPUS_SHA256, 'Corpus changed')
    tensor_review = review_values(trial, request, binding['request_sha256'], binding['result_sha256'],
                                  binding['training_contract'], kind='pilot')
    bound = dict(tensor_review['files']); bound[str(corpus)] = CORPUS_SHA256
    bound[str(PROJECTION.resolve())] = file_digest(PROJECTION)
    def recheck():
        _require(all(file_digest(Path(p)) == sha for p,sha in bound.items()), 'Pilot cache inputs changed')
    recheck()
    result = _json(trial/'result.json', binding['result_sha256'])
    endpoints = {}
    for kind in CACHE_CONTRACT['endpoints']:
        path = Path(result[kind]['checkpoint']).resolve()
        _require(path.is_relative_to((trial/'checkpoints').resolve()) and str(path) in bound,
                 'Checkpoint outside audited pair')
        payload = model = loader = None
        try:
            recheck()
            payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            saved_tokenizer = (payload['training_state']['contract']['tokenizer'] if kind == 'raw'
                               else payload['tokenizer_contract'])
            _require(digest(payload['cfg']) == digest(cfg_dict)
                     and digest(saved_tokenizer) == digest(expected_tokenizer), 'Loaded checkpoint identity differs')
            _finite_tree(payload['model_state'])
            cfg = OmegaConf.create(cfg_dict)
            loader = checkpoint_loader(cfg, payload, str(corpus), 'cpu')
            _require(digest(loader.tokenizer_contract) == digest(expected_tokenizer), 'Rebuilt tokenizer differs')
            ids = loader.train_data[:96].tolist()
            _require(token_digest(ids) == expected_token_sha256, 'Training cache tokens differ')
            model = GPT(cfg.model).to('mps').eval()
            load_model_weights(model, payload)
            del payload, loader
            payload = loader = None
            gc.collect()
            endpoints[kind] = dict(checkpoint=str(path), sha256=bound[str(path)],
                **check_model(model, cfg, family, ids, expected_token_sha256))
            recheck()
        finally:
            del model, payload, loader
            gc.collect()
            torch.mps.empty_cache()
    recheck()
    return dict(status='pilot_cache_checks_complete_pending_resource_review', spec=spec,
                token_sha256=expected_token_sha256, tokenizer_sha256=digest(expected_tokenizer),
                contract=CACHE_CONTRACT, endpoints=endpoints, tensor_review=tensor_review,
                files=bound, adopted=False)
