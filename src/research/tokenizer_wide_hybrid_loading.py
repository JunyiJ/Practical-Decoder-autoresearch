"""Prospective selected-export loading; not a benchmark worker or launch gate.

Caller must verify frozen plans, complete quality/tensor audits and source/runtime,
acquire locks, and start resource guards before calling. No CLI/dispatch here.
"""
import gc
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..models.gpt import GPT
from ..utils.checkpoint import read_checkpoint, load_model_weights, checkpoint_loader
from .protocol import digest, file_digest
from .tokenizer_wide_hybrid_config import rank_config
from .tokenizer_wide_hybrid_rules import FAMILIES, SEEDS, confirmation_specs
from .tokenizer_wide_hybrid_scores import review_trial, _json
from .tokenizer_wide_hybrid_timing import allocated_bytes, token_digest

CORPUS_SHA256 = '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_selected_exports(bindings, seed, selection, expected_tokenizer, expected_token_sha256):
    """Bindings are independent frozen requests/contracts/hashes, not result-derived.

    Each binding contains trial, request, request_sha256, result_sha256 and
    training_contract. Existing metadata score adapter rechecks the raw/H200
    pair. This function checks exported CPU tensors are finite before loading,
    but does not replace the independent exact-EMA/raw/optimizer tensor audit.
    Three models remain resident for the future timing worker.
    """
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1' or not torch.backends.mps.is_available():
        raise RuntimeError('Strict MPS required')
    _require(type(seed) is int and seed in SEEDS and set(bindings) == set(FAMILIES),
             'All three fresh families required')
    _require(isinstance(expected_tokenizer, dict) and bool(expected_tokenizer), 'Missing expected tokenizer')
    specs = {s['family']: s for s in confirmation_specs(selection) if s['seed'] == seed}
    # Complete all cheap request/config/corpus checks before loading any tensors.
    prepared = {}
    for family in FAMILIES:
        binding = bindings[family]
        _require(set(binding) == {'trial', 'request', 'request_sha256', 'result_sha256', 'training_contract'},
                 'Incomplete export binding')
        trial = Path(binding['trial']).resolve()
        request = binding['request']
        cfg = rank_config(trial, family, 512, seed, specs[family]['lr'], 1200, 'mps',
                          bottleneck_divisor=specs[family]['divisor'])
        _require(digest(request['spec']) == digest(specs[family]) and digest(request['config']) == digest(cfg)
                 and Path(request['result_path']).resolve() == trial/'result.json',
                 'Export request differs from selected fresh recipe')
        _require(digest(binding['training_contract'].get('tokenizer')) == digest(expected_tokenizer),
                 'Expected tokenizer differs between families')
        corpus = Path(cfg['data']['path'])
        _require(file_digest(corpus) == CORPUS_SHA256, 'Corpus changed')
        prepared[family] = (trial, cfg, corpus)
    models, configs, metadata, receipts = {}, {}, {}, {}
    fixed = None
    for family in FAMILIES:
        binding = bindings[family]
        trial, expected_cfg, corpus = prepared[family]
        receipt = review_trial(trial, binding['request'], binding['request_sha256'],
                               binding['result_sha256'], binding['training_contract'])
        result = _json(trial/'result.json', binding['result_sha256'])
        path = Path(result['h200']['checkpoint']).resolve()
        sha = receipt['checkpoint_hashes']['h200']
        _require(file_digest(path) == sha, 'Export changed before loading')
        payload = read_checkpoint(path, map_location='cpu')
        _require(file_digest(path) == sha, 'Export changed while loading')
        _require(digest(payload['cfg']) == digest(expected_cfg)
                 and digest(payload.get('tokenizer_contract')) == digest(expected_tokenizer),
                 'Loaded export config/tokenizer changed')
        # Review_trial verified names/dtypes/shapes with a meta construction.
        # Values here must be actual CPU tensors, not a meta-fixture success.
        _require(bool(payload['model_state']) and all(isinstance(v, torch.Tensor)
                 and v.device.type == 'cpu' and bool(torch.isfinite(v).all())
                 for v in payload['model_state'].values()), 'Nonfinite or non-CPU export tensors')
        cfg = OmegaConf.create(expected_cfg)
        loader = checkpoint_loader(cfg, payload, str(corpus), 'cpu')
        _require(digest(loader.tokenizer_contract) == digest(expected_tokenizer), 'Rebuilt tokenizer changed')
        ids = loader.train_data[:96].tolist()
        _require(token_digest(ids) == expected_token_sha256, 'Training timing tokens changed')
        _require(fixed is None or fixed == ids, 'Timing inputs differ between families')
        fixed = ids
        model = GPT(cfg.model).to('mps').eval()
        load_model_weights(model, payload)
        torch.mps.synchronize()
        models[family], configs[family] = model, cfg
        metadata[family] = dict(checkpoint=str(path), sha256=sha, seed=seed,
            family=family, divisor=specs[family]['divisor'], lr=specs[family]['lr'],
            config_sha256=digest(expected_cfg), tokenizer_sha256=digest(expected_tokenizer),
            parameters=sum(p.numel() for p in model.parameters()),
            parameter_storage_bytes=allocated_bytes(model.parameters()))
        receipts[family] = receipt
        del payload, loader
        gc.collect()
    # Bind the whole set again after all loading; never silently mix changing artifacts.
    for family in FAMILIES:
        binding = bindings[family]; trial, _, corpus = prepared[family]
        _require(file_digest(trial/'request.json') == binding['request_sha256']
                 and file_digest(trial/'result.json') == binding['result_sha256']
                 and file_digest(corpus) == CORPUS_SHA256, 'Bound inputs changed during triple loading')
        result = _json(trial/'result.json', binding['result_sha256'])
        _require(all(file_digest(Path(result[k]['checkpoint'])) == receipts[family]['checkpoint_hashes'][k]
                     for k in ('raw', 'h200')), 'Checkpoint pair changed during triple loading')
    return dict(models=models, configs=configs, metadata=metadata, token_ids=fixed,
                token_sha256=expected_token_sha256, receipts=receipts)
