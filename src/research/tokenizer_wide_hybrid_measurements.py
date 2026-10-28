"""Prospective cached correctness/timing callbacks; no loader or dispatcher.

Only tiny synthetic CPU kernels are tested so far. The public adapter requires
strict FP32 MPS and explicit width512 rank-policy models. It does not establish
checkpoint provenance or authorize bypassing campaign launch guards.
"""
import os
import time

import torch
from omegaconf import OmegaConf

from ..attention.cache import build_caches
from ..models.tokenizer_hybrid import RANK_POLICY, hybrid_contract, validate_hybrid_model
from .protocol import digest
from .tokenizer_wide_hybrid_config import LAYOUT
from .tokenizer_wide_hybrid_timing import FAMILIES, BATCHES, allocated_bytes, token_digest

RTOL = 3e-4
ATOL = 3e-5
CHUNKS = ((0, 32),) + tuple((p, p+1) for p in range(32, 96))


def _ids(ids):
    if ids.ndim != 2 or ids.shape[0] not in BATCHES or ids.shape[1] != 96 or ids.dtype != torch.int64:
        raise ValueError('Expected batch1/8 with96 int64 tokens')


def _finite_logits(logits, shape):
    if (logits.ndim != 3 or tuple(logits.shape[:2]) != tuple(shape)
            or logits.shape[2] <= 0 or logits.dtype != torch.float32
            or not bool(torch.isfinite(logits).all())):
        raise ValueError('Invalid/nonfinite FP32 logits')


@torch.no_grad()
def check_kernel(model, ids, cache_factory):
    """Full96 versus32+64 one-token cached calls, then reset/replay all96."""
    _ids(ids)
    full = model(ids)[0]
    _finite_logits(full, ids.shape)
    caches = cache_factory()
    if not caches:
        raise ValueError('No cache layers')
    error = 0.
    for replay in range(2):
        if replay:
            for cache in caches:
                cache.reset()
        for lo, hi in CHUNKS:
            logits = model(ids[:, lo:hi], cache=caches, start_pos=lo)[0]
            _finite_logits(logits, (ids.shape[0], hi-lo))
            target = full[:, lo:hi]
            torch.testing.assert_close(logits, target, rtol=RTOL, atol=ATOL, equal_nan=False)
            error = max(error, float((logits-target).abs().max()))
            del logits, target
    return dict(passed=True, max_absolute_error=error)


@torch.no_grad()
def measure_kernel(model, ids, cache_factory, *, synchronize, clock, memory):
    """Dependency-injected mechanics; production adapter supplies MPS APIs.

    Cache construction and input preparation are outside timed intervals. The
    decode loop is64 teacher-forced calls; no per-token synchronization is added.
    Numerical checks occur separately before any warmup/measurement sequence.
    """
    _ids(ids)
    caches = cache_factory()
    if not caches:
        raise ValueError('No cache layers')
    synchronize()
    started = clock()
    output = model(ids[:, :32], cache=caches, start_pos=0)[0]
    synchronize()
    prefill = clock()-started
    del output
    started = clock()
    for pos in range(32, 96):
        output = model(ids[:, pos:pos+1], cache=caches, start_pos=pos)[0]
        del output
    synchronize()
    decode = clock()-started
    state_bytes = allocated_bytes(b for c in caches for b in c.buffers())
    current, driver = memory()
    return dict(prefill_seconds=prefill, decode_seconds=decode, state_bytes=state_bytes,
                observed_mps_current_bytes=current, observed_mps_driver_bytes=driver)


def validate_mps_model(model, cfg, family):
    """Shared single-model geometry/device check; no checkpoint provenance."""
    if family not in FAMILIES:
        raise ValueError("Undeclared model family")
    contract = hybrid_contract(cfg.model)
    if (contract is None or contract['policy'] != RANK_POLICY or contract['width'] != 512
            or contract['layout'] != LAYOUT[family] or model.training):
        raise ValueError('Explicit eval width512 rank-policy family required')
    plain = lambda c: OmegaConf.to_container(OmegaConf.create(c), resolve=True)
    if digest(plain(model.cfg)) != digest(plain(cfg.model)):
        raise ValueError('Loaded model and cache config differ')
    validate_hybrid_model(model, contract)
    params = list(model.parameters())
    if not params or any(p.device.type != 'mps' or p.dtype != torch.float32 for p in params):
        raise ValueError('Model parameters must all be MPS FP32')
    for buffer in model.buffers():
        if buffer.device.type != 'mps' or (buffer.is_floating_point() and buffer.dtype != torch.float32):
            raise ValueError('Model buffers must be on MPS with FP32 floating values')


def mps_callbacks(models, configs, token_ids, expected_token_sha256):
    """Validate loaded-model geometry/device; return (check,measure) callbacks.

    Caller must hold global/run locks, run the resource watchdog and independently
    verify frozen source/data/config/checkpoint provenance and complete audits.
    No worker/CLI here; accepting model objects does not certify their weights.
    """
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1' or not torch.backends.mps.is_available():
        raise RuntimeError('Strict MPS required')
    if set(models) != set(FAMILIES) or set(configs) != set(FAMILIES):
        raise ValueError('All three model/config families are required')
    token_ids = tuple(token_ids)
    if token_digest(token_ids) != expected_token_sha256:
        raise ValueError('Timing token identity changed')
    for family in FAMILIES:
        validate_mps_model(models[family], configs[family], family)
    fixed = torch.tensor(token_ids, device='mps', dtype=torch.int64)
    inputs = {b: fixed.unsqueeze(0).repeat(b, 1) for b in BATCHES}

    def arguments(family, batch):
        if family not in FAMILIES or type(batch) is not int or batch not in BATCHES:
            raise ValueError('Undeclared model or batch')
        if models[family].training:
            raise ValueError('Model left eval mode')
        return models[family], inputs[batch], lambda: build_caches(
            configs[family], 'mps', batch_size=batch, dtype=torch.float32)

    def check(family, batch):
        return check_kernel(*arguments(family, batch))

    def measure(family, batch):
        return measure_kernel(*arguments(family, batch), synchronize=torch.mps.synchronize,
            clock=time.perf_counter, memory=lambda: (
                torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()))

    return check, measure
