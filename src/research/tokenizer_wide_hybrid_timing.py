"""Prospective inference-triple measurement contract; no loader or dispatcher.

Callbacks own actual cache checks/timing. This module cannot certify model,
checkpoint, tokenizer, device or tensor identity, or authorize a benchmark.
"""
import hashlib
import math
import struct

from .tokenizer_wide_hybrid_rules import FAMILIES, SEEDS, confirmation_specs

BATCHES = (1, 8)
WARMUPS = 2
REPEATS = 8


def token_digest(ids):
    """Exactly 96 GPT-2 training IDs, canonical unsigned little-endian uint32."""
    if len(ids) != 96 or any(type(x) is not int or not 0 <= x < 50257 for x in ids):
        raise ValueError('Expected 96 GPT-2 token IDs')
    return hashlib.sha256(struct.pack('<96I', *ids)).hexdigest()


def order(seed, repeat):
    if type(seed) is not int or seed not in SEEDS or type(repeat) is not int or repeat < 0:
        raise ValueError('Undeclared seed/repeat')
    offset = (SEEDS.index(seed) + repeat) % 3
    return FAMILIES[offset:] + FAMILIES[:offset]


def allocated_bytes(tensors):
    """Unique backing tensor storage, including unused capacity of a view.

    This measures tensor storage, not allocator overhead or deployment peaks.
    The caller supplies all cache buffers or all model parameters separately.
    """
    storages = {}
    for tensor in tensors:
        if tensor.device.type == 'meta':
            raise ValueError('Meta tensors do not measure allocated storage')
        storage = tensor.untyped_storage()
        size = storage.nbytes()
        if size:
            key = (str(tensor.device), storage.data_ptr())
            if key in storages and storages[key] != size:
                raise ValueError('Inconsistent storage identity')
            storages[key] = size
    return sum(storages.values())


def _finite(value, *, positive=False):
    return type(value) in (float, int) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def _correctness(value):
    if (set(value) != {'passed', 'max_absolute_error'} or value['passed'] is not True
            or not _finite(value['max_absolute_error'])):
        raise ValueError('Cache correctness callback did not pass')
    return dict(value)


def _measurement(value, batch):
    keys = {'prefill_seconds', 'decode_seconds', 'state_bytes',
            'observed_mps_current_bytes', 'observed_mps_driver_bytes'}
    if set(value) != keys:
        raise ValueError('Incomplete or unexpected timing fields')
    if not all(_finite(value[k], positive=True) for k in ('prefill_seconds', 'decode_seconds')):
        raise ValueError('Nonpositive or nonfinite timing')
    for k in keys - {'prefill_seconds', 'decode_seconds'}:
        if type(value[k]) is not int or value[k] <= 0:
            raise ValueError('Invalid allocated-memory observation')
    throughput = 64*batch/value['decode_seconds']
    if not math.isfinite(throughput):
        raise ValueError('Nonfinite throughput')
    return dict(value, decode_tokens_per_second=throughput)


def measure_triple(seed, selection, token_ids, expected_token_sha256, *, check, measure):
    """Callbacks (family,batch) use the same immutable tokens/selected models.

    Call only inside a separately guarded, provenance-verified worker after all
    quality/tensor audits. No such worker is implemented here. Tests inject
    synthetic callbacks; no model or corpus is loaded by this function.
    Exceptions propagate immediately, without retries or a completed report.
    """
    order(seed, 0)
    confirmation_specs(selection)
    # Snapshot input identity before callbacks; never let callbacks select data.
    token_ids = tuple(token_ids)
    if token_digest(token_ids) != expected_token_sha256:
        raise ValueError('Timing token identity changed')
    specs = [dict(s) for s in confirmation_specs(selection) if s['seed'] == seed]
    correctness = []
    # Global barrier: every family/batch passes before even warmup timing.
    for batch in BATCHES:
        for family in order(seed, 0):
            correctness.append(dict(batch=batch, family=family, **_correctness(check(family, batch))))
    rows, warmups, state = [], [], {}
    for batch in BATCHES:
        for phase, count, destination in (('warmup', WARMUPS, warmups), ('measured', REPEATS, rows)):
            for repeat in range(count):
                for position, family in enumerate(order(seed, repeat)):
                    sample = _measurement(measure(family, batch), batch)
                    key = (family, batch)
                    if key in state and state[key] != sample['state_bytes']:
                        raise ValueError('Allocated state changed between repeats')
                    state[key] = sample['state_bytes']
                    destination.append(dict(batch=batch, phase=phase, repeat=repeat,
                                            position=position, family=family, **sample))
    return dict(status='measurements_complete_pending_artifact_audit', seed=seed,
                selected_specs=specs, token_sha256=expected_token_sha256,
                fixed_train_token_ids=list(token_ids), token_hash_encoding='uint32-little-endian',
                context_tokens=128, prefill_tokens=32, decode_tokens=64,
                warmups=WARMUPS, repeats=REPEATS, correctness=correctness,
                warmup_records=warmups, records=rows, adopted=False,
                scope='Three models resident. Cache/model loading outside timed regions. '
                      'State bytes count unique backing cache buffers; allocator observations '
                      'are not single-model deployment peaks. Device/artifact identity requires '
                      'external verification. Teacher-forced training tokens; no test access.')
