"""New exact chunk-VJP prerequisite using saved weights; never trains a model."""
from array import array
import collections
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from research.output_anchored_tying_frequency_v1 import read, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'research/tokenizer-evidence/byte-output-chunk-pilot-001'
RECIPE = ROOT/'research/tokenizer-evidence/public-inference-correction-001/adoption/recipe.json'
CONFIG = ROOT/'research/tokenizer-evidence/byte-residual-readout-001/quality001/00/config.json'
CAP = 2*1024**2


def compose(p, codes, lengths, reference=False):
    import torch
    from torch.nn import functional as F
    from research.byte_embedding_v1 import embedding
    n, length = codes.shape
    valid = torch.arange(length, device=codes.device)[None, :] < lengths[:, None]
    indices = codes-(codes > 256).long()
    embedded = (F.embedding(indices, p['byte_weight']) if reference else embedding(p['byte_weight'], indices))
    embedded = embedded.masked_fill(~valid[:, :, None], 0)
    pooled = []
    for i, width in enumerate(range(1, 7)):
        weight, bias = p[f'filters.{i}'], p[f'filter_biases.{i}']
        if reference:
            values = F.conv1d(embedded.transpose(1, 2), weight.reshape(-1, 15, width), bias).transpose(1, 2)
        else:
            windows = embedded.unfold(1, width, 1).contiguous().view(n, length-width+1, 15*width)
            values = F.linear(windows, weight, bias)
        values = values.tanh()
        allowed = torch.arange(length-width+1, device=codes.device)[None, :]+width <= lengths[:, None]
        maximum = values.masked_fill(~allowed[:, :, None], -2).amax(dim=1)
        pooled.append(torch.where((lengths >= width)[:, None], maximum, torch.zeros_like(maximum)))
    y = torch.cat(pooled, 1)
    transformed = F.linear(y, p['highway_weight'], p['highway_bias']).relu()
    gate = F.linear(y, p['transform_weight'], p['transform_bias']).sigmoid()
    return F.linear(gate*transformed+(1-gate)*y, p['projection_weight'])


def chunks(lengths):
    groups = collections.defaultdict(list)
    for index, length in enumerate(lengths.tolist()): groups[max(6, length)].append(index)
    return [(length, indices[start:start+1024]) for length, indices in sorted(groups.items())
            for start in range(0, len(indices), 1024)]


def grouped(p, codes, lengths):
    import torch
    outputs, order = [], []
    for length, ids in chunks(lengths.cpu()):
        outputs.append(compose(p, codes[ids, :length], lengths[ids])); order.extend(ids)
    inverse = torch.argsort(torch.tensor(order, device=codes.device))
    return torch.cat(outputs)[inverse]


def upstream(ids, denominator, device, dtype):
    import torch
    return (((torch.tensor(ids, device=device)[:, None]*256+torch.arange(256, device=device)) % 17)-8).to(dtype)/(8*denominator)


def compare(actual, expected):
    import torch
    delta = (actual.detach().cpu().double()-expected.detach().cpu().double()).abs()
    assert bool(torch.isfinite(delta).all()) and bool((delta <= 4e-6+4e-5*expected.detach().cpu().double().abs()).all())
    return float(delta.max())


def run():
    import torch
    import tiktoken
    torch.set_num_threads(4)
    assert torch.backends.mps.is_available()
    rng = torch.random.get_rng_state().clone()
    assert sha(RECIPE) == '1a5029445e36f2701fa3e4b9d5bcf6af29ce21fbb07c54b0366bb027aa1d3823'
    receipt = read(RECIPE)['representative']['parent']; rawpath = Path(receipt['path'])
    assert rawpath.stat().st_size == receipt['size'] == 252854065
    assert sha(rawpath) == receipt['sha256'] == 'c086a81207231fac943c2c68b8b7a9dcf88e485d9add058d84d58e7d6d55b853'
    raw = torch.load(rawpath, map_location='cpu', weights_only=True)
    saved = {k.removeprefix('composer.'): v.clone() for k, v in raw['numerical']['parameters'].items() if k.startswith('composer.')}
    del raw
    assert len(saved) == 18 and sum(v.numel() for v in saved.values()) == 725235
    config = read(CONFIG); expected = config['data']['tokenizer']
    assert importlib.metadata.version('tiktoken') == expected['version']
    enc = tiktoken.get_encoding('gpt2')
    assert hashlib.sha256(json.dumps(sorted((t.hex(), r) for t, r in enc._mergeable_ranks.items())).encode()).hexdigest() == expected['vocabulary_sha256']
    packed = array('H', [256])*(50257*130); lens = array('H', [0])*50257
    for bs, token in enc._mergeable_ranks.items():
        row = array('H', [257, *bs, 258]); packed[token*130:token*130+len(row)] = row; lens[token] = len(row)
    packed[50256*130:50256*130+3] = array('H', [257, 259, 258]); lens[50256] = 3
    assert sys.byteorder == 'little' and packed.itemsize == 2
    identity = config['model_identity'][-1]
    assert hashlib.sha256(packed.tobytes()).hexdigest() == identity[6]
    assert hashlib.sha256(lens.tobytes()).hexdigest() == identity[7]
    codes = torch.frombuffer(packed, dtype=torch.uint16).long().reshape(50257, 130)
    lengths = torch.tensor(lens, dtype=torch.int64)
    representatives = {}; 
    for i, length in enumerate(lens): representatives.setdefault(length, i)
    ids = [representatives[k] for k in sorted(representatives)]+[0, 0, 50256]
    def parameters(device, dtype): return {k: v.to(device=device, dtype=dtype).clone().requires_grad_() for k, v in saved.items()}
    ref = parameters('cpu', torch.float64)
    expected_out = compose(ref, codes[ids], lengths[ids], reference=True)
    (expected_out*upstream(ids, len(ids), 'cpu', torch.float64)).sum().backward()
    checks = {}
    for device in ('cpu', 'mps'):
        p = parameters(device, torch.float32); c = codes[ids].to(device); ln = lengths[ids].to(device)
        out = grouped(p, c, ln)
        (out*upstream(ids, len(ids), device, torch.float32)).sum().backward()
        checks[device] = dict(output_max_error=compare(out, expected_out),
            gradient_max_errors={k: compare(p[k].grad, ref[k].grad) for k in p})
        if device == 'mps':
            full_grads = {k: v.grad.clone() for k, v in p.items()}
            for v in p.values(): v.grad = None
            for length, indices in chunks(ln.cpu()):
                y = compose(p, c[indices, :length], ln[indices])
                y.backward(upstream([ids[i] for i in indices], len(ids), device, torch.float32))
            checks['recomputed_mps'] = {k: compare(p[k].grad, full_grads[k]) for k in p}
    del ref, p, out, expected_out, full_grads, c, ln, y
    torch.mps.synchronize(); torch.mps.empty_cache()
    p = parameters('mps', torch.float32); c = codes.to('mps'); ln = lengths.to('mps')
    table = torch.empty((50257, 256), device='mps'); layout = chunks(lengths)
    torch.mps.synchronize(); started = time.monotonic()
    with torch.no_grad():
        for length, indices in layout: table[indices] = compose(p, c[indices, :length], ln[indices])
    torch.mps.synchronize(); forward_seconds = time.monotonic()-started
    assert bool(torch.isfinite(table).all())
    started = time.monotonic()
    for length, indices in layout:
        y = compose(p, c[indices, :length], ln[indices])
        y.backward(upstream(indices, 50257, 'mps', torch.float32))
    torch.mps.synchronize(); vjp_seconds = time.monotonic()-started
    assert all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in p.values())
    assert sorted(i for _, indices in layout for i in indices) == list(range(50257))
    assert torch.equal(rng, torch.random.get_rng_state()) and sha(rawpath) == receipt['sha256']
    return dict(completed=True, raw_receipt=receipt, fixture_rows=len(ids), checks=checks,
        full_rows=50257, chunks=len(layout), forward_seconds=forward_seconds, recomputed_vjp_seconds=vjp_seconds,
        mps_current_allocated_bytes=torch.mps.current_allocated_memory(), mps_driver_allocated_bytes=torch.mps.driver_allocated_memory(),
        gradients_finite=True, cpu_rng_unchanged=True, no_model_training=True, no_new_losses=True,
        scope='Single full-vocabulary component execution with synthetic upstream; not full-model integration, sustained throughput or quality evidence.')


def put(path, value):
    value = (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
    assert len(value)+sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) < CAP
    with path.open('xb') as stream: stream.write(value); stream.flush(); os.fsync(stream.fileno())


if __name__ == '__main__':
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert shutil.disk_usage(ROOT).free >= 10*1024**3+CAP
        OUT.mkdir()
        paths = [Path(__file__), RECIPE, CONFIG, ROOT/'research/BYTE_OUTPUT_CHUNK_PILOT_PROTOCOL.md',
            ROOT/'research/byte_embedding_v1.py', ROOT/'research/byte_composer_v2.py',
            ROOT/'research/output_anchored_tying_frequency_v1.py', ROOT/'src/research/cpu_prerequisite_guard.py',
            ROOT/'research/tokenizer-evidence/byte-output-composition-cost-001/result.json']
        pins = {str(p.relative_to(ROOT)): sha(p) for p in paths}
        put(OUT/'attempt.json', dict(input_sources=pins, output_cap_bytes=CAP))
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json', ROOT, timeout_seconds=180,
                    rss_limit_bytes=4*1024**3, reserve_bytes=10*1024**3):
                result = run()
                assert all(sha(ROOT/path) == value for path, value in pins.items())
            result.update(attempt_sha256=sha(OUT/'attempt.json'), guard_sha256=sha(OUT/'guard.json'))
            put(OUT/'result.json', result); print(json.dumps(result, sort_keys=True), flush=True)
        except BaseException:
            put(OUT/'failure.json', dict(traceback=traceback.format_exc())); raise
