"""Analytical full-vocabulary cost, without importing model or tensor code."""
from array import array
import ast
import collections
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from research.output_anchored_tying_frequency_v1 import read, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'research/tokenizer-evidence/byte-output-composition-cost-001'
CONFIG = ROOT/'research/tokenizer-evidence/byte-residual-readout-001/quality001/00/config.json'
COMPOSER = ROOT/'research/byte_composer_v2.py'
CAP = 1024**2


def count(n, length, widths, channels):
    conv = n*sum((length-w+1)*15*w*c for w, c in zip(widths, channels))
    dense = n*(2*525**2+525*256)
    return dict(forward_mac=conv+dense, convolution_mac=conv, dense_mac=dense,
        embedded_fp32_bytes=4*n*length*15,
        unfolded_windows_fp32_bytes=4*n*sum((length-w+1)*15*w for w in widths),
        convolution_values_fp32_bytes=4*n*sum((length-w+1)*c for w, c in zip(widths, channels)),
        pooled_fp32_bytes=4*n*525, composed_table_fp32_bytes=4*n*256)


def layout(groups, widths, channels):
    chunks = [(min(1024, n-start), length) for length, n in groups for start in range(0, n, 1024)]
    rows = [count(n, length, widths, channels) for n, length in chunks]
    totals = {key: sum(row[key] for row in rows) for key in rows[0]}
    assert sum(n for n, length in chunks) == 50257
    return dict(chunks=len(chunks), rows=50257, totals=totals,
        max_chunk={key: max(row[key] for row in rows) for key in rows[0]},
        nominal_forward_and_dense_backward_mac=3*totals['forward_mac'],
        forward_mac_ratio_to_existing_head=totals['forward_mac']/(1024*256*50257))


def run():
    import tiktoken
    config = read(CONFIG); expected = config['data']['tokenizer']
    assert importlib.metadata.version('tiktoken') == expected['version']
    enc = tiktoken.get_encoding('gpt2')
    vocabulary = json.dumps(sorted((t.hex(), r) for t, r in enc._mergeable_ranks.items())).encode()
    assert hashlib.sha256(vocabulary).hexdigest() == expected['vocabulary_sha256']
    assert enc.n_vocab == 50257 and set(enc._mergeable_ranks.values()) == set(range(50256))
    lengths = [0]*50257
    for raw, token in enc._mergeable_ranks.items(): lengths[token] = len(raw)+2
    lengths[50256] = 3
    packed = array('H', lengths); assert packed.itemsize == 2
    if sys.byteorder != 'little': packed.byteswap()
    codebook = config['model_identity'][-1]
    assert codebook[:2] == ['rsi_byte_codebook_v1', 'raw_bytes']
    assert hashlib.sha256(packed.tobytes()).hexdigest() == codebook[7]
    assert min(lengths) == 3 and max(lengths) == 130
    constants = {}
    for node in ast.parse(COMPOSER.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('WIDTHS', 'CHANNELS'):
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    widths, channels = constants['WIDTHS'], constants['CHANNELS']
    assert widths == (1, 2, 3, 4, 5, 6) and channels == (25, 50, 75, 100, 125, 150)
    parameters = 259*15+sum(c*(15*w+1) for w, c in zip(widths, channels))+2*(525**2+525)+525*256
    assert parameters == 725235
    histogram = collections.Counter(lengths)
    effective = collections.Counter(max(6, length) for length in lengths)
    fixed = layout([(130, 50257)], widths, channels)
    bucketed = layout(sorted(effective.items()), widths, channels)
    direct = sum(count(1, max(6, length), widths, channels)['forward_mac'] for length in lengths)
    assert direct == bucketed['totals']['forward_mac']
    assert fixed['totals']['forward_mac'] == 50257*count(1, 130, widths, channels)['forward_mac']
    assert bucketed['totals']['composed_table_fp32_bytes'] == 50257*256*4
    return dict(completed=True, code_lengths_sha256=codebook[7], vocabulary_sha256=expected['vocabulary_sha256'],
        length_histogram=dict(sorted(histogram.items())), effective_length_histogram=dict(sorted(effective.items())),
        mean_code_length=sum(lengths)/50257, fixed_padding=fixed, length_bucketed=bucketed,
        composer_parameters=parameters, augmented_model_parameters=16351859+parameters,
        existing_head_forward_mac=1024*256*50257,
        nominal_nine_run_raw_h200_tensor_bytes=2*4*(3*16351859+6*(16351859+parameters)),
        six_augmented_runs_extra_raw_h200_tensor_bytes=6*2*4*parameters,
        no_model_execution=True, no_random_draws=True, holdout_accessed=False,
        scope='Analytical MAC/tensor inventory only; no measured runtime/RSS, quality result, complete panel forecast or admission.')


def put(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
    assert len(raw)+sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) < CAP
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


if __name__ == '__main__':
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert shutil.disk_usage(ROOT).free >= 10*1024**3+CAP
        OUT.mkdir()
        paths = [Path(__file__), CONFIG, COMPOSER,
            ROOT/'research/BYTE_OUTPUT_COMPOSITION_COST_PROTOCOL.md',
            ROOT/'research/BYTE_OUTPUT_LEXICAL_REVIEW.md',
            ROOT/'research/byte_input_model_v6/model.py', ROOT/'research/byte_input_model_v3/codebook.py',
            ROOT/'research/output_anchored_tying_frequency_v1.py', ROOT/'src/research/cpu_prerequisite_guard.py']
        pins = {str(p.relative_to(ROOT)): sha(p) for p in paths}
        put(OUT/'attempt.json', dict(input_sources=pins, output_cap_bytes=CAP))
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json', ROOT, timeout_seconds=120,
                    rss_limit_bytes=1024**3, reserve_bytes=10*1024**3):
                result = run()
                assert all(sha(ROOT/path) == value for path, value in pins.items())
            result.update(attempt_sha256=sha(OUT/'attempt.json'), guard_sha256=sha(OUT/'guard.json'))
            put(OUT/'result.json', result)
            print(json.dumps(result, sort_keys=True), flush=True)
        except BaseException:
            put(OUT/'failure.json', dict(traceback=traceback.format_exc())); raise
