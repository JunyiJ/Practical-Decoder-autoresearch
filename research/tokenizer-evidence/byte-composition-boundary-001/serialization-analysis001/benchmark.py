"""CPU-only byte serialization measurement, separate from the running model."""
import ast
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'research/byte_input_model_v3/model.py'

def publish(name, value):
    with (OUT / name).open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())

def main():
    publish('attempt.json', dict(pid=os.getpid(), scope='Synthetic CPU bytes only; no model, GPU, checkpoint or scientific roles.',
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        active_quality_controller=17021, active_quality_session=22743,
        trials=3, rss_limit_bytes=1536*1024**2, timeout_seconds=90))
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    try:
        with CpuPrerequisiteGuard(OUT/'guard.json', ROOT, timeout_seconds=90,
                                 rss_limit_bytes=1536*1024**2, reserve_bytes=8*1024**3):
            import torch
            torch.set_num_threads(1)
            tree = ast.parse(SOURCE.read_text())
            method = next(n for n in ast.walk(tree)
                          if isinstance(n, ast.FunctionDef) and n.name == '_buffer_digest')
            original = ast.unparse(method)
            old = 'bytes(part.view(torch.uint8).reshape(-1).tolist())'
            new = 'part.view(torch.uint8).numpy().tobytes()'
            assert original.count(old) == 1
            changed = original.replace(old, new)
            namespace = dict(torch=torch, hashlib=hashlib)
            exec(compile(original, str(SOURCE), 'exec'), namespace)
            baseline = namespace['_buffer_digest']
            exec(compile(changed, '<direct-byte-serialization-candidate>', 'exec'), namespace)
            candidate = namespace['_buffer_digest']
            factories = [
                ('all_byte_values', lambda: torch.arange(256, dtype=torch.int64).to(torch.uint8).repeat(4096)),
                ('code_ids_shape', lambda: torch.arange(50257*130, dtype=torch.int64).reshape(50257,130)),
                ('code_lengths_shape', lambda: torch.arange(50257, dtype=torch.int64)),
                ('float_buffer', lambda: torch.arange(131073, dtype=torch.float32)/7),
            ]
            results = []
            for name, factory in factories:
                value = factory()
                before = value.clone()
                version = value._version
                reference = hashlib.sha256(value.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                assert baseline(None, value) == candidate(None, value) == reference
                timings = {'baseline':[], 'direct_bytes':[]}
                for trial in range(3):
                    order = [('baseline', baseline), ('direct_bytes', candidate)]
                    if trial % 2: order.reverse()
                    for label, operation in order:
                        start = time.perf_counter()
                        observed = operation(None, value)
                        timings[label].append(time.perf_counter()-start)
                        assert observed == reference
                assert value._version == version and torch.equal(value, before)
                # A byte change must remain detectable, with both paths agreeing.
                value.reshape(-1).view(torch.uint8)[0] ^= 1
                changed_hash = candidate(None, value)
                assert changed_hash == baseline(None, value) and changed_hash != reference
                results.append(dict(case=name, shape=list(value.shape), dtype=str(value.dtype),
                    bytes=value.numel()*value.element_size(), sha256_before=reference,
                    sha256_after_mutation=changed_hash, unchanged_input_during_measurement=True,
                    timings_seconds=timings))
                del value, before
            publish('result.json', dict(accepted=True, cases=results,
                changed_expression=dict(original=old, candidate=new),
                scope='CPU tensor chunk conversion and SHA256; no MPS transfer, integrated update or model quality measurement.',
                limitations=['Synthetic contents with actual codebook shapes.',
                    'Concurrent scientific training; timings are descriptive and not isolated-machine benchmarks.',
                    'No change to pinned sources or active controller.',
                    'No full-model equivalence or adoption claim.']))
        publish('STOP', dict(status='complete_cpu_serialization_analysis', no_replay=True))
    except BaseException as exc:
        publish('STOP', dict(status='failed_cpu_serialization_analysis', error=repr(exc), no_replay=True))
        raise

if __name__ == '__main__':
    main()
