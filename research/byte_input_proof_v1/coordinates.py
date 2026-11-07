"""Bounded durable coordinate evidence; no candidate imports.

Each gzip member stores interleaved actual FP32, reference FP32 and uint8 mask.
The consumer must call finish only after the data and metadata are durable.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path


def compare(out, name, actual, expected, atol, rtol):
    import numpy as np
    assert actual.shape == expected.shape
    shape = list(actual.shape)
    count = actual.numel()
    assert count == expected.numel()
    dtype = np.dtype([('actual', '<f4'), ('reference', '<f4'), ('pass', 'u1')])
    target = Path(out) / (name + '.coordinates.gz')
    passed, maximum = True, 0.
    digest = hashlib.sha256()
    a, b = actual.detach().reshape(-1), expected.detach().reshape(-1)
    # Chunk CPU transfer first; only the bounded CPU reporting chunk uses FP64.
    with target.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
            for start in range(0, count, 65536):
                av = a[start:start+65536].to(device='cpu').float().numpy()
                bv = b[start:start+65536].to(device='cpu').float().numpy()
                delta = np.abs(av.astype('float64') - bv.astype('float64'))
                mask = np.isfinite(av) & np.isfinite(bv) & (delta <= atol + rtol * np.abs(bv.astype('float64')))
                block = np.empty(len(av), dtype=dtype)
                block['actual'], block['reference'], block['pass'] = av, bv, mask
                payload = block.tobytes()
                digest.update(payload)
                stream.write(payload)
                passed = passed and bool(mask.all())
                maximum = max(maximum, float(delta[np.isfinite(delta)].max(initial=0.)))
        raw.flush()
        os.fsync(raw.fileno())
    record = dict(shape=shape, count=count, atol=atol, rtol=rtol,
                  dtype='interleaved_le_float32_float32_uint8', raw_bytes=count*9,
                  raw_sha256=digest.hexdigest(), passed=passed, max_finite_abs=maximum,
                  data_file=target.name)
    with (Path(out)/(name+'.json')).open('x') as stream:
        json.dump(record, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    # The assertion follows publication, preserving failures in full.
    assert passed, name
    return record


def review(out, metadata):
    """Reconcile every saved coordinate and mask without importing Torch."""
    import numpy as np
    record = json.loads(Path(metadata).read_text())
    dtype = np.dtype([('actual', '<f4'), ('reference', '<f4'), ('pass', 'u1')])
    digest, count, accepted = hashlib.sha256(), 0, True
    with gzip.open(Path(out)/record['data_file'], 'rb') as stream:
        while payload := stream.read(9*65536):
            assert len(payload) % 9 == 0
            digest.update(payload)
            block = np.frombuffer(payload, dtype=dtype)
            a = block['actual'].astype('float64')
            b = block['reference'].astype('float64')
            mask = np.isfinite(a) & np.isfinite(b) & (np.abs(a-b) <= record['atol'] + record['rtol']*np.abs(b))
            assert np.array_equal(block['pass'], mask.astype('uint8'))
            accepted = accepted and bool(mask.all())
            count += len(block)
    assert count == record['count'] and count*9 == record['raw_bytes']
    assert digest.hexdigest() == record['raw_sha256'] and accepted == record['passed']
    return accepted, count
