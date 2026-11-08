# Direct byte serialization: bounded CPU measurement

The previous cost analysis identified a specific avoidable operation in
`ByteInputModel._buffer_digest`: converting every uint8 byte through a Python
list before reconstructing bytes for SHA256. This analysis measures that
operation directly without changing the running scientific sources.

`benchmark.py` extracts the actual frozen method with Python AST. It replaces
exactly one expression, leaving flattening, chunk boundaries, CPU conversion,
contiguity and SHA256 unchanged:

```python
# Existing expression
bytes(part.view(torch.uint8).reshape(-1).tolist())
# Candidate expression
part.view(torch.uint8).numpy().tobytes()
```

The complete analysis finished in2.381seconds with385,187,840bytes peak RSS.
It used one CPU thread, no GPU/model/checkpoint access, no model seed roles,
and no shared research payload changes. Scientific training continued separately.
The finite guard was90seconds/1.5GiB RSS/8GiB free reserve. Source and benchmark
hashes are in `attempt.json`; terminal evidence is `guard.json` and `STOP`.

| Synthetic CPU buffer | Bytes | Median existing seconds | Median direct-byte seconds | Ratio |
|---|---:|---:|---:|---:|
| All256 byte values repeated |1,048,576|.005725|.000384|14.91x|
| Int64 code-ID shape50257x130 |52,267,280|.284271|.019357|14.69x|
| Int64 code-length shape50257 |402,056|.002280|.000148|15.40x|
| FP32 buffer across a chunk boundary |524,292|.002881|.0001875|15.37x|

Each case first matched the direct complete-buffer SHA256 oracle, then ran
three timed trials per method with alternating order. Every measured digest
matched; tensor contents and version remained unchanged. Flipping one byte
changed the hash and both paths agreed on the new digest. All timings and
before/after hashes are saved in `result.json`.

This establishes finite CPU serialization equivalence and a descriptive local
component speed advantage. Synthetic values have the real codebook shapes but
are not lexical codes. Concurrent training was active, so these are not isolated
machine benchmarks. There is no MPS-to-CPU transfer measurement, integrated
update speedup, full-model numerical-equivalence claim or quality result. Do not
apply the14.69x ratio to model training or extrapolate a replacement campaign ETA.

The next engineering action is a separately identified MPS/integration check
when the existing worker releases its resource allocation, preserving complete
content hashing and all mutation rejection requirements. The candidate changes
serialization only; it does not justify another learning-rate search merely to
check byte identity. Any integrated replacement must preserve exact hashes,
training results and saved identities under its own source binding. No pinned
source, active worker, scientific case, role, selection rule or tolerance was
changed by this analysis.
