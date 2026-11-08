# Actual-buffer hashing comparison

The corrected attempt passed exact digest and complete-validator mutation checks
for all14 real model buffers on CPU and MPS (28 buffer/backend cases,56 rejected
mutations with successful restoration). It reconstructed the first fresh adopted
checkpoint identity and copied all105 accepted parameter tensors into the
candidate. No sampling, forward/backward call, optimizer update, quality score,
new weight archive or scientific seed search occurred. CPU/MPS RNG stayed exact.

| Backend | Code-ID hash original ms | Direct bytes ms | Hash speedup | Full validation original ms | Direct bytes ms | Validation speedup |
|---|---:|---:|---:|---:|---:|---:|
| cpu | 282.386 | 17.330 | 16.30x | 298.239 | 27.854 | 10.71x |
| mps | 310.622 | 35.111 | 8.85x | 357.614 | 66.679 | 5.36x |

Times are medians of three alternating paired repeats after warmup. CPU and MPS
use the same actual14 buffers, including Boolean attention masks, FP32 position
buffers and the52,267,280-byte integer code-ID buffer. MPS timing synchronizes
around the measured call, and includes the existing chunked host transfer.
All complete validation, content hashing and rejection guards remain present.

The guard completed in33.3782 seconds, peak
RSS775,094,272bytes and final sampled MPS driver
allocation1,107,771,392bytes, within120seconds,
4GiB RSS,6GiB sampled MPS and8GiB disk reserve. New weight output was zero.

## Preserved harness failure

Attempt001 remains closed failed: its midpoint mutation used add_(1) on a Boolean
attention buffer, which PyTorch rejected before validation. That was a harness
operation error, not a digest mismatch or model-quality result. Attempt002 changes
only that operation to logical_not_ for Boolean tensors, retaining add_(1) for
other dtypes. Its plan binds the original source, plan, traceback, guard and STOP.
The checkpoint, sources under comparison, devices, buffers, midpoint locations,
repeat count, exact criteria and limits were unchanged. Historical reconstruction
uses no new training roles. No failed evidence was overwritten or relabeled.

## Applicability and next implementation

This extends the closed synthetic CPU serialization measurement to actual model
buffers, GPU transfers and the complete inherited validator. It supports a
validation-overhead improvement. It is not a training, forward, generation or
whole-system speedup measurement. Three timing repeats are descriptive, not a
confidence interval. The subclass still cannot enter exact-type training/state
consumers; those guards were not weakened.

The separate direct-hash-integration-source001 record maps seven additive source
files to the frozen originals. The model changes only byte serialization; imports
route new exact model/optimizer types consistently through state, training,
inference and cache consumers. Shared numerical identity/codebook definitions
and artifact formats remain unchanged. Exact reverse-diff and syntax checks
passed. These integrated sources are not yet runtime qualified or adopted.

Next: one bounded differential saved-checkpoint loading, forward/gradient/update
and cache comparison for the integrated sources, with descriptive timing. Reuse
existing mathematical/checkpoint/cache qualifications and this buffer evidence;
no quality sweep or replay of a closed fixture is justified by an implementation
hashing change.
