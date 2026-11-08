# Scoped inference repair qualified on the new baseline

The complete third attempt passed on the already trained byte-context slot00
H200 checkpoint. No training, checkpoint modification, quality reselection or
public-consumer adoption occurred. The incumbent recipe remains unchanged.

The correction explicitly supplies the already declared FP32 RMSNorm epsilon
inside a scoped ordinary no-grad context. It changes no module attributes or
global functions and builds no gradient graph. Leaving the scope restores the
original behavior exactly on the two fixed probe windows.

## Verified requirements

- Full 34,278-target CPU and MPS scoring, with 2,144 corrected normalization
  calls per backend. CPU scores exactly match the prior CPU reference. MPS's
  maximum target difference from the training-equivalent reference is
  1.335145e-5, below the unchanged 3e-4 limit.
- CPU/MPS mean CE difference 2.1248434e-5 and maximum target difference
  .0002725124, below the original 3e-5 and 3e-4 limits respectively.
- CPU/MPS cached B1/B8 prefill96 followed by 32 contiguous single-token steps,
  compared with full-context logits under the same correction. All 264 recorded
  chunk comparisons pass the inherited atol3e-5/rtol3e-4 requirement. The largest
  error-to-tolerance ratio is .195156, below 1. Both replays are bit-identical,
  including final cache contents, positions and reset behavior.
- Explicit-epsilon outputs remain unchanged. Unsupported dtype and width,
  training-mode gradients and torch.inference_mode calls are rejected as specified.
- All model weights, semantic inventories and CPU/MPS RNG states remain unchanged;
  no parameter gradients are created. Frozen inputs and applicable sources verify
  at entry and exit. A separate saved-evidence recomputation reproduced numerical
  errors, complete cache coverage, exact replay hashes and resource totals.

## Preserved recoveries and cost

Attempt001 failed while constructing an unsupported MPS float64 test tensor.
The separately frozen correction retains CPU's float64 rejection case and uses
representable float16 for MPS's equivalent non-FP32 rejection case. The repair
implementation itself did not change. See the
[harness recovery](FP32_RMSNORM_INFERENCE_HARNESS_RECOVERY.md).

Attempt002 completed numerical/cache checks but failed its final broad source
census because another task edited an unrelated report generator. Its result
was not accepted. Attempt003 explicitly excludes only that audited standalone
report, pins all other existing research/src Python sources, and reruns the
entire check. See the [source recovery](FP32_RMSNORM_SOURCE_RECOVERY.md).
All original plans, failed guards, tracebacks and partial evidence remain intact.
Transient lock contention occurred before some attempts started; no lock was
bypassed and no worker was interrupted.

Attempt durations were 38.960124, 42.389451 and 43.951386 seconds, totaling
125.300961 seconds within the original 600-second allowance. The final guarded
observation was about 2.05GiB peak RSS, .072GiB MPS driver allocation and 6.26MiB
combined attempt evidence, within their declared limits. It observed 12.24GB
free disk space; no capacity cleanup was required.

## Next required integration

The qualified dynamic scope is still opt-in. The public checkpoint consumer,
inference scorer and cache entrypoint do not yet apply it automatically. Add a
new source namespace around the adopted model-v6/inference-v5/cache-v4 path,
preserving the original modules and typed loading identity. Bind and test the
actual selected checkpoint through that entrypoint, including uncached/cached
agreement and unchanged weights/RNG. Do not rescore a closed quality comparison
or claim a new quality gain from this implementation repair.

The existing CPU scorer also has a separately demonstrated native-CE precision
defect. Public score integration must use the already verified stable FP32 CPU
reference, with MPS scoring unchanged apart from the intended RMSNorm correction.
The present check qualifies the RMSNorm scope; its CPU CE deliberately retains
the original reference and is not proof that native CPU CE is generally adequate.

## Bound evidence

- [Accepted plan](tokenizer-evidence/byte-context-transform-001/inference-epsilon003/plan.json), SHA256 `b70056237858f91b76f5c0e88bcc74750d8d2b03251d6f88dfbba87bdc3a6672`
- [Acceptance](tokenizer-evidence/byte-context-transform-001/inference-epsilon003/acceptance.json)
- [Result](tokenizer-evidence/byte-context-transform-001/inference-epsilon003/result.json)
- [Guard](tokenizer-evidence/byte-context-transform-001/inference-epsilon003/guard.json)
- [Implementation](fp32_rmsnorm_inference_v1.py)
