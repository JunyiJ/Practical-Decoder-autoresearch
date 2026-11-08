# Scoped public-inference repair candidate

Status: source prepared, runtime unqualified, not adopted. The active byte-context
quality panel continues with its frozen grad-enabled scorer. This candidate is
not imported by that panel or by the public checkpoint consumer.

## Demonstrated defect and applicability

The new byte-context baseline's full-validation CPU/MPS check failed after
training. Its immutable evidence is in
`tokenizer-evidence/byte-context-transform-001/00/`. The separate stage and RMS
diagnostics isolated a default-epsilon error, and `scoring-recovery001/` verified
the training-equivalent scoring workaround against the original full split and
unchanged tolerances. No failed historical experiment is being replayed.

Installed PyTorch 2.14.1, commit
`5c4886908584029761b579af026dcfb627c84070`, defaults missing epsilon in its MPS
no-grad fused RMSNorm to double machine epsilon. Its autograd/composite path uses
the input's FP32 epsilon. The local `research/block_norm_v1/spec.py` already
declares epsilon `2**-23`; its native module construction uses `eps=None`.
The exact upstream sources are
[MPS RMSNorm](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/operations/RMSNorm.mm)
and [dispatch/composite normalization](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/layer_norm.cpp).

Source inspection confirms both `byte_input_inference_v5.score` and
`byte_input_cache_v4.Session.logits` execute the same default-epsilon modules
under no-grad. This establishes exposure to the defect, not a new measured
quality score for the incumbent. The older quality conclusions remain closed.

## Smallest candidate

`fp32_rmsnorm_inference_v1.FP32RMSNormInference` is an explicit dynamic scope
around an existing no-grad inference call. It supplies the declared FP32 epsilon
only to default-epsilon, affine width256 RMSNorm calls. It rejects training,
`torch.inference_mode`, autocast, other dtypes and unsupported shapes. Ordinary
`torch.no_grad` is the supported execution mode. Calls with explicit epsilon are
unchanged. The installed `torch.overrides.TorchFunctionMode` provides this
scoped dispatch mechanism without a global function patch.

The frozen model validates module objects and scalar attributes. Mutating
`module.eps`, replacing modules, or installing hooks would violate that contract.
The candidate does none of those things, and it does not allocate gradient
graphs for the cache. Runtime qualification must still establish that the mode
intercepts every intended normalization call and preserves all model bindings.

## Next bounded verification

After the active panel releases the shared worker lock, admit a separate
implementation check on the already completed *new* byte-context slot00 H200
weights. Do not retrain, alter the panel scorer, or replace its primary reports.
Preserve the original failed reports and all historical restrictions.

Use the existing fixed first and worst validation windows (starts 0 and 15616)
for localization, then the complete 34,278-target split for correction coverage.
Require the original CPU/MPS mean CE error <=3e-5 and maximum target CE error
<=3e-4; require same-backend corrected no-grad versus training-equivalent
forward to meet those same tolerances. A failure is preserved and diagnosed,
not answered by increasing tolerance. Verify eight corrected calls per model
forward, unchanged module semantics and weights, absent gradients, unchanged RNG,
and unchanged behavior after exiting the mode.

Cached decoding needs a separate positive check on these same new baseline
weights: B1 and B8, prefill96 then 32 contiguous single-token steps, reset and
repeat. Compare ordinary corrected logits against cached corrected logits using
the existing cache qualification tolerances (`atol=3e-5`, `rtol=3e-4`, from
`byte_input_cache_proof_v1.py`), bound in the prospective plan before execution.
Verify the eight session-owned cache
tensors and position/reset invariants. Explicit-epsilon calls must retain their
original outputs; reject grad-enabled default-epsilon calls and unsupported dtype
or width. Do not infer cached correctness from an uncached score.

Keep this check within 10 minutes, 4GiB RSS, 6GiB MPS, 8GiB free-space reserve,
and 32MiB new evidence. Reuse original weights without another payload copy.
Freeze source/input receipts and exact cases before running, then report actual
cost. This is correctness qualification, not a quality campaign or a throughput
claim. Public adoption requires successful checks and a separately bound consumer
entrypoint; preparation alone does not change the adopted recipe.
