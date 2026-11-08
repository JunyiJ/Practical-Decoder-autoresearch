# Inference check recovery: representable rejected dtype on MPS

The first frozen check exited after 38.960124 seconds with a TypeError while
constructing `x.double()` on MPS. MPS cannot allocate float64 tensors, so the
intended invalid-dtype assertion was never reached. Original source, plan,
attempt, guard, failure, full CPU/MPS scores and completed CPU cache checks are
preserved in `byte-context-transform-001/inference-epsilon001/`.

Both full validation passes had completed: corrected CPU scores exactly matched
their reference, and corrected MPS maximum target error against its reference
was 1.335145e-5 (limit 3e-4). This is partial evidence only. The check failed and
did not qualify cached MPS inference or adopt a public implementation.

The new standalone v2 check uses a representable FP16 tensor for MPS's rejected
dtype test. CPU retains its original FP64 rejected-dtype case. Both test the
same requirement: the correction accepts only ordinary FP32 tensors. The repair
module is unchanged; width, gradient-mode, explicit-epsilon, full-split numerical,
B1/B8 cache, reset, weight, semantics and RNG cases retain their original limits.

This is a harness correction, not a scientific retry. No training, new roles,
checkpoint changes, quality reselection or tolerance changes occur. The new plan
binds every original attempt artifact and the unchanged implementation. It
charges prior execution against the original 600-second total, leaving a
561-second guard for this complete check. Both attempt directories count against
the original 32MiB evidence allowance. The same shared lock, RSS/MPS limits,
free-space reserve and extra headroom apply.

Only a complete successful v2 check can qualify this scoped implementation on
these new baseline weights. Public consumer integration and selected-checkpoint
qualification remain separate requirements.
