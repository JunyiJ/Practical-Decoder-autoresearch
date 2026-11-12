# Independent-batch gradient averaging: next bounded diagnostic

## Evidence and ranking

The completed first-update response study found lower own-batch CE in all 72
conditions, but positive parent-mean disjoint-probe changes at all three fixed
step fractions. At full step the mean probe change is +0.000304667 nats/token;
at 1/16 it is still +0.000007517. All three parents fail the frozen overshoot
criterion. See `GENERAL_RSI_UPDATE_RESPONSE_RESULTS.md` and its original scores.
Earlier no-decay and RAW-start diagnostics reduced drift without repairing it.

These observations justify testing the gradient's dependence on sampled batches.
They do not establish noise as the cause. The disjoint probes are finite; the
hybrid optimizer can transform even an unbiased raw gradient into a different
expected update. A lower learning rate reduces measured harm but lacks a locally
supported beneficial direction. The closed warmup confirmation also failed.
Thus batch averaging ranks ahead of a new warmup/LR sweep. Group-specific learning
rates remain unqualified: group norm differences alone do not establish which
group causes probe harm. No broad quality campaign is justified yet.

## Literature and applicability

McCandlish et al., [arXiv:1812.06162v1](https://arxiv.org/html/1812.06162v1),
section 2.2, equations 2.1–2.10, relate independently sampled gradient averages,
variance and local loss response. The independence condition applies to sampled
examples, not correlated tokens within our 512-token blocks. Section 2.4 warns
about short-horizon bias, conditioning, tuned step sizes and generalization;
section 2.2 reports mixed results for a preconditioned-gradient simplification.
This supports testing an average of independently drawn batches. It does not
supply our noise scale, optimal batch size or a guarantee for Muon/AdamW.

Inference: the current single-batch update may be overly specific to its sampled
text. Averaging independent natural gradients before the existing optimizer could
improve disjoint-probe loss. Alternatively, optimizer transformation, local geometry
or probe sampling may dominate, so extra batches may merely add cost. Preserve
both possibilities. No old scores enter the new comparison.

## Prospective controlled contrast

Use the three selected EMA parents, slots 1/3/5. For each, freeze eight fresh
paired groups before execution. Each group contains eight independently drawn
ordinary natural batches, each with the unchanged 1,024-target packing policy.
Control uses the first batch. Candidate averages gradients from all eight before
one global clip and one canonical no-decay optimizer update. Both start with
identical parent parameters and pristine optimizer state. LR is 0.00025, clip 1.
Freeze alternating arm order and unused seed roles. The same natural batch may
appear in the two paired arms, but neither arm updates twice on any batch.
There is no stale-gradient, cached-update or batch-reuse arm.

Factor eight is one bounded variance-reduction contrast, not a tuned optimum.
Under independent batch sampling it materially changes the gradient estimator
while fitting the local diagnostic budget. Do not test a batch-size grid or
rescale learning rate after seeing the result. Candidate processes eight times
as many targets per update; this is not an equal-token or equal-time quality claim.

Each parent has 64 fresh document-local 128-target probe windows, excluded from
all 64 update batches' source positions. Score the unchanged parent once and
each arm after its one update. Total: 24 pairs, 48 optimizer updates, 216 natural
minibatch forward/backward passes, 221,184 training targets, 51 probe evaluations
and 417,792 scored targets. Retain every first score, sampled input, clip statistic
and actual update norm by group. No validation, test, generation or model output.

Primary diagnostic support requires both conditions in every parent: the mean
candidate-minus-control probe CE is negative, and the candidate's mean change
from the unchanged parent is negative. Lower harm without improvement fails.
Report all paired cases and equal-parent means. Shared probes and eight cases
are not independent model seeds. If the criterion passes, design a separate fresh
multi-step comparison with explicit time/data controls before any quality claim.
If it fails, do not launch a larger-batch quality sweep or silently try another
batch factor. Reassess whether this continuation family has useful local evidence.

## New engineering, cost and admission

Implement only serial gradient averaging before the already qualified clip and
optimizer. Verify accumulation against independently evaluated microbatch gradients
and verify one optimizer advance, finite state and sampler input identity. The
single-batch path must agree with the existing update. Freeze engineering tolerances
before running these new checks. No repeat of full model/restart/document-scoring
prerequisites without a demonstrated requirement; no restart output is needed by
this one-step, nonresumable diagnostic. Keep the failed attempts if any.

Use serial microbatches so activations do not grow eightfold. Provisional complete
cost is 216 × 0.49 seconds for forward/backward/update allowance, 51 × 1.79 seconds
for scoring and 110 seconds for initialization, input binding and evidence,
about 307.1 seconds. The allowance uses the observed cold first update and maximum
probe score; measure actual eight-batch accumulation in the new fixture before
admission. Retain 64 MiB evidence plus 64 MiB overlap, no model serialization,
10 GiB reserve and 2 GiB extra headroom; RSS ceiling 4 GiB, MPS 6 GiB and total
600 seconds only if the complete measured forecast fits. Include fixture cost
in the final report and preserve all existing selected/fresh payloads.

Next implement the small accumulation integration and qualify only that new path.
Then freeze source hashes, complete fresh input inventories, paired roles, criterion
and measured resource admission. Check actual processes and acquire the shared
research lock nonblocking. No worker or new quality panel is active at this boundary.
A full quality campaign and its retained-output forecast require a separate future
design; this diagnostic admits neither model adoption nor paid/parallel compute.
