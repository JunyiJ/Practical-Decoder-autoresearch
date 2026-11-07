# Generated CPU optimizer-response proof complete

The NEW `tokenizer_optimizer_response_v1` component and independent scalar proof
preserve all 1047 predecessors, for 1050 immutable source bindings. The original
CPU-only component, proof, terminal review and all earlier commands are closed.
No numerical worker remains. See `optimizer-response001/{plan,result,guard,STOP}`
and `optimizer-response-review001.json`.

The test used tiny detached synthetic CPU FP32 states, capped at 4096 elements.
It loaded no model, RAW checkpoint, corpus or data sampler. No random fixture role
was assigned or used. Genuine Muon/AdamW steps occurred only on cloned synthetic
parameters, never on an actual model. No MPS API, validation/holdout encoding,
bootstrap, EMA, resume or tensor-file output occurred. The caller's complete state,
gradients and global CPU RNG were unchanged. This is a correctness prerequisite,
not a batch experiment or model-quality result.

## What passed

All 54 predeclared cases completed: tall, wide and square matrices; fresh, step7
and step1200 synthetic state; ordinary, constant and zero gradients; clipped and
unclipped scales. There were 18 clipped and 36 unclipped cases. An independent
scalar-loop oracle implemented IEEE bfloat16 rounding, normalization and five
Newton–Schulz iterations without using PyTorch's Muon helper, matmul or addmm.
Adam moments, bias correction, epsilon and independent decay were calculated
separately. The genuine response used the incumbent EpsilonOptimizer ancestor,
clipping wrapper, installed Muon/AdamW and exact nominal toy group LRs.

Maximum parameter discrepancy was 1.4901161e-8, maximum moment discrepancy
7.4505806e-9, maximum reconstructed Muon direction discrepancy 3.9041042e-5 and
clipping coefficient discrepancy 1.7448799e-8, all inside the original declared
tolerances. Those tolerances were frozen before execution and were not adjusted.
Two explicit missing-gradient cases skipped both decay and optimizer exposure;
active zero gradients still advanced their own state. Twenty malformed caller
state/gradient cases plus three invalid metadata-budget cases were rejected
without caller mutation. No foreign state or real RAW consumer was admitted.

The linear averaging control agreed exactly. In the deliberately generated
nonlinear control, response(mean gradient) minus mean(response) had maximum
absolute displacement .0002162680 for the matrix and .0000321353 for the vector.
Sequential state evolution also differed. These are toy absolute displacements,
not normalized effect sizes, actual endpoint measurements or evidence of better
training. They show why raw-gradient averaging alone cannot determine the
clipped, stateful optimizer response.

The CPU guard finished in .5833 seconds at peak RSS 312426496 B, under the original
120 s / 4 GiB RSS / 8 GiB reserve bounds. Entry free space exceeded the complete
9861394432 B floor and 12008878080 B target; no storage blocker or cleanup arose.
Metadata-only output stayed inside 2 MiB. A separate terminal metadata/source/
resource review verified all 54 Cartesian cases, original tolerances, all 1050
bindings, exact installed library hashes and terminal STOP without replaying any
optimizer steps. The actual inherited HybridOptimizer resolves to
`research/adamw_beta2_model_v1/runtime/train/_grouping.py`; the earlier general
source note also bound the equivalent older FFN wrapper, but this proof explicitly
binds the actual inherited implementation.

## Interpretation and next executable prerequisite

The earlier four-cohort result remains unchanged: covariance max/min1.031–1.067,
corrected signal max/min1.818–3.674, with denominator variation driving broad raw
noise/signal ratios. The new CPU test validates a measurement component. It does
not resolve that sampling uncertainty or establish batch16/32, an optimizer
change, a critical batch or a quality gain. Keep the native dense incumbent.

Using the evidence-grounded-research skill, admit only a NEW tiny generated
CPU/MPS response-transport proof as the next prerequisite. CPU arithmetic does
not establish target-device bf16 reductions, matrix response or optimizer-state
handling. Use the same deterministic Cartesian cases, independently calculated
scalar equations, genuine CPU/MPS responses and original tolerances. An explicit
new synthetic-state/device policy belongs in a new namespace: never change or
bypass the successful CPU-only v1 guard. Never import and execute the closed
proof script; reuse pure equations in a new proof, with exact source provenance.

`optimizer-response-mps-design001.json` specifies the controls, falsifiable
agreement prediction and full 2 MiB metadata / reserve / external / entry budget.
Use one strict MPS process and four threads with fallback0/fastmath0, 600 s,
RSS4 GiB/MPS6 GiB/reserve8 GiB and complete source/runtime/LIVE gates. The forecast
uses prior actual integration peaks plus512 MiB allowance; it is not an actual
new memory pass. No model, RAW, corpus, sampler, new seed or actual-model update
is admitted. A pass would enable only a separately designed full-state endpoint
consumer/integration review; a failure closes this attempt with no tolerance
rescue. No actual endpoint response measurement or quality panel is frozen.

McCandlish1812.06162v1 §2.4/A.1/E.3 remains the previously read primary basis for
conditioning, local-step, ratio-cancellation and generalization cautions. It does
not validate Muon/Adam target-device transport or local quality. Historical batch
contrary results and all closed MoE/prefix/dropout negatives remain unpooled.
Preparation is not numerical quality exploration; 70/20/10 remains proposed.
