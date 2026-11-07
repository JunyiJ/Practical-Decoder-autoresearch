# Supplied-logit proximity objective: CLOSED_ACCEPTED finite component only

Original proof pid75457 exited0 (session8403); source gate pid75370 exited0.
The new source gate ran first: .0051010409370064735s/RSS24166400B, AST/literal
inspection only, no Torch/objective/oracle execution. Runtime plan SHA
715edc898d8a7e5b7b08436006b546d19971304b86aa4ae57d1a9f5be756ca2a;
source gate plan SHA3595bb23f1c3e71f61ff8cfb8938022b224a6c31350a8f05e4aa5d0c710bafa9.

All30 positive reports (10 fixed literals x CPU64/CPU32/MPS32),116 rejection
transactions covering41 declared families, and10 CPU/MPS loss comparisons passed.
All156 output IDs were distinct. Sparse-layout and meta-device checks used only
CPU32 constructible payloads; no MPS sparse/FP64 coverage claim. CPU64/CPU32/MPS32
negative counts38/40/38. Max literal positives B2/T8/V31/support4. No full-vocab
neural qualification; invalid V50258 is an early shape rejection, not a CE probe.

Independent scalar shifted-LSE/analytic-gradient oracle consumed represented
literal logits/weights, never candidate outputs. All2838 gradient coordinates
passed fixed elementwise gates. Six genuine one-hot CE comparisons passed loss
and ALL612 gradient coordinates. CPU64 max loss/gradient errors
1.7763568394002505e-15/2.220446049250313e-16; CPU32
9.774362634829004e-7/8.68425538147477e-8; MPS32
5.383714154483243e-7/5.289412674525096e-8. CPU/MPS max scalar gap
9.5367431640625e-7 < unchanged3e-5. CPU64 loss1e-11/grad1e-10/rtol1e-10,
FP32 loss+grad4e-6/rtol4e-5 unchanged. Saved FP32 represented row masses span
0.9999999925494194..1.0000000298023224, so m*softmax-w was necessary; no
renormalization or exact-unit-mass assumption was introduced.

Every valid/invalid transaction preserved inputs/targets/populated grad slots and
global CPU/MPS RNG. autograd.grad produced gradients; objective source writes no
grad slots and performs no backward. Output transfer preceded FP64 conversion.
Full actual/oracle arrays and masks were published before assertions. Saved JSON
was independently reconciled read-only with exact shapes/all coordinates/pins;
no component, harness, target or scalar-oracle replay. The metadata review used
original tool stdin; no filesystem executed-source hash is claimed.

Whole MPS guard 1.052379458094947s/RSS402128896B/peak MPS11288576B under
600s/4GiB/6GiB/reserve8GiB. All4 CPU guards finished: startup.794360708s,
oracle.000865917s, CPU64.119888250s, CPU32.093204375s <120s. Original two-root
retained JSON293052B at review <2MiB. Review guard also finished. One sparse COO
construction emitted PyTorch's default sparse-invariant-check warning; valid
literal coordinates were retained and no source was changed/retried. No source,
case, dtype, device, backend or tolerance rescue. Original plans/sources/commands/
reports/guards/STOPs are CLOSED/no replay/no retry.

Objective top_proximity_objective_v1.py remains unchanged3657B/SHA
d425365fad2836f267f641632779213406c8802403d0700c7db7db216efe5de5.
Two new proof/reviewer sources extend future1253->1255, preserving predecessors;
closed-model-proof1236 and scientific-runtime1211 unchanged. Source/math acceptance
is not trained transfer, model causality, state/resume/cache/scoring/quality or
upstream reproduction. No model/corpus/tokenizer/optimizer/sampler/seed roles/
checkpoint/scoring/Triton/weights/copies/deletions/retirement. Initialized global
RNG was inspected/restored; zero seeds or RNG draws. All historical closed model,
index and CE failures remain closed/unqualified; no missing old outputs recaptured.
Native incumbent/holdout and future_two prospective panel unchanged.

Next: Bounded read-only TOP consumer design boundary per top-consumer-design-handoff001.md: trace actual incumbent supplied-logit/head/backprop/loss-reduction and optimizer-parameter source; state independent TOP training-only head/scale/control contract and complete prospective model costs before any new model source or execution. Decide distinct applicability without reopening or recapturing closed future-token/index/CE proofs; no scientific roles/panel admission.
