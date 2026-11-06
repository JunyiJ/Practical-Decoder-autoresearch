# MPS component and full-model integration passed

This continues the original [direction review](evidence-review.md); that note,
the CPU proofs and all prior scientific sources stay immutable. No new research
direction, scientific seed, quality panel or adoption decision is introduced.

New component003-mps independently passed six generated cases against the proved
CPU FP64 dense-sum equation oracle. Maximum absolute value/gradient error was
7.209284422771134e-7 at the predeclared atol4e-5/rtol4e-4. Ordinary and exact-tie/
zero-route cases include D256/H336. Lower-index ties, device-local cumulative
counters and no-record behavior agree; CPU/MPS RNG states are unchanged.
The original MPS guard passed in4.25s, peakRSS406,781,952B and driver19,693,568B.
There is no automatic fallback, model/corpus scoring or weight output.

The new tokenizer_moe_model_v1 namespace has its own exact policy and validators.
It first constructs the genuine native incumbent, then replaces only changed
FFNs. The native control preserves initialization and post-CPU RNG exactly;
changed arms preserve native non-FFN weights but consume additional FFN draws.
This is not a claim that changed-shape matrices are identical. Native vocabulary,
MLA, context128, readout scale4, learned final scale/bias and optimizer recipe
remain fixed. Actual inherited block RMSNorm epsilon2^-23 and final norm1e-5
are preserved, following source rather than stale descriptions.

| Arm | Parameters | Muon groups | AdamW groups |
|---|---:|---:|---:|
| dense672 | 28,492,416 | 44 | 44 |
| dense1344 | 30,556,800 | 44 | 44 |
| top2of4 expert336 | 30,560,896 | 80 | 48 |

Router weights use output-like initialization multiplier2, forward divisor4 and
constant nominal AdamW LR. Expert/body matrices use the inherited half nominal
Muon LR at width256. The auxiliary term is .01 times mean layer load loss;
validation CE excludes it. No noise, z-loss, shared experts, token dropping or
new optimizer hyperparameters were added.

Generated model001 passed all three arms:

- Native exact initialization/RNG/logits/CE/gradients, two optimizer steps and
  complete child optimizer states agree bitwise with the genuine incumbent.
- CPU/MPS full-model logits/CE/all-gradient comparisons agree, maximum gradient
  discrepancy9.499490261077881e-8. This is generated input, not corpus quality.
- Causality and zero future embedding gradients pass on both devices. Full/
  cached/reset inference passes B1/B8 at context128; cache196608B/item unchanged.
  Foreign cache/alias and out-of-vocabulary IDs are rejected.
- Eighteen own weights-only kind/architecture/missing/shape/dtype/nonfinite
  forgeries are rejected before model mutation. This does not prove RAW/EMA resume.
- Exact-tie dormant experts receive None gradients and neither decay nor update;
  active zero-gradient parameters shrink by exactly1-.00006 with fresh optimizer
  state. Per-parameter gradient/decay exposure counts are explicit optimizer
  metadata, separate from routed-token assignment counters.

Model guard passed17.09s, peakRSS3,186,769,920B and driver2,283,585,536B, against
original600s/4GiB/6GiB/8GiBreserve limits. The generated proof retained no tensors
on disk. Its complete metadata-only floor9,876,074,496B and+2GiBtarget passed
actual95.98GBfree. That is a prerequisite forecast, not a future panel budget.
All1,009 predecessors plus4new model files and the proof give1,014 verified
source bindings. These new model/proof files are now immutable too.

## Next executable work

Implement a new additive native-data/RAW-H200 transport and proof. Genuine
moe_model/backbone_template configuration must not masquerade as cfg.model in an
old dense validator. Bind original native token IDs/private samplers, own model
signature, route counters and per-parameter optimizer exposure. Inactive expert
optimizer slots may legitimately be absent; Adam step counts must equal each
parameter's actual updates, rather than blindly equal the global step. Validate
all tensors/scalars/CPU-MPS RNG/sampler layouts before destination/global mutation.
EMA covers learned parameters, not integer route counters. Keep CE and auxiliary
objective separately auditable and clipping records explicit about active tensors.

Then run all three actual real-data four-update fresh repeats and serialized2+2
resume with exact state-byte identities, full clipping, optimizer/EMA/H200 and
samplers/RNG, under original guards. Prefer sequential digest references to
retaining multiple full baselines; current generated peakRSS is already3.19GB.
Forecast the complete serial proof and serialization memory/output before launch;
no relaxed guard, partial arm set, silent retry or old proof replay. A new fixture
seed is permitted; no scientific seed assignment. The holdout stays unencoded.

Real generated workers with independent full validation, actual20-update resource
pilots, finite source/statistics contracts, complete all-output budget and unused
scientific seed census still precede any quality freeze. No structural gain,
quality result, training throughput or sparse inference speed is established.
