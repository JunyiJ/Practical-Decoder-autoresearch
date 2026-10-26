# Fixed attention-logit scaling quality results

Fixed attention-logit scaling comparison completed 36 physical trajectories and 72 correlated raw/H200 outputs, independently reviewed. Common warmup 0 updates. Frozen eligible winner: half. Fresh H200 width64/256 deltas versus tuned baseline: half-scale -0.006210/-0.009851; double-scale +0.005693/+0.000935. Adopt the frozen half recipe after both-width support.

Historical source `53302eb5ff0674a0babdabce24183dc952436c6a`. Result `kb_79cd2f5f81a48a6542e9f7a9e29f88d3a6b0c6af7d1c1c0bf5aa3053a92611b0`. All 72 checkpoint hashes/payloads, requests/results, full coverage, optimizer/RNG/schedule/H200 contracts, frozen selection and complete report/bootstrap were independently reviewed. No training/evaluation rerun or holdout access.

Controller active time 2,812.65 seconds; output before review 530.89 MiB. [Frozen protocol](ATTENTION_SCALE_PROTOCOL.md).

## Interpretation and adopted recipe

Adopt attention-logit multiplier **0.5 with peak LR 0.01**, keeping the reviewed four-layer MHA/SwiGLU backbone, independent decay 0.00006, zero warmup, 3,000-update cooldown starting at 2,000, and H200 inference weights. Half-strength scores were eligible in screening and then passed every frozen confirmation requirement against both tuned baseline and the fixed incumbent at widths 64 and 256. The old incumbent is archived; its historical findings are unchanged.

Fresh H200 mean CE improves from 1.516007823 to 1.509797344 at width 64, and from 1.432910572 to 1.423059505 at width 256. Half-minus-baseline paired seed deltas are −0.000108423 / −0.012046134 / −0.006476880 at width 64 and −0.009109756 / −0.004908308 / −0.015535138 at width 256. Every seed is negative, and mean gains 0.006210479 / 0.009851067 exceed 0.005. The rule requires a mean gain above 0.005 and improvement on every seed; it does not require each seed's gain to exceed 0.005. In particular, width-64 seed 2411 improves only slightly.

All three conditional text-bootstrap upper bounds are negative at both widths. The 4,096-token intervals are [−0.009015252, −0.003271856] and [−0.011971762, −0.007767043]. These intervals condition on the trained seeds and reused validation text; they do not establish training-seed population uncertainty or independent holdout generalization.

Baseline and double selected LR 0.005, while half selected the upper grid boundary 0.01. The confirmation gain is a selected-recipe result, including that LR difference, not an isolated causal effect of the score multiplier. The separate fixed-LR 0.005 screen also favored half by 0.009554228 on average, with both screening seeds improving; those two screening runs are not fresh fixed-LR confirmation. No boundary extension or retrospective reselection follows.

Double-strength scores worsen every fresh width-64 seed, mean +0.005692578, and have mixed width-256 seeds with mean +0.000935324 and intervals containing zero. Only the frozen half candidate is adopted. Six fixed-incumbent cells prospectively alias selected baseline trajectories; these are identities, not six independent repeated controls. No historical control pooling, holdout access or training/scoring rerun occurred.

Half's measured training time is 0.85% / 0.79% higher at widths 64 / 256. These are descriptive measurements, not a speed-equivalence or serving-performance finding. All arms use equal 3,000 updates and 3,072,000 targets; changed scores, gradients, clipping and LR can change optimization paths. There is no diagnosed entropy-collapse mechanism or optimal muP-transfer claim.

## Fresh confirmation means

Seeds 2411/2412/2413; 111,538 validation targets once each. H200 primary; raw correlated diagnostic.

| Family | Selected peak LR | H200 CE64 | H200 CE256 | Raw CE64 | Raw CE256 |
|---|---:|---:|---:|---:|---:|
| baseline | 0.005 | 1.516008 | 1.432911 | 1.523558 | 1.448574 |
| half | 0.01 | 1.509797 | 1.423060 | 1.518723 | 1.440763 |
| double | 0.005 | 1.521700 | 1.433846 | 1.530373 | 1.447894 |
| incumbent_reference | 0.005 | 1.516008 | 1.432911 | 1.523558 | 1.448574 |

## Paired contrasts

Positive is worse. Intervals use 4,096-token blocks; all three block sizes were audited.

| Endpoint | Width | Contrast | Mean delta | Seed deltas | Conditional 95% text interval | Primary support |
|---|---:|---|---:|---|---|---|
| H200 | 64 | half minus baseline | -0.006210 | -0.000108 / -0.012046 / -0.006477 | [-0.009015, -0.003272] | True |
| Raw | 64 | half minus baseline | -0.004836 | -0.001897 / -0.009070 / -0.003539 | [-0.007454, -0.002191] | Diagnostic |
| H200 | 64 | double minus baseline | +0.005693 | +0.007148 / +0.004642 / +0.005288 | [+0.002617, +0.008928] | False |
| Raw | 64 | double minus baseline | +0.006815 | +0.006293 / +0.005784 / +0.008367 | [+0.003791, +0.009879] | Diagnostic |
| H200 | 64 | double minus half | +0.011903 | +0.007256 / +0.016688 / +0.011765 | [+0.009337, +0.014639] | False |
| Raw | 64 | double minus half | +0.011650 | +0.008190 / +0.014854 / +0.011906 | [+0.008885, +0.014562] | Diagnostic |
| H200 | 64 | baseline minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 64 | baseline minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 64 | half minus incumbent_reference | -0.006210 | -0.000108 / -0.012046 / -0.006477 | [-0.009015, -0.003272] | True |
| Raw | 64 | half minus incumbent_reference | -0.004836 | -0.001897 / -0.009070 / -0.003539 | [-0.007454, -0.002191] | Diagnostic |
| H200 | 64 | double minus incumbent_reference | +0.005693 | +0.007148 / +0.004642 / +0.005288 | [+0.002617, +0.008928] | False |
| Raw | 64 | double minus incumbent_reference | +0.006815 | +0.006293 / +0.005784 / +0.008367 | [+0.003791, +0.009879] | Diagnostic |
| H200 | 256 | half minus baseline | -0.009851 | -0.009110 / -0.004908 / -0.015535 | [-0.011972, -0.007767] | True |
| Raw | 256 | half minus baseline | -0.007811 | -0.006812 / -0.004517 / -0.012105 | [-0.010229, -0.005354] | Diagnostic |
| H200 | 256 | double minus baseline | +0.000935 | -0.000656 / +0.009151 / -0.005689 | [-0.001880, +0.003439] | False |
| Raw | 256 | double minus baseline | -0.000680 | -0.003455 / +0.009561 / -0.008145 | [-0.003392, +0.001987] | Diagnostic |
| H200 | 256 | double minus half | +0.010786 | +0.008454 / +0.014059 / +0.009846 | [+0.007533, +0.013526] | False |
| Raw | 256 | double minus half | +0.007131 | +0.003356 / +0.014078 / +0.003960 | [+0.003896, +0.009964] | Diagnostic |
| H200 | 256 | baseline minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 256 | baseline minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 256 | half minus incumbent_reference | -0.009851 | -0.009110 / -0.004908 / -0.015535 | [-0.011972, -0.007767] | True |
| Raw | 256 | half minus incumbent_reference | -0.007811 | -0.006812 / -0.004517 / -0.012105 | [-0.010229, -0.005354] | Diagnostic |
| H200 | 256 | double minus incumbent_reference | +0.000935 | -0.000656 / +0.009151 / -0.005689 | [-0.001880, +0.003439] | False |
| Raw | 256 | double minus incumbent_reference | -0.000680 | -0.003455 / +0.009561 / -0.008145 | [-0.003392, +0.001987] | Diagnostic |

## Fixed-LR screens

| Contrast | H200 mean delta | Two seed deltas | Raw mean delta |
|---|---:|---|---:|
| LR0.0025: half minus baseline | -0.003476 | -0.008061 / +0.001109 | -0.003399 |
| LR0.0025: double minus baseline | +0.003636 | +0.000453 / +0.006819 | +0.004029 |
| LR0.0025: double minus half | +0.007112 | +0.008514 / +0.005709 | +0.007428 |
| LR0.005: half minus baseline | -0.009554 | -0.005618 / -0.013491 | -0.008658 |
| LR0.005: double minus baseline | +0.000005 | +0.002100 / -0.002090 | -0.000694 |
| LR0.005: double minus half | +0.009559 | +0.007718 / +0.011400 | +0.007964 |
| LR0.01: half minus baseline | -0.010419 | -0.011165 / -0.009674 | -0.007330 |
| LR0.01: double minus baseline | +0.001079 | -0.003872 / +0.006030 | +0.002429 |
| LR0.01: double minus half | +0.011498 | +0.007292 / +0.015703 | +0.009759 |

## Measured training costs

| Family | Width | Training seconds | Wall seconds | Ratio versus tuned baseline | Sampled RSS MiB |
|---|---:|---:|---:|---:|---:|
| baseline | 64 | 54.86 | 58.92 | 1.000 | 475.42 |
| baseline | 256 | 125.45 | 133.59 | 1.000 | 484.19 |
| half | 64 | 55.32 | 59.42 | 1.009 | 475.02 |
| half | 256 | 126.45 | 134.31 | 1.008 | 484.07 |
| double | 64 | 55.26 | 59.35 | 1.007 | 475.12 |
| double | 256 | 126.56 | 134.65 | 1.009 | 484.69 |
| incumbent_reference | 64 | 54.86 | 58.92 | 1.000 | 475.42 |
| incumbent_reference | 256 | 125.45 | 133.59 | 1.000 | 484.19 |

## Limits

- Adaptive development on reused validation; conditional text intervals are not seed-population uncertainty, independent holdout or adjusted significance.
- 36 physical trajectories yield 72 correlated raw/H200 endpoints. Reference aliases {'incumbent_reference': 'baseline'} were declared prospectively and are not independent repeated controls; no historical pooling.
- Only the frozen eligible width64 winner can be adopted; no alternate fallback, larger-width reselection, grid extension or completed-trial rerun.
- Equal finite LR tuning; selected-recipe contrasts include LR choice. A tuned baseline-only gain is attributed to LR selection, not attention scaling. Fixed-LR screens are separate.
- Equal 3000 updates and 3072000 targets, common final-reviewed residual topology/table/bias/prior/norm/input, warmup/cooldown/H200 and independent decay. Fixed1/.5/2 multiplier is applied once to post-RoPE QK/sqrt32 before causal masking and softmax. Unit path preserves historical arithmetic; cached K/V stay unscaled. Parameter layout and construction RNG are unchanged, but gradients, clipping and optimization paths can differ. Q/K weights can adapt. No diagnosed entropy collapse, isolated mechanism, paper replication or optimal muP transfer is claimed. Double-minus-half is reported separately.
- Measured training time is separate from quality; no equal-compute, serving-speed or peak-accelerator-memory claim.
- Final H200 tensors equal saved raw-parent shadows; endpoint evidence cannot reconstruct every historical snapshot. Earlier MPS reproducibility failures remain preserved without an identified cause.
- H200 checkpoints are inference-only; resume optimizer training only from their raw parent.
