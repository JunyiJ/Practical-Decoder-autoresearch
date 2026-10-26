# Gated FFN activation quality results

Gated FFN activation comparison completed 36 physical trajectories and 72 correlated raw/H200 outputs, independently reviewed. Common warmup 0 updates. Frozen eligible winner: incumbent_unchanged. Fresh H200 width64/256 deltas versus tuned SwiGLU: GEGLU -0.000404/+0.006786; ReGLU +0.007098/+0.006577. Retain the prior incumbent under the complete frozen rule.

Historical source `570a0fc35f6ab1139dece4f302ffc148a4a6fa41`. Result `kb_ee2cfb969a8029e552f6d178506d7e9e6c8eeb69e45ae104122e2b05aa2c757b`. All 72 checkpoint hashes/payloads, requests/results, full coverage, optimizer/RNG/schedule/H200 contracts, frozen selection and complete report/bootstrap were independently reviewed. No training/evaluation rerun or holdout access.

Controller active time 2,816.62 seconds; output before review 530.90 MiB. [Frozen protocol](GATED_ACTIVATION_PROTOCOL.md).

## Interpretation and retained recipe

Retain **SwiGLU with attention-logit scale 0.5 and peak LR 0.01**, the previously adopted four-layer MHA recipe, independent decay 0.00006, zero warmup, 3,000-update cooldown starting at 2,000, and H200 inference weights. GEGLU was the lowest screening mean, but its 0.002782548 gain versus tuned SwiGLU fell below the required 0.005. Retention was therefore frozen before fresh confirmation; no alternative promotion or threshold change follows.

Fresh H200 means at widths 64 / 256 are SwiGLU 1.511098040 / 1.425150793, GEGLU 1.510693884 / 1.431936621, and ReGLU 1.518195822 / 1.431728243. GEGLU-minus-SwiGLU is −0.000404156 at width 64, with seed deltas −0.000628967 / +0.002054515 / −0.002638015, and +0.006785828 at width 256, with every seed worse: +0.012424184 / +0.004164474 / +0.003768827. Its width-64 conditional text intervals include zero; all three width-256 intervals are positive. The 4,096-token intervals are [−0.005103039, +0.004022233] and [+0.003838367, +0.009711387].

ReGLU worsens every fresh seed at both widths, mean +0.007097782 / +0.006577450 versus SwiGLU, with all three conditional text intervals positive. ReGLU-minus-GEGLU is +0.007501938 at width 64 with every seed worse, and −0.000208378 at width 256 with mixed seeds and intervals including zero. Raw outputs remain correlated diagnostics; they do not replace the H200 decision endpoint.

SwiGLU selected LR 0.01 at the upper boundary, while GEGLU and ReGLU selected interior LR 0.005. These are selected-recipe comparisons including LR choice, not isolated activation effects. The separate fixed-LR 0.005 screen favored GEGLU by 0.006364440 on average with both screening seeds improving; that does not override the tuned comparator or constitute fresh fixed-LR confirmation. No grid extension, fallback or larger-width reselection occurred.

All arms have the same three bias-free matrices, hidden width 21d/8, initialization draws and parameter counts, but their initial functions, gradients and optimization paths differ. This is not a T5 replication or empirical proof of optimal muP transfer. Six fixed-incumbent cells prospectively alias SwiGLU trajectories and are not independent repeated controls. No historical control pooling, holdout access or completed-training/scoring rerun occurred. Conditional text intervals do not represent training-seed population uncertainty.

Measured GEGLU training time changes by +0.10% / +0.60% and ReGLU by −1.36% / +0.32% at widths 64 / 256. These are descriptive measurements, not serving-speed or equivalence claims. All arms use equal 3,000 updates and 3,072,000 targets.

## Fresh confirmation means

Seeds 2431/2432/2433; 111,538 validation targets once each. H200 primary; raw correlated diagnostic.

| Family | Selected peak LR | H200 CE64 | H200 CE256 | Raw CE64 | Raw CE256 |
|---|---:|---:|---:|---:|---:|
| swiglu | 0.01 | 1.511098 | 1.425151 | 1.520136 | 1.442389 |
| geglu | 0.005 | 1.510694 | 1.431937 | 1.517088 | 1.449938 |
| reglu | 0.005 | 1.518196 | 1.431728 | 1.525704 | 1.446767 |
| incumbent_reference | 0.01 | 1.511098 | 1.425151 | 1.520136 | 1.442389 |

## Paired contrasts

Positive is worse. Intervals use 4,096-token blocks; all three block sizes were audited.

| Endpoint | Width | Contrast | Mean delta | Seed deltas | Conditional 95% text interval | Primary support |
|---|---:|---|---:|---|---|---|
| H200 | 64 | geglu minus swiglu | -0.000404 | -0.000629 / +0.002055 / -0.002638 | [-0.005103, +0.004022] | False |
| Raw | 64 | geglu minus swiglu | -0.003048 | -0.004901 / +0.001026 / -0.005268 | [-0.007775, +0.001533] | Diagnostic |
| H200 | 64 | reglu minus swiglu | +0.007098 | +0.004756 / +0.012672 / +0.003866 | [+0.003777, +0.010463] | False |
| Raw | 64 | reglu minus swiglu | +0.005568 | +0.001268 / +0.010417 / +0.005018 | [+0.001851, +0.009128] | Diagnostic |
| H200 | 64 | reglu minus geglu | +0.007502 | +0.005385 / +0.010617 / +0.006504 | [+0.004300, +0.010524] | False |
| Raw | 64 | reglu minus geglu | +0.008615 | +0.006169 / +0.009391 / +0.010286 | [+0.005590, +0.011631] | Diagnostic |
| H200 | 64 | swiglu minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 64 | swiglu minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 64 | geglu minus incumbent_reference | -0.000404 | -0.000629 / +0.002055 / -0.002638 | [-0.005103, +0.004022] | False |
| Raw | 64 | geglu minus incumbent_reference | -0.003048 | -0.004901 / +0.001026 / -0.005268 | [-0.007775, +0.001533] | Diagnostic |
| H200 | 64 | reglu minus incumbent_reference | +0.007098 | +0.004756 / +0.012672 / +0.003866 | [+0.003777, +0.010463] | False |
| Raw | 64 | reglu minus incumbent_reference | +0.005568 | +0.001268 / +0.010417 / +0.005018 | [+0.001851, +0.009128] | Diagnostic |
| H200 | 256 | geglu minus swiglu | +0.006786 | +0.012424 / +0.004164 / +0.003769 | [+0.003838, +0.009711] | False |
| Raw | 256 | geglu minus swiglu | +0.007549 | +0.014863 / +0.005321 / +0.002463 | [+0.004386, +0.010724] | Diagnostic |
| H200 | 256 | reglu minus swiglu | +0.006577 | +0.011036 / +0.007866 / +0.000830 | [+0.003875, +0.009268] | False |
| Raw | 256 | reglu minus swiglu | +0.004378 | +0.011321 / +0.003566 / -0.001753 | [+0.001469, +0.007263] | Diagnostic |
| H200 | 256 | reglu minus geglu | -0.000208 | -0.001388 / +0.003702 / -0.002939 | [-0.002752, +0.002360] | False |
| Raw | 256 | reglu minus geglu | -0.003171 | -0.003542 / -0.001755 / -0.004216 | [-0.006193, -0.000170] | Diagnostic |
| H200 | 256 | swiglu minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 256 | swiglu minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 256 | geglu minus incumbent_reference | +0.006786 | +0.012424 / +0.004164 / +0.003769 | [+0.003838, +0.009711] | False |
| Raw | 256 | geglu minus incumbent_reference | +0.007549 | +0.014863 / +0.005321 / +0.002463 | [+0.004386, +0.010724] | Diagnostic |
| H200 | 256 | reglu minus incumbent_reference | +0.006577 | +0.011036 / +0.007866 / +0.000830 | [+0.003875, +0.009268] | False |
| Raw | 256 | reglu minus incumbent_reference | +0.004378 | +0.011321 / +0.003566 / -0.001753 | [+0.001469, +0.007263] | Diagnostic |

## Fixed-LR screens

| Contrast | H200 mean delta | Two seed deltas | Raw mean delta |
|---|---:|---|---:|
| LR0.0025: geglu minus swiglu | -0.000241 | +0.002957 / -0.003440 | +0.000848 |
| LR0.0025: reglu minus swiglu | +0.004448 | +0.004743 / +0.004153 | +0.006314 |
| LR0.0025: reglu minus geglu | +0.004689 | +0.001786 / +0.007593 | +0.005466 |
| LR0.005: geglu minus swiglu | -0.006364 | -0.010566 / -0.002163 | -0.005462 |
| LR0.005: reglu minus swiglu | -0.004225 | -0.008725 / +0.000275 | -0.003669 |
| LR0.005: reglu minus geglu | +0.002139 | +0.001841 / +0.002438 | +0.001793 |
| LR0.01: geglu minus swiglu | -0.002341 | +0.000229 / -0.004911 | -0.001183 |
| LR0.01: reglu minus swiglu | +0.003394 | +0.000170 / +0.006618 | +0.002959 |
| LR0.01: reglu minus geglu | +0.005735 | -0.000059 / +0.011528 | +0.004142 |

## Measured training costs

| Family | Width | Training seconds | Wall seconds | Ratio versus tuned SwiGLU | Sampled RSS MiB |
|---|---:|---:|---:|---:|---:|
| swiglu | 64 | 55.19 | 59.25 | 1.000 | 474.56 |
| swiglu | 256 | 125.80 | 134.26 | 1.000 | 486.61 |
| geglu | 64 | 55.24 | 59.40 | 1.001 | 475.37 |
| geglu | 256 | 126.56 | 134.87 | 1.006 | 484.58 |
| reglu | 64 | 54.43 | 58.55 | 0.986 | 476.42 |
| reglu | 256 | 126.21 | 134.71 | 1.003 | 486.79 |
| incumbent_reference | 64 | 55.19 | 59.25 | 1.000 | 474.56 |
| incumbent_reference | 256 | 125.80 | 134.26 | 1.000 | 486.61 |

## Limits

- Adaptive development on reused validation; conditional text intervals are not seed-population uncertainty, independent holdout or adjusted significance.
- 36 physical trajectories yield 72 correlated raw/H200 endpoints. Reference aliases {'incumbent_reference': 'swiglu'} were declared prospectively and are not independent repeated controls; no historical pooling.
- Only the frozen eligible width64 winner can be adopted; no alternate fallback, larger-width reselection, grid extension or completed-trial rerun.
- Equal finite LR tuning; selected-recipe contrasts include LR choice. A tuned SwiGLU-only gain is attributed to LR selection, not gated activation. Fixed-LR screens are separate.
- Equal 3000 updates and 3072000 targets. Final-reviewed attention scale, residual topology, table, bias/prior, readout norm, input scale, warmup/cooldown/H200 and independent decay are common. SwiGLU, GEGLU and ReGLU use SiLU, exact-erf GELU and ReLU respectively on the gate projection, with the same three bias-free matrices and hidden width21d/8. Parameter layout and construction RNG are unchanged; initial functions, gradients, clipping and optimizer paths can differ. No isolated mechanism, T5 replication or optimal muP transfer is claimed. ReGLU-minus-GEGLU is reported separately.
- Measured training time is separate from quality; no equal-compute, serving-speed or peak-accelerator-memory claim.
- Final H200 tensors equal saved raw-parent shadows; endpoint evidence cannot reconstruct every historical snapshot. Earlier MPS reproducibility failures remain preserved without an identified cause.
- H200 checkpoints are inference-only; resume optimizer training only from their raw parent.
