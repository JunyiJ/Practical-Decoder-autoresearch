# Complete-branch residual gain quality results

Complete-branch residual gain comparison completed 36 physical trajectories and 72 correlated raw/H200 outputs, independently reviewed. Common warmup 0 updates. Frozen eligible winner: incumbent_unchanged. Fresh H200 width64/256 deltas versus tuned Unit gain: Static 0.1 +0.021737/+0.029721; Learned 0.1 +0.026876/+0.020310. Retain the prior incumbent under the complete frozen rule.

Historical source `5622e4b66745397764284500b9504fe71bec150d`. Result `kb_6169eeca230e3dcd46f4f4f5006fd33907e1aaf109dedef5bc5a6a260d935c49`. All72 checkpoint hashes/payloads, requests/results, full coverage, optimizer/RNG/schedule/H200 contracts, frozen selection and complete report/bootstrap were independently reviewed. No training/evaluation rerun or holdout access.

Controller active time 2887.83seconds; output before review 531.32MiB. [Frozen protocol](RESIDUAL_GAIN_PROTOCOL.md).

## Fresh confirmation means

Seeds2451/2452/2453;111,538 validation targets once each. H200 primary; raw correlated diagnostic.

| Family | Selected peak LR | H200 CE64 | H200 CE256 | Raw CE64 | Raw CE256 |
|---|---:|---:|---:|---:|---:|
| unit | 0.01 | 1.507867 | 1.420070 | 1.513755 | 1.435713 |
| static_tenth | 0.01 | 1.529604 | 1.449791 | 1.534389 | 1.459711 |
| learned_tenth | 0.01 | 1.534743 | 1.440380 | 1.540085 | 1.455735 |
| incumbent_reference | 0.01 | 1.507867 | 1.420070 | 1.513755 | 1.435713 |

## Paired contrasts

Positive is worse. Intervals use4096-token blocks; all three block sizes were audited.

| Endpoint | Width | Contrast | Mean delta | Seed deltas | Conditional95% text interval | Primary support |
|---|---:|---|---:|---|---|---|
| H200 | 64 | static_tenth minus unit | +0.021737 | +0.006874 / +0.038502 / +0.019835 | [+0.019318, +0.024250] | False |
| Raw | 64 | static_tenth minus unit | +0.020634 | +0.005435 / +0.035543 / +0.020926 | [+0.017753, +0.023347] | Diagnostic |
| H200 | 64 | learned_tenth minus unit | +0.026876 | +0.017211 / +0.037899 / +0.025519 | [+0.024605, +0.029025] | False |
| Raw | 64 | learned_tenth minus unit | +0.026330 | +0.017349 / +0.035800 / +0.025841 | [+0.023751, +0.028946] | Diagnostic |
| H200 | 64 | learned_tenth minus static_tenth | +0.005139 | +0.010337 / -0.000604 / +0.005684 | [+0.003028, +0.007484] | False |
| Raw | 64 | learned_tenth minus static_tenth | +0.005696 | +0.011915 / +0.000258 / +0.004915 | [+0.003040, +0.008467] | Diagnostic |
| H200 | 64 | unit minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 64 | unit minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 64 | static_tenth minus incumbent_reference | +0.021737 | +0.006874 / +0.038502 / +0.019835 | [+0.019318, +0.024250] | False |
| Raw | 64 | static_tenth minus incumbent_reference | +0.020634 | +0.005435 / +0.035543 / +0.020926 | [+0.017753, +0.023347] | Diagnostic |
| H200 | 64 | learned_tenth minus incumbent_reference | +0.026876 | +0.017211 / +0.037899 / +0.025519 | [+0.024605, +0.029025] | False |
| Raw | 64 | learned_tenth minus incumbent_reference | +0.026330 | +0.017349 / +0.035800 / +0.025841 | [+0.023751, +0.028946] | Diagnostic |
| H200 | 256 | static_tenth minus unit | +0.029721 | +0.030482 / +0.022615 / +0.036065 | [+0.026457, +0.032854] | False |
| Raw | 256 | static_tenth minus unit | +0.023998 | +0.028232 / +0.014173 / +0.029588 | [+0.020042, +0.027590] | Diagnostic |
| H200 | 256 | learned_tenth minus unit | +0.020310 | +0.021817 / +0.009447 / +0.029666 | [+0.016688, +0.024219] | False |
| Raw | 256 | learned_tenth minus unit | +0.020022 | +0.023090 / +0.008862 / +0.028115 | [+0.016909, +0.023495] | Diagnostic |
| H200 | 256 | learned_tenth minus static_tenth | -0.009411 | -0.008665 / -0.013167 / -0.006399 | [-0.013192, -0.005537] | True |
| Raw | 256 | learned_tenth minus static_tenth | -0.003975 | -0.005142 / -0.005311 / -0.001473 | [-0.007168, -0.000734] | Diagnostic |
| H200 | 256 | unit minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 256 | unit minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 256 | static_tenth minus incumbent_reference | +0.029721 | +0.030482 / +0.022615 / +0.036065 | [+0.026457, +0.032854] | False |
| Raw | 256 | static_tenth minus incumbent_reference | +0.023998 | +0.028232 / +0.014173 / +0.029588 | [+0.020042, +0.027590] | Diagnostic |
| H200 | 256 | learned_tenth minus incumbent_reference | +0.020310 | +0.021817 / +0.009447 / +0.029666 | [+0.016688, +0.024219] | False |
| Raw | 256 | learned_tenth minus incumbent_reference | +0.020022 | +0.023090 / +0.008862 / +0.028115 | [+0.016909, +0.023495] | Diagnostic |

## Fixed-LR screens

| Contrast | H200 mean delta | Two seed deltas | Raw mean delta |
|---|---:|---|---:|
| LR0.0025: static_tenth minus unit | +0.055759 | +0.056281 / +0.055237 | +0.053847 |
| LR0.0025: learned_tenth minus unit | +0.031533 | +0.026231 / +0.036836 | +0.030914 |
| LR0.0025: learned_tenth minus static_tenth | -0.024225 | -0.030050 / -0.018401 | -0.022933 |
| LR0.005: static_tenth minus unit | +0.040441 | +0.044400 / +0.036483 | +0.038050 |
| LR0.005: learned_tenth minus unit | +0.027696 | +0.028187 / +0.027205 | +0.026858 |
| LR0.005: learned_tenth minus static_tenth | -0.012745 | -0.016213 / -0.009277 | -0.011192 |
| LR0.01: static_tenth minus unit | +0.039943 | +0.044051 / +0.035836 | +0.038831 |
| LR0.01: learned_tenth minus unit | +0.026575 | +0.019697 / +0.033452 | +0.025815 |
| LR0.01: learned_tenth minus static_tenth | -0.013368 | -0.024353 / -0.002383 | -0.013017 |

## Measured training costs

| Family | Width | Training seconds | Wall seconds | Ratio versus tuned Unit gain | Sampled RSS MiB |
|---|---:|---:|---:|---:|---:|
| unit | 64 | 55.20 | 59.63 | 1.000 | 475.11 |
| unit | 256 | 126.53 | 135.04 | 1.000 | 484.19 |
| static_tenth | 64 | 55.48 | 59.79 | 1.005 | 475.62 |
| static_tenth | 256 | 126.79 | 135.33 | 1.002 | 484.20 |
| learned_tenth | 64 | 59.43 | 64.03 | 1.077 | 475.18 |
| learned_tenth | 256 | 131.91 | 140.43 | 1.043 | 484.01 |
| incumbent_reference | 64 | 55.20 | 59.63 | 1.000 | 475.11 |
| incumbent_reference | 256 | 126.53 | 135.04 | 1.000 | 484.19 |

## Limits

- Adaptive development on reused validation; conditional text intervals are not seed-population uncertainty, independent holdout or adjusted significance.
- 36 physical trajectories yield 72 correlated raw/H200 endpoints. Reference aliases {'incumbent_reference': 'unit'} were declared prospectively and are not independent repeated controls; no historical pooling.
- Only the frozen eligible width64 winner can be adopted; no alternate fallback, larger-width reselection, grid extension or completed-trial rerun.
- Equal finite LR tuning; selected-recipe contrasts include LR choice. A tuned Unit gain-only gain is attributed to LR selection, not residual gain. Fixed-LR screens are separate.
- Equal 3000 updates and 3072000 targets. Final-reviewed gated activation, attention scale, residual topology, table, bias/prior, readout norm, input scale, warmup/cooldown/H200 and independent decay are common. Both complete post-projection attention and FFN branches, including projection biases, receive unit, fixed0.1 or learned per-channel gains before residual addition. Learned gains are eight independent width-D vectors initialized0.1 without RNG draws, unconstrained in sign, using AdamW base LR and the common independent decay. Static and learned variants initially match; learned gains add8D parameters and couple to optimization. H200 averages gain tensors explicitly; gains are never folded into averaged projections. No isolated mechanism, LayerScale replication or optimal muP transfer is claimed. Learned-minus-static is reported separately and is not an additional adoption gate.
- Measured training time is separate from quality; no equal-compute, serving-speed or peak-accelerator-memory claim.
- Final H200 tensors equal saved raw-parent shadows; endpoint evidence cannot reconstruct every historical snapshot. Earlier MPS reproducibility failures remain preserved without an identified cause.
- H200 checkpoints are inference-only; resume optimizer training only from their raw parent.

## Interpretation and retained recipe

Retain the existing unit-gain SwiGLU model with attention-logit scale 0.5 and peak LR 0.01. Unit gain was the best screen and exactly matched the fixed incumbent, so retention was frozen before confirmation. All three arms selected the upper LR boundary 0.01; no grid extension or larger-width reselection follows.

Static 0.1 gains worsen every fresh seed at both widths: mean H200 CE increases by 0.021737240 at width 64 and 0.029720798 at width 256. Learned gains initialized at 0.1 also worsen every seed, by 0.026876334 / 0.020310062. All three conditional text intervals are positive for both comparisons at both widths. The 4,096-token intervals are [+0.019317582, +0.024249675] / [+0.026456747, +0.032853595] for static minus unit, and [+0.024605299, +0.029025154] / [+0.016687804, +0.024218942] for learned minus unit.

Learned minus static is +0.005139094 at width 64 with mixed seeds, and −0.009410736 at width 256 with every seed improving. The latter contrast does not qualify either attenuated recipe against the unit-gain incumbent. It was a separate explanatory comparison, not an additional adoption gate. Conditional text intervals condition on these trained seeds and reused validation text; they do not quantify training-seed population uncertainty.

Selected LRs are identical in this round. The differences therefore do not include a selected-LR difference, although residual attenuation and learned parameters change gradients, optimizer histories and effective updates. Static and learned gains initially match, with learned gains adding 512 / 2,048 parameters at widths 64 / 256. Both complete post-projection branches include their biases; learned gains remain explicit in H200 checkpoints and are never folded into projections. This is not an isolated mechanism, LayerScale replication or optimal muP-transfer claim.

Static training time increases by 0.52% / 0.20%, and learned-gain time by 7.68% / 4.25%. These are descriptive training measurements, not serving-speed or equivalence claims. All arms use equal 3,000 updates and 3,072,000 targets. Six fixed-reference cells prospectively alias unit-gain trajectories and are not independent repeated controls. No historical control pooling, holdout access, training/scoring rerun or alternative promotion occurred.
