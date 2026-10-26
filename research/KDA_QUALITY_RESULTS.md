# KDA quality results

KDA comparison completed 24 physical trajectories and 48 correlated raw/H200 outputs, independently reviewed. Frozen choices: mha LR 0.005, kda LR 0.005; eligible winner: incumbent_unchanged. Fresh H200 KDA-minus-MHA deltas at widths 64/256: +0.004848/+0.007836. The complete frozen adoption rule is not met; retain the prior incumbent.

Historical source `b254d13a234b5d9075c802dbaa1e32f195553267`. Result `kb_efc90606f2c734ba312f765148f9524885a4be1439dafe1765079d12ef3d7aa7`. All 48 raw/H200 checkpoint hashes and payload contracts, requests/results, optimizer/decay/RNG/schedule/attention-alpha-convolution-head-FFN-shape/full-coverage contracts, frozen choices, aliases and complete report/bootstrap were independently reviewed. No training/evaluation rerun or holdout access.

Controller active time 6547.45 seconds; output before review 376.08 MiB. [Frozen protocol](KDA_QUALITY_PROTOCOL.md).

## Fresh confirmation means

Seeds 2271/2272/2273; 111,538 validation targets once each. H200 is primary; raw is a correlated diagnostic.

| Recipe | Decay | Selected LR | H200 CE64 | H200 CE256 | Raw CE64 | Raw CE256 |
|---|---:|---:|---:|---:|---:|---:|
| mha | 6e-05 | 0.005 | 1.522230 | 1.431696 | 1.526995 | 1.446416 |
| kda | 6e-05 | 0.005 | 1.527077 | 1.439532 | 1.531449 | 1.451928 |
| incumbent_reference | 6e-05 | 0.005 | 1.522230 | 1.431696 | 1.526995 | 1.446416 |

## Paired contrasts

Positive is worse. Intervals use 4,096-token blocks; all three block sizes were audited. Contrast support alone does not determine adoption.

| Endpoint | Width | Contrast | Mean delta | Paired seed deltas | Conditional 95% text interval | Primary support |
|---|---:|---|---:|---|---|---|
| H200 | 64 | kda minus mha | +0.004848 | -0.002018 / +0.003602 / +0.012958 | [+0.001256, +0.008183] | False |
| Raw | 64 | kda minus mha | +0.004454 | +0.001785 / +0.000545 / +0.011030 | [+0.000957, +0.007785] | Diagnostic |
| H200 | 64 | mha minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 64 | mha minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 64 | kda minus incumbent_reference | +0.004848 | -0.002018 / +0.003602 / +0.012958 | [+0.001256, +0.008183] | False |
| Raw | 64 | kda minus incumbent_reference | +0.004454 | +0.001785 / +0.000545 / +0.011030 | [+0.000957, +0.007785] | Diagnostic |
| H200 | 256 | kda minus mha | +0.007836 | +0.007242 / +0.010068 / +0.006198 | [+0.002883, +0.012156] | False |
| Raw | 256 | kda minus mha | +0.005513 | +0.005707 / +0.007765 / +0.003066 | [-0.000104, +0.010648] | Diagnostic |
| H200 | 256 | mha minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 256 | mha minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 256 | kda minus incumbent_reference | +0.007836 | +0.007242 / +0.010068 / +0.006198 | [+0.002883, +0.012156] | False |
| Raw | 256 | kda minus incumbent_reference | +0.005513 | +0.005707 / +0.007765 / +0.003066 | [-0.000104, +0.010648] | Diagnostic |

## Fixed-LR screens

| LR | Contrast | H200 mean delta | Two seed deltas | Raw mean delta |
|---|---|---:|---|---:|
| 0.00125 | kda minus mha | -0.007888 | -0.017467 / +0.001691 | -0.005992 |
| 0.0025 | kda minus mha | +0.007570 | -0.001219 / +0.016358 | +0.009569 |
| 0.005 | kda minus mha | +0.001575 | -0.007350 / +0.010500 | -0.000774 |

## Measured training costs

| Recipe | Width | Training seconds | Trajectory wall seconds | Ratio versus MHA | Sampled RSS MiB |
|---|---:|---:|---:|---:|---:|
| mha | 64 | 54.86 | 59.23 | 1.000 | 474.03 |
| mha | 256 | 125.63 | 133.67 | 1.000 | 482.83 |
| kda | 64 | 408.29 | 428.56 | 7.442 | 530.84 |
| kda | 256 | 558.94 | 580.78 | 4.449 | 546.01 |
| incumbent_reference | 64 | 54.86 | 59.23 | 1.000 | 474.03 |
| incumbent_reference | 256 | 125.63 | 133.67 | 1.000 | 482.83 |

## Limits

- Fresh development seeds and reused full validation remain adaptive evidence. Conditional text intervals are not seed-population uncertainty, independent holdout, or adjusted significance.
- 24 physical trajectories produce 48 correlated raw/H200 outputs. Exact prospective reference aliases are not independent repeats.
- Only the frozen width-64 winner is eligible for adoption. No alternate fallback, larger-width reselection, LR-grid extension or completed-trial rerun.
- The KDA adaptation derives alpha bottleneck D/4 under its named policy; the human default outside this policy is unchanged. Matrix width factors are prospective conventions, not demonstrated optimal muP transfer.
- KDA changes recurrence, causal convolution, positional mechanism, gating and initializer draws jointly. It is an all-KDA dense-SwiGLU local adaptation, not a reproduction of the cited hybrid system.
- Shared final-reviewed constant independent decay, 3,000-update cooldown and H200 policy; inner optimizer decay stays zero. Both arms selected the same peak LR; confirmation compares the family changes at that LR.
- Equal 3,000 updates and 3,072,000 targets do not imply equal compute. Training time is measured separately; no serving-speed or peak-accelerator-memory claim.
- H200 tensors match final raw-parent shadows, but saved endpoints cannot reconstruct every historical averaging snapshot. Preserve earlier MPS reproducibility failures without an identified cause.
- H200 checkpoints are inference-only; optimizer continuation requires the original raw training parent.

Both families selected LR 0.005 at the frozen upper boundary; no extension. The fixed incumbent prospectively aliases the six selected MHA confirmation cells, so these are not independent repeats. At width 64, KDA H200 improves seed 2271 but worsens 2272 and 2273. At width 256 it worsens all three seeds. All three conditional text intervals are positive at both widths. This is evidence about this local adaptation and budget, not a general judgment on KDA or the human reference implementation.
