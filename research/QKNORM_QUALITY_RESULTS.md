# QK normalization quality results

QK normalization completed 24 physical training trajectories and 48 correlated raw/H200 endpoints, all independently reviewed. Both families selected LR 0.005 at the upper grid boundary. QK RMS screening improved mean CE by 0.004858576, short of the frozen >0.005 criterion; incumbent retention was frozen before confirmation. Fresh QK-minus-standard H200 mean deltas at widths 64/256 are -0.005274/+0.007981. The unchanged MHA, 3,000 updates, cooldown starting at 2,000, H200 incumbent remains; no grid extension or posthoc adoption.

Historical source `2c8d77a35232107cbba9d57db9e0c9e359279fd2`. Result `kb_32f17c4c75d8825a53cf1d8f6bd8eb54fe19b0f324d992163e2a01d24c52744c`. All 24 raw and 24 H200 checkpoint hashes, requests/results, saved configurations, gain/optimizer states, RNG, schedules, shapes, full validation coverage, frozen selection/aliases and entire report were independently verified before documentation. Conditional bootstrap intervals were reconstructed independently. No training/evaluation rerun or holdout access.

Controller active time 1991.90 seconds (0.55 hours); output before review 353.88 MiB. [Frozen protocol](QKNORM_QUALITY_PROTOCOL.md).

## Fresh confirmation means

Seeds 2191, 2192 and 2193; 111,538 validation targets once each at context 128 and batch size 8. H200 primary, raw correlated diagnostic.

| Recipe | H200 CE64 | H200 CE256 | Raw CE64 | Raw CE256 |
|---|---:|---:|---:|---:|
| standard, LR.005 | 1.518116 | 1.430485 | 1.520760 | 1.445553 |
| qkrms, LR.005 | 1.512843 | 1.438466 | 1.517506 | 1.455690 |

## Paired confirmation contrasts

Positive is worse. Intervals condition on the three seed pairs; all 2,048/4,096/8,192 block sizes are audited. Fixed incumbent equals selected standard through prospective aliases.

| Endpoint | Width | Candidate minus reference | Mean delta | Seeds 2191 / 2192 / 2193 | 4,096-token conditional 95% interval | Primary support |
|---|---:|---|---:|---|---|---|
| H200 | 64 | qkrms minus standard | -0.005274 | -0.000151 / -0.011806 / -0.003864 | [-0.007651, -0.003114] | True |
| Raw | 64 | qkrms minus standard | -0.003254 | +0.002358 / -0.009942 / -0.002180 | [-0.005414, -0.001189] | Diagnostic |
| H200 | 64 | qkrms minus incumbent_reference | -0.005274 | -0.000151 / -0.011806 / -0.003864 | [-0.007651, -0.003114] | True |
| Raw | 64 | qkrms minus incumbent_reference | -0.003254 | +0.002358 / -0.009942 / -0.002180 | [-0.005414, -0.001189] | Diagnostic |
| H200 | 64 | standard minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 64 | standard minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 256 | qkrms minus standard | +0.007981 | +0.010742 / +0.007842 / +0.005358 | [+0.004521, +0.011497] | False |
| Raw | 256 | qkrms minus standard | +0.010137 | +0.013965 / +0.009158 / +0.007288 | [+0.007217, +0.012968] | Diagnostic |
| H200 | 256 | qkrms minus incumbent_reference | +0.007981 | +0.010742 / +0.007842 / +0.005358 | [+0.004521, +0.011497] | False |
| Raw | 256 | qkrms minus incumbent_reference | +0.010137 | +0.013965 / +0.009158 / +0.007288 | [+0.007217, +0.012968] | Diagnostic |
| H200 | 256 | standard minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 256 | standard minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |

## Fixed-LR screening

Two screening seeds 2181 and 2182. These contrasts preserve equal-LR evidence separately from selected recipes.

| LR | H200 QK minus standard | Paired deltas | Raw mean delta |
|---|---:|---|---:|
| 0.00125 | -0.008569 | -0.003044 / -0.014093 | -0.009344 |
| 0.0025 | +0.001454 | -0.001869 / +0.004777 | +0.001304 |
| 0.005 | -0.004859 | -0.006545 / -0.003173 | -0.003483 |

## Measured costs

| Recipe | Width | Training seconds | Trajectory wall seconds | Training ratio vs standard | Sampled RSS MiB |
|---|---:|---:|---:|---:|---:|
| standard | 64 | 54.90 | 59.00 | 1.000 | 473.77 |
| standard | 256 | 125.39 | 133.57 | 1.000 | 482.59 |
| qkrms | 64 | 64.06 | 68.04 | 1.167 | 475.63 |
| qkrms | 256 | 139.36 | 147.72 | 1.111 | 484.64 |

## Next bounded hypothesis

A separate output-gating comparison is under development: unchanged attention, constant-half weighted values and a zero-initialized learned headwise sigmoid gate. Resolve the backbone from this final review. Actual training/full-suite/MPS/resource checks and a new source contract/preregistration are required before launch. All completed studies remain closed.

## Limits

- Fresh development seeds and full validation remain adaptive evidence; conditional text intervals are not seed-population uncertainty, holdout, or adjusted significance.
- Twelve screens plus twelve confirmation trajectories produce 48 correlated raw/H200 endpoints. Six fixed-reference cells prospectively alias selected standard; they are not independent repeats.
- Both selected LRs hit 0.005 upper boundary. The finite grid remains closed; no claim of optimal LR or universal width transfer.
- Learned per-head RMS gains are shared across heads and positions and applied before RoPE with unchanged 1/sqrt(32) attention scale. This is not an exact L2-plus-learned-scalar paper replication.
- The local mean-gain criterion failed before fresh confirmation. Larger-width results cannot revise the frozen retention decision or promote another recipe.
- Equal 3,072,000 targets and 3,000 updates do not imply equal compute. Training time is measured separately; no inference-speed or peak-memory claim.
- Every H200 tensor equals its final raw-parent shadow; final checkpoints cannot reconstruct every historical online snapshot. Earlier independent MPS reproducibility failures remain preserved, and no identified cause is claimed.
- H200 outputs are inference-only; continuing the online optimizer requires the original raw parent.
