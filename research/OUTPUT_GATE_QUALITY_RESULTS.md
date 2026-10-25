# Attention-output gating quality results

Output-gating completed 36 physical trajectories and 72 correlated raw/H200 endpoints, all independently reviewed. All three arms selected LR 0.005 at the upper boundary. Static-half had the best screening mean, improving CE by 0.006703611, but seed 2202 worsened by +0.000584166; incumbent retention was frozen before confirmation. Fresh H200 mean differences at widths 64/256: static-half minus unchanged +0.004553/+0.006444, learned minus unchanged +0.002581/-0.000689, learned minus static-half -0.001972/-0.007133. Retain unchanged MHA, 3,000 updates, cooldown starting at 2,000, and H200; no grid extension or posthoc promotion.

Historical source `047008b4b59d8141f5463f793c39798fee006a17`. Result `kb_8e576a1e5481fc9c450caea48c026f1970fa74af7d69cea5a2afdb2597761c6d`. All 36 raw and 36 H200 checkpoint hashes, requests/results, saved configurations, gate and optimizer states, RNG, schedules, model shapes, full validation coverage, frozen selection and aliases and the entire report were independently verified before documentation. Every conditional bootstrap interval was independently reconstructed. No training/evaluation rerun or holdout access.

Controller active time 2950.19 seconds (0.82 hours); output before review 530.97 MiB. [Frozen protocol](OUTPUT_GATE_QUALITY_PROTOCOL.md).

## Fresh confirmation means

Seeds 2211, 2212 and 2213; 111,538 validation targets once each at context 128, batch size 8. H200 is primary; raw weights are a correlated diagnostic.

| Recipe | Selected LR | H200 CE64 | H200 CE256 | Raw CE64 | Raw CE256 |
|---|---:|---:|---:|---:|---:|
| none | 0.005 | 1.514142 | 1.432261 | 1.519004 | 1.448587 |
| static_half | 0.005 | 1.518694 | 1.438705 | 1.522764 | 1.452062 |
| sigmoid_headwise_zero | 0.005 | 1.516723 | 1.431572 | 1.521680 | 1.445858 |

## Paired confirmation contrasts

Positive is worse. Intervals condition on the three seed pairs; all 2,048/4,096/8,192 block sizes are audited. The fixed incumbent aliases selected unchanged attention. Primary support below describes each contrast; frozen local eligibility separately controls adoption.

| Endpoint | Width | Candidate minus reference | Mean delta | Seeds 2211 / 2212 / 2213 | 4,096-token conditional 95% interval | Primary support |
|---|---:|---|---:|---|---|---|
| H200 | 64 | static_half minus none | +0.004553 | -0.009174 / +0.018366 / +0.004466 | [+0.001693, +0.007500] | False |
| Raw | 64 | static_half minus none | +0.003760 | -0.010598 / +0.016663 / +0.005215 | [+0.000512, +0.007182] | Diagnostic |
| H200 | 64 | sigmoid_headwise_zero minus none | +0.002581 | -0.001679 / +0.006634 / +0.002787 | [+0.000025, +0.005001] | False |
| Raw | 64 | sigmoid_headwise_zero minus none | +0.002675 | -0.001879 / +0.005212 / +0.004693 | [-0.000226, +0.005403] | Diagnostic |
| H200 | 64 | sigmoid_headwise_zero minus static_half | -0.001972 | +0.007495 / -0.011732 / -0.001679 | [-0.004420, +0.000493] | False |
| Raw | 64 | sigmoid_headwise_zero minus static_half | -0.001084 | +0.008719 / -0.011451 / -0.000522 | [-0.003951, +0.001747] | Diagnostic |
| H200 | 64 | none minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 64 | none minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 64 | static_half minus incumbent_reference | +0.004553 | -0.009174 / +0.018366 / +0.004466 | [+0.001693, +0.007500] | False |
| Raw | 64 | static_half minus incumbent_reference | +0.003760 | -0.010598 / +0.016663 / +0.005215 | [+0.000512, +0.007182] | Diagnostic |
| H200 | 64 | sigmoid_headwise_zero minus incumbent_reference | +0.002581 | -0.001679 / +0.006634 / +0.002787 | [+0.000025, +0.005001] | False |
| Raw | 64 | sigmoid_headwise_zero minus incumbent_reference | +0.002675 | -0.001879 / +0.005212 / +0.004693 | [-0.000226, +0.005403] | Diagnostic |
| H200 | 256 | static_half minus none | +0.006444 | +0.006714 / +0.013166 / -0.000546 | [+0.004753, +0.008273] | False |
| Raw | 256 | static_half minus none | +0.003475 | +0.007395 / +0.006680 / -0.003649 | [+0.001514, +0.005559] | Diagnostic |
| H200 | 256 | sigmoid_headwise_zero minus none | -0.000689 | -0.001631 / +0.000862 / -0.001297 | [-0.003006, +0.001616] | False |
| Raw | 256 | sigmoid_headwise_zero minus none | -0.002729 | -0.002920 / -0.003295 / -0.001972 | [-0.005235, -0.000088] | Diagnostic |
| H200 | 256 | sigmoid_headwise_zero minus static_half | -0.007133 | -0.008345 / -0.012303 / -0.000750 | [-0.009751, -0.004661] | True |
| Raw | 256 | sigmoid_headwise_zero minus static_half | -0.006204 | -0.010315 / -0.009975 / +0.001677 | [-0.008933, -0.003460] | Diagnostic |
| H200 | 256 | none minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 256 | none minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 256 | static_half minus incumbent_reference | +0.006444 | +0.006714 / +0.013166 / -0.000546 | [+0.004753, +0.008273] | False |
| Raw | 256 | static_half minus incumbent_reference | +0.003475 | +0.007395 / +0.006680 / -0.003649 | [+0.001514, +0.005559] | Diagnostic |
| H200 | 256 | sigmoid_headwise_zero minus incumbent_reference | -0.000689 | -0.001631 / +0.000862 / -0.001297 | [-0.003006, +0.001616] | False |
| Raw | 256 | sigmoid_headwise_zero minus incumbent_reference | -0.002729 | -0.002920 / -0.003295 / -0.001972 | [-0.005235, -0.000088] | Diagnostic |

## Fixed-LR screening

Two screening seeds 2201/2202; keep equal-LR contrasts separate from selected recipes.

| LR | Candidate minus reference | H200 mean delta | Paired deltas | Raw mean delta |
|---|---|---:|---|---:|
| 0.00125 | static_half minus none | +0.007032 | +0.004817 / +0.009246 | +0.005417 |
| 0.00125 | sigmoid_headwise_zero minus none | +0.003227 | +0.003226 / +0.003228 | +0.001913 |
| 0.00125 | sigmoid_headwise_zero minus static_half | -0.003805 | -0.001591 / -0.006018 | -0.003505 |
| 0.0025 | static_half minus none | -0.000048 | -0.000035 / -0.000062 | -0.000960 |
| 0.0025 | sigmoid_headwise_zero minus none | -0.000011 | +0.001615 / -0.001637 | -0.000559 |
| 0.0025 | sigmoid_headwise_zero minus static_half | +0.000037 | +0.001650 / -0.001575 | +0.000401 |
| 0.005 | static_half minus none | -0.006704 | -0.013991 / +0.000584 | -0.007475 |
| 0.005 | sigmoid_headwise_zero minus none | -0.001687 | -0.005691 / +0.002318 | -0.002769 |
| 0.005 | sigmoid_headwise_zero minus static_half | +0.005017 | +0.008300 / +0.001734 | +0.004705 |

## Measured costs

| Recipe | Width | Training seconds | Trajectory wall seconds | Training ratio vs unchanged | Sampled RSS MiB |
|---|---:|---:|---:|---:|---:|
| none | 64 | 55.79 | 60.51 | 1.000 | 474.15 |
| none | 256 | 135.91 | 144.31 | 1.000 | 483.16 |
| static_half | 64 | 55.78 | 60.25 | 1.000 | 473.90 |
| static_half | 256 | 134.60 | 142.97 | 0.990 | 483.33 |
| sigmoid_headwise_zero | 64 | 59.65 | 64.21 | 1.069 | 479.06 |
| sigmoid_headwise_zero | 256 | 136.93 | 145.21 | 1.008 | 487.39 |

## Next bounded hypothesis

A separate SwiGLU-product normalization comparison is prepared: unchanged product, RMSNorm and LayerNorm before the down projection. Resolve its common attention and reference LR only from this final reviewed decision. Actual model, trainer, full-suite, MPS and resource gates and clean committed source plus a new finite preregistration are required before launch. All closed studies remain closed.

## Limits

- Fresh development seeds and full validation remain adaptive evidence; conditional text intervals are not seed-population uncertainty, holdout or adjusted significance.
- 18 screening and 18 confirmation trajectories produce 72 correlated raw/H200 endpoints. Six fixed-reference cells prospectively alias selected unchanged MHA; they are not independent repeats.
- All selected LRs hit the 0.005 upper boundary. The finite grid remains closed; no optimal LR or universal width-transfer claim.
- The learned sigmoid is query-local and per head, after weighted values and before concatenation and output projection. Its zero matrix starts at exactly 0.5, matching static-half values; no attention-sink mechanism is identified.
- Output-projection bias is unchanged. Static-half values do not halve the complete attention residual. Learned versus static is reported separately; a gain versus unchanged alone cannot establish a learned-gating benefit.
- The best screening arm failed every-seed eligibility before confirmation. Larger-width or confirmation observations cannot change that retention decision or promote another arm.
- Equal 3,072,000 targets and 3,000 updates do not imply equal compute. Training costs are measured separately; no serving latency or peak memory claim.
- Every H200 tensor equals its final raw-parent shadow; final checkpoints cannot reconstruct all 101 historical online snapshots. Earlier independent MPS agreement failures remain preserved, with no identified cause.
- H200 outputs are inference-only; optimizer resume requires original raw parents.
