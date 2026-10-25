# Training horizon and cooldown quality results

The horizon study completed 24 training trajectories and 48 correlated raw/H200 outputs, all independently reviewed. Frozen candidate: long6000_stretched. H200 mean validation CE changes at widths 64/256 versus fresh 3,000-update controls: -0.036435/+0.079981. It does not pass the complete frozen adoption rule; retain the prior 3,000-update H200 incumbent.

Historical source `280ea428792d4dd31780123d14846d5302d43227`. Result `kb_0d878166b1ec83ae3d35aa1090e4ee544a46b1283463a810a51da0e1cc82cc3f`. All 24 raw and 24 derived checkpoint hashes, requests, results, saved configurations, optimizer and RNG states, schedules, averaging counts, model shapes, full validation coverage, frozen selection, and report calculations were verified before documentation. The paired conditional bootstrap was reconstructed independently. No training or scoring was rerun, and the holdout remained closed.

Controller active time: 3425.37 seconds (0.95 hours); output before review 489.36 MiB. [Frozen protocol](HORIZON_QUALITY_PROTOCOL.md).

## Fresh confirmation scores

Means use fresh seeds 2171, 2172 and 2173. Each endpoint scores all 111,538 validation targets once, at context length 128 and batch size 8. H200 is the primary metric; raw endpoints are correlated diagnostics.

| Recipe | Updates | Cooldown start | Targets | H200 CE64 | H200 CE256 | Raw CE64 | Raw CE256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| control3000 | 3000 | 2000 | 3072000 | 1.514926 | 1.429409 | 1.519862 | 1.443889 |
| long6000_stretched | 6000 | 4000 | 6144000 | 1.478490 | 1.509390 | 1.485608 | 1.538587 |
| long6000_fixedtail | 6000 | 5000 | 6144000 | 1.481838 | 1.477672 | 1.490269 | 1.510866 |

## Paired confirmation contrasts

Positive means worse. Intervals condition on the three trained seed pairs and resample contiguous text blocks; they are not population confidence across training seeds. All three block sizes are audited.

| Endpoint | Width | Candidate minus reference | Mean CE delta | Paired seeds 2171 / 2172 / 2173 | 4,096-token conditional 95% interval | Primary quality rule |
|---|---:|---|---:|---|---|---|
| H200 | 64 | long6000_stretched minus control3000 | -0.036435 | -0.035690 / -0.034044 / -0.039571 | [-0.040965, -0.031322] | True |
| Raw | 64 | long6000_stretched minus control3000 | -0.034254 | -0.031971 / -0.033309 / -0.037484 | [-0.038456, -0.029684] | Diagnostic only |
| H200 | 64 | long6000_fixedtail minus control3000 | -0.033088 | -0.023647 / -0.035091 / -0.040526 | [-0.038524, -0.026772] | True |
| Raw | 64 | long6000_fixedtail minus control3000 | -0.029593 | -0.019099 / -0.032646 / -0.037035 | [-0.035368, -0.023079] | Diagnostic only |
| H200 | 64 | long6000_fixedtail minus long6000_stretched | +0.003347 | +0.012043 / -0.001047 / -0.000955 | [+0.001488, +0.005365] | False |
| Raw | 64 | long6000_fixedtail minus long6000_stretched | +0.004661 | +0.012872 / +0.000663 / +0.000448 | [+0.002338, +0.007220] | Diagnostic only |
| H200 | 256 | long6000_stretched minus control3000 | +0.079981 | +0.084790 / +0.077706 / +0.077448 | [+0.070121, +0.091446] | False |
| Raw | 256 | long6000_stretched minus control3000 | +0.094698 | +0.105821 / +0.088256 / +0.090018 | [+0.084433, +0.106381] | Diagnostic only |
| H200 | 256 | long6000_fixedtail minus control3000 | +0.048263 | +0.055863 / +0.050764 / +0.038162 | [+0.040234, +0.057376] | False |
| Raw | 256 | long6000_fixedtail minus control3000 | +0.066978 | +0.076791 / +0.069622 / +0.054520 | [+0.058591, +0.076808] | Diagnostic only |
| H200 | 256 | long6000_fixedtail minus long6000_stretched | -0.031718 | -0.028927 / -0.026942 / -0.039286 | [-0.035546, -0.028204] | True |
| Raw | 256 | long6000_fixedtail minus long6000_stretched | -0.027721 | -0.029030 / -0.018634 / -0.035498 | [-0.031670, -0.023961] | Diagnostic only |

## Measured costs

Costs include the shared training trajectory; wall time also includes raw/H200 scoring and export. Longer recipes have twice the training tokens and updates.

| Recipe | Width | Mean training seconds | Mean trajectory wall seconds | Training ratio versus control | Mean sampled RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| control3000 | 64 | 54.72 | 58.77 | 1.000 | 473.49 |
| control3000 | 256 | 125.70 | 133.87 | 1.000 | 482.88 |
| long6000_stretched | 64 | 109.53 | 115.07 | 2.002 | 473.90 |
| long6000_stretched | 256 | 251.35 | 262.68 | 2.000 | 482.67 |
| long6000_fixedtail | 64 | 109.42 | 114.96 | 2.000 | 473.98 |
| long6000_fixedtail | 256 | 251.11 | 262.72 | 1.998 | 482.36 |

## Frozen selection and decision

Six width-64 screening trajectories on seeds 2161 and 2162 froze **long6000_stretched** before any fresh confirmation. All three recipes then completed all six width/seed cells. The frozen winner fails full adoption; the original 3,000-update H200 incumbent remains. No alternative is promoted from confirmation.

The backbone has four MHA layers with head dimension 32, bias-free SwiGLU with hidden width 21d/8, RMSNorm blocks, final LayerNorm, RoPE and unscaled residuals. Training uses Muon/AdamW with muP base width 64, peak learning rate 0.005, cooldown to 10% of peak, independent decay 0.00006 per update, clipping at 1, zero dropout, context length 128 and batch size 8 with paired_blocks512 sampling. H200 initializes at cooldown start and updates every 10 steps. Control/stretched/fixed-tail recipes have 101/201/101 snapshots.

## Continuation

A separate finite QK RMS normalization and equal bounded LR-tuning round is prepared. It resolves its horizon from this final adoption, and must pass the full CPU test suite, serial MPS correctness and resource checks, clean committed source and new preregistration before launch. Completed studies remain closed; human-stopped campaign 006 remains stopped.

## Limits

- Adaptive validation on fresh development seeds; conditional text bootstrap is not seed-population uncertainty, holdout, or adjusted significance.
- Twenty-four training trajectories produce 48 correlated raw/H200 outputs; raw and averaged endpoints of one trajectory are not independent training runs.
- The 6,000-update recipes consume 6,144,000 targets versus 3,072,000 for 3,000 updates; extra tokens, updates, compute and cumulative independent decay are explicit. This is not an equal-compute comparison.
- Control/stretched/fixed-tail base learning-rate sums are 12.75/25.5/27.75 and nominal decay retention differs with update count. Cooldown shape, LR integral, averaging duration and training budget are not isolated causal effects.
- H200 snapshot counts are 101/201/101. Every derived tensor matches its final raw-parent shadow; final checkpoints do not independently reconstruct every historical online snapshot.
- Inherited direct MPS recurrence checks remain valid for unchanged averaging source; earlier independent MPS trajectory agreement failures are preserved. No exact independent MPS reproducibility or identified kernel cause claim.
- Only the frozen screening winner can replace the incumbent after fresh confirmation at both widths. Other recipes remain reported diagnostics; no posthoc promotion or grid extension.
- Derived averaged weights are inference-only; online optimizer continuation requires the original raw parent. No inference-speed or peak-memory claim.
- A fixed peak learning rate of 0.005 at the tested horizons does not establish optimal horizon, LR, width scaling or universal model quality.
