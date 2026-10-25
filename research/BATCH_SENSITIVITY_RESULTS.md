# Batch sensitivity results

All 48 distinct trials passed evidence review: 18 screening and 30 fresh confirmation/reference runs. The fixed deep reference aliases six selected B8 checkpoints prospectively, so there are no duplicate training runs. All checkpoint/raw request/result hashes, exact saved configurations, optimizer slots/groups/steps/width factors, sampler contracts and final private training RNG states were checked. Strict CPU loads and finite tensor checks passed. Every final training RNG matches 3000 fixed two-start draws at its seed; the saved test RNG is untouched. The report and all paired conditional bootstrap intervals were reconstructed from saved evidence. No training or evaluation was rerun.

Historical source: `e5bea92f91034bf6ddf6820fb927367c42b1cdb9`. Result: `kb_b1ff2c19f9435878b28fb3affcf3b7d0f93a8b674fa30124d5a7b9c3fe29a622`. Controller active time: 2755.10 seconds (0.77 hours); output before review: 348.35 MiB. [Frozen protocol](BATCH_SENSITIVITY_PROTOCOL.md).

## Frozen recipes and scores

B8 selected LR .0025 inside the grid; B16 and B32 selected .005 at the upper boundary. Choices and the alias manifest were frozen from seeds 2061/2062 before confirmation seeds 2071/2072/2073 or width 256. The finite grid is closed.

All tuned recipes use four-layer MHA head32, bias-free SwiGLU hidden21d/8, RoPE, Muon/AdamW muP base64, unscaled residuals and clipping1 at context 128. B8/16/32 use 3000/1500/750 updates, each 3,072,000 training targets, with the identical ordered paired_batch512 sample stream for each seed. Decay is .00006 at B8 and 1−(1−.00006)^(B/8) otherwise. This matches nominal cumulative multiplicative shrinkage only. Momentum/betas and clipping retain their per-update settings, so their token time scales change. This compares complete equal-token recipes, including LR choice; it does not isolate batch noise or equalize updates/compute.

CE is full-validation128, evaluation batch 8,111,538 targets once per checkpoint; means use the three confirmation seeds. Sampled training-log panels differ in size and are not the selection endpoint. Training seconds exclude final full-validation and process overhead.

| Recipe | LR | CE64 | CE256 | Training seconds64 | Training seconds256 |
|---|---:|---:|---:|---:|---:|
| d4_mha_b8 | 0.0025 | 1.588912 | 1.512314 | 54.13 | 125.91 |
| d4_mha_b16 | 0.005 | 1.596620 | 1.509756 | 34.63 | 96.03 |
| d4_mha_b32 | 0.005 | 1.594512 | 1.510024 | 20.96 | 85.80 |
| gelu_reference | 0.002 | 1.670114 | 1.537752 | 31.57 | 66.07 |
| shallow_reference | 0.0025 | 1.652344 | 1.531839 | 32.85 | 67.15 |

The fixed four-layer B8 reference is exactly the selected B8 row through the frozen alias; this is not an independent repeat. Every selected four-layer recipe beats the original GELU and cheaper two-layer SwiGLU references on every confirmation seed at both widths.

## Paired quality contrasts

Positive CE delta is worse. Intervals below use 4096-token blocks; all 2048/4096/8192 intervals were verified. The frozen near-neutral rule requires mean, every seed and all three text-interval upper bounds≤ +.01. This is a descriptive operational rule, not population non-inferiority or automatic adoption.

| Width | Candidate minus reference | Mean delta | Seeds2071 /2072 /2073 | Conditional 95% text interval | Pass |
|---:|---|---:|---|---|---|
| 64 | d4_mha_b16 − d4_mha_b8 | +0.007709 | -0.004299 / +0.013624 / +0.013801 | [+0.004507, +0.010941] | no |
| 64 | d4_mha_b32 − d4_mha_b8 | +0.005601 | -0.012174 / +0.013047 / +0.015929 | [+0.002489, +0.008766] | no |
| 64 | d4_mha_b32 − d4_mha_b16 | -0.002108 | -0.007874 / -0.000577 / +0.002127 | [-0.005038, +0.000795] | yes |
| 256 | d4_mha_b16 − d4_mha_b8 | -0.002558 | +0.000352 / -0.001730 / -0.006296 | [-0.005141, -0.000343] | yes |
| 256 | d4_mha_b32 − d4_mha_b8 | -0.002290 | -0.008240 / -0.000651 / +0.002021 | [-0.005945, +0.001245] | yes |
| 256 | d4_mha_b32 − d4_mha_b16 | +0.000268 | -0.008592 / +0.001079 / +0.008317 | [-0.003667, +0.004010] | yes |

At width 64, both larger batches have mean loss below +.01, but seeds 2072 and 2073 exceed the margin. B16 also exceeds the interval-upper-bound criterion at all three block sizes. B32 passes the interval criterion but fails the per-seed rule. At width 256, both larger batches have slightly lower mean CE and pass every component of the frozen tolerance rule, while each still worsens one seed. B16 has negative conditional text intervals at all block sizes; B32 intervals include zero. This does not establish population superiority. B32 versus B16 passes the tolerance rule at both widths, with mixed seed signs.

At fixed screening LR .0025 and width 64, B16-minusB8 is+.004195, B32-minusB8+.032623 and B32-minusB16+.028428; both screening seeds worsen in every contrast. Separate tuning substantially changes the observed recipe comparison, especially for B32. These screening results do not prove a universal LR scaling law or identify optimal LRs at width 256.

## Measured training cost and memory diagnostics

| Recipe | Time change64 vs B8 | Time change256 vs B8 | Targets/sec64 | Targets/sec256 |
|---|---:|---:|---:|---:|
| d4_mha_b16 | -36.03% | -23.73% | 88714 | 31989 |
| d4_mha_b32 | -61.27% | -31.85% | 146531 | 35803 |

Throughput above is fixed target count divided by mean training seconds. It measures this implementation on this Mac, not inference speed or equal-compute quality. All three four-layer recipes have 204,544 parameters at width 64 and 3,152,896 at width 256.

| Recipe | Mean RSS MiB64 /256 | Mean current MPS allocated MiB64 /256 |
|---|---:|---:|
| d4_mha_b8 | 473.43 / 481.96 | 2.77 / 36.63 |
| d4_mha_b16 | 473.77 / 483.25 | 3.04 / 36.90 |
| d4_mha_b32 | 473.65 / 445.61 | 3.60 / 38.42 |
| gelu_reference | 472.23 / 476.48 | 1.61 / 18.93 |
| shallow_reference | 457.33 / 476.03 | 1.59 / 18.71 |

These are recorded memory diagnostics, not peak accelerator or total unified-memory demand. Larger batches trade fewer optimizer steps for larger per-step workloads; this study does not causally separate their timing contributions.

## Decision and limitations

The historical four-layer SwiGLU/MHA LR .0025 quality incumbent remains unchanged. B16/B32 at width 256 are promising training-efficiency options under this token budget and tolerance, but their local seed failures prevent an all-width claim. No automatic recipe adoption, grid extension or larger-width reselection follows. All previous studies and the human-stopped campaign006 remain closed; MoE remains deferred.

Fresh controls are not pooled with earlier studies. Three development seeds use adaptively reused validation; conditional text bootstrap resamples text given these models and does not estimate population seed uncertainty or correct multiple comparisons. The test split stayed closed. Source, corpus, runtime, full knowledge snapshot and prerequisite artifacts were verified before documentation changes. The primary human kimi checkout was untouched.

Evidence: `research_runs/batch-sensitivity-001/{plan,selection,state,report,review}.json`, all per-trial requests/results/checkpoints, `research_state/batch-validation/completion-audit.py`, and the frozen/full completed knowledge snapshots.
