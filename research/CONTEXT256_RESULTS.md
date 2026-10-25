# Context256 follow-up results

All 54 distinct trials completed and passed evidence review: 18 width64 screening runs and 36 fresh confirmation/reference runs, with no aliases. Every checkpoint and raw request/result hash, saved configuration, sampler/optimizer contract, layer/head/QKV/FFN shape, strict CPU load and finite tensor check passed. The complete report and conditional text bootstrap were reconstructed from saved evidence. No training or evaluation was rerun, and the held-out test stayed closed.

Historical study source: `3937a7b6636737710a16802387fab9313156ca56`. Result: `kb_d91505192ed3a2498401f35382c6f55987efd969d97c229140ed6812c1bc7bef`. Measured controller active time: 4306.34 seconds (1.20 hours); output before review: 430.04 MiB. [Frozen protocol](CONTEXT256_PROTOCOL.md).

## Frozen selection and primary scores

Context128 selected LR .005 at the upper boundary; contexts256 and512 selected .0025 inside the grid. All three choices and the full manifest were frozen before fresh-seed confirmation or any width256 run. The grid remains closed.

All models use MHA head32, Muon/AdamW muP base64, independent decay .00006 and the paired_blocks512 sampler. Tuned models have four layers and bias-free SwiGLU hidden21d/8. Each run trains 3000 updates and 3,072,000 target positions. Primary scores are full-validation CE at common evaluation context128, 111,538 targets once per model, averaged over seeds2051/2052/2053. Training seconds exclude final full-coverage evaluation and process overhead.

| Recipe | LR | CE64 | CE256 | Training seconds64 | Training seconds256 |
|---|---:|---:|---:|---:|---:|
| d4_mha_c128 | 0.005 | 1.599709 | 1.510641 | 54.85 | 125.73 |
| d4_mha_c256 | 0.0025 | 1.601452 | 1.529642 | 59.07 | 137.69 |
| d4_mha_c512 | 0.0025 | 1.616875 | 1.539860 | 66.65 | 164.03 |
| gelu_reference | 0.002 | 1.667809 | 1.546304 | 32.15 | 65.88 |
| shallow_reference | 0.0025 | 1.663014 | 1.540596 | 32.93 | 66.96 |
| deep_reference | 0.0025 | 1.599282 | 1.516076 | 54.82 | 125.62 |

## Context contrasts

Positive means worse primary CE. Intervals shown use4096-token blocks; all2048/4096/8192 intervals were independently checked. The frozen near-neutral rule requires mean, every paired seed and all three interval upper bounds<=+.01. It is an operational descriptive flag, not population non-inferiority or automatic adoption.

| Width | Contrast | Mean delta | Seeds2051 /2052 /2053 | Conditional95% text interval | Near-neutral |
|---:|---|---:|---|---|---|
| 64 | d4_mha_c256 − d4_mha_c128 | +0.001743 | +0.000941 / -0.000772 / +0.005059 | [-0.004648, +0.007590] | yes |
| 64 | d4_mha_c512 − d4_mha_c128 | +0.017166 | +0.025341 / +0.010327 / +0.015830 | [+0.012888, +0.021125] | no |
| 64 | d4_mha_c512 − d4_mha_c256 | +0.015423 | +0.024400 / +0.011099 / +0.010771 | [+0.011643, +0.019446] | no |
| 256 | d4_mha_c256 − d4_mha_c128 | +0.019001 | +0.015137 / +0.024501 / +0.017365 | [+0.015450, +0.022420] | no |
| 256 | d4_mha_c512 − d4_mha_c128 | +0.029220 | +0.027982 / +0.036983 / +0.022694 | [+0.025848, +0.032373] | no |
| 256 | d4_mha_c512 − d4_mha_c256 | +0.010219 | +0.012845 / +0.012482 / +0.005329 | [+0.007660, +0.013163] | no |

Context256 is near-neutral versus128 at width64, with mixed seed signs. At width256, its mean loss is+.019001 and every seed is beyond+.01. Context512 loses to128 at both widths with every seed beyond+.01; it also loses to256 at both widths. The width256 512-minus256 mean is just above+.01, with two seeds beyond the margin. These are tuned-recipe comparisons, so the128 contrasts also include the selected LR difference (.005 versus.0025).

At fixed screening LR .0025 and width64, context256-minus128 is+.001490,512-minus128+.014352, and512-minus256+.012862; both screening seeds have positive deltas in every contrast. This separate two-seed panel supports the direction locally without identifying the cause of wider-model differences.

## References, native evaluation and cost

Context128 LR .005 versus fixed four-layer .0025 is+.000427 at64 (seed deltas+.005152/−.005393/+.001522) and−.005435 at256 (−.004762/−.013514/+.001971). The latter mean improves, but seed2053 worsens. The larger-width result cannot revise the frozen local choice or prove optimal-LR transfer. Context256 and512 both lose to fixed four-layer128 at width256 by+.013565 and+.023784; every seed exceeds+.01. The original GELU and cheaper two-layer SwiGLU references remain available in the complete report.

Native-context evaluation is secondary and changes available history and window boundaries. Native256 means are1.587263/1.513364 at widths64/256; native512 means are1.591313/1.509434. These cannot establish gains on the identical common128 task.

Relative to tuned context128, measured mean training time rises by7.69%/9.51% for context256 and 21.51%/30.46% for context512 (width64/256). Equal targets and updates do not equalize compute; these measurements do not establish inference speed or memory benefit.

## Limits and evidence

The historical four-layer SwiGLU/MHA LR .0025 quality incumbent remains unchanged. Fresh controls use the paired sampler and current source; do not pool them with historical studies. Three fresh development seeds still use adaptive validation. Conditional text bootstrap resamples text given these models and does not estimate population seed uncertainty or correct multiple comparisons. No pruning, grid extension, target-width reselection or holdout access occurred. Batch sensitivity remains a separate bounded follow-up; MoE remains deferred.

The audit initially used the prior study’s screening-stage label while adapting the review script and stopped on a missing-key check; correcting the audit-only filter to this study’s local stage and fixed LR completed the review. Training, raw evidence and selection were unchanged.

Evidence: `research_runs/context256-followup-001/{plan,selection,state,report,review}.json`, all per-trial requests/results/checkpoints and the frozen full knowledge snapshot. The primary human kimi checkout was untouched; all older studies and the permanent campaign006 stop remain closed.
