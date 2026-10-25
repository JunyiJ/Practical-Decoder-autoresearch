# Context interaction results

All 114 distinct trials completed and reviewed: 48 width 64 screening runs and 66 fresh-seed confirmation/reference runs. The frozen alias manifest is empty. All checkpoints, raw contracts, selection, report comparisons and conditional bootstrap intervals passed review. No training or evaluation was rerun for interpretation; the test split stayed closed.

Historical study source: `15d8ddb738a0466366abbebb20bdbd52304f54c1`. Result: `kb_117c97dfeee6a8db25365dfe3bbb1ed96ae5d352b11315e26ffd004159211368`. Measured controller active time 7683.46 seconds (2.13 hours), output before review 685.74 MiB. These replace the rough pilot proxy as completed-run measurements.

## Frozen recipes and measured primary scores

All means below are full-validation CE at common evaluation context 128, 111538 targets once per model, across seeds 2031/2032/2033. Lower is better. Every run trained 3000 updates with 1024 targets/update using paired_blocks512. Training seconds exclude final full-coverage evaluation and process overhead.

| Family | Frozen LR | CE64 | CE256 | Train seconds64 | Train seconds256 |
|---|---:|---:|---:|---:|---:|
| d2_mha_c128 | 0.005 | 1.658071 | 1.539010 | 33.14 | 66.91 |
| d2_mha_c512 | 0.005 | 1.672666 | 1.550701 | 41.77 | 86.40 |
| d2_gqa_c128 | 0.005 | 1.664115 | 1.537891 | 31.46 | 63.73 |
| d2_gqa_c512 | 0.0025 | 1.682595 | 1.551547 | 41.14 | 82.66 |
| d4_mha_c128 | 0.005 | 1.604367 | 1.507755 | 54.24 | 125.66 |
| d4_mha_c512 | 0.005 | 1.607994 | 1.527665 | 64.92 | 163.92 |
| d4_gqa_c128 | 0.0025 | 1.602048 | 1.526959 | 53.23 | 118.74 |
| d4_gqa_c512 | 0.005 | 1.613465 | 1.527418 | 63.45 | 156.43 |
| gelu_reference | 0.002 | 1.681139 | 1.546129 | 31.91 | 65.85 |
| shallow_reference | 0.0025 | 1.665943 | 1.538182 | 33.02 | 66.83 |
| deep_reference | 0.0025 | 1.598513 | 1.523177 | 54.29 | 125.54 |

Six of eight factorial choices selected upper-boundary .005; d2_gqa_c512 and d4_gqa_c128 selected .0025. Choices were frozen after 48 screens, before confirmation or width 256. No grid extension, larger-width reselection or automatic promotion.

## GQA quality eligibility

Candidate minus matched-depth/context MHA. The frozen rule requires mean, every seed and every 2048/4096/8192 conditional text-bootstrap upper bound to be at most +.01. The interval shown is 4096 blocks; every sensitivity interval was verified.

| Width | GQA cell | Mean delta | Seeds2031 /2032 /2033 | Conditional95% text interval | Eligible |
|---:|---|---:|---|---|---|
| 64 | d2_gqa_c128 | +0.006044 | -0.012535 / +0.011872 / +0.018794 | [+0.002877, +0.009050] | no |
| 64 | d2_gqa_c512 | +0.009929 | -0.005692 / +0.031145 / +0.004333 | [+0.005635, +0.013819] | no |
| 64 | d4_gqa_c128 | -0.002318 | +0.021060 / -0.019054 / -0.008961 | [-0.005584, +0.001107] | no |
| 64 | d4_gqa_c512 | +0.005471 | +0.007695 / +0.014629 / -0.005910 | [+0.002033, +0.008860] | no |
| 256 | d2_gqa_c128 | -0.001119 | +0.008442 / -0.013296 / +0.001498 | [-0.006119, +0.003756] | yes |
| 256 | d2_gqa_c512 | +0.000845 | -0.002240 / +0.004992 / -0.000215 | [-0.002306, +0.004093] | yes |
| 256 | d4_gqa_c128 | +0.019204 | +0.018877 / +0.015181 / +0.023553 | [+0.015682, +0.022461] | no |
| 256 | d4_gqa_c512 | -0.000246 | +0.003099 / -0.007919 / +0.004080 | [-0.003153, +0.002568] | yes |

No width 64 cell qualifies: each has at least one paired seed beyond +.01. Three width 256 cells qualify, but four-layer GQA at128 loses +.019204 CE and every seed exceeds +.01. These are operational quality eligibility findings only. Actual matched prefill/decode/cache measurements for these new depth/context recipes remain required. No inference efficiency or population non-inferiority claim follows from training time or parameter counts.

## Depth, context and learning-rate findings

Four layers improve every seed against matched two-layer recipes in all attention/context cells at both widths. Mean deltas span −.053705 to−.069130 at 64 and−.010932 to−.031255 at 256. This preserves the depth advantage under the fresh sampler at additional training cost; it is not equal-compute superiority.

Training at 512 gives worse mean primary context 128 CE in all eight matched comparisons. At 256, MHA512 minus MHA128 is +.011691 for two layers and+.019909 for four layers, with every seed worse. Four-layer GQA is nearly flat (+.000459, mixed seeds). Equal flattened targets do not equalize available context, sequence count or gradient variance.

Native 512 is secondary: two-layer MHA/GQA means at 64 are 1.648352/1.660021, at 256: 1.522882/1.524264; four-layer MHA/GQA means at 64 are 1.582108/1.588315, at 256: 1.497828/1.497201. At 256 native GQA-minus-MHA deltas are +.001382 (two layers) and−.000627 (four layers). These scores use different evaluation boundaries and longer available context than the primary 128 score, so they must not be treated as improvements on an identical evaluation task.

Selected four-layer MHA128 LR .005 versus its fixed .0025 reference is +.005854 at 64 (mixed seeds), but−.015422 at 256 (all three seeds better). The larger-width result cannot revise the frozen local choice or establish optimal-LR transfer; there was no width 256 grid. The historical .0025 quality incumbent remains in place, with this new adaptive comparison recorded separately.

## Interactions and limits

At fixed screening LR .0025 and width 64, the depth×attention×context mean is −.006490 (two seeds with mixed signs). Fresh confirmation uses separately selected LRs: its three-way mean is +.003905 at 64 (mixed signs) and−.021414 at 256 (all three negative). At 256 the depth×GQA contrast is +.020322 at128 and−.001092 at 512. These are tuned-recipe interactions including LR choice, not pure causal architecture effects or adjusted significance claims.

The sampler and RoPE source require fresh controls; historical measurements are preserved and not pooled. Three fresh development seeds remain adaptive validation. Conditional text intervals resample text blocks given these models and do not quantify population seed uncertainty. No test access, pruning, completed-trial reruns or study extension occurred. Context256 and batch sensitivity remain separate bounded follow-ups; MoE remains deferred.

Evidence: `research_runs/context-interaction-001/{plan,selection,report,review}.json`; screening freeze review is retained. The primary human `kimi` branch and GatedMLA edits were kept separate.
