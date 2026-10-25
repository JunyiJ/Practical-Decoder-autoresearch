# MoE quality results

MoE quality001 completed and reviewed48/48. All three families selected LR.0025 inside the frozen grid. MoE was the width64 screen winner. Fresh confirmation MoE-minus-dense mean CE is -0.031764/+0.002608 at widths64/256. The frozen winner fails the complete adoption rule; the historical dense quality incumbent is retained. Wider dense and MoE use additional stored capacity; comparisons are equal-token recipes, not equal compute or pure routing effects. Test split stays closed.

Historical source: `15f2cc655767908c1bae7d46fad60ef3f50f32ba`. Result: `kb_d0c34e1507f736aee3cbcc38a410480edb35bd788bb0f104a3be81dd8f86c0c7`. All48 checkpoint/raw request/result hashes and saved model, sampler, optimizer and runtime contracts were audited. Final private training RNG was reconstructed; saved test RNG is untouched. Every report aggregate and conditional bootstrap interval was independently reconstructed. No training or quality evaluation was rerun.

18 full-budget screening trials plus30 distinct confirmation/reference trials. Six fixed deep references alias selected dense checkpoints prospectively. Active controller time: 4701.44seconds (1.31hours). Output before review: 462.07MiB. [Frozen protocol](MOE_QUALITY_PROTOCOL.md).

## Recipes and validation scores

All three families independently selected LR.0025, an interior grid point, using width64 seeds2081/2082. The screen winner and complete manifest were frozen before fresh2091/2092/2093 confirmation at widths64/256. All runs process3,072,000 targets over3000 updates; final full-validation128 evaluates111,538 targets once per model. No historical pooling or holdout access.

| Recipe | LR | CE64 | CE256 | Training seconds64 | Training seconds256 |
|---|---:|---:|---:|---:|---:|
| dense | 0.0025 | 1.600833 | 1.515562 | 54.16 | 126.28 |
| dense_wide | 0.0025 | 1.572029 | 1.510540 | 59.07 | 176.43 |
| moe4_top2 | 0.0025 | 1.569069 | 1.518170 | 136.76 | 225.66 |
| gelu_reference | 0.002 | 1.675995 | 1.542497 | 31.61 | 65.90 |
| shallow_reference | 0.0025 | 1.652372 | 1.531293 | 32.50 | 66.93 |

## Paired confirmation contrasts

Positive CE delta is worse. Seed values are2091/2092/2093. Intervals use4096-token text blocks; all2048/4096/8192 intervals are preserved and verified in the report. They are conditional on these trained models and this reused validation text, not population seed uncertainty.

| Width | Candidate minus reference | Mean delta | Paired seed deltas | Conditional95% text interval |
|---:|---|---:|---|---|
| 64 | dense_wide − dense | -0.028804 | -0.030953 / -0.038773 / -0.016686 | [-0.031749, -0.025726] |
| 64 | moe4_top2 − dense | -0.031764 | -0.026852 / -0.040457 / -0.027984 | [-0.035465, -0.027982] |
| 64 | moe4_top2 − dense_wide | -0.002960 | +0.004101 / -0.001684 / -0.011298 | [-0.006076, +0.000434] |
| 64 | dense − gelu_reference | -0.075162 | -0.085012 / -0.073726 / -0.066747 | [-0.081155, -0.068713] |
| 64 | dense − shallow_reference | -0.051539 | -0.054474 / -0.052593 / -0.047551 | [-0.056381, -0.045986] |
| 64 | dense − deep_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] |
| 64 | dense_wide − gelu_reference | -0.103966 | -0.115965 / -0.112498 / -0.083434 | [-0.110379, -0.097063] |
| 64 | dense_wide − shallow_reference | -0.080343 | -0.085426 / -0.091365 / -0.064238 | [-0.085760, -0.074083] |
| 64 | dense_wide − deep_reference | -0.028804 | -0.030953 / -0.038773 / -0.016686 | [-0.031749, -0.025726] |
| 64 | moe4_top2 − gelu_reference | -0.106926 | -0.111864 / -0.114182 / -0.094731 | [-0.113956, -0.099383] |
| 64 | moe4_top2 − shallow_reference | -0.083304 | -0.081326 / -0.093050 / -0.075535 | [-0.089875, -0.075633] |
| 64 | moe4_top2 − deep_reference | -0.031764 | -0.026852 / -0.040457 / -0.027984 | [-0.035465, -0.027982] |
| 256 | dense_wide − dense | -0.005022 | -0.001789 / -0.002196 / -0.011081 | [-0.008950, -0.001039] |
| 256 | moe4_top2 − dense | +0.002608 | +0.006216 / +0.003292 / -0.001685 | [-0.000219, +0.005411] |
| 256 | moe4_top2 − dense_wide | +0.007630 | +0.008006 / +0.005488 / +0.009396 | [+0.003459, +0.012052] |
| 256 | dense − gelu_reference | -0.026935 | -0.030467 / -0.035193 / -0.015146 | [-0.031897, -0.021453] |
| 256 | dense − shallow_reference | -0.015731 | -0.018642 / -0.020898 / -0.007652 | [-0.019275, -0.011485] |
| 256 | dense − deep_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] |
| 256 | dense_wide − gelu_reference | -0.031957 | -0.032256 / -0.037389 / -0.026227 | [-0.035570, -0.028112] |
| 256 | dense_wide − shallow_reference | -0.020753 | -0.020431 / -0.023094 / -0.018733 | [-0.023378, -0.018165] |
| 256 | dense_wide − deep_reference | -0.005022 | -0.001789 / -0.002196 / -0.011081 | [-0.008950, -0.001039] |
| 256 | moe4_top2 − gelu_reference | -0.024327 | -0.024250 / -0.031901 / -0.016831 | [-0.030094, -0.017890] |
| 256 | moe4_top2 − shallow_reference | -0.013123 | -0.012426 / -0.017607 / -0.009337 | [-0.017434, -0.008143] |
| 256 | moe4_top2 − deep_reference | +0.002608 | +0.006216 / +0.003292 / -0.001685 | [-0.000219, +0.005411] |

At width64, MoE versus wider dense has mixed seed signs and a4096-block interval including zero. At width256, MoE is worse than wider dense on every confirmation seed and worse than ordinary dense on two of three seeds. The local MoE gain therefore does not transfer under the frozen rule. Wider dense improves every seed versus ordinary dense at both widths, but it was not the frozen screen winner and cannot be promoted retrospectively from this round. These observations motivate later prospective work, not a revised selection. Extra stored capacity and routed computation change together; the study does not isolate routing as the cause.

## Fixed-LR screening and costs

Screening contrasts below use LR.0025 on both sides and seeds2081/2082; they are distinct from the confirmation evidence.

| Candidate minus reference | Mean delta | Seed2081 | Seed2082 |
|---|---:|---:|---:|
| dense_wide − dense | -0.022393 | -0.009869 | -0.034916 |
| moe4_top2 − dense | -0.029350 | -0.020674 | -0.038025 |
| moe4_top2 − dense_wide | -0.006957 | -0.010805 | -0.003109 |

| Recipe | Time change64 vs dense | Time change256 vs dense |
|---|---:|---:|
| dense_wide | +9.06% | +39.71% |
| moe4_top2 | +152.51% | +78.70% |

Final logged routing diagnostics (three confirmation seeds; fractions of top2 assignments):

| Width | Minimum per-expert assignment fraction | Maximum | Mean auxiliary loss |
|---:|---:|---:|---:|
| 64 | 0.1714 | 0.3120 | 1.0132 |
| 256 | 0.1738 | 0.3003 | 1.0151 |

Each row summarizes the last sampled training batch across four layers and three seeds. These are not cumulative token-exposure statistics or evidence that balanced routing caused the quality change. Auxiliary loss is excluded from validation CE.

Training seconds exclude process overhead and final full validation. Ordinary dense stores/activates204,544 and3,152,896 parameters at widths64/256; wider dense333,568 and5,217,280. MoE stores334,592 and5,221,376 and activates205,568 and3,156,992 per token including routers. These counts do not equalize measured compute or establish inference efficiency.

## Decision and continuing research

Only the width64 screen winner can qualify; confirmation mean gain>.005 and every seed improves at widths64 and256 versus tuned dense and fixed deep reference.

MoE does not pass the complete rule. The prior dense incumbent remains, while all narrower observations are preserved.

The next finite round tests constant, cosine-decay and late-cooldown learning rates on the reviewed incumbent with fresh constant and historical references. Implementation, resume correctness, source/resource checks and a separate preregistration are required before launch. This is a future protocol, not measured evidence. The continuing human quality objective remains active; all completed studies and campaign006 remain closed.

## Limitations

- Fresh development seeds on adaptively reused Shakespeare validation; no holdout or population/seed uncertainty claim. Conditional text bootstrap is descriptive.
- MoE has roughly the active FFN weights of ordinary dense and roughly the stored FFN weights of wider dense, plus router weights; no equal compute, pure routing attribution or inference benefit claim.
- Top2-of4 half-width experts, fixed auxiliary coefficient.01 and explicit engineering router scaling; no optimal MoE muP or universal architecture claim.
- Experts without gradients skip update and independent decay; active expert exposure and per-parameter cumulative shrinkage are not matched.
- Sampled per-layer routing snapshots do not measure whole-run balance; current MPS allocation is not peak unified memory.
- Deep reference aliases selected dense in six cells prospectively; it is not an independent repeat. No completed-trial rerun, grid extension or larger-width reselection.

Evidence: `research_runs/moe-quality-001/{plan,selection,state,report,review}.json`, all immutable per-trial artifacts, `research_state/moe-guard-validation/completion-audit.py`, and full knowledge snapshots. The human kimi checkout remains untouched.
