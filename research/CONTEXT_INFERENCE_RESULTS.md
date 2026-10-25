# Context GQA inference results

All 36 paired workload jobs completed and passed evidence review: 18 immutable saved checkpoints, 324 measured paired repetitions, 399.19 seconds of controller active time. All checkpoint/config/training contracts, raw requests/results, cache counts, timing statistics, source/corpus/runtime/full-bank hashes and engineering gates were verified. No training, quality rescoring, checkpoint rewrite or held-out test access occurred.

Historical benchmark source: `6fbd8d1850ca45090e498107fcd80a50a73d56ea`. Result: `kb_b5512bbe8a3a2d4ad2bb9dbdb9a8cc8592153213b3204e5068c8666c6453a360`. Frozen protocol: [context inference](CONTEXT_INFERENCE_PROTOCOL.md).

## Findings

Every tested GQA setting halves actual allocated and logically used KV-cache storage at matched capacity. None passes the prospective operational efficiency gate across every seed and workload. No setting gives the required 10% decode speed benefit throughout the panel. The memory benefit is real, but the guard against latency regressions/variability fails. This does not prove GQA is slower on every workload or unsuitable for a different memory-constrained deployment.

| GQA setting, width 256 | Prefill median change range | Decode median change range | Allocated KV change | Pass all-workload latency guard |
|---|---:|---:|---:|---|
| d2_gqa_c128 | -2.90% to +4.58% | -0.31% to +4.76% | -50% | no |
| d2_gqa_c512 | -4.03% to +2.48% | -3.11% to +9.32% | -50% | no |
| d4_gqa_c512 | -5.54% to +0.53% | -4.81% to +8.42% | -50% | no |

Ranges span the 12 individual seed/workload median paired ratios per setting. They are descriptive ranges, not confidence intervals. Lower time is better; throughput reciprocates time. At batch 8/full capacity, MHA→GQA cache allocation is 4→2 MiB for 2L128, 16→8 MiB for 2L512 and 32→16 MiB for 4L512. Cache storage is only one part of total memory.

The two-layer context128 setting keeps every prefill/decode median within the 5% latency guard, but fails its upper-quartile guard. For seed 2033/B1/P32, decode median is +4.76% and its upper quartile is +39.98%. Longer-context settings also contain median decode regressions beyond 5%; the worst are +9.32% for 2L512 and +8.42% for 4L512. Preserve the variability rather than selecting favorable repetitions or rerunning completed cells.

## Frozen acceptance rule and interpretation

The human accepted +.01 CE nats/token as the quality ceiling. These three settings previously met the common-evaluation-context128 rule requiring mean, every paired seed and all three conditional text-bootstrap upper bounds ≤ +.01. That is operational quality eligibility on adaptive validation; it does not establish population or native512 non-inferiority.

Before pilots/timing, this diagnostic froze engineering gates: every seed and workload must have median paired decode ratio ≤ .90 and upper quartile ≤ 1 for speed benefit, or actual allocated cache ratio ≤ .75 for memory benefit. Both routes also require every prefill/decode median ≤ 1.05 and upper quartile ≤ 1.10. All settings pass the memory criterion but fail the common latency guard. These numerical timing thresholds are prospective engineering choices under the authorized objective, not human-specified tolerances or statistical confidence claims. No thresholds were revised after results.

The historical four-layer SwiGLU/MHA quality incumbent remains unchanged. Any future adoption with a different workload, memory constraint or latency tolerance needs an explicit prospective decision; these measurements must remain intact. Context256 and batch sensitivity are separate bounded follow-ups, not extensions or reruns of this panel.

## Raw-result overview

Each row reports the median candidate/reference time change from nine alternating pairs, separated by seed. Three warmups per model and pilot measurements are excluded. All models have width 256; MHA matches depth/context and the original selected recipe.

| Setting | Seed | Batch | Prompt/decode | Prefill change | Decode change | Decode paired-ratio IQR |
|---|---:|---:|---|---:|---:|---|
| d2_gqa_c128 | 2031 | 1 | 32/32 | +1.02% | +0.32% | [0.9905, 1.0166] |
| d2_gqa_c128 | 2031 | 1 | 64/64 | +4.58% | +1.30% | [1.0073, 1.0524] |
| d2_gqa_c128 | 2031 | 8 | 32/32 | -0.74% | +2.06% | [0.9918, 1.0351] |
| d2_gqa_c128 | 2031 | 8 | 64/64 | -1.05% | +1.06% | [0.9021, 1.0159] |
| d2_gqa_c128 | 2032 | 1 | 32/32 | +1.26% | +0.86% | [1.0034, 1.0132] |
| d2_gqa_c128 | 2032 | 1 | 64/64 | -1.50% | +1.37% | [1.0090, 1.1036] |
| d2_gqa_c128 | 2032 | 8 | 32/32 | +2.56% | +1.39% | [1.0048, 1.0255] |
| d2_gqa_c128 | 2032 | 8 | 64/64 | -1.24% | +0.81% | [0.8974, 1.0169] |
| d2_gqa_c128 | 2033 | 1 | 32/32 | +2.20% | +4.76% | [1.0156, 1.3998] |
| d2_gqa_c128 | 2033 | 1 | 64/64 | +0.02% | -0.31% | [0.9845, 1.0012] |
| d2_gqa_c128 | 2033 | 8 | 32/32 | +4.47% | +2.97% | [0.7196, 1.0321] |
| d2_gqa_c128 | 2033 | 8 | 64/64 | -2.90% | +0.93% | [0.9002, 1.0168] |
| d2_gqa_c512 | 2031 | 1 | 128/128 | +2.48% | +2.31% | [1.0092, 1.1341] |
| d2_gqa_c512 | 2031 | 1 | 256/256 | +1.14% | -3.11% | [0.9604, 1.0657] |
| d2_gqa_c512 | 2031 | 8 | 128/128 | -3.74% | +2.06% | [1.0042, 1.1253] |
| d2_gqa_c512 | 2031 | 8 | 256/256 | -2.85% | -0.93% | [0.9842, 1.0757] |
| d2_gqa_c512 | 2032 | 1 | 128/128 | +1.07% | +1.48% | [1.0043, 1.0986] |
| d2_gqa_c512 | 2032 | 1 | 256/256 | -1.88% | -2.88% | [0.9546, 1.0616] |
| d2_gqa_c512 | 2032 | 8 | 128/128 | -3.44% | +9.32% | [1.0209, 1.1409] |
| d2_gqa_c512 | 2032 | 8 | 256/256 | -2.67% | -1.73% | [0.9755, 1.0550] |
| d2_gqa_c512 | 2033 | 1 | 128/128 | -2.54% | -0.90% | [0.9294, 0.9999] |
| d2_gqa_c512 | 2033 | 1 | 256/256 | -4.03% | -2.42% | [0.9624, 1.0549] |
| d2_gqa_c512 | 2033 | 8 | 128/128 | -3.17% | +1.47% | [0.9326, 1.0223] |
| d2_gqa_c512 | 2033 | 8 | 256/256 | -2.70% | -0.06% | [0.9942, 1.0943] |
| d4_gqa_c512 | 2031 | 1 | 128/128 | -2.32% | +7.90% | [1.0098, 1.0860] |
| d4_gqa_c512 | 2031 | 1 | 256/256 | -4.58% | +5.76% | [0.9782, 1.0655] |
| d4_gqa_c512 | 2031 | 8 | 128/128 | -3.68% | +2.28% | [0.9682, 1.0370] |
| d4_gqa_c512 | 2031 | 8 | 256/256 | -2.24% | +4.43% | [0.9965, 1.0827] |
| d4_gqa_c512 | 2032 | 1 | 128/128 | +0.53% | -4.81% | [0.9303, 1.0113] |
| d4_gqa_c512 | 2032 | 1 | 256/256 | -1.13% | +1.35% | [0.9688, 1.0385] |
| d4_gqa_c512 | 2032 | 8 | 128/128 | -3.06% | +8.42% | [1.0210, 1.1036] |
| d4_gqa_c512 | 2032 | 8 | 256/256 | -4.96% | -0.64% | [0.9900, 1.0613] |
| d4_gqa_c512 | 2033 | 1 | 128/128 | -1.38% | +2.39% | [0.9682, 1.1753] |
| d4_gqa_c512 | 2033 | 1 | 256/256 | -1.44% | -1.87% | [0.9731, 1.0497] |
| d4_gqa_c512 | 2033 | 8 | 128/128 | -5.54% | +1.61% | [0.9601, 1.0217] |
| d4_gqa_c512 | 2033 | 8 | 256/256 | -4.03% | +0.69% | [0.9635, 1.0251] |

## Limits and evidence

FP32 MPS, four CPU threads, teacher-forced cached sequential decoding, identical validation-prefix inputs within each pair, both models resident, one Mac. Timing includes the normal Python/model path but excludes load, cache allocation/reset, copying, sampling and serving overhead. Current MPS allocation is not true peak or isolated per-model memory; peak is unavailable. Three checkpoint seeds and repeated timings do not independently replicate hardware conditions. Longer contexts use different workloads, so cross-context time differences are not architecture speedup estimates.

Full/cached-logit correctness passed in every worker at the frozen atol1e-4/rtol2e-5. Completion review checked the saved worker correctness records without rerunning inference, and independently rebuilt statistics and gates from raw timings. No benchmark retry occurred.

Evidence: `research_runs/context-inference-001/{plan,report,review,state}.json`, per-job `request.json`/`result.json`, and the complete frozen shared-knowledge snapshot. All older studies and human-stopped campaign006 remain closed. The human kimi checkout was untouched.
