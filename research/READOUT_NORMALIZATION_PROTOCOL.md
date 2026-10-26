# Final readout normalization quality protocol

Status: implemented and CPU-verified on isolated research source; serial MPS and resource gates remain before registration or launch. All 2,272 collected project cases passed, including twelve actual final-normalization geometry/training/resume cases and 204 new metadata/controller/audit cases. These establish CPU correctness, not quality or MPS behavior. The completed embedding-scale review retained input scale 1, peak LR 0.005, independent decay 0.00006 and zero warmup on the four-layer MHA/cooldown/H200 incumbent. All closed studies and permanent stop006 remain closed; the human quality objective continues.

## Question and inherited recipe

Compare the final normalization immediately before the vocabulary readout: gain-and-bias LayerNorm (`layer_affine`), gain-only LayerNorm (`layer_no_bias`), and gain-only RMSNorm (`rms`). All use explicit epsilon 0.00001. Existing residual-block RMSNorms remain unchanged. The affine arm preserves the existing final normalization.

Inherit input embedding scale and LR only through the complete audited `embedding-scale-quality-001` report and its frozen adoption decision. If that round does not qualify, retain its fixed incumbent, rather than its best screen or larger-width result. If it qualifies, inherit exactly the selected arm and LR. The new controller accepts scales 1, 0.5, 0.25 and LRs 0.0025, 0.005, 0.01, with zero warmup; any different predecessor requires a separate protocol review.

Keep four-layer MHA, head dimension 32, bias-free SwiGLU hidden dimension 21d/8, RoPE, unscaled residuals, muP base width 64, Muon/AdamW, independent decay 0.00006, clipping 1, zero dropout, context 128, batch 8, and the paired_blocks512 sampler. Each trial receives 3,000 updates and 3,072,000 training targets. Linear cooldown begins at update index 2,000 and reaches multiplier 0.1 on the final update. Return H200 spaced-tail averages from 101 snapshots at completed steps 2,000 through 3,000 every 10 steps. Raw final weights are diagnostic and remain the only optimizer-resumable checkpoint.

Fresh controls share this exact recipe. Never pool their quality scores with historical studies. No old grid, completed trial, or human-stopped campaign is reopened.

## Initialization, parameterization and interpretation

The new `mha4_readout_norm` policy binds kind and epsilon, along with the inherited explicit input-scale policy. The legacy no-policy constructor remains `nn.LayerNorm(dim)`. For explicit affine, specify the same epsilon, gain and bias. Bias-free LayerNorm and RMSNorm remove exactly d final-bias parameters and retain a d-vector gain. Standard norm parameters initialize deterministically; actual shared-weight/RNG equality must be verified, not assumed from shapes.

The final norms have the following definitions, with statistics over the feature dimension:

- LayerNorm: `(x - mean(x)) / sqrt(mean((x - mean(x))^2) + epsilon) * gain`, with or without learned bias.
- RMSNorm: `x / sqrt(mean(x^2) + epsilon) * gain`.

Keep the existing matrix/vector optimizer classification and width factors. Bias removal changes optimizer groups and potentially the global clipping norm. Normalization changes gradients and feature statistics. This is no empirical guarantee of optimal LR or muP transfer. Equal bounded LR tuning remains necessary.

RMS versus affine LayerNorm changes bias availability and normalization formula. RMS versus bias-free LayerNorm has matching parameter count but changes both centering and denominator statistics. Do not claim that either contrast identifies one causal mechanism. Selected confirmation contrasts include LR choice; fixed-LR screens are reported separately.

Bias-free LayerNorm and RMSNorm have identical parameter shapes but different semantics. Saved/current/constructed normalization kind and epsilon must agree for inference and resume. Reject mismatches, mutable-config disguises, and unreviewed combined attention, FFN, depth, tying, or normalization overrides. Derived H200 checkpoints preserve the exact normalization contract and cannot resume optimization.

Literature motivates the comparison, without predicting this model's result: [Layer Normalization](https://arxiv.org/abs/1607.06450), [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467). Only abstracts were consulted during preparation; this is not a paper reproduction.

## Prospective finite selection

Use the same LR grid 0.0025, 0.005, 0.01 for all three families. Screening uses fresh seeds 2321 and 2322 at width 64: exactly 18 full-budget trajectories. Freeze one LR per family by mean final H200 full-validation CE, ties lower LR. Freeze the best selected family by mean, ties affine, bias-free LayerNorm, RMSNorm.

A changed-normalization family is locally eligible only if its mean gain exceeds 0.005 CE and every screen seed improves against both the fixed inherited incumbent and equally tuned affine family. An affine LR-only winner must beat the fixed incumbent by the same local rule and is explicitly attributed to LR selection. No fallback to another arm after failure. No larger-width reselection.

Run all three selected recipes on fresh confirmation seeds 2331, 2332, 2333 at widths 64 and 256: 18 trajectories. Include six fresh fixed-incumbent references unless the affine family selected the exact inherited LR and complete recipe, in which case prospectively alias those six cells. This yields exactly 36 or 42 physical trajectories and 72 or 84 correlated raw/H200 outputs. Aliases are not independent repeats.

Only the frozen locally eligible family can replace the incumbent. It must improve mean CE by more than 0.005, improve every confirmation seed, and have negative upper endpoints for all three conditional text intervals at both widths against each required reference. No grid extension, pruning, promotion from raw weights, or post hoc rule changes.

## Evaluation and reporting

Primary score: every one of the 111,538 validation targets once, common context 128 and batch 8. Retain raw 2,048-token block sums/counts. Report paired conditional text bootstrap intervals for blocks of 2,048, 4,096 and 8,192 tokens, 2,000 draws, seed 20261024. These intervals condition on the trained seeds and adaptively reused text; they do not measure seed-population uncertainty or independent holdout evidence.

Report six comparisons per width: bias-free versus affine, RMS versus affine, RMS versus bias-free, and all three versus fixed incumbent. Include all raw/H200 paired results, nine fixed-LR screening contrasts, every seed, boundary choices, parameter counts, measured training times and resource metrics. No inference-speed or equal-compute claim follows from training time or parameter counts. The held-out test remains closed.

## Correctness and resource gates before launch

The predecessor audit and complete CPU suite have passed before publication. The original proposal and test evidence remain preserved in the preparation folder; this status update changes documentation only. Bind the full shared knowledge bank, including negative/failed findings and the memory-guard clarification, and finish all remaining serial MPS/resource gates before registration. The complete checklist is:

1. Actual CPU shared initialization/RNG/parameter-count checks; independent double-precision normalization formula and all-gradient comparison; causal and cached/full-logit checks at widths 64/256 for each family.
2. Actual trainer/optimizer/resume tests: moments/steps/group factors, independent decay, matching schedule, private train/reset-validation/untouched-test RNG, H200 noninterference and export equality, strict reloads and rejection of same-shaped semantic mismatches.
3. Required project regression and complete controller/predecessor/selection/alias/report tests, including independent bootstrap reconstruction and first-failure/no-retry behavior.
4. Six serial MPS pilots, one per family and width, using the upper LR grid value. Verify training, strict reloads, full validation, cached logits, optimizer/RNG/averaging contracts, and measured process/device allocations. Pilots are correctness/resource observations excluded from quality selection.
5. A bounded resource estimate with 50% margin, final source review, clean commit/push, fresh-seed recheck and a uniquely named preregistered run. Never run competing model workloads.

Hard limits: 42 trials, 2,400 seconds per trial, 21,600 seconds per round, 4,096 MiB output artifacts, serial one-worker execution, first failure stops without automatic retry. Verify the inherited bootstrap calls `start_memory_guard`: nominal 0.5-second polling of process peak RSS (4,096 MiB limit) and current MPS driver allocation (6,144 MiB limit), exits 86/87. This sampled guard is not instantaneous total-system or peak-accelerator enforcement. MPS current allocation must not be labelled peak memory. No duration/storage prediction is frozen before actual pilots.

At completion, audit every distinct raw/derived checkpoint and raw request/result hash, source/corpus/runtime/full-bank contract, all saved shapes/configuration/optimizer/RNG/schedule/averaging/coverage fields, frozen selections and aliases, full report and independently reconstructed bootstrap. Preserve failed evidence; never rerun completed model work merely to recover interpretation. Only after that audit may findings update the incumbent, learning guide and report. Continue the human quality-improvement goal through later finite rounds until interruption.
