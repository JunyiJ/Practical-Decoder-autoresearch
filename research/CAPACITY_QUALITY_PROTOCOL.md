# Dense capacity quality protocol

This is a separate finite round under the human’s continuing model-quality objective. All completed studies and the permanent stop on campaign 006 remain closed. Primary human branch `kimi` is untouched. One local worker runs at a time; no external compute or held-out test access.

## Question and common schedule

Compare where added dense capacity is placed: four layers with standard SwiGLU hidden width 21d/8 (`dense4`), four layers with hidden width 21d/4 (`wide4`), and six layers with hidden width 21d/8 (`dense6`). All use MHA with head dimension 32. These recipes have different parameter counts and compute; this is an equal-token quality comparison, not an isolated causal effect of capacity location.

MoE quality001 observed wider dense improving all fresh seeds at widths 64 and 256, but it was not the frozen screen winner and was not promoted. That observation motivates this new prospective experiment with fresh controls. Earlier depth gains do not establish an optimal six-layer scaling policy.

The completed and independently reviewed schedule-quality-001 adopted final-third linear cooldown. Its historical source is `e01f7920fc9b3efe3c32bd50a38412ca34f9c1b4`, findings commit `1ae1a8314993ab95bff2432ea8775293d45aef5f`, result `kb_c6a54b5a4fd0d7ef57510cdad7fb7d5072d26235a795961177913ca3bea3bcbd`. Cooldown improved every confirmation seed at both widths, with mean CE deltas −.071693/−.075062 against tuned constant LR. The predecessor’s plan, selection, report and review are bound into this new lineage; the adoption decision is independently reconstructed by the new controller.

Every capacity arm uses that same cooldown: multiplier 1 through zero-based step 2000, then `1 - .9*(i-2000)/999`, ending at .1 on step 2999. No warmup or restart. The fixed reviewed incumbent is standard four-layer dense with peak LR .005. Each architecture receives the same bounded peak-LR grid [.00125, .0025, .005]; no linear or square-root depth/FFN LR scaling is assumed. No completed LR grid is extended.

## Fixed model, training and scaling contracts

Context 128, batch 8, `paired_blocks512`, 3,000 optimizer updates and 3,072,000 training targets per trial. Each update draws two 512-token chunks with private train RNG and reshapes them to eight 128-token examples. Fresh controls use the same sampler; historical panels are never pooled.

All arms use bias-free SwiGLU, MHA head dimension 32, RoPE, RMSNorm blocks with final LayerNorm, zero dropout, unscaled residual additions, Muon/AdamW hybrid, muP base width 64, clipping 1 and independent per-step decay .00006. Adam moments and Muon momentum retain their existing per-update definitions. The configured base LR is captured before optimizer restore; a common absolute-step schedule multiplier is applied once to each group’s immutable base rate.

The new `depth6_dense_unscaled` contract is narrow: exactly six layers, widths 64/128/256, base width 64, standard SwiGLU 21d/8, bias-free FFN, MHA head 32, context 128 and the stated norm/RoPE/dropout settings. Hidden-width, KV/group, MoE, latent/head, attention-bias, causal, positional-encoding and residual-scaling overrides are rejected. Older four-layer contracts still reject six layers. Existing default behavior is preserved.

Hidden matrix initialization remains fan-in based; readout initialization is multiplied by sqrt(d/64), and its forward output divided by d/64. Muon hidden matrix base rates are peak LR / sqrt(d/64); embedding, readout and vector Adam groups use peak LR. No depth multiplier or residual rescaling is introduced. These algebraic rules are not proof of optimal muP or depth transfer. Extra blocks consume RNG before readout creation, so full initialization is not identical across architectures. Wider FFNs also change matrix aspect ratios and RNG consumption.

Calculated parameter counts at vocabulary 65 are:

| Family | Width 64 | Width 128 (correctness only) | Width 256 |
|---|---:|---:|---:|
| dense4 | 204,544 | 798,208 | 3,152,896 |
| wide4 | 333,568 | 1,314,304 | 5,217,280 |
| dense6 | 302,592 | 1,188,864 | 4,712,448 |

The formula is `132d + depth*(4d² + 6d + 3dh)`. Actual model and saved-checkpoint counts must match. Synthetic forward/backward/update diagnostics measure finite activations, true gradients and updates at all three widths; one sample per shape is a correctness check, not a full-horizon stability or quality finding.

## Finite screening and confirmation

Exactly 18 full-budget screens at width 64: three families × three peak LRs × new development seeds 2121/2122. Select each family’s LR by mean final full-validation CE, tie lower LR. A capacity candidate is locally eligible only if its mean improvement exceeds .005 and every seed improves against both tuned dense4 and the fixed reviewed incumbent (.005 on the common cooldown). Choose the eligible candidate with lowest mean CE, tie family name. Freeze all LRs, the architecture winner and the confirmation manifest before any fresh confirmation or width-256 result.

Fresh confirmation uses selected recipes for all three families at widths 64/256 and seeds 2131/2132/2133: 18 runs. Fresh fixed references are two-layer GELU constant .002, two-layer SwiGLU constant .0025, original four-layer dense constant .0025, and reviewed four-layer dense cooldown .005: up to 24 runs. The cooldown incumbent reference prospectively aliases selected dense4 only if its selected LR is exactly .005. The original constant deep reference never aliases a cooldown recipe. No different architecture aliases another. Including screening, the exact total is therefore 54 or 60 distinct runs. Aliases are the same checkpoint, not independent repetitions.

No pruning, grid expansion, larger-width reselection or historical pooling. Recheck all five development seeds against every prior plan before registration. Fixed .0025 screening contrasts are reported separately from tuned confirmation contrasts.

Only the frozen architecture winner can replace the current incumbent. Require mean CE gain >.005 and every confirmation seed better at both widths against tuned dense4, the fixed reviewed cooldown incumbent and the original deep constant reference. Otherwise retain the incumbent and record narrower findings. The three-way candidate comparison includes LR choice and different compute; it is not a pure depth or FFN effect.

## Evaluation and uncertainty

Primary metric is final-checkpoint full validation at context 128 and evaluation batch 8: 111,538 targets, each scored once. Preserve raw 2,048-token block losses and counts. Independently reconstruct paired conditional text bootstraps at block sizes 2,048/4,096/8,192, 2,000 draws, seed 20261024. These intervals condition on the trained models and adaptively reused text; they do not quantify training-seed population uncertainty, independent holdout performance or adjusted significance.

The inherited +.01 CE near-neutral summary remains descriptive and does not override the quality-improvement adoption rule. Training sampled logs are diagnostics, not the ranking metric. Report actual training time, parameter counts and resource measurements; no equal-compute, optimal-transfer or inference-speed claim.

## Preconditions, budgets and completion

Before registration: review source and scaling; pass the full project suite, 59 configuration/synthetic controller cases, 10 actual shape/group/causality checks and 12 exact full-trainer resume cases; inspect nine one-update synthetic activation/gradient diagnostics. Model tests run only after the predecessor’s training has exited.

Commit clean source locally, then run serial MPS correctness/resource pilots at widths 64/256 for dense4, wide4, dense6 and all distinct fixed-reference shapes/schedules: 12 pilots, each 40 updates and 40,960 targets, seed 9303. Strictly reload finite CPU weights and optimizer state, verify sampler/private RNG, untouched test RNG, LR schedule/endpoints, layer/head/FFN shapes/counts and full 111,538-target validation. Pilot quality is excluded from research. No automatic retry on failure.

Hard research limits: 60 trials, 1,800 seconds per trial, 28,800 seconds per round, 4,096 MiB output, sampled process RSS 4,096 MiB and MPS driver allocation 6,144 MiB. One worker, first failure stops, no automatic retry. Pilot-derived worst-case no-alias time and checkpoint projections with 50% margin must fit before registration. Those are projections, not measured full-run durations; sampled/current allocation is not a true device peak.

Push reviewed and tested source to origin/main without force, freeze a full knowledge snapshot and unique `capacity-quality-001` preregistration, then launch from a read-only plan. Source stays frozen until completion review. Audit every distinct checkpoint and raw request/result hash, configuration, sampler/RNG/resume/optimizer/schedule/depth/head/FFN/count/coverage contract, exact seeds, frozen choices and aliases, full report and independent bootstrap/adoption calculation. Do not rerun completed training or evaluation to recover interpretation. Append findings, update best-observed configuration and the learning guide/HTML, commit and push, then continue the human’s quality objective through a separately justified finite round.
