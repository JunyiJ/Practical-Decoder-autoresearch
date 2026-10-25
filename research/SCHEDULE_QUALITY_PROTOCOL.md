# Schedule quality experiment

This is a separate finite experiment under the human's continuing quality objective. All earlier studies remain closed, including MoE quality001 and permanently stopped campaign006. The held-out test stays closed. The run plan, committed source, prerequisite review and full knowledge snapshot must be frozen before research training.

## Question and resolved backbone

Does reducing the learning rate within a fixed 3000-update budget improve final validation quality over constant learning rate? This is a hypothesis, not an observed improvement.

The completed, independently audited MoE quality001 selected MoE at width64, but it failed confirmation at width256. Its result is `kb_d0c34e1507f736aee3cbcc38a410480edb35bd788bb0f104a3be81dd8f86c0c7`. Therefore this round uses the retained **four-layer dense SwiGLU MHA** for every schedule arm. Wider dense was promising in that round but was not its frozen winner; it is not substituted post hoc here. Bind the final MoE review and its report/selection hashes in the new plan. The controller reconstructs that adoption decision and rejects incomplete or inconsistent reviews. Its prospective MoE branch is unused in this round.

Fixed architecture: bias-free SwiGLU hidden21d/8, four layers, MHA head dimension32, RoPE, RMSNorm blocks with the existing final LayerNorm, unscaled residuals, `depth4_unscaled`, dropout0. Width64 for selection; widths64/256 for confirmation. Muon/AdamW hybrid, muP base_width64, clipping1, independent per-update decay .00006. Context128, batch8, paired_blocks512, 3000 updates and 3072000 training targets. All schedule arms have identical shapes, initialization and paired training exposure within seed. Fresh controls prevent pooling with historical sampler/source contracts.

The original Muon matrix LR width factor and Adam parameter-specific factors are preserved. Scheduling applies the same scalar to every optimizer group's configured base rate. Decay remains independent of LR at every update, so changing LR changes the update/decay balance and LR integral. Equal updates and targets do not establish equal compute or an optimal muP transfer rule.

## Schedules and finite selection

Each schedule receives the same peak/base LR grid: .00125, .0025, .005.

- Constant: multiplier1 at every update.
- Cosine: .1 + .9*(1+cos(pi*i/2999))/2 for zero-based update i=0..2999.
- Linear cooldown: multiplier1 through i=2000, then 1-.9*(i-2000)/999; final multiplier.1.

No warmup or restarts. Both decay arms end at10% of peak, with a gradient contribution on the final update. Capture each group's base LR before optimizer-state loading; never compound a multiplier onto a previously scheduled rate. Schedule kind, horizon, floor and cooldown boundary are bound in the resume contract. A changed nonconstant horizon is rejected. Absence of a schedule preserves legacy behavior; explicit constant is opt-in.

Exactly18 full-budget screening trials: three schedules × three LRs × fresh seeds2101/2102, width64. Interleave schedules and reverse their order for the second seed. Recheck all historical plans for seed reuse before preregistration.

Select each schedule's LR by mean final full-validation CE; ties choose lower LR. Freeze all three choices before any confirmation or width256 trial. A decaying schedule is locally eligible only if its screen mean improves by more than .005 CE and every screen seed improves against both tuned constant and fixed constant .0025. Among eligible schedules choose lowest mean, then schedule name. If none qualifies, record no winner. Do not revise this choice using larger-width results.

Confirm every selected schedule on fresh seeds2111/2112/2113 at widths64/256:18 trials. Add fresh fixed references: original two-layer GELU LR.002, cheaper two-layer SwiGLU LR.0025, retained four-layer dense SwiGLU constant LR.0025, three seeds at both widths. The deep reference prospectively aliases the six selected-constant cells exactly when its selected LR is.0025; otherwise all six deep references are distinct. Thus **48 or54 distinct trials**. Resolve and freeze this manifest before confirmation. An alias is the same evidence, not an independent repeat. No pruning, grid/horizon extension, historical pooling or larger-width reselection.

Only the frozen screen winner can replace the incumbent after completion audit. It must improve mean CE by more than .005 and every seed at both widths against tuned constant and the fixed deep incumbent. The historical dense reference and fixed incumbent coincide in this round. Otherwise preserve narrower findings and retain the incumbent.

## Evaluation and interpretation

Primary score: final checkpoint, full validation at context128/batch8,111538 targets exactly once in disjoint windows. Never rank sampled training-time panels or best intermediate checkpoints. Preserve all actual LR snapshots, sampled logs and final checkpoints. Fixed-LR .0025 screening contrasts are separate from confirmation contrasts that include LR selection.

Report paired seed deltas and conditional text-block bootstrap at blocks2048/4096/8192,2000 draws, seed20261024. This is adaptive validation and conditional uncertainty over this text, not independent holdout or population/seed uncertainty. The inherited +.01 CE near-neutral summary is descriptive and does not replace the stricter quality-improvement adoption rule. Report training time separately; do not infer serving/inference speed.

## Implementation, validation and launch gates

The opt-in trainer uses an absolute update index and immutable configured group rates. Hybrid optimizer loading refreshes its flattened group view because child loads replace group dictionaries. Trusted result validation requires every expected metric step and exact finite schedule fields. Historical requests remain unchanged.

CPU validation recorded before pilots:96 focused synthetic/protocol cases;14 actual trainer tests covering dense/MoE, widths64/256, all schedules, checkpoint/resume trajectory and explicit-constant versus legacy-default equivalence;765 remaining project tests. The 14 and765 cases cover779 unique tests on matching source. These are correctness checks, not quality results. Test logs and source hashes are saved in `research_state/schedule-validation/cpu-review.json`.

Before research launch, complete10 serial actual MPS correctness/resource jobs: each of three schedules at widths64/256, plus original GELU and shallow SwiGLU references at both widths. Each uses40 updates,40960 targets, seed9301; schedule arms use LR.005, controls their fixed rates. Scale decay horizons and cooldown boundary proportionally. Require save/reload, finite strict CPU weights and optimizer state, correct final group rates, reconstructed private train RNG, untouched test RNG and full111538-target validation. Pilot quality and timing are excluded from research findings. No competing training.

The pilot driver requires the completed predecessor, clean locally committed source, matching CPU artifacts and no live training. It uses a fresh output directory,180 seconds/job,2160 seconds total,1024MiB output, at most12 jobs (10 for this dense round), and stops on first failure. Preserve failed artifacts; a repair uses a new commit and new pilot directory after diagnosis. Push implementation only after pilot and resource gates pass.

The controller hard ceiling is60 trials; this resolved dense manifest permits at most54. Further caps:1800 seconds/trial,28800 active seconds/round,4096MiB output,4096MiB process RSS and6144MiB MPS driver allocation. One worker; first failure stops, no automatic retry. Memory watchdog observations are sampled, not guaranteed allocation peaks. Validate a worst-case no-alias resource projection from fresh pilots with50% margin before freezing the plan; this is a proxy, not a measured research duration.

Finish source review, clean committed/pushed source, prerequisite review, full append-only knowledge snapshot including MoE findings, unique preregistration, immutable code/data/runtime/plan hashes before launch. Keep source frozen through completion review. At completion independently verify every raw request/result/checkpoint and saved configuration/schedule/sampler/RNG/optimizer/shape/coverage contract, selection/aliases, complete report and bootstrap, and quality decision before recording findings or adoption. Never rerun completed training to recover interpretation.

## Literature context

[SGDR](https://arxiv.org/abs/1608.03983v5) and the [official cosine formula](https://docs.pytorch.org/docs/main/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html) motivate cosine decay. [Wen et al.](https://arxiv.org/abs/2410.05192v3) study stable-then-decay language-model schedules. This round omits warmup and restarts and fixes3000 updates with a10% floor; it is not a direct WSD replication or evidence of a horizon-free optimum. Published results motivate the hypothesis, not a local quality claim. Literature record:`kb_af813e4a493cf7a926fc355d5697ce457c5182ca193c6e3194f2b9a4b348db8d`.
