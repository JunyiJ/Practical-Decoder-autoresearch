# Weight-averaging quality protocol

This protocol is prepared for a distinct bounded round under the human’s continuing quality objective. It becomes operative only after the completed capacity predecessor is independently reviewed, the gates below pass, and a unique read-only plan is preregistered. The plan binds the exact resolved backbone, source, corpus, runtime, prerequisite evidence and full knowledge snapshot. Preparation does not authorize competing training or imply a launch. All completed studies and permanent stop 006 remain closed.

## Motivation and evidence limits

Our completed schedule experiment established an all-seed quality gain from final-third cooldown at both tested widths. Averaging nearby training iterates is a separate way to change the returned model without adding training tokens. Whether it improves an already decayed endpoint is uncertain: smoothing can suppress noise, but including older weights can also lag optimization.

Primary sources consulted October 25:

- [Anytime Pretraining, v2](https://arxiv.org/html/2602.03702v2), Meterez et al., August 2026: studies averaging with constant and inverse-square-root schedules on substantially larger language models, alongside a linear-regression analysis. It motivates testing averages but does not establish an improvement for this small Muon/cooldown recipe. We are not reproducing its anytime claim or transferring its theory to this setting.
- [Early Weight Averaging, v2](https://arxiv.org/abs/2306.03241v2), Sanyal et al., December 2023: reports benefits from checkpoint averaging in larger pretraining experiments, with learning rate and checkpoint spacing affecting results. That is motivation, not a prediction of a gain here.

## One trajectory, several returned models

Resolve the exact architecture, peak LR and cooldown only from the FINAL reviewed capacity adoption. If its frozen winner qualifies, use that recipe; otherwise use the reviewed four-layer standard dense cooldown .005 incumbent. Bind predecessor plan/report/review/selection and the adopted config. No partial capacity result, larger-width posthoc choice or historical pooling may choose this backbone.

Fixed training: 3,000 updates, 3,072,000 targets, context 128, batch 8, paired_blocks512, the resolved dense MHA SwiGLU architecture and exact Muon/AdamW muP/decay/clipping/schedule recipe. No LR tuning is added: the question is averaging on this fixed trajectory, not jointly optimizing a new averaging/LR family.

Each fresh trajectory stores its ordinary final online weights and three exponential tail averages. The online weights alone produce training gradients and optimizer updates; averages never feed back into training or change sampled batches. At completed update 2,000, initialize each average from the online parameters. Thereafter update it every 10 completed updates, including update 3,000. The half-lives are 10, 50 and 200 optimizer updates, with beta = 2^(-10/H) for each snapshot recurrence:

`A_j = beta*A_(j-1) + (1-beta)*theta_(2000+10*j)`, j=1..100, A_0=theta_2000.

This is a spaced exponential tail average, not per-step EMA or a uniform checkpoint mean. The initial tail snapshot has final weight 2^(-1000/H). No zero initialization or bias correction is used. All learned parameters—including embedding, readout, norms and biases—are averaged in float32. Persistent nonlearned RoPE buffers must be copied unchanged from the final online state, not averaged. The model architecture and inference math remain identical. Never pair an averaged model with the online optimizer for training continuation.

The eight physical training trajectories yield 32 model evaluations: raw plus three averages per trajectory. These are correlated outputs of the same training runs, not 32 independent trainings or seeds. Reporting shared trajectory cost avoids inventing a separate raw training-time baseline. Additional shadow copies, updates, final evaluation and checkpoint output have real overhead and require measurement in pilots.

## Bounded prospective selection

Two width-64 screening seeds, 2141/2142, whose nonuse must be verified before registration, train once each and score all four returned states on full validation. Select the lowest-mean averaging half-life, tie shorter half-life, before any fresh confirmation or width-256 result. A frozen average is locally eligible only if mean CE improves by more than .002 and both seeds improve versus their own raw endpoint. The smaller .002 minimum is prospective for a same-trajectory, same-architecture averaging change; it does not alter any earlier .005 capacity/schedule rule. It represents about .2% perplexity, not a universal statistical significance threshold.

Regardless of local eligibility, confirm all three averages and raw endpoints on fresh seeds 2151/2152/2153 at widths 64/256: six more training trajectories. Only the frozen screening choice may replace the incumbent. Require mean gain >.002, every seed improving and every conditional text-bootstrap upper bound below zero at both widths. If the screen choice was ineligible, retain the incumbent regardless of later results. No new half-life, re-selection, pruning or grid extension after registration. Confirm all planned outputs even if no local choice qualifies.

Primary evaluation remains full validation at context 128, batch 8, 111,538 targets scored once. Preserve raw block losses/counts and paired conditional bootstraps at 2,048/4,096/8,192 tokens, 2,000 draws, seed 20261024. These are adaptive validation and conditional text uncertainty, not seed-population or held-out evidence. The historic GELU, shallow and constant-deep references remain preserved; fresh duplicate training of them is unnecessary for the narrowly paired averaging question. The fresh raw resolved incumbent is the comparator in every trajectory.

## Implementation and correctness gates before launch

- Explicit optional averaging config, immutable half-lives/start/interval/horizon, sequential completed-step checks and strict named-parameter identity/shape/dtype contracts. Disabled mode must preserve existing checkpoint/metric behavior.
- Shadow state and counters saved with raw training state; exact continuation must reproduce raw parameters, optimizer state, private/global RNG and each average. Reject missing, malformed, mismatched, nonfinite or out-of-step averaging state. Resume before, at and after averaging begins must work exactly on CPU.
- Prove synthetic recurrence against an independently computed weighted sum; demonstrate that collecting averages does not change online parameters or consume RNG. No model training tests while capacity research is active.
- Derived inference checkpoints retain strict model/config/version/lineage metadata and a hash of the raw parent. They carry no online optimizer or resumable training state; resuming them must fail explicitly. Full coverage evaluates each derived checkpoint once without rewriting the raw checkpoint.
- Full project tests and actual serial MPS train/save/reload/full-validation pilots at both production widths after capacity completion/review. Check the resolved backbone, all three half-lives, intermediate resume states and raw-versus-tracked online math. MPS numerical checks are tolerance-based unless exactness is observed; no blanket bitwise-reproducibility claim.
- Fresh source/scaling/resource review. Hard caps: eight training trajectories, four final outputs each, 1,800 seconds per trajectory including all scoring, 14,400 seconds round, 2,048 MiB output, process RSS 4,096 MiB and MPS driver allocation 6,144 MiB, one worker, first failure stops, no automatic retry. Recompute conservative time/memory/checkpoint projections from pilots and verify seed nonuse before registration. No resource projection is measured yet.
- Clean reviewed/tested source committed and pushed, full shared knowledge, unique finite preregistration and read-only plan before launch. Completion must independently audit raw and derived lineage, exact parameter/buffer payloads, averaging metadata/counters, sampler/RNG/optimizer contracts, full coverage, frozen choice, report/bootstrap/adoption and resource limits. Never rerun completed training to reconstruct a missing average.

A launch requires the complete preceding gates and unique registration. This protocol does not imply an averaging quality improvement or permit reopening a completed study.

## Verification limits and pilot contract

The worker performs one training trajectory, scores its raw endpoint, then exports and scores half-lives 10/50/200 sequentially. A failure preserves partial artifacts and stops; no automatic retry. The final audit compares each derived parameter exactly with its registered final shadow state in the raw parent and preserves every nonlearned buffer. Final checkpoints alone cannot reconstruct all 101 historical online snapshots; recurrence correctness relies on independently calculated synthetic tests and the reviewed frozen trainer code. Do not claim independent reconstruction of the complete historical averaging path.

The prerequisite pilots comprise six serial jobs: tracked 40 updates, raw 40 updates, and resume the final 20 updates from the tracked midpoint at each width 64/256, seed 9304. All 18 full-coverage outputs are correctness/resource evidence excluded from research quality results. CPU continuation must be exact; MPS online/resume comparison uses atol=1e-4 and rtol=2e-5. The finite pilot caps are 180 seconds per job, 1,800 seconds total and 1,024 MiB output. Require the completed predecessor, no live research process, a matching full CPU review and clean committed source before pilots. The separate prerequisite review must record actual outcomes; this protocol does not assert that they have passed.

The resource forecast uses five width-64 and three width-256 trajectories. Scale tracked 40-update pilot training to 3,000 updates, add the measured four-output overhead, and apply a 50% time margin. Count the raw checkpoint including its three shadow states plus three derived inference checkpoints, allow 25 MiB of per-trajectory metadata, and apply a 50% disk margin. These are forecasts; observed per-job and round limits remain mandatory.

## Calculated coefficient lag, not a quality prediction

| Half-life in optimizer updates | Per-snapshot beta | Initial snapshot weight | Final snapshot weight | Weighted mean lag in updates |
|---|---:|---:|---:|---:|
| 10 | 0.500000000 | 7.88860905e-31 | 0.500000000 | 10.000 |
| 50 | 0.870550563 | 9.53674316e-07 | 0.129449437 | 67.250 |
| 200 | 0.965936329 | 0.03125 | 0.034063671 | 274.706 |

These finite recurrence coefficients sum to one. Larger half-life weights older models more strongly, which may smooth noise or lag improvement; neither effect has been measured here. The101 snapshots are correlated, not101 independent training seeds.
