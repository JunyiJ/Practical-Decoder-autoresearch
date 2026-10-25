# Training horizon and cooldown quality protocol

This is a separate finite study of fresh training trajectories. It does not resume or extend any completed study. The final independently reviewed averaging result adopted half-life 200; that returned-weight policy is fixed for every arm here. Historical raw cooldown, constant deep, shallow and GELU references remain preserved. Campaign 006 remains permanently stopped.

## Question and motivation

The completed schedule study's saved logs show improving sampled validation near update 3,000, while LR is also decreasing. Those logs neither establish convergence nor predict a gain from more training. This study asks whether a fixed increase to 6,000 updates improves final quality, and whether stretching cooldown or preserving its duration matters at the same longer budget. It does not tune LR or search for an optimal horizon.

## Fixed architecture and training recipes

Keep the four-layer MHA model with head dimension 32, bias-free SwiGLU hidden width 21d/8, RMSNorm, RoPE, zero dropout and unscaled residuals. Use Muon/AdamW with muP base width 64, peak LR 0.005, independent per-step decay 0.00006, clipping 1, context 128, batch 8 and paired_blocks512 sampling. Widths 64 and 256 have 2 and 8 heads, and FFN hidden widths 168 and 672. No attention, FFN, depth, normalization or optimizer change accompanies the horizon comparison.

| Recipe | Updates | Cooldown starts at zero-based step | Training targets | Final LR |
|---|---:|---:|---:|---:|
| control3000 | 3,000 | 2,000 | 3,072,000 | 0.0005 |
| long6000_stretched | 6,000 | 4,000 | 6,144,000 | 0.0005 |
| long6000_fixedtail | 6,000 | 5,000 | 6,144,000 | 0.0005 |

LR stays at 0.005 before cooldown and decreases linearly through the final update. Exact discrete base-LR sums are 12.75, 25.50 and 27.75. These calculated sums are not effective update norms or measured compute. Per-update betas, momentum, clipping and decay stay unchanged. Nominal cumulative decay retention is approximately 0.835266 for 3,000 updates and 0.697669 for 6,000; actual weights also receive gradients. More updates double data exposure and change cumulative shrinkage. The comparison against 3,000 updates is neither equal-token nor equal-compute, and cannot isolate a pure cooldown effect. The two 6,000-update arms share update/token budgets but differ in cooldown shape and LR integral.

Same-seed recipes use the same initialized active weights and ordered training input/target stream. Longer runs extend that private sampler stream; they do not resample a different prefix. This correspondence does not imply exact agreement between independent MPS trajectories. Existing evidence of independent-run differences remains preserved.

## Returned weights and evaluation

Half-life 200 is fixed by the completed averaging study's final review. Initialize one FP32 parameter average at the cooldown boundary, then update every 10 completed optimizer steps, with coefficient `2^(-10/200)`. All learned parameters are averaged; persistent nonlearned buffers come from the raw endpoint. The numbers of snapshots are 101, 201 and 101. Half-life stays in optimizer-update units, not a fraction of horizon. The changed averaging duration is an explicit part of each recipe. No half-life tuning or width-dependent reselection is allowed.

Evaluate both the averaged endpoint and raw endpoint for every trajectory. The average is the primary returned model; raw endpoints are diagnostics. Averages never feed into online gradients or the optimizer. Derived checkpoints are inference-only, and cannot resume the online optimizer; preserve every raw parent and full optimizer/RNG state. A general controller fallback to raw weights exists only for a predecessor that did not adopt averaging; this study resolves its final policy to H200 before registration.

Each final evaluation scores all 111,538 validation targets once at context 128 and evaluation batch 8. Training logs use evaluation every 300 updates with 30 sampled batches and are not the selection metric. The held-out test remains closed. Paired conditional text bootstrap uses blocks of 2,048, 4,096 and 8,192 tokens, 2,000 draws and seed 20261024. This is adaptive validation on fresh development seeds; text intervals condition on trained models and reused validation text, not population uncertainty across training seeds.

## Frozen selection and confirmation

Run six full-budget width-64 screens: all three recipes on development seeds 2161 and 2162. A longer recipe is locally eligible only when its mean primary CE improves by more than 0.005 and both seeds improve versus the fresh 3,000-update control. Choose the eligible recipe with lowest mean CE; ties use recipe name. If none qualifies, freeze retention of the control.

Freeze that selection and the complete confirmation manifest before any confirmation result or width-256 run. Confirm all three recipes at widths 64 and 256 on fresh seeds 2171, 2172 and 2173: 18 further trajectories. Exactly 24 physical trajectories produce 48 correlated model outputs. There is no pruning, aliasing, grid extension, posthoc alternative promotion or historical pooling.

Only the frozen locally eligible winner may replace the incumbent. At both widths it must improve mean primary CE by more than 0.005, improve every confirmation seed and have all three conditional text interval upper bounds below zero versus the fresh control. Report both longer-versus-control contrasts and the direct fixed-tail-versus-stretched contrast, for both primary and raw endpoints. An unselected alternative cannot be promoted using confirmation results.

## Source, scaling and resource gates

Existing explicit schedule and averaging contracts bind total horizon, start step, interval and resume state. Synthetic tests check exact endpoint multipliers, discrete LR sums, snapshot counts, fixed manifests and selection failures. Actual CPU trainer tests check optimizer/model/RNG agreement after saved-state continuation, sampler stream reconstruction, derived export counters and rejection of changed-horizon resumes at both widths. These checks establish implementation behavior, not optimal muP or depth transfer.

After predecessor completion, run six actual serial MPS correctness/resource pilots: three recipes at both widths, shortened to 30 updates/start 20, 60/start 40 and 60/start 50. Use H200 and interval 10, producing 2, 3 and 2 snapshots. Each pilot trains, strictly loads the raw and derived checkpoint, checks finite model/optimizer/shadow state, sampler RNG, schedule/counters/shapes and full raw/average validation coverage. Pilot scores are excluded from research. No independent MPS weight-equality requirement is inferred: averaging arithmetic and noninterference were directly tested under the unchanged trainer implementation in the predecessor. Bind those tests and all earlier failure/repair records in the full knowledge snapshot.

Hard limits: 24 trajectories; 2,400 seconds per trajectory including both full evaluations; 21,600 active seconds per round; 4,096 MiB output; sampled process RSS at most 4,096 MiB and MPS driver allocation at most 6,144 MiB. One worker only. Stop on the first failure and preserve it; no automatic retry. Current/sampled memory is not peak accelerator memory. Recompute a conservative runtime/disk forecast from measured serial pilots and historical completed trajectories before registration. Report measured training, evaluation and wall times, storage and sampled allocations separately from forecasts. Lower CE at higher cost is not automatically an efficiency improvement.

Before launch require a complete test pass, reviewed clean committed/pushed source, serial device pilots, explicit resource review, a fresh historical seed scan, the full shared knowledge bank and one uniquely named immutable preregistration. At completion independently verify all raw/derived checkpoint and request/result hashes, contracts, optimizer slots, final sampler RNG, shapes, counters, coverage, frozen selection, entire report and bootstrap calculations before updating the best config and learning guide. Do not rerun completed training to recover interpretation. Continue to the next justified finite quality question until human interruption.
