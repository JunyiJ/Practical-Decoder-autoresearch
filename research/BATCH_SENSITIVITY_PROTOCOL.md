# Batch sensitivity at equal token exposure

This separately bounded round completes the authorized batch8/16/32 robustness
question. It does not reopen any completed architecture, context or inference
study. Four-layer SwiGLU/MHA at context128 remains the architectural reference;
no GQA, MLA, MoE, depth or FFN change is combined with batch size.

## Frozen design

Tune three cells: batches8/16/32, context128, widths64/256, four layers, head32,
bias-free SwiGLU hidden21d/8, RoPE, RMSNorm, zero dropout, unscaled residuals,
depth4_unscaled, Muon/AdamW muP base64 and global clipping1. Initialization and
width factors remain unchanged. Shape/algebra tests do not prove optimal-LR
transfer to either a new width or a new batch size.

Train3000/1500/750 updates respectively: each recipe sees3,072,000 target
positions. Independent per-update decay is delta(B)=1-(1-.00006)^(B/8), computed
with log1p/expm1 for B16/32, and exactly .00006 for B8. This matches nominal
cumulative multiplicative shrinkage across the horizon, not the complete
regularization effect in the presence of gradients. Momentum/betas and clipping
stay unchanged per update. Their time scale in tokens consequently changes;
this is an equal-token batch-recipe comparison, not an isolated noise experiment
or an equal-update/equal-compute comparison. No linear or square-root LR scaling
rule is assumed.

The new explicit paired_batch512 sampler draws fixed pairs of512-token chunks
and groups1/2/4 such draws into batches8/16/32 at context128. Each larger batch
therefore contains the exact ordered inputs and shifted targets of2/4 consecutive
B8 training updates at the same seed. Fixed two-start RNG calls preserve the
private training RNG stream across grouping. Initialization does not depend on
batch size. Evaluation uses a separate RNG. Default independent sampling and
historical paired_blocks512 remain supported unchanged; the new layout is bound
in the checkpoint/resume contract. Fresh controls use the new named layout; do
not pool them with older studies.

Screen exactly18 width64 full-token-budget runs: batches8/16/32 x LRs.00125/
.0025/.005 x fresh seeds2061/2062. Interleave batch cells, reverse on the second
seed. Independently select each batch LR by lowest mean final full-validation128
CE, tie lower LR. Freeze all three choices and the whole confirmation/alias
manifest before any fresh-seed or width256 job. No pruning, grid extension or
larger-width reselection, even for boundary winners.

Confirmation: all three selected batches x widths64/256 x seeds2071/2072/2073
=18 runs. Add fresh B8/context128 original2L GELU LR.002, cheaper2L SwiGLU
LR.0025 and fixed4L SwiGLU LR.0025 references,18 runs. If selected B8 LR.0025
matches the fixed deep reference, alias its six jobs prospectively instead of
duplicating training. Total distinct runs are48 or54. Check all five seeds
against historical plans before preregistration. Preserve older repeat references.

## Evaluation, contrasts and interpretation

Primary endpoint is full-validation context128 batch8:111,538 targets scored
once per checkpoint. It is identical across all batch recipes. Held-out test
stays closed. The sampled training-log validation panel uses the training batch
and30 evaluation iterations, so its token count differs across batches; it is a
secondary diagnostic and never selects the LR. Logging intervals300/150/75
updates give the same nominal token spacing, with the trainer's zero-based step
convention; endpoint scores come from the final checkpoint.

Store2048-token raw block loss sums/counts. Use paired conditional text bootstrap
at2048/4096/8192 blocks,2000 draws, seed20261024. Report B16-minusB8,
B32-minusB8 and B32-minusB16 at each width, with every seed and all intervals;
report each selected cell versus all three fixed references. Separately report
fixed-LR.0025 screening contrasts. Selected-recipe comparisons include LR choice,
update count and matched decay, not a pure batch causal attribution.

The prospective descriptive near-neutral flag requires the mean, every seed and
all three conditional text interval upper bounds<=human+.01 CE. It is not
population non-inferiority, a multiplicity-adjusted test or automatic adoption.
Record measured training seconds/throughput, allocated memory diagnostics and
parameter counts; current MPS allocation is not true peak. No inference benefit
follows from training throughput. The historical four-layer MHA .0025 quality
incumbent remains unchanged. Conditional text bootstrap does not estimate
training-seed population uncertainty on this adaptively reused validation corpus.

## Prerequisites and resource bounds

Before launch require focused and full tests, exact ordered stream/RNG and
initialization checks, optimizer/decay/step/token checks, bounded serial engine
selection/alias/failure tests, checkpoint resume-layout checks and actual serial
MPS train/reload/full-validation pilots at every batch and both widths. Pilots
use seed9291 outside research panels and are excluded from research findings.
Review the implementation and full shared knowledge, commit/push clean source,
then freeze a unique readonly plan and full knowledge snapshot. Keep the human
kimi checkout separate and untouched.

Hard limits:54 trials,1200seconds/trial,21600seconds/round,2048MiB output,
one worker, first failure stops, no automatic retry. A conservative planning
proxy is1.5 times the maximum historical four-layer/context128 duration for
36width64 and18width256 jobs; checkpoints use historical maxima with25%margin.
This is a feasibility proxy, not a batch-speed measurement. Pilots must verify
actual execution and resource feasibility; no competing training is permitted.

At completion verify all distinct checkpoint/raw request/result hashes,
configs/sampler/RNG/optimizer/steps/tokens/layer/head/QKV/FFN contracts, finite
strict CPU loads, full coverage, seed panels, frozen LR choices and alias
manifest, report and independent paired bootstrap, source/corpus/runtime/fullbank.
Append findings, update best-context/guide/HTML, commit/push documentation and
preserve all evidence. Never rerun completed trials for interpretation. All older
studies remain closed, especially the human-stopped campaign006.
