# Prospective serial versus parallel attention/FFN blocks

This finite study is not yet registered. The implementation was integrated only
after the weight-tying study completed its full independent audit and documentation.
All 3,030 unique CPU cases passed, including 18 actual geometry/trainer cases and
193 new metadata cases. Source and test hashes stayed unchanged throughout serial
validation. Passed temporary checkpoint batches were inventoried before removal;
all logs, inventory receipts and older failed evidence remain preserved. CPU
correctness is not quality evidence. Four serial MPS pilots and independent
checkpoint, branch-math and resource reviews remain required before registration.
The final predecessor retained legacy untied tables, no output bias, affine final
LayerNorm, unit input scale, peak LR .005, independent decay .00006, zero warmup,
3,000 updates with cooldown from 2,000, and H200 returned weights.

For block input `x`, let `a = A(N1(x))` and `u = x + a`. The serial control
returns `u + F(N2(u))`; the candidate returns `u + F(N2(x))`. Both retain the
existing separate RMS normalizers, modules, parameter shapes and constructor
draw order. The candidate changes initial functions and gradients. Attention
is added exactly once and the floating-point addition order stays `(x+a)+f`.
“Parallel” describes dependencies; this implementation launches ordinary eager
operations and does not claim concurrent kernels or fused projections.

PaLM section 2 motivates this structural question but reports small quality
degradation at 8B and none at 62B. Its large-scale fused-projection speed result
does not predict local MPS speed or quality. This experiment retains separate
RMS norms, MHA, attention biases and the reviewed optimizer/initialization;
it is not a reproduction of the PaLM architecture bundle.
Source: https://arxiv.org/html/2204.02311v5

The narrow `mha4_separate_norm_residual_topology` policy requires explicit
inherited table-sharing, output-bias/prior, final-normalization and input-scale
contracts. Four layers, MHA head32, SwiGLU21D/8, RMS blocks, RoPE, unscaled
residuals, base64 and widths64/128/256 are required. Unreviewed branch scales,
normalizer sharing, MoE and other backbone overrides are rejected. Named
optimizer roles and width factors stay unchanged; this is not proof of optimal
muP transfer. Save/load and optimizer construction bind constructed/current/
saved topology even though state shapes are identical. Legacy absence is serial.

Proposed finite panel: 12 width64 screens, serial/parallel × LR .0025/.005/.01
× fresh2381/2382. Freeze each LR by mean H200 (ties lowerLR), then the minimum
mean family (tie serial). A changed-family winner requires mean gain>.005 and
every seed better than tuned serial and the fixed predecessor. A serial LR-only
winner is evaluated against the fixed predecessor and attributed separately.
Failed best means retention, with no alternate fallback. Run all12 selected
confirmation cells at widths64/256 with fresh2391/2392/2393, plus0/6 fresh fixed
serial references. Alias only when selected serial LR and the whole recipe
match. Total24/30 physical trajectories,48/60 correlated raw/H200 outputs.

Only the eligible frozen winner may adopt: gain>.005, every fresh seed better,
and all2048/4096/8192 conditional text interval upper bounds<0 at both widths
against the required references. No grid extension, pruning, larger-width
reselection or holdout. Report fixed-LR screens separately. Selected-LR results
are recipe comparisons, not an isolated gradient mechanism.

Retain the final reviewed common3000-update,3072000-target, context128B8
paired_blocks512, cooldown/H200 recipe unless the terminal predecessor contract
requires otherwise; do not infer a new setting from partial results. Full111538
validation targets once per output,2000 bootstrap draws seed20261024; adaptive
validation and conditional text uncertainty do not establish seed-population
inference. Tying, prior, bias, normalization, input scale, decay and reference LR
are bound to the completed weight-tying review; partial results never choose them.

Before registration: actual independent forward/all-gradient graph checks,
initial tensors/RNG/counts/group identity, causal cache, changed-topology and
config-deletion rejection; CPU trainer resume before/at/after H200 start,
private train/resetval/untouched test RNG, canonical table and passive H200;
full regression; four serial MPS40-update upperLR.01 pilots with independent
branch/checkpoint/optimizer/RNG/fullcoverage audits. Pilot seed is 9417; four cells cover serial and parallel separate-norm blocks
at widths 64 and 256. The research manifest is frozen only after all gates pass. CPU scratch storage must have a bounded retention
plan preserving failed evidence; disk preflight alone does not prevent a full
suite from accumulating too many temporary checkpoints.

Hard proposed limits30 trials,2400s/job,21600s/round,4096MiB research outputs;
existing sampled4096MiB process RSS/6144MiB current MPS driver guard, one worker,
first-failure-stop/no-auto-retry. Require50%margin resource forecast and adequate
disk headroom, complete source/scaling review, clean published source, full
knowledge snapshot, repeated seed-reuse scan and a new unique preregistration.
No competing model work is allowed. Research source is frozen once a unique
new run is launched; all earlier studies remain closed.

The separate prototype includes a final-only predecessor resolver that binds the
complete reviewed weight-tying plan/state/report/selection and independently
reconstructs report bootstrap arithmetic. It preserves the resolved table-sharing
kind together with bias, full training prior, final norm, input scale and LR. The
new controller has exactly12 screens and24/30 total physical trajectories. Its
independent report auditor reconstructs six width-specific selected-recipe
comparisons (primary and raw, all three block lengths) and three fixed-LR screen
contrasts. Synthetic fixtures do not establish model correctness or quality.
