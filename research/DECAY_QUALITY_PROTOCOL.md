# Prospective decay comparison on the cooldown/H200 recipe

Preparation only. No registered plan or launch. Resolve the common architecture
and peak reference LR only from the completed, independently reviewed
FFN-hidden-normalization round. Its partial screens or width256 measurements
must not select the next backbone. Preserve all closed studies and stop006.

## Question and evidence boundary

Test whether the old independent-decay anchor remains suitable after the
model changed from two-layer GELU to four-layer SwiGLU, higher peak LR,
cooldown and H200 returned weights. This is a new recipe interaction question;
it does not reopen the earlier fixed-LR decay studies or extend their grids.

Nine historical decay probes are recorded in
`research_state/decay-quality-validation/historical-review.json`. At width64,
zero/half/double decay changed CE by +.003513/+.001515/+.005433 respectively.
The old .00006 anchor survived those probes; nearby changes did not establish
a transferable improvement. These reused seeds11/22/33 and the earlier
two-layer recipe. Do not pool their scores or infer an optimum for the new
recipe. The newer horizon comparison worsened width256 when training longer,
but changed tokens, schedule and cumulative decay together; it does not
identify insufficient decay, overfitting or a mechanism.

## Intervention and scaling

Three arms, in tie-preference order: anchor=.00006, weak=.00002,
strong=.00018. The outer values bracket the anchor by a factor of three;
this is a prospective engineering choice, not a predicted optimum. No zero
arm or fine local sweep. Do not extend the panel after seeing results.

Keep the existing HybridOptimizer unchanged. At each optimizer update, each
parameter with a gradient is multiplied once by (1-delta), before the Muon or
AdamW update. Inner optimizers have weight_decay=0. This includes embeddings,
readout, matrices, gains and biases with gradients; absent-gradient parameters
are not decayed. Decay is independent of LR, its cooldown multiplier and width.
No inverse-width, LR-proportional or batch scaling is introduced.

Across3000 updates, nominal shrinkage alone is (1-delta)^3000. It is not a
prediction of final norms: optimization adds parameter-dependent updates on
every step. H200 includes a weighted sequence of those updates and decays.
The three arms intentionally do not match cumulative shrinkage. At matched
LR, the only configured difference is delta. Tuned comparisons also include
the separately selected peak LR. No causal mechanism or optimal-muP claim.

Common architecture: final-reviewed FFN norm and output-gate choices, otherwise
4L MHA/head32, dense bias-free SwiGLU21/8, RMSNorm blocks/final LayerNorm,
unscaled residuals, RoPE, zero dropout. Existing Muon hidden LR/sqrt(D/64),
AdamW vector/embedding/readout base LR and all momentum/beta/epsilon/clipping
rules stay unchanged. All arms start from identical common parameters and
private sampler RNG for a given width/seed. Resume binds delta exactly.

## Finite prospective panel

Proposed unused screen seeds2241/2242, confirmation2251/2252/2253; recheck all
plans at registration. Screen width64: three decay arms x peak
LR[.00125,.0025,.005] x two seeds =18 full-budget trajectories. For each arm,
choose lowest mean final H200 full-validation CE, ties lower LR. Choose the
lowest selected arm mean, ties anchor/weak/strong. Require mean gain>.005 and
both paired seeds negative against fixed incumbent, and tuned anchor for a
nonanchor arm. If the best arm fails, freeze retention; no alternative fallback.

Freeze all choices and eligibility before fresh confirmation at widths64/256:
three selected arms x three seeds x two widths =18 trajectories. Add six fresh
fixed references unless selected anchor exactly matches the reference LR and
full recipe, in which case prospectively alias those six cells. Total36 or42
physical trajectories;72 or84 correlated raw/H200 outputs. An alias is not an
independent repeat. Only the eligible frozen winner may replace the incumbent:
mean gain>.005, every fresh seed negative and all three conditional text
interval upper bounds<0 at both widths versus the required references.

Report all weak-anchor, strong-anchor, strong-weak and selected-fixed-reference
contrasts at both widths, for H200 primary and raw diagnostic. Separately
report all nine fixed-LR screening contrasts. No pruning, larger-width
reselection, old-run pooling, completed-run retry or holdout access.

Common training: context128/B8, paired_blocks512,3000 updates/3072000targets,
cooldown starts2000 and ends at.1peak, H200 snapshots every10 steps2000..3000
(101 snapshots),clip1. Primary full validation is111538targets once each at
context128/B8. Conditional text bootstrap blocks2048/4096/8192,2000draws,
seed20261024. Reused adaptive validation and conditional text uncertainty do
not establish held-out or seed-population significance.

## Launch gates and completion audit

No model tests or training while FFN training is active. Develop separately.
Before launch: final predecessor audit; independent algebraic update-order and
no-double-decay tests, zero/absent-gradient behavior, all actual optimizer
groups/states/width factors, strict state reload and changed-delta rejection;
actual trainer CPU continuation across cooldown/H200, private train/test RNG,
raw/averaging noninterference. Preserve default behavior and old evidence.
Run appropriate full regression after integration, then six serial shortened
MPS train/reload/full-validation pilots across three coefficients and two
widths, excluded from research. Cache/model arithmetic must stay unchanged.

Hard limits:42 trajectories,2400s/job,21600s/round,4096MiB artifacts and sampled
process RSS,6144MiB current MPS driver allocation, one worker, first failure
stops/no automatic retry. Current allocation is not peak. Pilot-derived time
and disk forecasts plus50%margin must fit without raising limits. No paid
compute. Clean reviewed source committed/pushed and new unique readonly
plan/full-bank snapshot before serial launch.

At completion verify every raw/derived checkpoint, request/result hash,
exact configuration, decay/optimizer slot/group/step contract, sampler RNG,
tokens, common norm/gate/QKV/FFN shapes,101-snapshot metadata and full coverage.
Independently reconstruct selection, aliases, all reports and bootstrap
intervals. Append findings and update best config and guide/HTML only after
review. A completed round is followed by another prospective quality question
until the human interrupts.
