# Attention-output gating quality protocol

This prospective protocol becomes frozen only with its unique registered run plan.
QK normalization completed and passed its independent audit. Its frozen decision
retains standard MHA at peak LR 0.005, 3,000 updates, cooldown starting at 2,000
and H200 returned weights. The deterministic resolver verifies that complete
review; no partial result or later width result changes this backbone.

## Why this direction

Earlier stage-1 residual and attention dropout experiments repeatedly failed
their local improvement rules, as did many small independent-decay changes.
Those results do not prove universal harm under today's different recipe, but
they make another fine-grained dropout or decay grid a low-priority next step.
The completed horizon study also failed width-256 confirmation; its reversal
does not identify overfitting as the cause and does not justify reopening it.

Qiu et al. report benefits from query-dependent sigmoid gates after attention's
weighted-value output. Their experiments use much larger models and different
data, optimizers and budgets. This motivates a new local hypothesis, not a
predicted gain, measured attention-sink problem, or exact reproduction.
[Primary paper, sections 2.2 and 3](https://arxiv.org/html/2505.06708v1).

The repository already contains the human's GatedMLA implementation. This
proposal isolates output gating on the retained MHA backbone, without latent
compression or changes to that implementation or the primary kimi/KDA checkout.

## Exact proposed intervention and amplitude control

Three arms: `none`, `static_half`, and `sigmoid_headwise_zero`. Keep four layers,
head dimension 32, SwiGLU 21d/8, bias-free FFNs, RMSNorm blocks/final LayerNorm,
RoPE, unscaled FFN residuals and zero dropout. Resolve QK normalization and the
fixed incumbent LR only from the completed QK review. Hold them constant across
these arms except for the explicitly bounded per-arm LR tuning below.

Let X[B,S,D] be the existing pre-normalized input to attention and Y[B,H,S,32]
its weighted-value output, before concatenation and the output projection.
The three outputs are Y, .5Y, and Y * sigmoid(X Wg^T), respectively. Learned
scores have shape [B,S,H], transposed/broadcast only over the head channel axis.
Wg has shape [H,D], no bias, and starts exactly at zero without an RNG draw.
Thus learned and static-half arms initially have identical attention outputs;
the static arm controls initial attenuation. Gates depend on the current query
input only and do not add, alter or regate historical KV cache entries.
The existing output-projection bias stays unchanged: multiplying Y by .5 does
not halve that additive bias or the complete attention residual branch.

This initialization is our prospective design choice, not a claim that the
paper uses the same initialization. The static control is not a parameter- or
compute-matched control for learning a gate. A quality difference does not alone
identify sparsity, sink removal or a particular optimization mechanism.

Each learned layer adds D^2/32 parameters: four-layer additions are 512/8,192 at
widths 64/256. Default and static arms add none. Common model initialization
draws must remain identical across all arms. The learned matrix is a hidden
two-dimensional projection with both axes scaling with width, so the proposed
optimizer uses the existing Muon shape adjustment and 1/sqrt(D/64) width LR
factor. Other parameter roles and independent decay remain unchanged. Zero
initialization, sigmoid geometry and small head count need explicit checks;
this algebra is not an empirical optimal-LR transfer guarantee.

Use the explicit `mha_head32_output_gate` scaling policy and reject unreviewed gate bias,
temperature, initialization, placement, shape, attention type, residual or
normalization overrides. Bind the full resolved model config into resume
contracts. Missing/default `none` must preserve state keys, common weights,
RNG and forward arithmetic exactly. Do not loosen old optimizer guardrails.

## Finite panel and frozen selection

Fresh screen seeds 2201/2202 and confirmation seeds 2211/2212/2213 must be
checked against every prior frozen plan before registration. Screen three arms
at width 64 and peak LRs [0.00125, 0.0025, 0.005]: 18 physical trajectories. Select each
arm's LR by mean final H200 full-validation CE, ties lower LR. Then freeze the
best selected arm by mean CE, with exact ties favoring none, then static_half,
then learned. Do not revisit that winner after confirmation. Require local
mean gain >0.005 and both seed deltas negative against the fixed incumbent;
nondefault winners must also satisfy that rule against equally tuned none.
If the best arm fails these criteria, freeze incumbent retention.

Confirm all three selected arms at widths 64/256 on the three fresh seeds:
18 trajectories. Add six fresh incumbent-reference runs unless the selected
none arm has exactly the resolved incumbent LR/config, when references alias
those six runs prospectively. Total 36 or 42 physical trajectories, producing
72 or 84 correlated raw/H200 endpoints. Aliases are not independent repeats.
No pruning, grid extension, larger-width reselection, or alternative fallback.

Only the frozen eligible winner may replace the incumbent. Require mean
gain >0.005, every fresh-seed delta negative and all three conditional-text
interval upper bounds negative at both widths versus the fixed incumbent,
and versus equally tuned none for a nondefault winner. Always report learned
minus static-half as a separate comparison; without its support, do not claim
a benefit specifically from learned gating over constant attenuation.

## Shared training, evaluation and limits

Retain 3,000 updates, cooldown beginning at 2,000 to 10% of peak, H200 snapshots every
10 completed updates from 2,000 through 3,000 (101 snapshots), context 128/B8 paired_blocks512,
3,072,000 training targets, Muon/AdamW muP base 64, independent decay 0.00006, clip 1.
Final H200 full-validation CE is primary; raw final CE remains a correlated
diagnostic. Score 111,538 targets once each at context 128/B8. Preserve sampled
training logs without selecting on them. Use paired text blocks 2,048/4,096/8,192,
2,000 bootstrap draws, seed 20261024. Adaptive validation and conditional text
uncertainty are not held-out evidence or population uncertainty across seeds.

Hard caps: 42 physical jobs, 2,400 seconds/job, 21,600 seconds/round,
4,096 MiB artifacts/RSS and 6,144 MiB current MPS driver allocation. These are limits,
not cost forecasts or peak-memory measurements. Pilot-based forecasts must fit
them before launch, without increasing limits in response to failed gates.
One serial worker, first failure stops, no automatic retry or paid compute.

## Required launch gates

The implementation must pass independent forward and input/parameter gradient
checks, causal query-local behavior, full/cached decoding, common RNG/init,
parameter counts and optimizer roles/scales, finite actual updates, zero-init
gate gradients, strict save/load, changed-gate resume rejection, H200 raw/shadow
and RNG noninterference. Run the full project suite after predecessor completion.
Then run six serial MPS training/reload/full-validation/cache/gradient pilots
(three arms x two widths); exclude their scores and timings from research rankings.
The separate prerequisite review records exact executed tests, source hashes,
pilot evidence and resource forecasts. Preparation alone does not satisfy a gate.

Freeze a deterministic predecessor resolver, full knowledge snapshot, exact
source/runtime/corpus hashes, resource review, finite manifest and independently
reviewed controller. Commit/push clean isolated source and register a unique
run only after all gates pass. Audit every physical raw/derived checkpoint,
request/result, learned gate optimizer state, RNG, schedule, H200 count,
coverage, frozen choices/aliases, all report aggregates and independently
reconstructed bootstrap before any adoption. Preserve all closed studies,
failed evidence, human work and campaign 006's permanent stop.
