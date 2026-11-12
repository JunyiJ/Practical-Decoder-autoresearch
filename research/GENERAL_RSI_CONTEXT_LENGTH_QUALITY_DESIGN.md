# Fresh equal-time context128/context256 comparison

## Evidence, mechanism and limits

The common-target training-span diagnostic found CE96-minus-CE48 negative in all
three selected native models and both sources, mean−0.014213562nats/token. Earlier
48-minus16 gains were larger; individual windows sometimes worsened. Original
scores and criteria are in `general-rsi-context-prefix-001/analysis.json` and its
plan. This establishes useful context inside128, not a benefit beyond128.

The isolated context pilot preserves all28,492,416 parameters, initializer order,
optimizer policy and1,024 targets/update. New128 matches old128 exactly through
one update. Mean pilot times were0.352604/0.348422seconds at128/256, with no claim
of sustained speed superiority. The active-EMA integration passes exact full
restart/replay, endpoint export, identical-target coverage and fixed first outputs.
See `GENERAL_RSI_CONTEXT_LENGTH_PILOT_RESULTS.md` and
`general-rsi-context-length-integration-001/completion-review.json`.

The applicable primary literature is the previously read Khandelwal et al.,
[1805.04623v1](https://arxiv.org/html/1805.04623v1), sections3–4.1, on truncation
with common eligible targets, and Dai et al.,
[1901.02860v3](https://arxiv.org/html/1901.02860v3), sections3.1–3.2 and4.3, on
fixed-segment context limits and matched baselines. Their architectures, scale and
word-level tasks differ. We do not implement recurrence or claim their measured
context lengths or gains apply locally. Local gains can diminish or reverse.

Hypothesis: joining adjacent128-token segments into256-token segments reduces
missing-context loss enough to improve the complete recipe at equal training time.
The control distinguishes that gain from more wall time, more parameters or a new
optimizer. A failure of the gate rejects adoption at this fixed setting, without
claiming every longer-context design fails. A positive primary result with no
common128 benefit would localize the advantage mainly to the available inference
context rather than improved short-context weights. These are prospective tests.

This is fresh training. Recent selected-parent continuation diagnostics all failed
to improve their parents; none enters these pairs. Existing four-hour native
training improved on two hours. It supplies the inherited horizon and LR, not
pooled confirmation scores or an optimum. No new LR screen is added: both arms
use the established native LR0.0025, and the question concerns this fixed recipe.

## Frozen comparison

Six serial runs, three fresh paired bundles, initialized from scratch. Order is
128/256,256/128,128/256 across bundles. Each run receives14,400 synchronized
training-update seconds, including sampling, transfer, forward/backward, clipping,
optimizer and averaging. Clock overshoot is at most one update. Evaluation,
checkpointing and generation are reported outside that clock. Audited lost work
is charged on recovery. No seed replacement, pooling or stopping by probe score.

Both arms use ordinary CE, FP32 D256/four MLA blocks, RoPE, the same88 parameters,
Muon/AdamW, clip1 and independent decay0.00006. Two sampled512-token blocks are
reshaped to8×128 or4×256, preserving flattened inputs and targets per paired
sampler draw. Faster runs may consume a longer prefix of that same stream.
The inherited cross-document training packing remains explicit. Context identity,
causal masks and rotary-buffer length are the only architecture-contract changes.

Use inherited linear cooldown over each run's final third to LR ratio0.1 and
start EMA at the first update beginning at or after two-thirds of the budget.
Use the same127,184,813,977ns half-life in both arms, inherited from the native
recipe. Each update's averaging decay uses elapsed time between update starts.
Final time-EMA is primary, RAW secondary. No best checkpoint selection.

The plan freezes fresh role values, bootstrap seed and source/runtime hashes
before launch. No historical training seed is used as fresh confirmation.

## Evaluation and decision

Score every validation document separately, every target after its first token
exactly once, including EOT:100,416 targets in108 documents. Both arms score
identical target IDs and document membership. Primary inference uses each arm's
native context length, with nonoverlapping chunks and context reset at chunk
boundaries. Thus available prefix differs intentionally with the recipe. Retain
all per-token scores, document sums/counts and source aggregates. Corrected FP32
RMSNorm remains active. Test stays unopened.

For each pair compute time-EMA CE256-minus-CE128. Adopt256 only if all three
are negative, their mean is at most−0.02nats/token, the one-sided99% paired-document
bootstrap upper bound is below zero, and neither source regresses by more than
0.05nats/token on the three-pair mean. Use10,000 PCG64 resamples of108 document IDs,
average paired seed loss sums before resampling, divide by sampled target counts,
and take the linear99th percentile. Confidence is conditional on these documents
and seeds. Otherwise retain128; do not select a seed or reinterpret a near miss.

Also score every256 endpoint at128 as a fixed secondary contrast against its
paired128 endpoint. RAW and common128 results diagnose the full-recipe result;
they cannot rescue a failed primary gate. Retain RAW probes on the inherited
fixed eight documents at one-third and two-thirds of each run, with native length.
They are diagnostic only. Do not rerun historical validation panels.

Generate24 first outputs from each final time-EMA endpoint,144 total, using the
unchanged12 prompts, greedy and temperature0.8/top-k40, seed42 reset per prompt
and decoder, max64new tokens, EOS50256, no rerolls. Uncached full-prefix inference
is qualified; these prompts and outputs fit128, so they do not test eviction or
long-prompt ability. Retain token journals and text. Apply the existing repetition,
relevance and consistency rubric with arm/seed identities hidden until annotations
are saved. Agent ratings remain exploratory and cannot rescue a failed CE gate.

## Complete resource admission

Measured new time-EMA restart size is445,008,823bytes and inference payload size
113,998,811bytes. The complete retained-output ceiling is12×128MiB inference
files, two512MiB active/temporary restart slots,512MiB metadata and5GiB failed-
output allowance. Add10GiB reserve and2GiB extra headroom. Total required free
space at initial admission is21,474,836,480bytes. Existing retained payloads are
already charged in free space. Never admit just the first run because it fits.

The scientific tranche has36hours total and22,800seconds per slot, including
attempts, with at most two attempts per slot. The forecast includes24training
hours, conservative full-score/probe time derived from the training-document
integration, six fixed-generation sets, checkpoint I/O, initialization, logging
and remaining failure/runtime variability allowance. Exact measured components
and free space are recorded in plan/admission. Serial MPS only, RSS4GiB/MPS6GiB.

Only a verified transient restart predecessor may be retired after its successor
is read back, hash-verified and bound to complete resume state. Use exact receipt,
consumer checks and durable journals under the shared lock. Final endpoints,
failures and first outputs stay retained. If capacity changes, execute authorized
exact audited cleanup; do not weaken criteria or reduce confirmation.

Capacity009 retired24 accepted obsolete pilot/worker payloads from the closed
readout-mixture and MLA-gate studies,6,727,436,212bytes. It verified705 protected
files unchanged, including their scientific weights and selected references.
This narrow standing-authorized retirement supersedes older retain-all wording
only for those exact engineering artifacts. Original evidence remains unchanged.
