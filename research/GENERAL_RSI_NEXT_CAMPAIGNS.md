# General RSI: human-directed continuation, 2026-11-09

The human requested this sequence: finish the existing longer-training comparison
unchanged; compare native and byte at equal Mac training time with suitable
schedules; add fixed generation measurements alongside validation loss; then
investigate transfer at another size or on broader held-out domains.

This prospective document supplements the active state. It does not modify
`GENERAL_RSI_PLAN.md`, the exposure plan, source hashes, seeds, endpoints or gates
of any launched experiment. No new panel is admitted by this document alone.

## 1. Finish the six-run exposure comparison

The frozen authority is `tokenizer-evidence/general-rsi-exposure-001/plan.json`.
All six endpoints are accepted. Three paired H200 differences (long minus short)
are -0.1981092561, -0.2056842069 and -0.1967594173 nats/token. Mean loss changed
from5.1356050685 to4.9354207751. The one-sided99% document-bootstrap upper bound
is-0.1812101383, and both source means improved. All frozen gates passed.
`general-rsi-exposure-001/decision.json` records the decision and
`adopted-recipe.json` retains all three long-run averaged checkpoints without
best-seed selection. Failure/replay evidence remains preserved. Test is unopened.
The longer recipe is validated on this corpus; generation and equal-time
superiority remain separate questions.

## 2. Native versus byte at equal Mac training time

Measured evidence is in `tokenizer-evidence/general-rsi-next-001/timing.json`,
derived from original confirmation slots04–09, including hashes of the source
logs and results. At equal 10,240 updates, byte's mean H200 advantage was
0.052604 nats/token, but it took roughly 1.83 times the logged update time.
Native has 28,492,416 parameters; byte has 16,351,859. Equal updates establish
a parameter/quality tradeoff, not a training-time win. The currently positive
exposure results also show that more updates can materially change quality.

Hypothesis: native can spend its throughput advantage on additional updates
and reduce or reverse byte's equal-update CE advantage. Alternative: spelling
sharing remains sufficiently sample-efficient for byte to win at equal time.
Either result identifies the appropriate recipe for a local time budget.

Use a fresh campaign with the same data, vocabulary, context128, D256/four
blocks, optimizer grouping and document-wise scoring. Three fresh paired seed
bundles; identical training-sampler streams within pairs, with different prefix
lengths permitted by throughput. Do not reuse old seeds as fresh confirmation.
No context, readout, optimizer-family or data change is bundled with this test.

Primary budget: equal **synchronized training-update wall seconds** on this Mac.
Include sampling/device transfer, forward/backward, clipping, optimizer and
averaging; synchronize MPS before and after timed work. Separately report total
worker time including initialization, evaluation, checkpoint I/O and generation,
plus failures/replay costs. This distinction must be explicit in every result.
The new worker must qualify timer boundaries, budget overshoot (at most one
update), restart accounting and thermal/runtime logging before admission.
Interrupted execution must never silently erase spent compute from accounting.

Candidate budget is 7,200 training seconds per confirmation arm, anchored to
the measured roughly 6,500-second byte short runs. This is a prospective design
choice, not a measured optimum. Native and byte receive the same live clock
budget; neither is stopped at the other's update count. Cooldown must span each
run's own budget, not borrow an intermediate point from a longer schedule.
Use the inherited linear cooldown over the final third to ratio0.1, expressed
against accumulated training time. Averaging must also have an explicitly
matched time window/decay; select and freeze its conversion using throughput
qualification, before any screen result. Retain RAW as a secondary endpoint.

Give each arm the same two LR candidates (.00125, .0025) and 1,800-second
screen budget per candidate, selecting final averaged validation CE within arm.
Confirm chosen LR from fresh initialization for 7,200 seconds, with its own
full-budget schedule. These proposed values may change only on prelaunch
engineering/cost evidence, with a documented revision, never after scores.

Before first screen, freeze exact clocks, averaging, seed roles, run order,
source/runtime/data hashes, generation protocol and the complete resource
contract. Proposed quality gate: all three paired averaged CE deltas in the
same direction, mean magnitude at least0.02, one-sided99% paired document
bootstrap excluding zero, and no source regression greater than0.05, symmetric
for native/byte. Freeze bootstrap seed and tie handling then. Generation scores
are separate diagnostics and cannot rescue failed CE gates.

Provisional training cost: four screens ×1,800s plus six confirmations ×7,200s
=50,400s (14h), before overhead. Admission requires measured evaluation,
generation, serialization/restart and failure allowance, not a blanket estimate.
All ten RAW/averaged pairs alone project about1.794GB from measured native/byte
weight sizes; add complete restart plus atomic serialization overlap, metadata,
all existing retained outputs and the10GiB reserve. Audit available capacity
and execute authorized exact cleanup if necessary before admitting the panel.
No partial panel or relaxed confirmation to fit disk. One local worker only.

## 3. Fixed generation evaluation for the new comparison

`general-rsi-generation-eval-v1.json` fixes 12 continuation prompts, two decoding
conditions and the scoring rubric before new outputs. Eight inherited prompts
provide continuity; four explicit small scenarios probe entity/state consistency.
They are diagnostics, not a representative held-out language benchmark.

Generate only from every final primary confirmation endpoint: 12 prompts ×2
decoders ×6 models =144 first completions, at most9,216 generated tokens.
Use greedy and temperature0.8/top-k40, seed42 reset per prompt/decoder, max64
new tokens, EOS50256, no rerolls. Save prompt, token IDs, exact decoded text,
seed, stop reason, checkpoint hash and decoder/runtime hashes. Never select
examples based on appearance. Current exposure outputs remain original evidence.

Report repeated generated-token trigram fraction and a terminal repeated-phrase
flag, together with length and EOS/truncation rates. A short or empty completion
must not appear successful merely because it has no repetition opportunities.
Separately score relevance and consistency using the fixed 0/1/2 rubric, with
quoted evidence and an explicit unassessable category. Hide arm/seed identities
in deterministic shuffled score sheets until annotations are saved. State the
actual annotator; agent ratings are not independent human validation. Do not
use lexical overlap as a semantic relevance score or a factuality guarantee.
Report each decoder separately, per-seed means and all prompt-level outputs;
no blended CE/generation score and no post-result decoder tuning.

`general_rsi_generation_metrics_v1.py` implements the token-level repetition,
length and termination statistics without loading models. Six synthetic boundary
cases and four invalid-input checks are recorded in
`tokenizer-evidence/general-rsi-next-001/scorer-check.json`. These are engineering
checks, not model-quality results. Generation-worker integration and blinded
semantic annotation remain prerequisites for the new panel.

## 4. Transfer after the equal-time result

Default next question: does the equal-time ordering hold on broader domains?
This is cheaper than another full training panel and directly probes whether
educational/Wikipedia specialization explains the result. Before loading any
new evaluation text into scoring, freeze source versions/licenses, domain
definitions, document selection, token budget, exclusions, deduplication against
training/validation, aggregation and all checkpoints to compare. Evaluate both
arms and all three seeds. Do not inspect the original test or select domains
according to model scores. New text is a prospective transfer holdout, not
additional training data. Findings remain conditional on those domains.

If evidence instead makes parameter scaling the discriminating question, record
that rationale and freeze one additional model size with fresh paired training,
separate throughput calibration and full output/resource forecast. Do not vary
size and corpus together. Neither transfer variant is yet launched or claimed
fully specified; its exact cost and gate must follow the equal-time findings.

## Literature and limits

- Hoffmann et al., [arXiv:2203.15556v1](https://arxiv.org/html/2203.15556v1),
  AppendixB: schedule length affects terminal quality; mismatched decay horizons
  confound comparisons. Their cosine schedules and much larger scale do not
  establish that our linear schedule or7,200s budget is optimal.
- Holtzman et al., [arXiv:1904.09751v2](https://arxiv.org/html/1904.09751v2),
  sections5.3–6: repetition depends on decoding, and distributional statistics
  do not adequately measure coherence. This supports fixed decoder controls
  and separate rubric ratings. Our tiny models,64-token continuations and
  agent ratings do not reproduce their large-model human-evaluation study.
- Penedo et al., [arXiv:2406.17557v2](https://arxiv.org/html/2406.17557v2),
  sections3.1,4.1–4.2: educational filtering has domain-dependent benefits and
  tradeoffs. Broader transfer is a hypothesis, not implied by current CE.
  Their model/data scale differs substantially from our finite10M-token corpus.

## Current engineering boundary — 2026-11-10

`general-rsi-time-qualification-001/review.json` accepts the new time-budget
training context on both MPS arms. A12-second budget produced33 native updates
(12.158s) and19 byte updates(12.461s), each with at most one-update overshoot.
The complete optimizer/model/sampler/RNG/time-EMA state matches exactly after
restore and replay with the original measured clock durations. Live clocks vary;
this test does not assert numerical equality under different time schedules.
Explicit lost-work charges survive capture/restore. Checkpoint I/O is excluded
from the training clock and reported separately. No validation selection occurred.

Measured restart payloads are445,008,055 and247,897,649bytes. The review records
a prospective ten-pair forecast plus two512MiB restart slots,64MiB metadata/header
margin and10GiB reserve. Available space exceeds that forecast. Recheck exact
capacity at admission; this is not permission to omit any retained output.

Next executable action: integrate durable recovery accounting and fixed generation
into the production worker. Qualify incomplete-update/replay journaling and decoder
boundaries, then freeze matched-time EMA half-life, fresh roles, source hashes,
run order and complete panel resources before launching the first LR screen.
The qualified document scoring and old restart fixtures need no repeat.
