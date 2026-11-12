# General RSI: human-directed continuation, 2026-11-09

## Current boundary — 2026-11-12 Pacific

The pseudo-repetition continuation panel is complete. All six execution reviews
passed and all144 first outputs received fixed identity-hidden agent ratings.
Ordinary CE wins the frozen validation rule: pseudo-minus-CE mean+0.144614982
nats/token, all three pairs regressing, ordinary-CE99% upper bound−0.129389331.
The candidate reduces greedy terminal loops17/36→8/36 but has lower descriptive
relevance ratings. This is a valid negative for the complete recipe at this setting.
See `GENERAL_RSI_PSEUDO_REPETITION_RESULTS.md` and
`tokenizer-evidence/general-rsi-pseudo-repetition-001/completion-review.json`.
All12 final weights and144 first outputs remain retained; test is unopened.

The common-state and accumulated-state optimizer diagnostics are complete.
The latter reviewed all1,494updates/45probes across nine fresh trajectories. At
100matched natural updates, zero-minus-CE mean loss is+0.018368 and pseudo-minus-
zero adds+0.035367nats/token, both positive in all three parents. See
`GENERAL_RSI_PSEUDO_ACCUMULATED_RESULTS.md`. This supports accumulated effects;
ordinary CE also drifts upward from parents, so it is not a model-improvement claim.

The joint-gradient integration is now qualified in all three arms: full CE,
half CE, and half CE plus half pseudo before one clip/optimizer advance. All264
parameter-gradient checks matched exactly, and all three complete save/restore/
replay checks passed. See `GENERAL_RSI_JOINT_GRADIENT_INTEGRATION_RESULTS.md`.
Measured update times are0.343/0.354/0.486seconds; total qualification111.703seconds.
The fresh joint-gradient diagnostic is also complete:900 updates/45 probes,
all execution checks passed. At100 natural updates, joint-minus-full CE is
+0.019226 and joint-minus-half CE+0.021136nats/token, both unfavorable in all
three parents. Synthetic repetition probability and its previous-copy ratio fall.
The frozen joint-benefit criterion fails. See `GENERAL_RSI_JOINT_DIAGNOSTIC_RESULTS.md`.
The posthoc saved-window audit finds prediction costs in both sources and in
windows with and without repeated trigrams. No quality model is adopted.

The CE continuation drift audit is complete. All three 1,200-second controls
regress against their starting parents on full validation: mean RAW+0.050811,
time-EMA+0.026287nats/token. Both sources and all retained400/800-second subset
probes regress. Raw parameter-group norms fall to83.3–83.5% of their parent norms.
See `GENERAL_RSI_CE_DRIFT_RESULTS.md`. These are existing-score and weight audits;
no validation was rerun. Norm shrinkage does not establish decay causality.

The retained-weight norm-restoration diagnostic is complete. All27 conditions
passed. Readout-only scaling lowers training-window CE in all six endpoints,
mean−0.070219 RAW and−0.058423 EMA, but every endpoint remains worse than its
parent. Other-parameter and joint scaling harm every endpoint. See
`GENERAL_RSI_NORM_RESTORATION_RESULTS.md`. No model is adopted.

The group-decay integration and fresh diagnostic are complete. All1,200 updates/
63 evaluations pass. Removing readout decay lowers CE in all three pairs,
mean−0.003509; removing both lowers it by−0.009567. Every arm still regresses
from its parent, including no decay at mean+0.019654. See
`GENERAL_RSI_CONTINUATION_DECAY_RESULTS.md`. No recipe is adopted.

The RAW-versus-time-EMA starting-weight diagnostic is complete. All 600 updates/
36 evaluations pass, including 528 exact loaded-tensor checks. RAW starts reduce
relative drift by mean −0.007399 nats/token, but every endpoint still trails the
unchanged EMA parent. See `GENERAL_RSI_CONTINUATION_START_RESULTS.md`. No model
is adopted.

The first-update response diagnostic is complete. All24 updates/171 evaluations
pass. Mean disjoint-probe loss changes are positive at scales1/16,1/4 and1,
so all three parents fail the frozen overshoot criterion. All72 update-batch
comparisons improve. See `GENERAL_RSI_UPDATE_RESPONSE_RESULTS.md`. Reduced harm
at smaller steps is not a useful model gain.

The independent-batch averaging diagnostic is also complete:48 updates and51
probe evaluations passed. Eight-batch means remain above every unchanged parent,
and the pooled paired difference is +0.000010555 nats/token. All three parents
fail the frozen criterion. See `GENERAL_RSI_BATCH_AVERAGING_RESULTS.md`. Close
this diagnostic without a larger-batch sweep. Next implement the common-target
prefix ablation in `GENERAL_RSI_CONTEXT_PREFIX_NEXT.md`, within the current128-token
limit. Prior loops fit within95 tokens, so this does not claim an eviction cause.
No worker or quality panel is active. Selected/fresh weights and all first outputs
remain retained; test is unopened.

## Completed priority — 2026-11-11 Pacific

The native-horizon panel, frozen decision and144 generation ratings are complete.
The separately authorized composer FP32 quality panel also completed all six
runs and144 blind generation ratings. Its frozen result is inconclusive:
mean H200 FP32-minus-existing CE−0.002731575nats/token, one-sided99% upper
+0.002311393, with two favorable pairs and one unfavorable pair. Neither
superiority direction passes. This does not establish equivalence. The original
failed numerical qualification is unchanged and its0.01% threshold was not used
as a quality veto. See `GENERAL_RSI_COMPOSER_FP32_QUALITY_RESULTS.md` and
`tokenizer-evidence/general-rsi-composer-fp32-quality-001/completion-review.json`.

No panel in the historical sequence below remains to be rerun. Next use the
research skill for an evidence-grounded boundary review of retained generation
failures and relevant primary literature, then rank justified new hypotheses.
No new training panel is admitted. State records the next executable action;
all selected and fresh checkpoints and original first outputs remain retained.

## Latest human priority — 2026-11-10 Pacific

The human explicitly authorized: "Yes, let's run that training-quality comparison
after the current panel finishes. Keep the original failure recorded, but don't
use the 0.01% threshold as a quality veto."

Finish all six native-horizon slots, their frozen decision and fixed generation
evaluation first. Then execute the separately identified composer FP32 training
comparison in `GENERAL_RSI_COMPOSER_FP32_QUALITY_NEXT.md`, after its bounded
integration checks and prospective resource admission. This is the next priority
ahead of other optional research. Do not interrupt or change the active panel.
Preserve the original FP32 and reuse failures and their frozen rules. The new
authorization supersedes their prohibition on a subsequent FP32 quality study,
but does not reclassify those failures or authorize a reuse arm. FP64 direction
agreement, including the old 1e-4 threshold, is diagnostic in the new study.

Sections below retain the earlier sequence and its historical prerequisites;
completed studies must not be repeated. The active state identifies progress.

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

The new durable journal and decoder integration passed in
`general-rsi-io-qualification-001/`. All48 first engineering completions are retained
and excluded from the future scientific panel. Generation preserved complete
training state. A same-boot supervisor timestamp bounds an unfinished update;
missing exit evidence rejects automatic recovery.

The assembled worker passed publication, readback, exact audited retirement and
endpoint checks in `general-rsi-worker-qualification-001/`. Source review found
that the initial launcher reset the slot timeout on retry. Launcher v2 corrects
that cumulative resource limit; AST equality confirms its tested execution and
retirement functions are unchanged. Explicit remaining-budget checks passed.

`GENERAL_RSI_EQUAL_TIME_DESIGN.md` and `general-rsi-equal-time-001/plan.json`
now freeze the new panel before scientific scores. It has four 1,800-second
screens and six 7,200-second confirmations. The admission includes all retained
outputs, active serialization overlap,256MiB metadata,5GiB failed-write allowance
and10GiB reserve. It fits actual free space. The20-hour tranche includes failures.
Next action: run slot00, review it, and follow the frozen serial run order.
