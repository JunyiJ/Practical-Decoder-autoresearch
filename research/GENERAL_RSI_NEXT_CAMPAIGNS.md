# General RSI: human-directed continuation, 2026-11-09

## Latest human priority — cross-size transfer, 2026-11-12 Pacific

The human requested: "update the RSI research plan and tell the other thread to
focus on testing which findings transfer across model sizes on my Mac mini,
after finishing the current experiment."

This section takes precedence over the older next-action sections below. Finish
all six slots of `general-rsi-context-length-quality-001`, execution reviews,
frozen decision, common128 evaluation and fixed first-generation evaluation
unchanged. Do not interrupt the worker or change its sources, seed roles,
schedules, endpoints or criteria. `GENERAL_RSI_PLAN.md` is hash-bound into that
experiment and remains unchanged. This live continuation plan carries the new
research priority. Existing findings and closed campaigns stay closed.

After that completion, the objective is to identify which measured effects
persist, weaken or reverse as model size increases within local Mac capacity.
A lower score from another adjustment at D256/four blocks is no longer the
default objective. Local cross-size evidence is a step toward LLM relevance,
not proof of billion-parameter or GPU-cluster transfer. Continue autonomously
within the existing local authority, one MPS worker, no paid remote compute or
subagents. Preparation that reads existing evidence may proceed during training;
new accelerator qualification waits for the shared worker lock.

### Evidence and first question

- Equal-token byte composition improves CE by0.052604nats/token with42.61% fewer
  parameters, but costs about1.84x guard time. Equal7200-second training instead
  favors native by0.122102nats/token across three fresh pairs. Original records:
  `tokenizer-evidence/general-rsi-panel-001/decision.json` and
  `tokenizer-evidence/general-rsi-equal-time-001/decision.json`; readable reports:
  `GENERAL_RSI_CONFIRMATION_RESULTS.md`, `GENERAL_RSI_EQUAL_TIME_RESULTS.md`.
- The native model has28,492,416 parameters. Its two50257x256 vocabulary tables
  contain25,731,584 parameters, about90.3% of the total. This unusually large
  vocabulary-table share is a concrete reason to test a larger transformer body.
  Byte composition still consumes GPT-2 token IDs and retains an independent
  output table; it is not tokenizer-free BLT.
- Books/Python transfer was inconclusive. Python mean differences were sensitive
  to formatting-heavy excerpts. Preserve `GENERAL_RSI_TRANSFER_RESULTS.md` and
  `GENERAL_RSI_TRANSFER_DIAGNOSIS.md`; do not relabel them broad-domain success.
- Composer-only FP32 superiority was inconclusive. Repetition suppression lost
  its quality comparison; recent fresh-optimizer continuation diagnostics did
  not beat their parents. Preserve `GENERAL_RSI_COMPOSER_FP32_QUALITY_RESULTS.md`,
  `GENERAL_RSI_PSEUDO_REPETITION_RESULTS.md`, and
  `GENERAL_RSI_BATCH_AVERAGING_RESULTS.md`. Do not make more local precision,
  repetition or continuation sweeps the next default activity.
- Added prefix96 versus48 improves common-target CE by0.014214nats/token in the
  retained models (`GENERAL_RSI_CONTEXT_PREFIX_RESULTS.md`). The completed128/256
  panel accepts256: mean native-context CE difference−0.095711917 and common128
  difference−0.022220934. Generation remains weak. See
  `GENERAL_RSI_CONTEXT_LENGTH_QUALITY_RESULTS.md`.

Default first hypothesis: byte/native quality and cost tradeoffs depend on the
ratio of vocabulary/composer work to transformer-body work. At fixed vocabulary,
a growing body should dilute the fractional input-parameter saving. Whether
byte's equal-token loss advantage persists, and whether its runtime penalty
shrinks, are separate falsifiable predictions, not assumed scaling laws.
Measure each ratio rather than inferring it from total parameter count.
If the completed context panel supplies a stronger, cheaper discriminating
cross-size question, rank context128/256 against this hypothesis explicitly and
select ONE finding first. Do not combine context and input-representation changes
in the same contrast or launch an all-factor grid.

### Literature basis and limits

Primary sources reviewed for this boundary:

- Hoffmann et al., [2203.15556v1](https://arxiv.org/html/2203.15556v1),
  sections1 and3, AppendixB: model size, training tokens and schedule horizon
  jointly determine comparisons. Their much larger, largely non-repeated data
  regime does not supply an optimal token ratio for our finite10M-token corpus.
- Yang et al., [2203.03466v1](https://arxiv.org/html/2203.03466v1), sections3–6.1:
  standard-parameterization optima can shift with width; specific optimization
  hyperparameters transfer under muP, with explicit limits and no general
  regularization-transfer guarantee. Our mixed Muon/AdamW implementation is not
  qualified muP. Do not assume unchanged LR is equally suitable at every size,
  and do not bundle an unqualified parameterization change into this study.
- Pagnoni et al., [2412.09871v1](https://arxiv.org/html/2412.09871v1), sections3,
  5.1 and5.3: byte-module overhead and relative quality can change with scale and
  training budget. Their dynamic-patch encoder/decoder and400M–8B experiments
  differ from our token-input composer. These are a rationale for measuring an
  interaction, not evidence that our architecture inherits their gains.

### Bounded next execution sequence

1. After current-panel completion, inventory retained measurements and calculate
   parameter allocation for candidate sizes without training. Choose a declared
   width/depth scaling path that meaningfully changes body share, rather than
   adding vocabulary parameters. Target three feasible sizes including the
   reference; at least two with one larger body are needed for a cross-size
   comparison. If only two fit the measured cost, call it two-size replication,
   not a scaling law. Never claim broad extrapolation from a narrow size range.
2. Qualify only newly required shapes, initialization, numerical execution,
   complete restart and scoring. Run a bounded engineering timing/memory pilot
   under the shared lock, with no validation-based arm selection. Measure both
   arms at each candidate size. Do not rerun historical prerequisites or build
   general infrastructure without a demonstrated need.
3. Freeze a new prospective protocol before scientific scores. Use fresh paired
   seeds and at least three pairs per size for a confirmed quality claim. Keep
   corpus, tokenizer, output head, context, packing, precision and optimizer
   family controlled except the declared finding and size axis. State the
   initialization mapping and shared sampler policy. Use equally bounded LR
   screening if needed, with separate screen/confirmation roles; otherwise state
   that the result concerns transfer of a fixed recipe, not each size's optimum.
   Historical scores provide context and planning anchors, never fresh pairs.
4. For byte/native, use equal-token endpoints as the primary mechanism contrast,
   with complete horizon-matched schedules and fixed averaging. Declare whether
   token exposure is common across sizes; report distinct corpus size, sampled
   targets and reuse explicitly. Do not infer compute optimality. Report loss
   curves against tokens, estimated training FLOPs and synchronized Mac time,
   with operator assumptions and uncounted work stated. Intermediate curve
   points are descriptive, not replacements for terminal matched-budget runs.
   Any equal-time endpoint claim needs its own prospective schedule/endpoint
   contract. A context-first protocol must define the analogous matched-target
   and native/common-context controls before launch.
5. Predeclare the within-size effect and change of effect across sizes, practical
   thresholds, seed-level results, uncertainty and multiplicity handling. Do not
   call a nonsignificant interaction equivalence, a one-seed winner transfer, or
   separate significance tests proof that two effects differ. Use a small,
   prospectively frozen, deduplicated broader-domain evaluation selected without
   model scores. Keep the original test unopened. Existing validation is a
   development set; repeated reuse is not independent external confirmation.
   Generation diagnostics remain separate from prediction-loss decisions.
6. Admit the WHOLE selected comparison only after forecasting screening, fresh
   confirmation, evaluation, generation if required, replay/failure time,
   all retained endpoints, optimizer/averaging state, atomic serialization
   overlap and metadata. Preserve10GiB reserve plus2GiB extra headroom and
   current memory guards. Reuse the existing36-hour tranche ceiling as the
   initial planning bound; record any justified prospective revision before
   scientific results. Never shrink a launched confirmation or run only a
   convenient arm to fit. Apply standing-authorized cleanup safeguards when
   needed. Record measured infeasibility and the smallest concrete capacity
   remedy if no meaningful cross-size panel fits; do not revert silently to
   same-size tuning or repeat unchanged capacity scans.

Success means an interpretable cross-size result, including a vanished or
reversed benefit. Report local measurements, external evidence and inference
separately. Keep Mac throughput conclusions distinct from hardware-independent
compute estimates. Do not claim larger-LLM transfer until evidence beyond the
local tested range exists. Commit/push substantive source and evidence, update
state, and continue the next executable item under this priority without asking
again for authorization already given.

### Selected cross-size question — 2026-11-13

The completed context finding was ranked against the default. Select byte/native
equal-token transfer first, at D256 depths4/8/16. The body grows4× and its native
parameter share rises9.69%→30.03%; arithmetic byte savings fall42.61%→33.01%.
`GENERAL_RSI_CROSS_SIZE_BOUNDARY.md` records evidence, contrary results,
literature, controls and cost. ExactCPU depth4 parity and all sixMPS engineering
cells pass. Four new-depth full-state replays pass with activeEMA. Short-pilot
byte/native time ratios fall1.909→1.421. This is not sustained runtime evidence.
Fresh broader dataset002 contains16parents and32,768targets. Its original failed
filter remains preserved. See `GENERAL_RSI_CROSS_SIZE_ENGINEERING_RESULTS.md`.

The new endpoint/worker integration002 passes after preserving001's pre-training
data-adapter failure. All24first engineering outputs remain retained. The
scientific protocol is frozen in `general-rsi-cross-size-quality-001/plan.json`
and `GENERAL_RSI_CROSS_SIZE_QUALITY_DESIGN.md`:18fresh equal-token runs, fixed
seed roles/order, within-size rules, direct interaction and multiplicity handling.
The full forecast is51.226hours including probes, measured generation, completed
engineering and30% margin, under the prospective54-hour tranche. Full storage
admission passes with658MB beyond the required12GiB reserve/headroom and all
projected outputs. Slots00–05, the first fresh paired bundle across all three depths, passed
execution reviews. Byte-minus-native H200CE differences are-0.049465058,
-0.018119530 and-0.013540404 at depths4/8/16, with1.840×,1.648× and1.441×
training time. The descriptive16-minus-4 interaction is+0.035924654.
The first bundle supports dilution of the quality advantage and runtime penalty,
but the two larger-size gains are below0.02. Three-bundle confirmation and the
frozen interaction decision remain unresolved. All12endpoints and144first
outputs from bundle1 are retained. Slots06–07 complete the second depth8 pair:
byte-minus-native H200 CE is-0.030494703 with1.603× training time. Both depth8
pairs favor byte, with magnitudes straddling0.02. Eight runs,16endpoints and
192first outputs are retained. The third pair and other second-bundle sizes
remain pending. Slots08–09 complete the second depth16 pair: H200 byte-minus-
native CE is+0.006673365, versus-0.013540404 in bundle1. Training ratios are
1.440740× and1.440717×. Mixed signs preclude either arm meeting the frozen
all-seed sign rule at depth16. The full interaction and other sizes remain
pending. Slots10–11 complete the second depth4 pair with byte-minus-native
H200 CE -0.061117546 and1.844763× training time. The second bundle's
16-minus-4 interaction is+0.067790911; its runtime-penalty ratio16/4 is0.780989.
Both completed bundles descriptively support dilution. Third-bundle confirmation
and the frozen panel decisions remain pending. Slots12–13 complete the third
depth16 pair: byte-minus-native H200 CE is+0.001170895.
All three depth16 differences are−0.013540404, +0.006673365 and+0.001170895.
Their descriptive mean is-0.001898715. Neither arm meets the frozen
all-seed sign requirement; this does not establish equivalence. The third depth4/8
pairs and frozen panel inference remain pending. Fourteen runs,28endpoints and
336first outputs are retained. Full-panel capacity passes. Admit fixed slot14,
depth4 native bundle3, after live capacity and execution checks.
Continue every frozen run unchanged. See
`GENERAL_RSI_CROSS_SIZE_INTEGRATION_RESULTS.md`. Do not repeat completed depth
qualification or change bound sources, roles, schedules or criteria.

## Current experiment boundary — 2026-11-12 Pacific

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
this diagnostic without a larger-batch sweep.

The natural-text prefix diagnostic is complete:1,152 sequences passed. Added
prefix96 versus48 lowers CE in all three parents and both sources, mean−0.014213562.
The frozen context-use criterion passes within the existing limit. See
`GENERAL_RSI_CONTEXT_PREFIX_RESULTS.md`. This does not establish benefit beyond128
or an eviction cause for prior short loops.

The isolated128/256 pilot is complete: all50 updates pass, complete256 restart
and replay are exact, and wrong-context loading rejects before mutation. Mean
update times are0.352604/0.348422seconds, without a sustained-speed claim. One
CPU-digest harness failure and its separately identified repair remain retained.
See `GENERAL_RSI_CONTEXT_LENGTH_PILOT_RESULTS.md`. The changed active-EMA,
identical-target evaluator and generation paths are now qualified:28 updates,
exact complete restart/replay,176 exact exported tensors and24 first engineering
outputs. See `GENERAL_RSI_CONTEXT_LENGTH_INTEGRATION_RESULTS.md`.

The fresh six-run128/256 comparison is frozen in
`tokenizer-evidence/general-rsi-context-length-quality-001/plan.json`, under
`GENERAL_RSI_CONTEXT_LENGTH_QUALITY_DESIGN.md`. Three paired bundles each receive
14,400 training seconds per arm. Adoption requires all pairs improve, mean
CE256-minus-CE128 at most−0.02,99% bootstrap upper below zero and the fixed
source guard. Quality beyond128 remains untested. Capacity009 recovered
6,727,436,212 bytes from24 obsolete engineering payloads and verified705 protected
files unchanged. Full admission includes all retained outputs, serialization
overlap, metadata, failure allowance,10GiB reserve and2GiB extra headroom.
All six runs and execution reviews are complete. Context256 passes the frozen
CE gate: paired differences−0.097954103,−0.095261660,−0.093919989; mean−0.095711917,
99% upper−0.083908521. Both source means improve. Common128 EMA mean difference
is−0.022220934. All144 first outputs received identity-hidden agent ratings;
generation results are mixed and greedy terminal loops rise14/36→16/36.
See `GENERAL_RSI_CONTEXT_LENGTH_QUALITY_RESULTS.md` and the panel's
`completion-review.json`. All12 final weights and144 first outputs remain
retained. The original test is unopened. No worker is active at panel closure.
Next rank this completed finding against byte/native and select one cross-size
comparison under the latest priority above. No scientific cross-size training
is admitted yet. Do not repeat completed prerequisites.

## Historical completed priority — 2026-11-11 Pacific

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

## Historical human priority — 2026-11-10 Pacific

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
