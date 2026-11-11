# Queued composer FP32 training-quality comparison

## Authority and order

Authorized by the human on 2026-11-10 Pacific, after discussing the uncalibrated
0.01% arithmetic threshold. Finish the current six-slot native-horizon panel,
its frozen decision and all144 first-generation evaluations before executing
this study's numerical work. This note queues a new study; it does not admit
training, assign fresh seeds, or modify any frozen source or old failure.
Use a separate identity, `general-rsi-composer-fp32-quality-001`.

## Evidence and unresolved question

Measured local evidence:

- `tokenizer-evidence/general-rsi-composer-fp32-fixture-001/attempt-01/failed-comparison.json`
  and `completion-review.json`: the first nonzero rank-one case had direction
  relative error0.0002945274107749496 versus FP64. Maximum parameter difference
  was4.695743395510821e-8 and momentum difference5.456968139486802e-13. Their
  respective gates passed. The direction gate failed, so repeat, restart and
  remaining gradient checks were not reached. No training-quality outcome exists.
- `GENERAL_RSI_MOMENTUM_PRECISION_RESULTS.md`: on saved momentum arrays FP32
  reduced paired output differences119–173 times versus BF16 for three matrices.
  FP64 also amplified input perturbations. This did not test actual full training.
- `GENERAL_RSI_EQUAL_TIME_RESULTS.md`: native won the equal-time comparison.
  This new byte-only ablation cannot establish superiority over native.
- No local calibration connects the old direction threshold to validation loss.
  The closed qualification remains failed under its original contract.

Primary external evidence: Keller Jordan, *Muon: An optimizer for hidden layers
in neural networks*, 8 December2024, [Definition, Eliminating alternatives and
Tuning the coefficients](https://kellerjordan.github.io/posts/muon/), reread for
this decision. The note uses BF16 finite Newton-Schulz iterations and reports
that imperfect orthogonalization can retain training performance. Its polynomial
approximation discussion is not a tolerance on our FP32-versus-FP64 rounding
error. Its hardware, models and much larger batches differ from this Mac's byte
composer. It supports measuring quality, not assuming FP32 is superior or harmless.

Inference: changed composer arithmetic may change optimization enough to improve
or worsen final prediction, or may have no detectable effect at this budget.
Small first-update parameter errors do not establish any of these outcomes.
The new information is measured training quality, not another attempt to pass
the old numerical threshold. A gain would justify further practical comparison;
a loss would reject the recipe at this setting; an inconclusive result would
not establish equivalence or harmlessness.

## Proposed discriminating comparison

Two arms only: the existing byte recipe and the same recipe with FP32 composer
Muon arithmetic. No batch reuse, new native arm, architecture change or data
change. Audit normalization, momentum, Nesterov, polynomial coefficients/steps,
shape scaling and weight decay against the installed baseline before freezing.
Keep these policies identical, with the precision change explicit. If the old
standalone component differs in other arithmetic policy, use new integration
source and disclose that difference; do not call a bundled change precision-only.

Use three fresh paired seed bundles, identical initial weights and sampler
streams,10240updates per arm,1024targets/update, and fixed LR0.0025 for both.
This is the previously selected byte LR, not a fresh tuning result. Equal
updates isolate the change under matched data exposure; measure synchronized
training time and total worker time separately. It is not an equal-time win.
Retain the established update-based final-third cooldown to0.1 and H200 EMA,
RAW secondary, the same model/data and all108 validation documents100416targets.
No historical scores are pooled into fresh confirmation. No short pilot CE gate.

Proposed primary inference uses the existing symmetric superiority rule:
all three paired H200 CE deltas favor the winner, mean gain at least0.02nats,
paired-document99% upper bound below zero in that direction, and no source mean
regression above0.05. Freeze bootstrap seed, resampling, order and all roles
before training. A failure to establish either direction is inconclusive, not
proof of equivalence. Do not weaken this rule after viewing quality scores.

Use the existing12 prompts and two fixed decoders on all six final H200 models:
144 first outputs, retained without rerolls. Blind arm/seed identity for the
fixed relevance/consistency rubric; report actual annotator identity, repetition,
length/EOS/truncation, each decoder and each seed. Generation is a separate
quality diagnostic and cannot rescue failed CE superiority. Test stays unopened.

## Necessary checks and admission

Only new integration checks are required: correct optimizer groups/updates,
finite gradients/state/weights, invalid-state rejection, complete optimizer and
sampler/averaging checkpoint roundtrip and uninterrupted-versus-restored replay
within the same recipe, plus whole-update timing. Existing scorer, corpus and
generation prerequisites remain qualified. Use new source and fixture identities;
do not retry the old failed fixed-gradient qualification.

Record FP64 update discrepancies as diagnostics. Neither1e-4 nor a substitute
cross-precision agreement threshold is a quality veto. Correctness failures
such as nonfinite updates, wrong state restoration or unintended policy changes
still require repair or explicit failure classification. A legitimate divergent
training run is preserved as an outcome, not repaired by changing seeds or LR.

Prospective cost: six runs total61440updates. At historical byte throughput of
roughly0.6seconds/update this is about10.2training hours, but FP32 throughput is
unmeasured. Provisional ceiling24hours including evaluation, generation, failures
and I/O, plus at most900seconds for the new integration fixture. Measure the
candidate before admission and revise the prospective ceiling if justified;
never silently shorten confirmation to fit it.

Planning storage caps: twelve final weights at64MiB each (768MiB), two512MiB
restart/serialization slots (1024MiB),256MiB metadata,5GiB failed-write allowance,
2GiB for new qualification outputs, plus10GiB reserve:19GiB free required before
new numerical work, subject to measured checkpoint sizes and complete retention
inventory. Existing payloads, the rest of the active panel and filesystem changes
must be accounted for at admission. This is a forecast, not a current capacity
blocker or launch approval. Use exact authorized cleanup if admission needs it.

Next executable action after current-panel completion: audit and integrate the
two optimizer arms outside frozen namespaces, freeze a bounded fixture with
arithmetic agreement diagnostic-only, verify new state/restart/timing behavior,
then freeze the scientific panel and measured full resource contract. Use the
nonblocking shared lock and one local worker. No paid compute or subagents.
