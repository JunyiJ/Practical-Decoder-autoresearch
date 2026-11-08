# Current work: CPU reference repaired; third bundle running

The first nonlinear candidate completed all 1,200 updates, then failed the fixed
maximum per-target CPU/MPS CE limit (.000364233 > .000300). The original failed
guard, both reports, logs and RAW/H200 weights remain preserved. A 2.477-second
saved-weight diagnostic isolated the discrepancy to native CPU cross-entropy:
identical logits gave CPU error up to .000364087 versus CPU64, while MPS was
within 8.9e-7. Transformer and context-transform differences were much smaller.

The separately frozen full-split CPU-reference verification took 55.144 seconds.
An explicit shifted FP32 reduction agreed with CPU64 within 1.94e-6 across all
three completed checkpoints. Every original CPU report reproduced exactly;
all original MPS quality reports remain byte-identical. Corrected CPU/MPS maximum
errors are 1.91e-5, 3.06e-5 and 1.75e-5, passing unchanged limits. See the
[recovery decision](tokenizer-evidence/byte-context-transform-001/cpu-reference-recovery-decision.md).
No training was repeated and no checkpoint was rewritten.

The new `quality-recovery002` plan imports slots00..02 and runs only original
unattempted slots03..08. Plan SHA:
`2f55b9e24888232ed6c2bbe5f41ae8b68bd6e9122eeca343de1973dcefb21aef`.
Dispatcher PID70581 / exec53076 and final slot08 linear worker91678 are live;
144 updates were observed. Eight runs are accepted. Slot07 byte completed with
H200 CE4.571279233, mean CPU/MPS error2.052775e-8 and maximum target error2.050400e-5,
passing unchanged limits. All three paired byte/tanh comparisons favor byte:
tanh-minus-byte H200 deltas are+.065875265,+.076989235,+.052111667.
In the later two pairs, final-online differences are only+.001474464 and−.003865640;
the larger averaged gaps identify an interaction worth analyzing, not permission
to switch endpoints. The first baseline online curve retains its old-scorer caveat.
Only original slot08linear remains. No new model is adopted.
Prior exec99662 is terminal failed. Do not restart on an observation
timeout or resume a superseded dispatcher. The source/count-derived sampler
limitation applies explicitly to slots00 and02; slot01's observed digest is kept.

First-bundle unchanged H200 losses: byte4.550480418, linear4.643989454,
tanh4.616355683. Tanh loses to byte, so the all-seeds-negative adoption criterion
cannot pass. Complete the fixed panel to characterize its outcome, without
changing roles, controls, endpoints, tolerances or learning rate. The retained
output cap remains1.5GiB; 2,151.110 seconds of prior guarded execution are charged
to the original10-hour allowance. Latest measured free space is12.63GB; no cleanup is needed.

After full completion, run:
`PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m research.byte_context_transform_review_v3 2f55b9e24888232ed6c2bbe5f41ae8b68bd6e9122eeca343de1973dcefb21aef`.
The independent reviewer covers imported evidence, original primary-score identity,
fixed comparisons, numerical checks and corrected resource accounting.
On accepted review, run `.venv/bin/python -m research.byte_context_final_analysis_v1`
to derive all paired learning curves, averaging differences and charged execution
cost from saved evidence. This analysis source is prepared; no premature panel
analysis or new model execution has occurred.

The [first-bundle analysis](tokenizer-evidence/byte-context-transform-001/first-bundle-analysis001/analysis.md)
is complete using immutable logs/reports only. Tanh's late training loss remains
higher, its H200 score is worse than its final online score, and two saved-weight
windows have mean local tanh derivatives near.15–.16. These are limited,
descriptive findings, not causal evidence for an initialization or EMA change.
Primary literature and contrary applicability limits are recorded. Actual
training consumed84.35% of the first three guarded runs; scheduled online scoring
consumed10.28%. No further model experiment is selected by this analysis.

The public [RMSNorm inference repair](FP32_RMSNORM_INFERENCE_REPAIR.md) remains
prepared but unqualified and unadopted. After the panel/review release the shared
lock, execute its existing bound plan e81386f597509df3e3929102b3c6e04cd486aad2ec6d52c55b4ffdf89f96b4bf.
The byte incumbent is unchanged and the autonomous goal remains active.
All sections below are historical snapshots, not current process state.

# Current work: byte-context quality recovery active

This continuation completed the first baseline training, diagnosed a genuine
runtime inference defect, verified the correction on the full original split,
and resumed the original unattempted comparisons. No model has been adopted.

Original controller54382 exited1 after slot00 finished1,200 updates. Mean
CPU/MPS error1.84e-5 passed3e-5, but maximum target error.003434 failed.0003.
All original weights/logs/reports remain immutable. Two new saved-weight
diagnostics isolated the error to RMSNorm in block0, not byte composition or
final LayerNorm. The installed PyTorch2.14.1 MPS no-grad fused RMSNorm defaults
to double epsilon; its training/composite path uses the intended FP32 epsilon.
The scoped scorer enables forward graphs, detaches losses, and calls no backward.

Full-split recovery passed the ORIGINAL limits: mean error2.126655e-5 and maximum
error.0002725124. CPU target scores are bit-identical, weights/gradient slots/RNG
unchanged, zero new training. See
[recovery decision](tokenizer-evidence/byte-context-transform-001/recovery-decision.md).
The completed baseline is imported into quality-recovery001, plan SHA
badc7b0daa9ff8455444d08fc5b7dcc6717033122077500253a66ae86cefd3cd.
Its original terminal sampler digest was not published before failure; a clearly
labeled expected state is derived from the frozen seed/call count, not claimed
as recovered runtime state. The independent reviewer checks this limitation.

A second, narrow resource-accounting defect was caught during slot01: v2 assigned
ORIGINAL_WEIGHTS to the new folder, omitting old weights and double-counting new
ones. Its source stays frozen. A [bound resource wrapper](tokenizer-evidence/byte-context-transform-001/resource-recovery-decision.md)
corrects that constant for remaining workers. The current scientific worker was
not interrupted. Worker61923 completed successfully, and wrapper PID63565/exec99662
terminated only the obsolete stopped dispatcher61922. The wrapper now runs
original slots02..08 with the corrected resource roots. Resource-plan SHA
7092e2d46b9e9991ee05d91f26741c201b12b02c883ca4e3728f6df43a9da2e2.

At this handoff, slots00 and01 are accepted. The first linear control finished
1,200 updates and scored H200 CE4.6439894544 versus baseline4.5504804177
(lower is better), a single-seed difference of +.0935090367. This is a partial
result, not the full-panel conclusion. Its CPU/MPS mean error2.8808344e-5 and
maximum target error.0002630055 pass the unchanged limits. Its observed sampler
digest matches the baseline's explicitly source/count-derived expected state.
The completed slot01 resource receipt verifies274,028,002 retained bytes, well
below the1.5GiB cap; the original defective guard counter remains preserved.

Slot02 tanh worker65917 is active (160 updates observed). Poll exec99662
and actual records; dispatcher61922 has been superseded and must not be resumed.
Do not restart from an observation timeout. After completion, run
`PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m research.byte_context_resource_recovery_v1 review`.
The three arms, roles, .0025 LR,1,200 updates,H200, strict .02 comparisons and
statistical/numerical tolerances remain fixed. No training has been repeated.
The full18-weight cap1.5GiB, reserve8GiB plus2GiB headroom and10-hour allowance
remain; original elapsed costs are carried. No storage blocker exists.

The broader autonomous goal remains active. The public incumbent inference path
has not yet been changed. A separately scoped epsilon repair and bounded
qualification are now source-prepared: [repair scope](FP32_RMSNORM_INFERENCE_REPAIR.md).
The prospective plan is `byte-context-transform-001/inference-epsilon001/plan.json`,
SHA e81386f597509df3e3929102b3c6e04cd486aad2ec6d52c55b4ffdf89f96b4bf.
No runtime check has executed and the repair is not adopted. After the quality
panel and independent review finish, use the free shared lock to run the bound
check on the new baseline weights. It covers full-split scoring and B1/B8 cached
decoding/reset with unchanged tolerances, no training, and a ten-minute cap.
Historical studies and failed fixtures remain closed. Previous sections below
are historical, not the current worker state.

# Current work: fresh byte-context transform comparison running

The previous goal turn made progress by completing and preserving the sharing
geometry analysis. The new evidence review selected a bounded context-side
transformation question on the adopted byte model; see
[protocol](BYTE_CONTEXT_TRANSFORM_PROTOCOL.md) and
[admission](tokenizer-evidence/byte-context-transform-001/quality-admission.md).

The 32-update CPU/MPS pilot passed in 55.675 seconds. The baseline's gradients,
updates, optimizer state and EMA match the official v6 baseline exactly on each
backend. Linear/tanh analytic gradients and serialized-weight checks passed.
This is implementation evidence, not a quality result.

A fresh fixed .0025 panel is now active: byte, linear, tanh across three paired
bundles, 1,200 updates each, rotated order, unchanged strict gain against both
controls. Plan SHA e5d26b5af0cf473f62abe4d25b0555127a52c5531b38b9a7b3faf46f0517586c.
Controller exec session54382/PID57142 started slot00 worker57144; 45 updates were
observed before this handoff. No quality trajectory was accepted at that time.
Use live session and per-slot records to determine subsequent status. No restart
based on an observation timeout. No numerical or scientific failure has occurred.

Conservative full forecast: 18,838 seconds (~5.23 hours), bounded by 10 hours total
and one hour per trajectory. Retained output cap1.5GiB plus reserve8GiB and extra
headroom2GiB fits the measured13.73GB free. No capacity recovery is needed.
The independent reviewer is frozen before outcomes. After all nine complete,
run it against the exact plan SHA; only then consider acceptance and consumer
qualification. The byte incumbent remains unchanged; no new model is adopted.

The autonomous goal remains active. Do not repeat the closed sharing experiment,
reopen old failed projection fixtures, or pool their scores into this panel.
Historical sections below retain their original evidence.

# Current result: shared-table comparison completed without adoption

Allnine fresh output-anchored MLA table-sharing trajectories and the independent
saved-result review are complete. Shared mean CE4.681936 is nearly native4.683425,
but worse than matched4.665970 on every seed. The quality gates fail. The byte
bulk-hash-adoption001 incumbent remains unchanged. No numerical worker is active.

[Complete results](OUTPUT_ANCHORED_TYING_RESULTS.md) and
[closure](tokenizer-evidence/output-anchored-tying-001/closure.json) bind all18retained
weight files, fixed protocol, independent statistics and saved-loss frequency
analysis. Targets seen1–9times in training contribute67.9% of the net sharing
penalty; this is exploratory evidence, not a causal or subgroup significance claim.

The [accepted-weight geometry analysis](tokenizer-evidence/shared-table-geometry-001/analysis.md)
is complete: 1.942 seconds, no training or scoring. Shared vectors are closer to
matched output vectors, but rare targets do not show the largest role mismatch.
This weakens a simple rare-specific distortion explanation and does not justify
an automatic adapter. Next examine context-side transformation applicability to
the byte incumbent using original mixture evidence and contrary primary literature.
The [projection evidence review](SHARED_TABLE_PROJECTION_EVIDENCE.md) retains
historical closed-study restrictions. No new candidate or quality panel is selected.
Do not replay or rescue the completed sharing comparison.

The autonomous research goal remains active. Historical sections below retain
their original evidence; current status is in TOKENIZER_RESUME_STATE.json.

# Active research: byte-input quality improvement adopted

The authorized autonomous research goal remains active. The byte-input quality
panel is now closed: all 18 trajectories and 21,600 updates are accepted, the
independent complete-panel review passed, and the fixed representative checkpoint
loaded successfully through the plan-derived consumer on MPS. No worker is
currently running. The original failed controller remains closed failed.

## Current result and adoption

[Scientific adoption recipe](tokenizer-evidence/byte-composition-boundary-001/adoption001/recipe.json)
and [closure](tokenizer-evidence/byte-composition-boundary-001/adoption001/closure.json)
bind the exact model, plan, checkpoint consumer, first fresh raw-byte representative
(slot10), source inventory, independent review and loading integration. The
representative was specified before final outcomes and was not selected by loss.
Historical incumbent recipes and all frozen source namespaces remain unchanged.

| Fresh bundle | Native CE | Real-byte CE | Code-control CE | Bytes minus native | Bytes minus codes |
|---|---:|---:|---:|---:|---:|
| fresh0 | 4.673415 | 4.573299 | 4.641407 | -0.100116 | -0.068108 |
| fresh1 | 4.693385 | 4.583503 | 4.675757 | -0.109882 | -0.092254 |
| fresh2 | 4.673116 | 4.584103 | 4.661347 | -0.089013 | -0.077244 |

Mean differences are -0.099670086120078 nat/token versus native and
-0.07920209240912808 versus codes. Every fresh seed difference is negative;
all three frozen conditional-text bootstrap upper bounds are negative for both
contrasts. The original primary threshold of -0.02 is passed. The independent
review reproduced the saved statistics within the unchanged 1e-12 tolerance.
These findings are conditional on three trained bundles and reused validation
text, not population-seed confidence or holdout evidence. The code control does
not isolate morphology or match all n-gram sharing statistics.

The [final review](tokenizer-evidence/byte-composition-boundary-001/recovery-final-review001/review.json)
finished in 4.1218 seconds, peak RSS65,273,856 bytes, under its original 300-second
and 1GiB limits. It verified all original/recovery acceptance routes, the 36
retained checkpoint hashes, source/asset bindings, selection and fixed gates.
The [actual consumer integration](tokenizer-evidence/byte-composition-boundary-001/checkpoint-consumer-runtime001/result.json)
loaded all105 parameter tensors (16,351,859 parameters) on MPS in3.3347 seconds;
its guard finished in4.3377 seconds, peak RSS716,357,632 bytes. CPU/MPS RNG was
unchanged. It performed zero training updates and zero new quality evaluations.
Accepted model, resume, averaging, inference and cache proofs were reused.

[Measured costs](tokenizer-evidence/byte-composition-boundary-001/cost-analysis002/analysis.md)
describe the original pre-optimization implementation: first-fresh raw-byte
training updates took4.1538x native;
cached32-token decoding takes1.9010x at batch1 and1.9921x at batch8.
The model has42.61% fewer trainable parameters, but no speed, memory or equal-time
quality advantage is claimed. Timing scope and variability remain in that record.

## Preserved execution failure and successful recovery

Original quality001 session22743/PID17021 failed after slot14 completed all1,200
updates: the output inventory raced checkpoint temporary-file publication and
raised FileNotFoundError. Its exact historical disappearing path is unproven;
a separate deterministic fixture reproduced the race. Original source, logs,
RAW receipt and failed guard remain immutable, with its missing final RSS sample
explicitly recorded. See [recovery evidence](tokenizer-evidence/byte-composition-boundary-001/guard-recovery001/README.md).

The separate frozen recovery plan restored slot14's verified saved RAW and
completed the identical post-training suffix with zero new updates. Only unused
original slots15–17 trained. Recovery controller session72451/PID79938 exited0
with complete_recovered_quality_panel_pending_independent_review. Its final
slot17 guard finished in3846.4259 seconds with910,147,584-byte peak RSS; all1,200
updates,41 averages and34,278 validation targets passed. Its sampler identities
match both fresh2 controls. All7,199,511,764 retained weight bytes remain present.
The subsequent independent review and additive adoption closure complete this
panel without rewriting the original failure or its scientific rules.

## Current implementation improvement and next executable action

The [hash-only recipe](tokenizer-evidence/byte-composition-boundary-001/direct-hash-adoption001/recipe.json)
was the first adopted equivalent implementation; subsequent implementations
and the current bulk-hash recipe are recorded below.
It keeps the same mathematical model, parameter count, first-fresh checkpoint,
artifact formats and quality evidence. The loader is
`research.byte_input_checkpoint_consumer_v2.load`, with the recipe's implementation,
plan and acceptance receipts under the shared lock and CPU/MPS resource guards.
Original sources, scientific gates and prior recipes remain unchanged.

[Integrated differential evidence](tokenizer-evidence/byte-composition-boundary-001/direct-hash-integration-runtime001/README.md)
matched CPU/MPS cached B1/B8 logits, B8T128 outputs and all105 gradients, plus
initial/step1/step2 complete captured and validated training-context states.
The two updates include optimizer allocation and the first averaging snapshot.
Eight numerical updates total took place; no quality evaluation or weight archive
was added. Fresh sampler roles62100600001/2 are now reserved/closed, not reusable.
The integration guard finished in62.3996seconds with2,888,744,960bytes peak RSS.
A separate public-loader call succeeded on MPS in1.2761seconds with unchanged RNG.

CPU's two-step total improved1.777x and MPS's2.576x. On MPS, the original took
3.1601/3.2952seconds and optimized1.3449/1.1607seconds for steps1/2. This is
short-fixture timing, not sustained throughput or an equal-time quality result.
The earlier full-panel cost record applies to the previous implementation.
[Real-buffer evidence002](tokenizer-evidence/byte-composition-boundary-001/direct-hash-runtime002/README.md)
provides the separate14-buffer CPU/MPS exact hash and mutation checks; its
Boolean-mutation harness failure001 remains preserved.

The exact [source map](tokenizer-evidence/byte-composition-boundary-001/direct-hash-integration-source001/source-map.json)
transfers unchanged algorithms to model-v4/state-v3/training-v3/inference-v2/cache-v2.
It is a historical preparation record; subsequent runtime and adoption evidence
above supplies qualification. No exact-type guards were weakened.

The subsequent [padding comparison002](tokenizer-evidence/byte-composition-boundary-001/padding-runtime002/README.md)
is closed not adopted:14/16 fixed cases failed exact numerical equality. Maximum
output difference4.84e-8 and gradient difference9.06e-6 remain numerical results,
not harness defects. Component timing improved roughly10x, which does not justify
transparent adoption. The original singleton-serialization failure001 is preserved;
recovery reused both completed negative CPU records byte-for-byte and kept all
cases/criteria fixed. No quality or optimizer update was performed by that check.

A [warmed-step profile](tokenizer-evidence/byte-composition-boundary-001/hash-baseline-profile001/README.md)
of the adopted baseline used three numerical updates, fresh sampler roles
62100700001/2 and no quality evaluation. One profiled MPS step took1.1216seconds.
Six model.validate calls used0.3943seconds cumulative, including0.1966seconds
nested hashing; these overlapping times must not be added. Repeated scalar
validation is a plausible remaining overhead. Data checks used only0.0153seconds.
The profile does not attribute GPU kernels or establish sustained throughput.

The [batched-finite implementation](tokenizer-evidence/byte-composition-boundary-001/finite-batch-adoption001/README.md)
was adopted before the optimizer-state change below. Its historical recipe is
[finite-batch-adoption001/recipe.json](tokenizer-evidence/byte-composition-boundary-001/finite-batch-adoption001/recipe.json),
using checkpoint_consumer_v3 with bound implementation/plan/acceptance receipts.
It retains the same selected weights, mathematical model and scientific evidence.
Sources are model-v5/state-v4/training-v4/inference-v3/cache-v3. All finite/content/
structural checks remain, with105 parameter predicates aggregated before one host
Boolean read. The prior hash-only recipe remains immutable historical evidence.

The fixed validator comparison passed20 invalid-state rejections/restorations.
MPS validation median fell68.080ms to46.628ms; CPU timing was essentially unchanged.
The integrated exact output/gradient/cache/state comparison passed, but its two
MPS updates showed no overall gain (0.9944x). That contrary result remains preserved.

A separate prospective warmed/interleaved comparison completed20 paired updates,
four warmups plus16 measurements, with fresh roles62100900001/2. Median candidate/
reference latency0.919603 (8.04% lower),15/16 paired wins, and phase medians below1
passed the predeclared5%/12wins/no-phase-regression gate. All20 paired histories
and full final parameter/optimizer/averaging/sampler/scheduler/RNG states matched
exactly. This is local short-recipe timing, not a confidence interval or1200-step
throughput claim. Integration roles62100800001/2 and timing roles62100900001/2 are
reserved/closed. The public loader succeeded on MPS in1.3249seconds.

The preceding [optimizer-state implementation](tokenizer-evidence/byte-composition-boundary-001/optimizer-state-adoption001/README.md)
is adopted via [optimizer-state-adoption001/recipe.json](tokenizer-evidence/byte-composition-boundary-001/optimizer-state-adoption001/recipe.json).
It keeps model-v5, cache-v3 and selected-checkpoint loader-v3 unchanged and uses
optimizer-v6/state-v5/training-v5/inference-v4 for training and export. The recipe
separately binds these training sources; its existing model loader receipt remains
valid. All original state checks remain; only device Boolean aggregation changes.

The read-only saved-state comparison passed 40 fixed invalid-state rejection
transactions before mutation. MPS state validation fell from 51.963 to 13.139 ms,
and complete optimizer validation from 99.584 to 58.066 ms. CPU had no meaningful
gain. Two device-name harness failures remain closed and preserved; completed
CPU evidence was copied unchanged and only the unfinished MPS work continued.

A prospectively frozen training comparison passed exact CPU update-one/update-two
states and restore-at-one continuation to two. Twenty MPS paired updates matched
all histories and complete final states exactly. Four warmup pairs and 16 measured
pairs yielded **17.73% lower median latency**, 15/16 wins, and improvements in both
averaging phases. All prospective timing gates passed. CPU roles 62101000001/2 and
MPS roles 62101100001/2 are closed/reserved. This is short local timing, not a
1,200-step speed claim or a new quality result. Separate speedups must not be multiplied.

The new training/export consumers also restored the original accepted RAW exactly,
reproduced its H200 export exactly and loaded its H200 parameters exactly, with
no new updates or weight files. Total numerical work was 45 updates; no quality
panel or holdout was scored. Full details, costs and failure lineage are linked above.

The refreshed [profile](tokenizer-evidence/byte-composition-boundary-001/optimizer-baseline-profile001/README.md)
measured 0.206869 s cumulative buffer hashing and 0.195348 s deterministic byte-
embedding backward in one early warmed update. Its three-update roles 62101200001/2
are closed. Profile times overlap, include device waits and are not kernel timings.

The current [bulk-hash implementation](tokenizer-evidence/byte-composition-boundary-001/bulk-hash-adoption001/README.md)
is adopted via [bulk-hash-adoption001/recipe.json](tokenizer-evidence/byte-composition-boundary-001/bulk-hash-adoption001/recipe.json).
It uses model-v6 with the existing optimizer improvements, state-v6/training-v6/
inference-v5/cache-v4 and checkpoint_consumer_v4. Complete-buffer CPU transfer and
the buffer protocol replace small transfer chunks and Python byte copies, while
preserving every content check and original SHA256 digest.

All 72 real-buffer mutation checks passed. Complete MPS model validation fell
from 46.693 to 33.494 ms. The integrated CPU/MPS cache, CPU update/resume and MPS
20-pair final-state comparisons were exact. The warmed timing gate passed with
**8.62% lower median latency**, 16/16 wins and no averaging-phase regression.
CPU roles 62101300001/2 and MPS roles 62101400001/2 are closed/reserved. This remains
a short local comparison; combined improvement versus native is unmeasured.

The public loader succeeded and original RAW capture/H200 export/load remained
exact. This continuation used 48 numerical updates including profiling, about
82 guarded seconds, zero quality evaluations and zero new weight archives.
The [active-row gradient prototype](tokenizer-evidence/byte-composition-boundary-001/active-byte-runtime001/README.md)
is closed not adopted. Existing-data analysis found54 active byte rows in the
first1024tokens and median58 across261 complete nonoverlapping windows. The new
comparison kept every2048-position chunk and its order, and tested seven fixed
cases on CPU/MPS. All forwards and CPU gradients were exact. MPS gradients differed
in the one-row full-bound case (14coordinates,max1.67847e-4) and four-row chunk-edge
case (48coordinates,max2.28882e-5). Real-input cases, all-row control and singleton
were exact. A59.75% lower representative primitive latency does not override the
numerical failure. The adopted bulk-hash model remains unchanged. No training
update, quality score or weight archive was produced by this check.

A separate [shape/stride diagnostic](tokenizer-evidence/byte-composition-boundary-001/active-byte-shape-diagnostic001/README.md)
found K2048 differences at output widths1–14, unchanged by retaining the original
stride. A genuinely new guarded primitive retained original arithmetic for those
small shapes and all partial chunks;281 primitive comparisons passed exactly.
The [full-model comparison](tokenizer-evidence/byte-composition-boundary-001/active-byte-training001/README.md)
then passed all105 gradients, cache, CPU resume and MPS20-pair state equality, but
its2.96% median latency reduction missed the frozen5% gate. It is closed not adopted;
no public loader or artifact qualification was attempted. The adopted bulk-hash
implementation remains active. All new sampler roles are closed/reserved.

Isolated primitive median saving was8.540ms. The old0.195s cProfile frame included
device dependency waits and instrumentation and is not isolated kernel cost.
Do not use the component percentage to overstate full-training gains.

No worker is active. The user's input/output embedding discussion reveals that the
42.61% parameter saving is relative to the untied native baseline. A tied version
of that body would have15,626,624parameters versus16,351,859for the byte model;
this is arithmetic, not quality evidence. Next inspect historical tying experiments
and restrictions, then use the evidence-grounded research skill and primary sources
to decide whether a genuinely new comparison is justified. Preserve closed studies
and scientific gates; do not replay old tests or change unfavorable timing criteria.

The remaining sections preserve earlier prerequisite and execution history;
statements about pending work below describe those historical stages. The current
state and next action above supersede them.

## Accepted integrated recovery

The [independent state review003](tokenizer-evidence/byte-composition-boundary-001/state-review003/review.json)
accepted all six B8T128/V50257 native/raw-byte/nonlexical cases on CPU and MPS,
234 rejection transactions, complete tensor digests covering 1,126,737,138
coordinates, and all 12 checkpoint archives at acceptance. Those obsolete
numerical RAW payloads were subsequently retired in the exact audit below;
original evidence and accepted review receipts remain. The proof verifies empty
optimizer state, complete capture/read/restore, changed LR and training mode,
exact CPU/MPS RNG draws, and a genuine next forward/backward/update after restore.
All logits, losses, gradients, parameters, optimizer state and RNG matched
uninterrupted execution bit-for-bit. Full byte-embedding gradient values are
also saved, allowing a future mismatch to be measured directly.

The worker finished in 237.268 seconds. Peak RSS was 3,394,977,792 bytes under
4 GiB; total recorded output was 2,911,379,673 bytes under 8 GiB. The sampled
6 GiB MPS guard passed; its final 1,375,420,416-byte observation is not a peak.
Independent review took 1.940 seconds and 46,727,168 bytes peak RSS.

This qualifies the finite numerical checkpoint scope for model-v3/state-v2.
Its mathematical predecessors are the preserved model-v2 independent finite
integration proof and the independently checked fixed-order embedding primitive
with an exact source-composition review. It is not a fresh independent oracle
for the entire model, a complete training-resume proof, or quality evidence.

## Preserved failures

Checkpoint001 failed constructing a raw-byte alias rejection: the smallest
composer parameter had no same-shaped peer. Checkpoint002 fixed only that
fixture with fresh roles, then exposed nonrepeatable MPS byte-embedding gradients.
Restored state/RNG/forward matched, but `composer.byte_weight` next gradients did
not. It completed all CPU cases and native MPS; nonlexical MPS was not reached.
Both attempts remain closed failed, with original sources, plans, failure logs
and diagnostic snapshots preserved. Their obsolete RAW payloads were later
retired under current human authorization; see the exact audit below. No thresholds or unfavorable seeds were changed into passing results.

The [isolated diagnosis](tokenizer-evidence/byte-composition-boundary-001/embedding-accumulation-diagnosis002/result.json)
reproduced native MPS accumulation differences for a mixed-magnitude upstream.
An exact-sum control remained repeatable. Fixed-order grouped reduction passed
independent float64-error and exact-repeatability checks. The accepted
[primitive/source review](tokenizer-evidence/byte-composition-boundary-001/embedding-recovery-review001/review.json)
binds the new embedding, composer-v2, model-v3 and state-v2 sources. Fresh003
then passed the unchanged complete checkpoint requirements. All original
entries, roles and evidence remain closed; never replay them.

## Accepted complete training context

The [training-context proof](tokenizer-evidence/byte-composition-boundary-001/training-math001/result.json)
and [independent saved review](tokenizer-evidence/byte-composition-boundary-001/training-review001/review.json)
accepted all six native/raw-byte/nonlexical CPU/MPS cases: 36 actual updates and
222 invalid transactions. Each four-step trajectory saves at step 2, then matches
uninterrupted and resumed steps 3–4 exactly, including private sampled IDs, global
RNG, clipping/exposure history, schedule, optimizer and averaging state. All
malformed payload and live-context checks preserve state on rejection.

The independent reviewer reconstructs final EMA from saved steps 2, 3 and 4 in
FP64 across 122,392,268 coordinates. Maximum observed absolute error is
3.572e-7, within the unchanged 4e-6 + 4e-5*abs(reference) gate. This uses the
original real train/validation identity and paired sampling, without holdout
scoring. It qualifies the four-update fixture, not 1200-step quality, inference
export, cached scoring or the campaign worker.

The worker took 387.480 seconds, peaked at 3,562,684,416 bytes RSS, and retained
4,393,040,580 recorded output bytes, within its frozen 4 GiB RSS / 7 GiB output
envelope. The sampled 6 GiB MPS guard passed; its final observation is not a peak.
Admission took 81.148 seconds. Independent review took 6.443 seconds and peaked
at 1,246,756,864 bytes RSS. The 222 rejection checks are not training runs.

[Acceptance](tokenizer-evidence/byte-composition-boundary-001/training-acceptance001/acceptance.json)
and the [local payload manifest](tokenizer-evidence/byte-composition-boundary-001/training-acceptance001/local-payload-manifest.json)
bind the original evidence. All entries and role assignments are closed, with
no replay. Reuse accepted checks; do not expand validation without a concrete
consumer requirement or demonstrated defect. Original failures remain intact.

## Accepted inference consumers and resource pilots

[inference-review001](tokenizer-evidence/byte-composition-boundary-001/inference-review001/review.json)
accepted all six existing-parent CPU/MPS exports and full-validation consumers.
Every parent EMA weight matched exactly (122,392,268 coordinates), loaded logits
matched the accepted direct model, and all205,668validation targets were covered.
Independent FP64 CE checked2,148targets from first/block-boundary/tail windows;
all target aggregation and byte accounting were independently checked. The run
used zero new training updates, took100.092s, and peaked at1,414,561,792B RSS.
Saved review took4.530s. Scope does not make the four-step scores quality evidence.

[cache-review001](tokenizer-evidence/byte-composition-boundary-001/cache-review001/review.json)
checks the complete saved coverage/gate/replay receipts:12 CPU/MPS B1/B8 cases,
1584chunk comparisons covering694,752,768logit coordinates, full128 context,
exact reset replay and84invalid transactions. The frozen worker compared cached
outputs with accepted full model-v3 outputs under unchanged native cache gates.
Maximum absolute difference1.3113e-6; maximum normalized gate ratio.026215.
The run took52.502s and peaked at1,703,641,088B RSS. The saved review is not a new
independent model-math oracle. Sources remain frozen and closed.

[Resource pilot002](tokenizer-evidence/byte-composition-boundary-001/pilot-math002/result.json)
and [saved review003](tokenizer-evidence/byte-composition-boundary-001/pilot-review003/review.json)
accepted all three actual-corpus20-update MPS arms, complete RAW/H200 ancestry,
20,480training targets per arm, eight EMA snapshots, full validation and cached
correctness. The complete run took198.573s and peaked at1,202,176,000B RSS.
Its final MPS guard observation is not a peak. Per-update observations reached
3,623,665,664B MPS driver allocation, below6GiB. Saved review took1.990s.

| Arm | Mean update seconds | Maximum update seconds | Full validation seconds |
|---|---:|---:|---:|
| Native |0.753146|0.861219|3.813279|
| Raw bytes |3.124641|3.329712|8.499450|
| Nonlexical codes |3.139232|3.373064|8.402696|

These are implementation-inclusive short-pilot measurements, not steady-state
1200-step guarantees or a quality ranking. Byte composition is about4.1x slower
per observed update despite fewer parameters. Use this complete measured cost
in the next scientific design; do not choose arms/rates/criteria from pilot CE.

Original pilot001 completed20native updates but failed before weight publication:
its driver passed `(path,payload)` to `training.write_new(context,path)`. Original
source/roles/logs/STOP remain closed. New worker-v2 corrected only the call and
removed redundant capture, then used distinct fresh roles for all three pilots.
A later saved-only reviewer failed on spelling `mps` versus canonical `mps:0`;
reviewer-v3 corrected that metadata expectation without replaying any training.
[Recovery records](tokenizer-evidence/byte-composition-boundary-001/pilot-recovery001/recovery.json)
and [reviewer diagnosis](tokenizer-evidence/byte-composition-boundary-001/pilot-review-recovery001/diagnosis.json)
preserve both failures. A separate pre-admission wrong CLI hash is recorded in
`pilot-launch-correction001.json`; it caused no attempt or role consumption.

## Frozen scientific comparison (now completed)

The original [quality protocol](tokenizer-evidence/byte-composition-boundary-001/quality001/protocol.md)
and machine-readable plan freeze18 trajectories: nine balanced learning-rate
screens followed by three fresh paired bundles across native, raw-byte and
nonlexical-code arms. Each trajectory runs1200updates and retains final RAW and
H200 weights. All three rates are selected before confirmation begins. Raw bytes
must improve fresh mean CE by more than.02nat/token, improve every seed and pass
all predeclared conditional-text bounds. Lexical-content support is a separate
contrast against nonlexical codes. Holdout remains unencoded and unscored.

[Admission](tokenizer-evidence/byte-composition-boundary-001/quality001/admission/review.json)
completed in87.122s, accepting source/resource bindings, unused-role census and
synthetic selection/statistics checks. It did not train or establish quality.
The actual controller was then launched once:PID17021, exec session22743;
initial native screen slot00 PID17022, LR.000625. Real updates and the first
non-mutating diagnostic evaluation were observed. These PIDs identify the launch,
not a permanent assertion of liveness; query the session/process and latest slot
before acting. Do not restart from a stale progress index.

The complete retained-output cap is10GiB, initial free requirement20GiB including
8GiB reserve and2GiB extra headroom. Pilot-based conservative runtime forecast is
22.66hours; hard controller budget36h20m, with7200s per slot,4GiB RSS and6GiB
sampled MPS allocation. These are estimates and guardrails, not guarantees.
The source implements the actual1200-step experiment; accepted prerequisites
are reused rather than replayed.

## Completed capacity recovery and retained evidence

[Exact recovery](tokenizer-evidence/byte-composition-boundary-001/capacity-recovery001/result.json)
retired38 obsolete numerical RAW payloads totaling9,992,314,882bytes:14 failed
checkpoint fixtures,12 accepted state fixtures and12 four-step training fixtures.
Dependencies were closed; shared lock, process/consumer checks, exact size/SHA
allowlists and durable deletion journals preceded deletion. Current human
permission supersedes older self-authored retain-all/failed-payload labels only
for these audited files. Original source/log/config/metrics/receipts and oracle
arrays, selected/fresh scientific weights and latest20-step pilot parents remain.
15,293 evidence files were verified retained. Historical manifests describe
original acceptance; deleted-payload saved reviews must not be replayed.
Free space rose to22,613,397,504bytes and the20GiB requirement was satisfied.
The storage blocker is resolved; do not repeat broad unchanged cleanup scans.

## Historical execution progression before completed adoption

A minimal [direct-byte-hash candidate](byte_input_direct_hash_v1.py) now exists
outside the frozen sources. Its only overridden method uses the already
CPU-measured serialization substitution; an AST check confirms the exact scope.
It has no runtime or training-consumer qualification. The [integration note](tokenizer-evidence/byte-composition-boundary-001/serialization-integration001/README.md)
records the inherited-source binding and the focused CPU/MPS comparison needed
when resources are free. This adds no training or qualification run to the
active panel and does not claim an integrated speedup.

The first rate's three screens (slots00–02, LR.000625) are accepted. Full
validation CE is4.896433671 for native,5.039635859 for real bytes and5.098909072
for nonlexical codes. Real bytes improve over the code control by.059273213
nat/token but lose to native by.143202188. Those initial screens did not select
a rate or establish a confirmed gain. The [fixed-bin saved-loss comparison](tokenizer-evidence/byte-composition-boundary-001/screen-diagnostic002/analysis.md)
shows that both composition models improve after unseen inputs, with an
additional real-byte advantage. This exploratory observation is not a new
acceptance criterion or a causal proof of lexical sharing.

The [training-only control-structure analysis](tokenizer-evidence/byte-composition-boundary-001/control-structure002/analysis.md)
quantifies an existing interpretation limit: 97.644% of training-weighted
three-byte content fragments recur across token types for real bytes, versus
0.682% for the screening code control. Byte frequency concentration also differs.
These measurements do not alter the control or imply learned filter behavior;
they limit causal explanations of any eventual code-relative gain. The analysis
took 2.952 seconds without a model or GPU. Its original packed-byte reader failure
and exact one-expression correction are preserved separately.

Slot02 completed all1200updates,41averaging snapshots and1,228,800sampled
targets; its finished guard recorded3795.112seconds. Its acceptance binds the
full validation, inference and saved-state review. The controller automatically
started slot03, real bytes at LR.00125, PID69790. That run has now completed
and is accepted: all1200updates,41averaging snapshots,1,228,800sampled targets
and34,278validation targets. Its full validation CE is4.6377457941683184,
BPB2.056278280301206. The finished guard recorded3766.939seconds and911,147,008
bytes peak RSS. The saved RAW/H200 pair totals318,299,358bytes.

This is a reduction of.4018900651005106nat/token against the same real-byte
arm at LR.000625, with matching final train/validation sampler hashes. It is
within-arm screening evidence, not a selected-rate or fresh-confirmation gain
over native. The subsequent [slot04 control](tokenizer-evidence/byte-composition-boundary-001/quality001/slots/04/acceptance.json)
also completed all 1,200 updates, 41 averaging snapshots and 34,278 validation targets.
Its full validation CE is 4.723060961468433 and BPB 2.0941052189230005; the guard
finished in 3,793.685 seconds with 916,193,280 bytes peak RSS. Its retained RAW/H200
pair totals 318,299,422 bytes. All saved-result acceptance checks passed.

At LR 0.00125, raw-minus-control CE is -0.08531516730011468 nat/token with matching
final train/validation sampler hashes. The subsequent [slot05 native baseline](tokenizer-evidence/byte-composition-boundary-001/quality001/slots/05/acceptance.json)
is now accepted after all 1,200 updates, 41 averaging snapshots and 34,278 validation
targets. Its full validation CE is 4.695642038340503 and BPB 2.081948249007118.
Its finished guard recorded 926.216 seconds and 1,294,925,824 bytes peak RSS;
the retained RAW/H200 pair totals 563,319,826 bytes.

The complete screening panel gives the following full validation scores:

| Learning rate | Native | Real bytes | Nonlexical codes | Bytes minus native |
|---|---:|---:|---:|---:|
| 0.000625 | 4.896434 | 5.039636 | 5.098909 | +0.143202 |
| 0.00125 | 4.695642 | 4.637746 | 4.723061 | -0.057896 |
| 0.0025 | 4.771982 | 4.569071 | 4.662836 | -0.202910 |

At LR 0.00125, raw-minus-native CE is -0.0578962441721842 nat/token. Final
train/validation sampler hashes match across all six completed screens. The
ranking reversed between the first two rates, illustrating learning-rate
sensitivity. These rates use the same screening bundle, not independent fresh
confirmations. All rates are now selected below; no confirmed quality gain is established.

The [slot06 control at LR 0.0025](tokenizer-evidence/byte-composition-boundary-001/quality001/slots/06/acceptance.json)
is also accepted: all 1,200 updates, 41 averaging snapshots and 34,278 validation
targets. Full validation CE is 4.662835553613667 and BPB 2.0674025483605565.
Its finished guard recorded 3,835.412 seconds and 909,885,440 bytes peak RSS;
the retained RAW/H200 pair totals 318,299,422 bytes. Final sampler hashes match
the other screens. Its CE is 0.06022540785476593 lower than the control's middle-rate
screen. Selection was deferred until all nine screens were complete, as required.

The [slot07 native baseline at LR 0.0025](tokenizer-evidence/byte-composition-boundary-001/quality001/slots/07/acceptance.json)
is accepted after all 1,200 updates, 41 averaging snapshots and 34,278 validation
targets. Full validation CE is 4.771981672719794 and BPB 2.1157956263898243.
Its finished guard recorded 931.604 seconds and 1,296,924,672 bytes peak RSS;
the retained RAW/H200 pair totals 563,319,826 bytes. Final sampler hashes match
the other screens. Its CE is 0.07633963437929125 higher than native at the middle
rate. Preserve this measured unfavorable rate comparison; no case or gate changes.

The [slot08 real-byte screen at LR 0.0025](tokenizer-evidence/byte-composition-boundary-001/quality001/slots/08/acceptance.json)
is accepted after all 1,200 updates, 41 averaging snapshots and 34,278 validation
targets. Full validation CE is 4.569071185341896 and BPB 2.0258294129407908.
Its finished guard recorded 3,827.581 seconds and 915,505,152 bytes peak RSS;
the retained RAW/H200 pair totals 318,299,358 bytes, with both hashes verified.
All nine screens have matching final train/validation sampler hashes.

The controller froze the [complete rate selection](tokenizer-evidence/byte-composition-boundary-001/quality001/selection.json)
before starting fresh confirmation: native LR 0.00125, real bytes LR 0.0025,
and nonlexical codes LR 0.0025. Each is the minimum complete validation CE for
that arm under the original rule. At the selected rates, real-byte screening CE
is 0.12657085299860693 lower than native and 0.09376436827177148 lower than the
code control. These are tuning observations from the single screening bundle,
not fresh confirmation, conditional-text inference, or an adoption decision.

The [selected-rate saved-loss analysis](tokenizer-evidence/byte-composition-boundary-001/screen-diagnostic003/analysis.md)
reuses the original frequency/length bins without additional model execution.
Real bytes improve over native after all five preceding-input frequency groups;
unseen inputs account for about 21% of the net screening gain. However, real
bytes still lose on unseen prediction targets (+0.382529 nat/token) and targets
appearing 1–9 times (+0.018928). These selected-screen observations limit claims
about rare-token prediction; they are exploratory, not fresh confirmation or new
acceptance endpoints. The analysis took 0.477 seconds and left the panel unchanged.

The [first fresh native confirmation, slot09](tokenizer-evidence/byte-composition-boundary-001/quality001/slots/09/acceptance.json)
is accepted at the selected LR 0.00125: all 1,200 updates, 41 averaging snapshots,
1,228,800 sampled targets and 34,278 validation targets. Full validation CE is
4.6734148945894 and BPB 2.0720932041303604. The finished guard recorded 927.673
seconds and 1,131,331,584 bytes peak RSS. Both retained RAW/H200 hashes were
verified; the pair totals 563,319,826 bytes. Keep screening scores separate from
the fresh panel; no pooling or replacement of unfavorable comparisons.

The [first fresh real-byte confirmation, slot10](tokenizer-evidence/byte-composition-boundary-001/quality001/slots/10/acceptance.json)
is accepted at selected LR 0.0025: all 1,200 updates, 41 averaging snapshots,
1,228,800 sampled targets and 34,278 validation targets. Full validation CE is
4.573299025850737 and BPB 2.0277039478973724. Its finished guard recorded
3,822.640 seconds and 910,868,480 bytes peak RSS. Both retained RAW/H200 hashes
were verified; the pair totals 318,299,358 bytes.

Fresh0 real-byte-minus-native CE is -0.10011586873866296 nat/token. Final
train/validation sampler hashes and complete validation target identities match.
This is one accepted fresh paired improvement. It does not satisfy the full
three-bundle criterion by itself; the remaining fresh bundles, conditional-text
bounds and final independent review remain pending. No overall quality gain or
adoption decision is established.

The [first fresh code-control confirmation, slot11](tokenizer-evidence/byte-composition-boundary-001/quality001/slots/11/acceptance.json)
is accepted at selected LR 0.0025: all 1,200 updates, 41 averaging snapshots,
1,228,800 sampled targets and 34,278 validation targets. Full validation CE is
4.641407008058075 and BPB 2.057901585013273. Its finished guard recorded
3,840.059 seconds and 911,081,472 bytes peak RSS. Both retained RAW/H200 hashes
were verified; the pair totals 318,299,422 bytes.

The first fresh bundle is now complete, with matching final train/validation
sampler hashes and complete target identities across all three arms:

| Fresh bundle | Native CE | Real-byte CE | Code-control CE | Bytes minus native | Bytes minus codes |
|---|---:|---:|---:|---:|---:|
| fresh0 | 4.673415 | 4.573299 | 4.641407 | -0.100116 | -0.068108 |

Fresh0 real-byte-minus-code CE is -0.06810798220733805 nat/token; code-minus-native
is -0.032007886531324914. Real bytes improve over both comparators in this bundle.
This is one complete paired bundle, not all-seed acceptance, a conditional-text
confidence result, or an isolated causal proof of lexical sharing. Do not pool
screening scores or alter the remaining original comparisons.

The [second fresh real-byte confirmation, slot12](tokenizer-evidence/byte-composition-boundary-001/quality001/slots/12/acceptance.json)
is accepted at selected LR 0.0025 on fresh1: all 1,200 updates, 41 averaging
snapshots, 1,228,800 sampled targets and 34,278 validation targets. Full validation
CE is 4.583503197906659 and BPB 2.03222826171238. Its finished guard recorded
3,849.814 seconds and 921,157,632 bytes peak RSS. Both retained RAW/H200 hashes
were verified; the pair totals 318,299,358 bytes. At its acceptance both paired
comparators were pending. The completed code control is recorded below; the
native baseline remains pending. Do not substitute fresh0 or screening scores
for that comparator or call this a second complete three-way bundle yet.

The [second fresh code-control confirmation, slot13](tokenizer-evidence/byte-composition-boundary-001/quality001/slots/13/acceptance.json)
is accepted at LR 0.0025 on fresh1: all 1,200 updates, 41 averaging snapshots,
1,228,800 sampled targets and 34,278 validation targets. Full validation CE is
4.67575744487748 and BPB 2.0731318413243067. The finished guard recorded
3,825.896 seconds and 910,999,552 bytes peak RSS. Both retained RAW/H200 hashes
were verified; the pair totals 318,299,422 bytes.

Fresh1 real-byte-minus-code CE is -0.0922542469708203 nat/token. Final train and
validation sampler hashes and complete validation target identities match across
slots12/13. This is the second fresh paired improvement over the code control,
not a second complete three-way bundle or full lexical-content acceptance. The
matching native baseline is still pending; do not substitute another bundle's
native score, pool screens, or compute a partial-panel adoption decision.

At that milestone fourteen of 18 trajectories were accepted. The original controller automatically
started slot14, native at LR 0.00125 on fresh1, PID 74101. Both controller and
worker processes and actual updates were observed. The original rotated order
for this bundle is real bytes, code control, then native. Complete the four
remaining original trajectories; recheck the live session and current slot
before acting on recorded process IDs.

The original controller subsequently stopped on slot14's guard failure. Follow
the recovery controller and current action at the top of this handoff; never
restart session22743 or its closed controller. Never replace an unfavorable seed
or alter scientific gates. Neither controller adopts automatically.

A [small final saved-data reviewer](byte_input_quality_final_review_v1.py) is now
prepared outside the frozen sources. Its [preparation note](tokenizer-evidence/byte-composition-boundary-001/final-review-preparation001/README.md)
records subsecond synthetic arithmetic checks and the bounded execution command.
It reuses accepted per-run numerical reviews, independently reconstructs full-panel
selection and bootstrap decisions from saved target losses, and verifies retained
weight/evidence hashes. Run only after successful full-panel termination and shared
lock availability. It has not accepted a panel or changed the incumbent.

The ignored live progress file and quality001 slot/guard/STOP records are the
continuation anchors. Binary checkpoints/oracle arrays remain local; Git contains
sources, plans, metadata, logs and receipts. No quality gain or adoption is
established. The broader autonomous research goal remains active.

## Complete-run cost evidence

[Cost analysis002](tokenizer-evidence/byte-composition-boundary-001/cost-analysis002/analysis.md)
uses the selected screens and first complete fresh bundle separately, verifying
saved acceptance bindings without model execution. In fresh0, real bytes have
4.1538 times native training-update cost, 1.9010/1.9921 times cached next-32-token
decode time at batch1/8, and 3.1369 times batch8 prefill time. Smaller parameters
and archives do not imply lower runtime or sampled training driver allocation.
Five-repeat min/median/max timings and source/evidence hashes are retained;
variable full-forward timings are not discarded. This is descriptive cost
analysis, not an equal-wall-time quality comparison or a new adoption gate.
Continue the original panel and final review, then the already prepared focused
implementation investigation after shared-lock release. No extra training was
launched and no frozen source changed.

## Existing-data cost analysis

[Cost analysis001](tokenizer-evidence/byte-composition-boundary-001/cost-analysis001/analysis.md)
separates measured pilot throughput/storage from static implementation work.
The composed path rehashes52,669,336codebook bytes at least six times per update;
its dense int64 codebook also offsets the FP32 parameter-storage saving in a
parameters-plus-codebook accounting. These are source-derived costs, not a
profiled attribution of the4.15x slowdown. The running equal-exposure comparison
and all accepted requirements remain unchanged. A later direct-byte serialization
profile can target this concrete overhead without another quality sweep.

## First complete scientific screen accepted

[Slot00 acceptance](tokenizer-evidence/byte-composition-boundary-001/quality001/slots/00/acceptance.json)
binds a complete native1200-update run at LR.000625:1,228,800sampled targets,
41H200 snapshots, full34,278-target validation and saved RAW/H200 review.
Validation CE is4.896433671383986nat/token and2.170975007311271bits/byte.
This is one screening observation, not a selected rate, fresh confirmation,
comparison gain or adoption. All nine screens must complete before selection.

Its guard finished in912.551s, peak RSS1,298,251,776bytes; raw/H200 archives total
563,319,826bytes and remain local. Full metadata, curves, validation losses,
inference receipts and saved-review bindings are retained. The controller
accepted the slot and automatically launched raw-byte slot01 at the same rate,
PID22816 under the existing session22743/controller PID17021. Query the live
session and latest slots before acting; these process IDs describe this handoff.

## Serialization overhead measured independently

[CPU serialization analysis001](tokenizer-evidence/byte-composition-boundary-001/serialization-analysis001/analysis.md)
completed in2.381s without model/GPU/checkpoint execution. Replacing Python-byte
lists with direct contiguous NumPy bytes preserved all four finite SHA256 cases
and byte-mutation detection; the52.3MB int64 code-ID shape hashed14.69x faster
by median of three alternating-order CPU trials. This is a component timing,
not an integrated training speedup. It identifies a narrow later optimization;
pinned scientific sources and the ongoing raw-byte run are unchanged. Preserve
full content checks in any later integration; do not infer end-to-end gain or
reopen closed scientific attempts from this result.

## Padding work quantified from existing training data

[Padding analysis001](tokenizer-evidence/byte-composition-boundary-001/padding-analysis001/analysis.md)
reconstructed the exact original267,688training IDs without encoding validation
or holdout and without model/sampler execution. Code lengths including boundaries
average5.3334 and peak at18, versus the fixed130positions processed. Exact
paired-block support gives an expected batch maximum14.0673. Cropping only unused
trailing padding would remove91.90% of the six filter banks' forward multiply-add
count under this distribution, not91.90% of total runtime. No cropped model has
been implemented or qualified; current scientific sources remain unchanged.
This is the second evidence-backed implementation target after direct-byte hashing.

## Second complete scientific screen accepted

[Slot01 acceptance](tokenizer-evidence/byte-composition-boundary-001/quality001/slots/01/acceptance.json)
binds the complete raw-byte1200-update run at LR.000625. Its41H200 snapshots,
1,228,800sampled targets, complete validation, cache/serving receipts and saved
RAW/H200/optimizer review passed. Native and raw-byte final train/validation
sampler hashes match. The guard finished in3708.092s with917,602,304bytes peak
RSS; update time averaged3.027185s, totaling3632.622s. Retained raw/H200 weights
for this slot total318,299,358bytes. The completed run satisfies its limits.

At this one screening rate, raw-byte CE is5.039635859268829nat/token versus
native4.896433671383986, a worse delta of+.143202187884843. Raw-byte BPB is
2.2344678250956105. Preserve this unfavorable observation as scientific evidence;
it is not an implementation failure. It neither selects a learning rate nor
settles the overall hypothesis: complete all nine screens and the predeclared
fresh confirmations. No tolerance, seed, control or horizon changed.

The controller automatically advanced to nonlexical-code slot02 at LR.000625,
PID45576 under session22743/controller PID17021. Actual updates were observed,
with no guard failures. Verify live state before acting; do not relaunch from
these recorded IDs. Two of18trajectories are accepted; the incumbent is unchanged.

## Existing first-rate losses: exploratory diagnostic

[Screen diagnostic001](tokenizer-evidence/byte-composition-boundary-001/screen-diagnostic001/analysis.md)
stratifies already saved native/raw-byte losses at LR.000625, after verifying
accepted bindings and original train/validation IDs. The overall raw-byte loss
increase is+.143202nat/token. Targets seen10–99times in training contribute
+.181558to that increase, offset partly by unseen and very frequent target groups.
Loss after an input token absent from training improves by-.923899nat/token,
which is consistent with shared input composition helping unseen tokens but is
not causal or confirmatory evidence. The pending nonlexical control is already
the discriminating comparison; no added model run is warranted. Fixed diagnostic
bins, all counts and exact additive contributions are saved. Preserve the full
scientific gates and complete all original screens/confirmations.
