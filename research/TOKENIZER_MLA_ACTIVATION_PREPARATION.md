# Tokenizer gate-activation preparation

Status, October 31, 2026: activation campaign001 frozen after complete prerequisite
review; both guarded pilots and audited pilot cleanup completed. Quality screening is next. Campaign005 remains closed, with the
hybrid not adopted. Existing selected models, every fresh pair and failures remain
protected. The earlier MoE draft is retained as a historical proposal. The October 28
GQA activation draft in TOKENIZER_ACTIVATION_PREPARATION.md also remains unchanged;
this separate MLA proposal follows completion of the later attention studies.

## Why this family now

The prospective three-arm MoE study needs **18,782,075,912 free bytes** under the
existing allowance method, including the 8 GiB reserve. Only 15,515,467,776 bytes
were free at the audit: a **3,266,608,136-byte gap**. Even the calculated final
raw/H200 tensor payload plus reserve is 15,562,137,728 bytes before headers,
RNG, logs or extra overlap. This assumes all experts acquire optimizer state;
idle experts must not be used to justify capacity. No pilot was attempted.

Defer MoE without dropping its wider-dense control, reducing fresh confirmation,
or deleting prior protected evidence. Select the already proposed same-shape
activation fallback instead. This explores another requested architecture family
while retaining the adopted width256 RoPE MLA backbone and a complete paired
confirmation panel. MoE remains queued for sufficient safely available capacity.

The two-arm activation projection requires **15,169,995,168 bytes**, leaving
345,472,608 bytes at that observation. This is a narrow, time-dependent margin,
not launch authorization. It includes two selected screening pairs, all six
fresh raw/H200 pairs, conservative optimizer/EMA/export overlap, per-trajectory
allowances, 256 MiB campaign metadata and the unchanged 8 GiB reserve. Pilot and
losing-screen cleanup requires complete audits and durable boundary receipts.
Current free space must be checked again before freeze and every dispatch.

Evidence: [storage projection](tokenizer-evidence/moe-preparation-001/ffn-storage-projection.json)
and [independent meta-optimizer crosscheck](tokenizer-evidence/moe-preparation-001/storage-crosscheck.json).
These are shape calculations, not observed training memory, time or quality.
Existing protected files already consume disk; observed free space takes no
credit for their deletion. Source references in the projection bind the
pre-implementation baseline, not a promise that later source remains identical.

## Hypothesis and proposed finite comparison

Compare ordinary SwiGLU to an exact GELU gate on the adopted width256,
four-layer, quarter-latent RoPE MLA backbone. Both use three bias-free FFN
matrices with hidden width 672, identical initialization draws, untied GPT-2
tables, block RMSNorm, final affine LayerNorm and attention score scale 0.5.
For normalized block input x:

- SwiGLU: W_down [SiLU(W_gate x) * (W_up x)].
- GELU gate: W_down [GELU_exact(W_gate x) * (W_up x)], with
  GELU_exact(z) = z [1 + erf(z / sqrt(2))] / 2.

Parameter counts and shapes match (28,492,416 parameters each), but functions,
gradients and optimization paths differ. No FFN product normalization, extra
gain, bias, approximate GELU or ReLU arm is permitted by the new contract.

The historical character study did not adopt GEGLU: fresh H200 deltas were
-0.000404 at width64 and +0.006786 at width256. ReGLU was worse at both widths.
This motivates a narrow GELU-only comparison under the changed subword/MLA
condition, not a positive forecast or a reopening of that study. Do not pool
character and subword metrics. See [historical results](GATED_ACTIVATION_RESULTS.md).

Proposed finite design: two 20-update resource pilots; three equal peak-LR
choices (0.000625, 0.00125, 0.0025) per arm on one screening seed; select each
arm's lowest H200 CE with lower-LR tie break; then both selected recipes on
three fresh seeds with alternating arm order. Quality trajectories use 1,200
updates, cooldown from 800, H200 start 800, batch8 and context128. Exact seeds,
requests, source/runtime/data identities and limits remain to be frozen.

Proposed adoption: GELU must improve mean H200 CE by more than 0.02 versus the
equally tuned fresh SwiGLU control, improve every fresh seed, and have negative
one-sided conditional-text upper bounds at blocks 2048/4096/8192 using 2,000
common paired resamples. Freeze RNG and quantile conventions and their tests
before results. Report raw endpoints, all seed deltas, actual training and
inference costs. No posthoc grid extension or quality-neutral cost route.
Validation is adaptively reused; the test holdout remains closed.

## Completed implementation prerequisite

`tokenizer_mla_gate_v1` validates exactly the width256 MLA backbone plus explicit
gate kind/policy. The existing architecture and gate validators delegate only
for that named contract; old policies retain their exact field sets and errors.
Checkpoint signatures distinguish gate semantics despite identical shapes.

154 targeted tests passed under a 120-second CPU prerequisite guard, with
558,956,544-byte peak RSS and the 8 GiB reserve intact. Checks cover exact
SwiGLU incumbent weight/logit/optimizer parity; full parameter coverage/counts;
independent exact-GELU value/input/parameter gradients; full versus cached logits;
causality; same-shape checkpoint mismatch; mutation and unsupported-combination
rejections; and historical gated-activation contract regressions. No optimizer
step, training pilot, quality evaluation or checkpoint file was performed by
these new tests. This is not a complete numerical or training-readiness claim.

Evidence: [test review](tokenizer-evidence/activation-preparation-001/contract-tests-v1-review.json),
[log](tokenizer-evidence/activation-preparation-001/contract-tests-v1.log),
[guard](tokenizer-evidence/activation-preparation-001/contract-tests-v1-guard.json).

## Next prerequisites, before launch

1. New-contract optimizer/RNG/EMA resume, H200 export/load, tokenizer binding,
   objective separation and cache isolation passed the RAM-serialization tests
   recorded below. Filesystem-producer integration remains required; preserve
   both harness failures and do not retry the closed hybrid numerical fixture.
2. Finite two-arm request/selection/statistical rules and seed registry passed
   22 tests (details below). Implement/test bounded artifact manifests, actual
   review/cleanup receipts and the retention-aware serial controller next.
   No launch through old hybrid controllers.
3. Reconcile the implemented model/optimizer/serialization resource projection,
   source/runtime/input pins and all prerequisites. Account for any test artifacts
   and metadata growth. Freeze the full plan before guarded MPS pilots; require
   accepted pilot resource evidence before quality runs.
4. Keep one worker, four CPU threads, strict MPS for training, all existing
   memory/time/storage guards, no paid compute, no subagents and closed holdout.
   If current storage fails, record the exact gap and safe options; never start
   a partial confirmation panel or remove protected pairs.


## Serialized state prerequisite completed October 31

Four new bounded integration tests passed for SiLU and exact GELU. Four-step
synthetic CPU updates with random token IDs produced bit-exact resumed online
weights, optimizer states, EMA and global/loader RNG relative to uninterrupted
execution. Production H200 derivation/export/load accepted the new contract,
retained the raw byte hash and RNG, matched EMA parameters, rejected optimizer
resume from averaged weights and rejected duplicate export. Production tokenizer
loading accepted the local corpus/tokenizer binding and rejected a changed
identity. Reset/interleaved caches and CE-only objective checks passed. No corpus
split was sampled or scored.

Real torch checkpoint serialization was held in RAM through an IO adapter; no
checkpoint file was created or deleted. This preserves capacity but does not
test filesystem publication. The successful 180-second/4 GiB/8 GiB-reserve guard
finished in 17.48 seconds with peak RSS 3,352,215,552 bytes.

Two harness failures remain preserved with their exact source, logs and guards:
v1 incorrectly requested logit-z and stopped before optimizer updates; v2 used
an uncanonicalized /tmp key for releasing a RAM buffer, causing two failures
after resume loading while both cache/objective cases passed. Production source
was unchanged across these attempts. The corrected v3 used the CE-only API and
canonical paths. These are new activation unit fixtures, not retries of the
closed failed hybrid numerical fixture or of an experimental trajectory.

The runtime and checkpoint, EMA, optimizer, schedule, loader, MLA/cache and
filesystem-writer source hashes match the completed005 baseline. Only the three
activation-contract files differ. This supports review of unchanged primitives,
not automatic reuse of hybrid numerical cases or authorization to launch an
activation campaign. [State prerequisite review](tokenizer-evidence/activation-preparation-001/state-prerequisite-review.json).

Next: implement/test the finite two-arm request and selection rules, exact fresh
seed registry and conditional-text statistics, then retention-aware serial
controller and filesystem-producer integration. Bind the state tests, review
source applicability and reconcile exact serialization/metadata capacity before
freezing all prerequisites and launching guarded MPS pilots. Retain all earlier
protected pairs, archives and failures. The plan is still unfrozen.


## Finite rules and requests validated October 31

The draft now fixes pilot seed3591, screening seed3601 and fresh seeds
3611/3612/3613. A recursive audit of seed-named fields in3,024 recorded
plan/request files found no reuse; its complete hashed inventory is preserved.
This is local nonreuse evidence, not population independence.

Two20-update pilots use LR0.0025, cooldown/EMA start13 and EMA interval1
(eight H200 snapshots). Six full screens use the three declared rates for both
arms, reversing arm order at the middle rate. Independent minimum H200 CE
selects each arm's LR, with lower-LR ties. Six full fresh runs alternate arm
order across the three seeds. The complete finite design has14 trajectories.
All nine conditional LR-selection branches are explicitly saved.

The single GELU-minus-SiLU contrast uses2,000 PCG64 resamples, RNG seed20261031
reset separately for each block size. Common text draws apply to all paired
models/seeds; paired seed loss sums are averaged before resampling and divided
by sampled target counts, including the tail. Bounds use sorted linear
interpolation and quantile1-0.05/3 for the three declared block sizes. Adoption
requires mean delta<-0.02, every fresh seed<0 and every upper bound<0.
The helper returns arithmetic eligibility only, never adopted=True.

22 tests passed, including independent loop-oracle weighted-tail bootstrap
comparison, strict threshold boundaries, invalid data/panel rejection, exact
request geometry/schedules, full ordered progression and persisted selection/
cleanup barriers. Missing, duplicated, reordered, failed or extra trials and
raw scores masquerading as H200 are rejected. Both selected screen pairs and
every fresh pair remain protected; only four losing screens are candidates
after complete audits. No deletion or training was performed.

[Finite design](tokenizer-evidence/activation-preparation-001/finite-design-v1.json),
[tests](tokenizer-evidence/activation-preparation-001/rules-tests-v1-review.json),
[seed audit](tokenizer-evidence/activation-preparation-001/seed-audit.json),
[source applicability](tokenizer-evidence/activation-preparation-001/rules-applicability-review.json).
Only two pure research modules were added since model/state validation.
These rules are not a production dispatcher or an executable freeze. Next
implement/test the bounded serial controller and filesystem producer, actual
review/retention receipts and complete resource/source/input/runtime binding.
Recheck full capacity before freezing and launching pilots.

## Controller core and filesystem transport validated October 31

The separate activation `Session` ledger now records exclusive immutable requests,
acceptances, selection and boundary receipts in a hash chain. It reconstructs
state from those receipts rather than a mutable progress cache. Exactly one
pending trial is allowed. Acceptance requires request/result/pair-bound training,
evaluation, tensor and resource audit receipts plus the complete producer-file
inventory, including metrics and summaries. STOP, failure/partial outputs, orphan
requests, altered artifacts, unreviewed extra files and premature selection block
progress. A failed attempt cannot be automatically retried.

Boundary receipts must explain exactly the four successful pilot files or eight
losing-screen files, preserve pinned non-checkpoint evidence and selected pairs,
and correspond to actual absence. All fresh pairs remain required. The ledger
has **no launch or deletion API**: it verifies ordering and receipt provenance,
not the scientific truth of a supplied audit verdict. It cannot replace the
independent scientific reviewer or the complete cleanup/process audit executor.

49 tests passed in54.10s; the guard finished54.21s with241,418,240-byte peak RSS,
180s/4GiB limits and the8GiB reserve intact. A complete14-trajectory synthetic
filesystem sequence retained16 endpoint files: two selected screening pairs plus
all six fresh pairs. Tests use tiny synthetic tensor payloads, real advisory
locks, actual free-space checks and the unchanged bounded torch checkpoint,
metric and summary writers. Independent torch.load verifies the saved tiny
tensors. These are transport fixtures, not model tensor audits or corpus runs.
No research checkpoint was created or deleted, and no closed fixture reran.

The initial44-test version passed. Its reporting wrapper then raised a KeyError
while reading the source fingerprint after pytest and its guard had completed;
an independent receipt records this error and the unchanged executed source.
A subsequent review added immediate producer-inventory pinning; the final49-test
run includes mutation/extra-output tests. Both versions' evidence is preserved.
Only the new session module differs from the previously tested source; model,
optimizer, EMA, tokenizer, checkpoint and bounded writers remain unchanged.

[Session review](tokenizer-evidence/activation-preparation-001/session-prerequisite-review.json),
[final test review](tokenizer-evidence/activation-preparation-001/session-tests-v2-review.json).

Next connect an executable activation worker/controller, independent scientific
acceptance and the complete retention executor. Bind actual source/runtime/input
identities, inference protocol and full resource projection before freezing the
plan. The ledger's synthetic verdicts and completed transport tests do not allow
pilots yet. Recheck the15,169,995,168-byte full-panel gate at freeze and dispatch.

## Worker and independent acceptance adapters validated October31

The activation worker body now connects the existing bounded output sink,
strict-MPS resource recorder and memory watchdog, training/export, full raw/H200
validation and pilot cache checks. A separate parent adapter reconstructs expected
input/training identities, checks coverage/score arithmetic and training metrics,
independently maps the pair for tensor/optimizer/EMA inspection, verifies resource
trace agreement, and only then publishes acceptance audits into the session ledger.
The result envelope now satisfies both the recorder and ledger and pins a separate
measurement document. No closed hybrid execution controller was modified or used.

76 tests passed in100.56s with1,149,943,808-byte peak RSS under the180s/4GiB/
8GiB-reserve guard. These include both actual activation models after20 synthetic
CPU updates each and eight EMA snapshots. Audits check all optimizer parameter
states, group/rate identity, Adam update counters, checkpoint geometry/contracts,
finite values and exact export equality. Wrong gate, step, EMA count, optimizer
state/rate/counter, parent identity and corrupted exports are rejected. This does
not independently replay the EMA recurrence or establish corpus quality; the
previous resume/EMA numerical tests remain separately applicable.

Six more tests passed for parent acceptance, including corrupted coverage, cache
reports, truncated traces, worker status and evidence changed during review.
Worker orchestration and parent integration tests explicitly mock MPS/training
or tensor/input operations; they are not pilot evidence. The actual tensor audits
are tested separately in RAM. CPU cache allocations also match the worker's
196,608-byte batch1 and1,572,864-byte batch8 expectations for both activations.
No checkpoint/model timing, MPS pilot, corpus training or holdout scoring occurred.

[Worker review](tokenizer-evidence/activation-preparation-001/worker-prerequisite-review.json)
records source applicability, tests and limits. The only source changes from the
prior session prerequisite are two new activation adapters and result-envelope
integration in the ledger; model/training/optimizer/tokenizer/transport primitives
remain unchanged. All tests passed; no new failed test attempt or closed-fixture
retry occurred in this iteration.

The worker prerequisite alone did not complete dispatch or retention. The next
iteration below records those additions; inference and final freeze remain open.

## Dispatch and retention execution prerequisites validated October31

The new executable entry binds the exact frozen plan, conditional finite design,
source/runtime, inputs, prerequisite receipts and full remaining storage. It uses
the reviewed bounded process supervisor with two inherited locks, a parent/slot
seal, guarded child preflight, positive exit and no-live-process-group checks.
Independent parent acceptance runs under its own guard. Every completed dispatch
pins both guard identities and the immutable process/acceptance chain. STOP,
failed/pending attempts, missing completion or partial cleanup block continuation.

The retention executor independently derives four eligible successful pilot files
or eight losing-screen files from complete audits and saved selection. It checks
actual research processes, hashes every explicit allowlisted path, journals intent
and deletion, then verifies the complete preserved inventory. Any incomplete or
failed attempt is permanent and cannot be retried. Selected screen pairs and all
fresh pairs remain required; prior protected campaigns and archives are outside
this executor's scope. No research cleanup was performed in this iteration.

24 tests passed in34.99s; guard35.10s, peak RSS239,960,064bytes, with180s/4GiB/
8GiB-reserve limits intact. Real harmless subprocess tests validate inherited
locks and positive/nonzero exit handling. The connected sealed-entry/parent test
runs actual guards, metadata/trace acceptance and ledger logic, with explicit
process and tensor/input doubles. It is not evidence of an actual MPS launch.
The disposable14-trajectory sequence exercises both real deletion boundaries,
retains16 tiny files (two selected screens plus all six fresh pairs), and rejects
changed journals/weights, incomplete reviews and an injected second-unlink failure.

The first attempt passed21 tests and rejected two subprocess fixtures at preflight:
their1MiB output cap was smaller than the fixed4MiB log cap. Both source/test
snapshots and failure evidence are preserved. Corrected fixtures use8MiB; no
production resource limit changed. Review also added a separately pinned child
entry guard and the connected success test. This did not retry a closed fixture.

Only two new activation research modules differ from the prior worker prerequisite;
all existing model/training/optimizer/EMA/tokenizer/transport sources and runtime
are unchanged. [Dispatch/retention review](tokenizer-evidence/activation-preparation-001/dispatch-retention-prerequisite-review.json).

Next finish finite inference protocol/runner and independent acceptance, then the
complete source/runtime/input applicability and exact artifact-capacity audit.
Latest free space15,580,561,408bytes exceeds the full15,169,995,168-byte projection
by410,566,240bytes; this observation is not the final launch capacity review.
Freeze all prerequisites before two guarded MPS pilots. No executable activation
plan is frozen, no corpus training has launched, and the holdout remains closed.


## Finite inference execution prerequisite validated October31

The inference contract now fixes all three selected fresh H200 pairs at seeds
3611/3612/3613, batches1/8, first96 training IDs, context128, prefill32 and64
teacher-forced decode calls. Four full/cached/reset correctness checks must pass
before timing starts. Two warmups and eight measured repeats alternate arm order
by seed and repeat. Synchronization brackets the timed intervals; cache creation
and model loading stay outside them. Independent review reconstructs every
sequence and throughput value, excludes warmups, computes seed-specific medians,
then the median of three paired GELU-over-SiLU latency ratios. Cost is descriptive;
there is no timing-dependent selection or quality-neutral adoption route.

The activation-specific loader rebinds accepted fresh raw/H200 pairs, configs,
tokenizer/training IDs and export lineage, checks finite CPU values, then validates
actual FP32 MPS geometry and parameter storage. Both models remain resident.
Unique allocated cache storage is196,608bytes times batch; allocator observations
must not be described as single-model deployment peaks. No test data is used.

A separate serial entry requires the complete accepted quality panel, both audited
retention boundaries and dispatch proof. It guards child preflight/worker and
parent acceptance, checks inherited locks and process identity, and pins each
seed's producer inventory, resource guards and completed transport. Saved process
descriptors are checked against the original request across later invocations.
STOP, changed proof, pending/future/orphan attempts and failures forbid progression.
Final summary requires all three accepted pairs and never itself adopts a model.

The unchanged metadata writer has three bounded inference buckets named3511,
3512,3513; the new contract explicitly maps them to scientific seeds3611,3612,3613
inside a new activation campaign root. Bucket names do not select training seeds
or reuse old campaign artifacts. Each seed has a5MiB output cap including its4MiB
log; the16MiB whole-inference cap fits inside the existing256MiB metadata reserve.
No new model/checkpoint/cache files are written by inference.

65 tests passed in37.67s; the guard finished37.77s with241,385,472-byte peak RSS
under180s/4GiB/8GiB-reserve limits. Timing/order/aggregation and CPU export-value/
lineage checks are real. The connected three-seed entry/worker/parent tests use
explicit approved-quality/process/loading/GPU doubles with real bounded filesystem
artifacts and guards. No actual MPS benchmark, corpus training, model-quality result
or research cleanup occurred. No failed test attempt or closed-fixture retry.

Source applicability: three new activation inference/timing modules and one
existing entry change to bind the finite inference contract. Existing model,
training, optimizer, EMA, tokenizer, timing kernels, loaders and bounded transport
primitives are unchanged; runtime matches the prior dispatch prerequisite.
[Protocol](tokenizer-evidence/activation-preparation-001/inference-protocol-v1.json),
[review](tokenizer-evidence/activation-preparation-001/inference-prerequisite-review.json).

Before freezing, implement the final independent campaign decision wrapper to
rebind all retained quality and accepted inference evidence, recompute the six
fresh endpoints' full-coverage paired statistics and report raw endpoints/training
costs/inference ratios. Apply only the declared strict quality criterion. Complete
exact implemented-shape/optimizer/artifact accounting and current capacity, cached
tokenizer/input and source/runtime/prerequisite applicability review. Latest free
space15,552,487,424bytes leaves382,492,256bytes beyond the full projection. This is
not a final launch audit. No activation plan has been frozen and no pilot launched.


## Final decision and executable freeze completed October31

The independent final wrapper rebinds all accepted quality, retention, dispatch,
checkpoint and inference evidence, validates raw/H200 coverage and score arithmetic,
and reconstructs paired H200 statistics from all six fresh endpoints. It reports
actual training/worker elapsed costs and measured memory alongside inference
ratios. Only the existing strict quality criterion can adopt GELU; if it passes,
seed3611 is the prospectively fixed representative export, never the best observed
seed. All six fresh pairs remain protected. Terminal failure/incomplete publication
cannot retry. 34 synthetic final-decision/freeze-entry tests passed in3.17s under
the180s/4GiB/8GiB-reserve guard (237,453,312-byte peak RSS). No scientific result
is inferred from these tests. [Final prerequisite](tokenizer-evidence/activation-preparation-001/final-prerequisite-review.json).

The prefreeze audit constructed both implemented activation models and optimizer
groups on the meta device:28,492,416 parameters,216,929,456 optimizer bytes,
65,536 persistent-buffer bytes and558,969,520 raw/H200 payload bytes per pair.
The unchanged allowance is788,953,140bytes per trajectory, including extra overlap,
headers and15percent allowance. All nine selected-LR branches peak at eight retained
pairs; with256MiB metadata and8GiB reserve the full requirement is15,169,995,168bytes.
Inference remains within existing metadata buckets and writes no weight files.
Actual CPU tokenizer/input identities and cached vocabulary hashes were verified;
no forward pass, optimizer update, checkpoint access or holdout score occurred.
Runtime and model/training/optimizer/cache/transport primitives remain unchanged.
[Capacity/applicability audit](tokenizer-evidence/activation-preparation-001/capacity-applicability-v1.json).

Campaign001 now has a complete executable plan, exact source/runtime/input and all
prerequisite references. Plan SHA256
`508b922253f7eef8ae6fafe05cb393aefe4bb0f58225e8573857e920933016c4`;
source SHA256
`7a4c02946a31820a7b1ff36b8cc4be42659a4fc4a62424cbc4ea5f32d6cbc487`.
Two20-update pilots precede six1200-update screens and six fresh runs; each stage
requires actual acceptance and audited retention barriers. The initial freezer
completed and validated publication but failed afterward while constructing its
external review: `file_digest(__file__)` supplied a string instead of Path. The
original procedure and failure diagnostic are preserved. An independent read-only
review verified the same actual/copy plan, empty campaign census, prerequisites,
source/runtime and current capacity without republishing or running training.
[Freeze review](tokenizer-evidence/activation-preparation-001/campaign001-freeze-review.json).

The next action is committing/pushing this frozen implementation and evidence,
then exactly the first guarded MPS pilot. No source edits or existing-attempt
retries. Model memory/time/cache readiness remains subject to actual pilot guards.


### October31 activation001 pilots and audited retention completed

Both20-update MPS pilots passed independent tensor/EMA, full-validation, cache
and resource acceptance. SiLU/GELU worker times were46.97/57.55seconds; peak RSS
1,839,251,456/1,837,727,744bytes and driver memory2,501,705,728bytes for both.
Raw/H200 cached parity passed at batches1/8; actual state196,608/1,572,864bytes.
These are feasibility checks, not comparative quality evidence or adoption.

The frozen retention boundary deleted exactly four audited pilot weights,
1,118,318,800 logical bytes, preserving all59 other campaign files. Manifest,
journal, inventory and terminal guard were independently verified; complete
portable text evidence is saved. Deleted pilot weights cannot be re-audited.
Observed free15,573,012,480bytes exceeded remaining requirement15,169,847,947.
No selected/fresh/failure checkpoint was deleted; test holdout remains closed.
Next is the first of six1200-update screens under the unchanged frozen source.
[Review](tokenizer-evidence/activation-preparation-001/campaign001-pilot-boundary-reviewed.json).


### October31 activation001 first quality screen accepted

SiLU at LR0.000625, seed3601, completed1200updates and independent acceptance.
Full-validation rawCE4.863816353, H200CE4.900551548;211.99s training and275.87s
worker duration. PeakRSS1,720,582,144bytes and MPS driver2,501,705,728bytes stayed
within frozen limits. This is one adaptive screening cell, not an activation
comparison or adoption. All checkpoints retained; test holdout remains closed.
The matched exact-GELU screen is now running serially.
[Accepted-prefix review](tokenizer-evidence/activation-preparation-001/campaign001-quality-prefix-01-review.json).


### October31 activation001 first matched screening pair

Both LR0.000625 cells (seed3601,1200updates) passed independent acceptance.
H200 CE: SiLU4.900551548, exact GELU4.898298625; GELU-minus-SiLU=-0.002252923.
Raw CE: SiLU4.863816353, GELU4.862704081. H200 was worse than raw by0.036735195
and0.035594544 respectively. Track the averaging pattern without changing the
frozen selection endpoint or introducing a posthoc EMA study into this panel.
This small single-screen-seed contrast is not adoption or fresh confirmation.
GELU training214.05s/worker278.14s; peakRSS1,721,024,512bytes, driver2,501,705,728bytes.
All guards passed; no cleanup and holdout stays closed. Slot02 GELU LR0.00125 running.
[Review](tokenizer-evidence/activation-preparation-001/campaign001-quality-prefix-02-review.json).


### October31 activation001 middle-rate GELU screen accepted

Exact GELU LR0.00125, seed3601, completed1200updates and independent acceptance.
Raw CE4.727150151; H200 CE4.726793996. Raising LR from0.000625 improved H200
by0.171504628 on this screening seed, much larger than the first-rate gate
contrast0.002252923. H200 now slightly improves raw by0.000356154; the earlier
averaging disadvantage therefore does not persist across these two GELU rates.
No general EMA conclusion, changed endpoint or adoption follows.
Training213.41s/worker297.45s; peakRSS1,718,665,216bytes, driver2,501,705,728bytes.
All guards passed, checkpoints retained, holdout closed. Matched SiLU LR0.00125
is running as slot03. [Review](tokenizer-evidence/activation-preparation-001/campaign001-quality-prefix-03-review.json).


### October31 activation001 middle-rate paired screen accepted

SiLU LR0.00125, seed3601, completed1200updates and independent acceptance.
RawCE4.723106871/H200CE4.722533288; GELU-minus-SiLU H200 delta=+0.004260708
at this rate, reversing the lower-rate delta=-0.002252923. Increasing LR from
0.000625 improves H200 by0.178018260 for SiLU and0.171504628 for GELU. Thus
within this one screening seed, LR effects are much larger than the gate
contrast; neither screening difference establishes an activation winner.
H200 slightly improves SiLU raw by0.000573584 at this rate. No endpoint change.
Training212.66s/worker297.91s, peakRSS1,650,491,392bytes, driver2,501,705,728bytes.
All guards passed, no cleanup, holdout closed. Slot04 SiLU LR0.0025 is running;
the final GELU rate follows before any frozen selection.
[Review](tokenizer-evidence/activation-preparation-001/campaign001-quality-prefix-04-review.json).


### October31 activation001 SiLU three-rate screen completed

SiLU LR0.0025, seed3601, completed1200updates and independent acceptance.
RawCE4.908889975/H200CE4.805960225. Its three H200 endpoints at ascending
LR0.000625/0.00125/0.0025 are4.900551548/4.722533288/4.805960225. The middle
rate is lowest on this screening seed; the highest rate is worse by0.083426937
despite H200 improving its own raw endpoint by0.102929750. No fresh-seed or
general-optimum claim. Persisted selection waits for the final GELU screen.
Training210.84s/worker325.27s, peakRSS1,706,131,456bytes, driver2,501,705,728bytes.
All guards passed; no cleanup, all current pairs retained, holdout closed.
Slot05 GELU LR0.0025 is running; then save selection and audit losing-screen
cleanup before fresh confirmation.
[Review](tokenizer-evidence/activation-preparation-001/campaign001-quality-prefix-05-review.json).


### October31 activation001 screen selection and cleanup complete

All six1200-update screens passed independent acceptance. H200 full-validation CE:

| LR | SiLU | Exact GELU | GELU minus SiLU |
| --- | --- | --- | --- |
|0.000625|4.900551548|4.898298625|-0.002252923|
|0.00125|4.722533288|4.726793996|+0.004260708|
|0.0025|4.805960225|4.807614079|+0.001653854|

The frozen minimum-H200 rule saved LR0.00125 for both arms. This screening seed
favors SiLU by0.004260708 at the selected rate; no fresh-seed/adoption conclusion.
Final GELU rawCE4.906256948/H200CE4.807614079;207.35s training/332.93s worker;
peakRSS1,697,267,712bytes, driver2,501,705,728bytes. All guards passed.

Independent selection review confirmed both saved minima and exact losing slots
00/01/04/05. Frozen cleanup removed their eight weights (2,236,641,184 logical
bytes), preserving all240 other campaign files including the four selected
weights. Independent post-cleanup replay verified selection, journals, terminal
guards, retained hashes and remaining capacity:14,436,069,376free bytes versus
14,050,805,346 required including the8GiB reserve and all future fresh pairs.
Deleted losing weights cannot be re-audited; complete portable text remains.

Next: fresh seeds3611/3612/3613 at the saved rates, alternating arm order. No
source/endpoint/grid change, holdout scoring or protected checkpoint deletion.
[Selection review](tokenizer-evidence/activation-preparation-001/campaign001-screen-selected-review.json),
[retention review](tokenizer-evidence/activation-preparation-001/campaign001-screen-released-review.json).


### October31 activation001 first fresh SiLU accepted

Fresh SiLU seed3611 at saved LR0.00125 completed1200updates and independent
acceptance. RawCE4.707916448/H200CE4.722778111; H200 is worse by0.014861663
in this cell. This unpaired endpoint is not an activation conclusion, and the
frozen H200 endpoint remains unchanged. Training206.79s/worker338.83s; peakRSS
1,697,906,688bytes and driver2,501,705,728bytes; all guards passed.
Matched GELU seed3611 is running as slot07. Every fresh raw/H200 pair remains
protected; no cleanup or holdout scoring.
[Review](tokenizer-evidence/activation-preparation-001/campaign001-quality-prefix-07-review.json).


### October31 activation001 first fresh pair accepted

Both seed3611 recipes at saved LR0.00125 passed independent acceptance.
SiLU rawCE4.707916448/H200CE4.722778111; GELU rawCE4.707687207/H200CE4.722655893.
GELU-minus-SiLU H200 delta=-0.000122218 (raw delta=-0.000229241): nearly identical
on this seed, far smaller than the strict mean0.02 adoption improvement. No
adoption or equivalence conclusion from one pair; all three pairs and frozen
conditional-text/inference/final reviews remain required. H200 worsened both
raw endpoints by about0.0149 on this seed; preserve the frozen endpoint.
GELU training206.89s/worker346.67s, peakRSS1,698,004,992bytes, driver2,501,705,728bytes.
All guards passed, every fresh pair protected, holdout closed. Slot08 GELU
seed3612 now runs first in the predeclared alternating arm order.
[Review](tokenizer-evidence/activation-preparation-001/campaign001-quality-prefix-08-review.json).


### October31 activation001 fresh GELU seed3612 accepted

GELU seed3612 at saved LR0.00125 completed1200updates and independent acceptance.
RawCE4.702213629/H200CE4.698578620; averaging improves this endpoint by0.003635009.
The matched SiLU seed is still pending, so this is not a paired conclusion.
Training205.52s/worker353.74s; peakRSS1,695,776,768bytes, driver2,501,705,728bytes.
All guards passed, no cleanup or holdout scoring. Matched SiLU seed3612 runs as
slot09; every fresh raw/H200 pair and selected screen remains protected.
[Review](tokenizer-evidence/activation-preparation-001/campaign001-quality-prefix-09-review.json).


### October31 activation001 second fresh pair accepted

Both seed3612 recipes at saved LR0.00125 passed independent acceptance. SiLU
rawCE4.699689255/H200CE4.695734123; GELU rawCE4.702213629/H200CE4.698578620.
GELU-minus-SiLU H200 delta=+0.002844497, reversing seed3611's-0.000122218.
The frozen every-fresh-seed-improves condition therefore fails; GELU cannot
qualify under this panel's adoption rule regardless of seed3613. This does not
establish population superiority or equivalence. Complete the final pair and
inference/final review as planned; no premature terminal receipt or changed rule.
SiLU training205.09s/worker378.39s; peakRSS1,695,023,104bytes, driver2,501,705,728bytes.
All guards passed; no cleanup/holdout scoring. Slot10 SiLU seed3613 running;
all fresh raw/H200 pairs and selected models remain protected.
[Review](tokenizer-evidence/activation-preparation-001/campaign001-quality-prefix-10-review.json).


### October31 activation001 final-seed SiLU accepted

SiLU seed3613 at saved LR0.00125 completed1200updates and independent acceptance.
RawCE4.701461123/H200CE4.712309397; averaging worsens this cell by0.010848274.
The matched GELU endpoint is pending. Training205.24s/worker393.60s; peakRSS
1,514,586,112bytes, driver2,501,705,728bytes. All guards passed; no cleanup or
holdout scoring. Final GELU seed3613 runs as slot11. Then the finite inference
panel and independent final review remain required; the every-seed quality
condition already failed on3612. All fresh and selected pairs stay protected.
[Review](tokenizer-evidence/activation-preparation-001/campaign001-quality-prefix-11-review.json).


### October31 activation001 complete fresh quality panel

All12 quality trajectories and both pilots passed independent acceptance.
Both families use their independently selected LR0.00125. Fresh H200 CE:

| Seed | SiLU | Exact GELU | GELU minus SiLU |
| --- | --- | --- | --- |
|3611|4.722778111|4.722655893|-0.000122218|
|3612|4.695734123|4.698578620|+0.002844497|
|3613|4.712309397|4.714239386|+0.001929989|

Mean GELU-minus-SiLU=+0.001550756: GELU is slightly worse on average and loses
two of three seeds. Both strict mean improvement and every-seed improvement
conditions fail. No population-equivalence claim; conditional-text intervals,
finite inference and independent terminal review still must be recorded.
Final GELU rawCE4.703934880/H200CE4.714239386;205.06s training/401.30s worker;
peakRSS1,692,663,808bytes, driver2,501,705,728bytes. All guards passed.

Training is complete; do not dispatch another training step. First inference
pair seed3611 is running in physical storage slot3511, followed serially by
3612/3613 and summary. These writer bucket names do not reuse old scientific
seeds or artifacts. Preserve all16 selected/fresh checkpoint files, all other
evidence and the closed test. No cleanup or model adoption.
[Full accepted-prefix review](tokenizer-evidence/activation-preparation-001/campaign001-quality-prefix-12-review.json).


### October31 activation001 inference failed; no retry

First inference seed3611 (storage slot3511) exited1 after39.24s. Its output
census enumerated control/process-inference-3511.json.tmp; the parent atomically
replaced that file before Path.stat, raising FileNotFoundError inside worker
budget(). The exact traceback, failed worker guard, process receipt, request,
attempt and failure latches are preserved. Worker guard failed at34.43s with
RSS821,805,056bytes; this was a file-census race, not a memory-limit crossing.
No complete timing result or accepted inference receipt was published.

Independent read-only audit confirmed no research worker/process group remains,
all12 quality trajectories and both boundaries still verify, and all16 retained
selected/fresh checkpoint hashes match. Original source/plan stayed unchanged.
GELU's completed quality failure remains supported; inference costs and the
frozen final wrapper are incomplete. Do not report a formal terminal receipt.

Do not retry or patch activation001, rerun training, or regenerate weights.
The next engineering step is a separate versioned inference-only continuation:
reproduce the race using disposable filesystem metadata; test bounded census
synchronization/revalidation without dropping transient bytes or weakening
unknown/symlink/hardlink/cap checks; bind all original accepted training,
selection, retention and checkpoint evidence; preserve frozen source and use
explicit applicability for any external adapter. Freeze a new dedicated root
and exact unchanged timing/quality/resource contract before measurements.
This is planned recovery work, not an approved result or completed fix.
[Failure review](tokenizer-evidence/activation-preparation-001/campaign001-inference-failure-review.json).
