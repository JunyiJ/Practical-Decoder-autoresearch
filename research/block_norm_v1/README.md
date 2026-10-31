# Block normalization prerequisites v1

This directory provides a prospective normalizer specification, CPU reference
checks and a separately namespaced unmodified runtime baseline. A named candidate model policy now has construction/optimizer-coverage tests.
Raw/H200 signature serialization now passes bounded synthetic tests. Synthetic CPU populated-optimizer and interrupted replay checks now pass.
CPU normalization-reference gradients and cache numerics now pass. Strict MPS numerical checks now pass. Generated-text tokenizer/trainer entry now passes. Production controller and
MPS optimizer/resume integration remain unvalidated; no quality training is ready. Frozen original/recovery source and evidence are unchanged.

`spec.py` binds kind, width256, FP32, epsilon2^-23, affine/bias and pre-attention/
pre-FFN placement. The control uses exact native RMSNorm with `eps=None`; the
candidate uses native LayerNorm with explicit2^-23 and learned bias. The factory
initializes constants without RNG draws. Runtime validation rejects wrong module
type, shape, epsilon, affine/bias, frozen gains, extra buffers, mixed/shared/aliased
normalizers, non-FP32 parameters/input and autocast. Eight distinct normalizers
are required. Signature encoding rejects mismatches even at matching widths.
This is signature metadata testing, not a torch checkpoint load/resume test.

The test oracle computes separate FP64 equations and analytic input/gain/bias
derivatives; it does not differentiate through native normalization. Cases cover
random/shifted, zero/constant, near-zero and shifted near-constant inputs. Actual
execution is native FP32 on CPU. Six control cases require bit-exact output,
input gradients and gain gradients against the incumbent native RMS path.

Tolerances were fixed before the first run and did not change afterward. Ordinary
output/input-gradient/gain-gradient atol are3e-5/3e-3/3e-4, rtol2e-4; shifted
near-constant LayerNorm uses8e-4/0.2/3e-3, rtol2e-4 because FP32 mean rounding is
amplified by the reciprocal scale. Bias gradients use rtol2e-5/atol3e-5. These are
bounded agreement checks, not bit-exact analytic equivalence or training-stability
claims. Observed maxima are recorded per case in the review. Near-constant
LayerNorm output/input-gradient/gain errors were0.0002103/0.01536/0.0009631;
ordinary random/shifted output errors stayed below6.26e-7.

Attempt001 passed all66 tests but FAILED its enclosing audit: the final metadata
hash loop passed a string to file_digest, which requires Path. Its log, failed
guard, report and full source snapshot are preserved. Attempt002 changed only
that wrapper conversion, reran under a new evidence prefix, passed66 tests and
verified original source, all13 frozen recovery modules, closed-root inventories
and noncheckpoint hashes unchanged. Spec/test code and tolerances were identical.
Guard002 finished0.388s, peakRSS285,900,800bytes, four CPU threads,8GiB reserve.
No corpus, checkpoint, training, MPS or cleanup operation was performed.

Do not rerun/overwrite either evidence attempt. Next map full-model/optimizer/
checkpoint/cache integration and source applicability without editing frozen
runtime artifacts or bypassing RMS-only validators. The prospective panel's
2.186GB storage deficit remains a training gate; do not poll unchanged capacity
or narrow the design merely to fit. See TOKENIZER_BLOCK_NORM_PREPARATION.md.


## Runtime integration baseline

See [INTEGRATION_CONTRACT.md](INTEGRATION_CONTRACT.md) for the import map, source
applicability, prospective edit allowlist and required full-model tests. The43
module copy under `runtime` is byte-identical to its original source. It passed
five guarded tests: import isolation, exact initial state/RNG/logits/loss and
old-new cached state/output, optimizer group equivalence and RMS-only rejection.
No candidate policy or optimizer update was exercised. Attempt001 stopped before
tests on sandbox PermissionError; attempt002 passed unchanged sources using
authorized process access. All snapshots/reports remain immutable. Do not rerun
those evidence attempts or invoke the copied Hydra CLI.

The baseline commit's staged whitespace check reported only six inherited lines: the exact baseline
and two immutable snapshots each retain original `attention/mha.py:106` and
`attention/mla.py:66` whitespace. These bytes are intentionally unchanged to
preserve origin hashes. No new authored whitespace error was found.


## Named policy construction

The new `runtime/models/block_norm.py` and three explicit model-file changes
passed40 tests (attempt001), then41 (attempt002 adding direct named-RMS versus
original equivalence). The runtime code was identical across attempts. Both
source snapshots are immutable. The current runtime has diverged from the
unmodified baseline only as recorded in `model-policy-delta-001.json`; historical
baseline evidence applies to its own saved sources, not silently to later edits.

New RMS control initial weights/RNG/full logits/loss and old-new cached execution
are bit-exact. LayerNorm adds2,048 parameters, preserving shared initial weights
and RNG, with all eight new biases uniquely grouped in AdamW at base LR. Config,
constructed module, epsilon/dtype/autocast and same-shape metadata rejection
checks passed. These are not gradient, serialized resume or H200 export tests.

Next bind actual normalization signatures in new raw/H200 checkpoint producers
and strict loaders, test serialized transport/EMA/resume, then complete numerical
cache/gradient and MPS checks. No old source or recovery runtime may change.
The original standalone spec remains unchanged. Training capacity remains gated.

The two model-policy source snapshots likewise preserve the two inherited
attention whitespace lines each; source hashes take priority over reformatting
immutable captures. Authored model changes have no whitespace-check errors.


## Serialized signature transport

The checkpoint writer/loader and H200 exporter now bind the dedicated constructed
normalization signature and validate exact typed metadata, saved configuration
and tensor inventory/shape/dtype/finite values.26 CPU checks passed on fresh
synthetic raw/H200 files for both arms. H200 gains/biases match a separate FP64
two-snapshot oracle; export leaves parent bytes and RNG unchanged. All four
fixture files (684,354,160bytes) are retained under an explicit size/SHA manifest.
No trained checkpoints or corpus were accessed. Source delta and exact tested
snapshot are recorded under checkpoint-transport-prerequisite-001.

This supersedes the prior pending-transport note. Empty optimizer group transport
and final EMA accumulator restoration passed; populated optimizer-state and
interrupted trajectory/loader RNG replay remain pending. Tighten new-policy raw
resume requirements and test those before declaring resume ready. No candidate
cache/gradient/MPS or quality claim. No evidence attempt may be overwritten.


## Populated resume replay

30 CPU checks now establish bit-exact batch1/context2 primitive replay for both
arms over three synthetic updates with an interruption after the first. Model,
Muon/AdamW slots, schedule, EMA, losses, samples and CPU/loader RNG agree exactly.
The new copied `utils/block_norm_resume.py` checks complete raw state before
weight loading; malformed groups, rates, slots, counters and RNG/EMA are rejected.
All source deltas and test snapshots are retained. Two raw fixture files and ID
files total890,211,626bytes; no cleanup or original corpus access occurred.

The accompanying three-test read-only H200 regression remains exact on prior
synthetic fixtures and preserves old-policy resume behavior. These results
supersede the earlier populated-resume pending note, but do not prove production
trainer/tokenizer integration, independent gradients, candidate cache parity or
MPS stability. Next complete numerical/cache checks without new binary fixtures
where feasible, then resource and controller readiness. Training remains gated.


## CPU cache and normalization-gradient checks

33 checks now pass: full/cached batch1/2/8/context17 and batch1/context128,
continuation offsets, exact causality/reset, independent interleaved requests and
invalid-cache rejection before any write. The copied cache/GPT/new-policy modules
now bind norm signature, layer, shape/dtype/device, distinct storage and contiguous
positions; old-policy behavior is unchanged. The delta explicitly extends the
cache-file allowlist for these guards. No attention math changed.

An explicit FP64 normalizer reference shares the FP32 attention/FFN/readout and
checks outputs, CE, all parameter gradients and eight norm-input gradients.
Maximum errors: logits1.073e-6, CE0, parameter gradients1.863e-8, norm-input
gradients1.863e-9. Original RMS full-model gradients are bit-exact. Cache/full
maximum logit error8.345e-7; storage196,608bytes/batch item for both arms.
Tolerances were frozen before execution; no binary fixture/update/checkpoint
load or MPS operation occurred. This supersedes earlier CPU-cache pending notes.

Next inspect existing strict MPS guards and predeclare backend tolerances for
bounded synthetic numerical checks, keeping RSS4GiB/driver6GiB, reserve8GiB,
serial work and fallback disabled. Trainer/tokenizer/controller/statistics and
finite-plan freeze still remain, and training capacity is still insufficient.


## Strict MPS numerical checks

11 checks pass with fallback=0 and fast_math=0, four CPU threads and the original
RSS4GiB/MPS-driver6GiB/reserve8GiB/600s guards. Runtime code is unchanged from the
CPU cache checks. CPU–MPS max logit error8.941e-7, MPS full/cache1.073e-6, and
CPU-reference parameter-gradient error1.933e-7; reset/CE errors0. Original MPS
RMS logits/CE are bit-exact; its gradients pass the declared tighter tolerance.
All tolerances were frozen before the first attempt. No optimizer/checkpoint/
corpus or binary-fixture work occurred. Guard3.340s, peakRSS1,154,957,312bytes and
peak sampled driver1,159,659,520bytes. This supersedes MPS-numerics pending notes.

Next actual tokenizer-bound trainer entry on generated synthetic text with its
own validated no-bias prior/tokenizer binding; never substitute the real-corpus
prior onto synthetic data or silently weaken validation. Plan storage first and
use one guarded worker. MPS optimizer/resume, controller/retention/statistics and
finite seed/order freeze still remain. No quality pilot while capacity blocked.


## Generated-text trainer integration

12 CPU checks now pass through actual run_training with the GPT-2 loader and an
independently derived generated-text no-bias binding: two updates per arm at
batch1/context128, complete raw resume preflight, exact saved-EMA/H200 equality,
raw/H200 tokenizer loading, metadata mismatch rejection and output contracts.
The paired512 batch8 sampler separately matches independently sampled starts.
There is no fixture test split and no real corpus/holdout access. Runtime unchanged.

Attempt001 failed before updates due to an invalid fixture cooldown start1 for
a two-update horizon; attempt002 used start0. Both plans/snapshots/outputs remain
immutable. Successful fixtures total1,118,435,545bytes; failed text40,902bytes.
Guard00212.437s, RSS1,419,116,544bytes,4threads/120s/8GiBreserve. This supersedes
pending-tokenizer/trainer-entry notes, not full-trainer interrupted replay or
MPS optimizer/resume readiness. Next bounded strict MPS optimizer/resume, then
controller/retention/statistics and exact finite-plan freeze; quality remains
gated on complete capacity. No old attempt may be rerun or overwritten.

The two exact trainer source snapshots preserve inherited attention whitespace;
the failed pytest log preserves emitted trailing spaces. Staged whitespace
diagnostics are confined to these immutable captures; authored changes are clean.


## MPS resume failure: no readiness claim

The strict MPS resume attempts001/002 are closed and preserved. Attempt001 exposed
`mps` versus `mps:0` device-name rejection; the new-policy preflight now accepts
those device0 aliases and validates MPS RNG byte-vector length. Attempt002 passed
five RNG and three device/version rejection checks without mutation, and resumed
RMS updates, but failed the fixed numerical replay tolerance (atol2e-6/rtol2e-4):
one element in a172,032-element tensor differed3.050663508e-6. LayerNorm was not
run, and later optimizer/EMA/RNG agreement assertions were not reached. No
production or quality readiness follows; no tolerance relaxation or old retry.

Both attempts preserved existing source/evidence/fixtures and used only in-memory
checkpoint transport, no new binary files. All failure logs/source snapshots stay
immutable, including emitted/inherited whitespace. Generic wrapper scope text
mentions archive hashes, but failure occurred before those final properties were
emitted; no archive hash/payload was retained. Next separately planned diagnostic
must emit named intermediate state/RNG/gradient/update evidence before assertions,
with a predeclared numerical control. No assumed root cause or quality launch.


## New RMS diagnostic completed; failed acceptance gate unchanged

Trace001 separately observed exact full checkpoint restoration (tensor values/
layouts, optimizer/EMA/schedule metadata and RNG), identical batches/draws/losses,
and the first differences in embedding gradients. A fresh uninterrupted repeat
also diverged: final max parameter delta4.9025e-5/5,472elements outside the unchanged
threshold, versus resumed3.9563e-6/3elements. This supports backend variation
independent of resume, but not a precise kernel cause or numerical acceptance.

Installed Muon orthogonalization explicitly uses BF16 intermediates despite FP32
parameters/gradients/saved slots. Its potential amplification remains an inference.
Runtime unchanged. Guard7.896s/RSS2,297,741,312B/driver1,191,608,320B;8synthetic RMS
updates, no new binary files/quality/holdout. Intermediate archive hash and full
trace retained; payload stays ephemeral. All earlier failed attempts remain closed.
Next separately plan embedding-backward deterministic controls and Muon sensitivity
checks, without silent backend/precision/tolerance changes or quality launch.


## Isolated kernel diagnostics

Repeated-token embedding backward returned8different hashes in8identical passes
both with deterministic algorithms off and strict on; unique-ID controls were
bit-exact. Max repeat delta1.8626e-9, FP64 oracle error1.5134e-9; no warning/error/
fallback. The strict flag alone is insufficient on this local backend. Installed
Muon repeated exactly on identical synthetic inputs, but1.86e-9 perturbations
produced9.766e-6/1.266e-5 parameter differences at256x64/672x256, identical across
modes. BF16 input observations reconstruct the formula rather than instrumenting
optimizer internals. This supports sensitivity, not full causal trace replay.

Both new diagnostics completed with all guards and frozen evidence unchanged;
no runtime changes, binary artifacts, model training, quality or holdout use.
Global deterministic settings restored; failed resume gate remains unchanged.
Next standalone deterministic MPS embedding-gradient prototype (unique-token
membership reduction/copy candidate), with prospective formula/FP32/repeat/resource
checks and no CPU fallback. Do not adopt into runtime without explicit delta and
both-arm numerical/replay/resource validation. No old failed fixture reruns.


## Standalone grouped embedding prototype passed; runtime unchanged

11 tests passed, including88backward calls over11patterns with deterministic
algorithms off/on. Native forward is exact, gradient hashes are identical within
and across modes, FP64 oracle maxerror1.0026e-8 under fixed1e-7/2e-5 tolerance;
unique/zero/constant and unused rows are exact. Batch1 and paired batch8/context128,
repeated/unique IDs and actual noncontiguous tensors were covered. Guard5.613s,
RSS624,197,632B/driver1,082,884,096B; no optimizer/model training or binary outputs.

`grouped_embedding_prototype.py` is not imported by the runtime. It performs
MPS unique-token membership matmul and distinct-row copy, with native forward.
It is first-order only, fixed50257x256, at most1024tokens and no extra embedding
options. Next explicit versioned backward-policy/config/checkpoint/source binding,
then both-arm full-model equivalence/gradients/transport/repeat/resume checks.
Do not silently alter old policies, reinterpret old checkpoints, change Muon
precision, loosen the failed gate, or claim full-model/campaign readiness.

Versioned grouped embedding integration passed33 checks under `grouped-embedding-integration-plan-001.json`: both-arm initialization/CPU gradients/native MPS forward-cache, four full-model gradient repeats per1x128/8x17 shape, RAM raw transport and old/new checkpoint/cache rejection. V2 is isolated prerequisite code only. Optimizer resume, H200, full8x128 readiness and quality-plan freeze remain pending. See `grouped-embedding-integration-analysis.json` and `grouped-embedding-integration-decision.json`; prior failed attempts stay closed.

New v2 MPS repeat/resume prerequisite passed both arms:16 synthetic updates at batch1/context128; exact intermediate gradients/states and zero final model/optimizer/EMA error. Original tolerance and Muon precision unchanged. Eight malformed-state rejections per arm, RAM raw transport only. See `grouped-embedding-resume-analysis.json`, plan001 and decision. H200/file transport, full8x128 and controller/resource/quality gates remain pending.

V2 actual MPS batch8/context128 trainer and persisted H200 prerequisite passed9 checks:two generated-text updates per arm, exact EMA replacement, raw resume preflight/H200 rejection, tokenizer/parent binding and CPU/MPS cached inference. Four binaries plus metadata retained under `block-norm-v2-h200-fixture-001`, SHA-bound by `grouped-embedding-h200-fixture-manifest.json`. Controller/statistical/retention/inference and actual-corpus/full-panel gates remain pending; no quality conclusion.

Pure finite campaign rules/statistics passed67 CPU checks:14trajectory specifications,18suggested transitions, per-arm three-LR selection/lower-LR ties, allfresh protection, strict gates and tail-weighted common paired bootstrap (independent error3.469447e-18). Test seeds are not a frozen launch plan. `campaign-rules-decision.json` records scope; immutable ledger/config/source/provenance/dispatch remain pending. No new weights or training.

## Read-only campaign contracts passed, October31,2026

The isolated preparation contract passed66 guarded CPU checks. It builds and
validates exact v2 configurations for both20-step pilots, six1200-step screens
and all54 fresh-slot/LR-selection combinations (62 checks of configurations).
Pilot cooldown and H200 begin at13of20 with EMA interval1; quality keeps800of1200
and interval10. No output directory or model is created by the builder. Changed
fields, wrong slots, path aliases and source inventory drift are rejected.

The preparation manifest binds369 source files, the unchanged original source
and13 recovery modules, strict four-thread environment, installed Muon source,
base recipe, corpus bytes, tokenizer and train/validation identities. The actual
validation has34,278 targets/111,536 bytes:267 full128-target windows plus102,
and16 full2048-target statistical blocks plus1510. Only train and validation
raw-text slices were encoded; test was neither tokenized nor scored. This binds
intended coverage, not evaluator loss provenance.

Guard:4.02235s, peak RSS272,367,616B, free disk9,226,452,992B;120s/4GiB/8GiB
guards passed. No MPS/model/optimizer work, checkpoint writes, cleanup or new
quality result. Earlier fixtures, closed campaigns and runtime source stayed
unchanged. The manifest always has launch_permitted=false; fixture seed values
are unreserved. A future immutable ledger must bind the selected LR and exact
request/config/source, one pending worker, accepted prefix, STOP/failure and
retention receipts. Full validation/inference and worker/parent prerequisites,
actual-corpus pilots, scientific seed census and full-panel capacity remain.

Evidence: `campaign-contract-plan-001.json`, `campaign-contract-001-review.json`,
`campaign-contract-analysis.json`, `campaign-contract-preparation-manifest-001.json`
and `campaign-contract-decision.json` in the block-norm preparation evidence directory.

## Synthetic append-only ledger passed, October31,2026

The isolated ledger passed42 bounded CPU checks. Its32-event replay accepts
exactly two pilots, six screens and six fresh confirmations, with one pending
request, per-arm selected LR, exact accepted prefix and boundary indices.
Content-addressed configuration/request/receipt objects and sequenced event
hashes reject edits, missing files, aliases and stale callers. A nonblocking
file lock serializes writes. Durable STOP/failure retains the pending identity
and blocks later acceptance; source drift does not prevent recording STOP.

A caller must retain the expected head outside the ledger: a hash chain alone
cannot detect removal of a valid suffix. That check passed, but external head
persistence/recovery is still a controller integration task. Interrupted or
invalid operations may leave valid unreferenced objects; these are preserved
and cannot advance accepted state. Partial event writes fail closed.

This version only accepts explicitly synthetic parent and retention receipts.
It cannot launch workers, validate real checkpoints or authorize cleanup, and
always reports launch_permitted=false. The passing replay is not real-artifact
acceptance. Guard:13.82545s, peak RSS276,971,520B, free disk9,223,426,048B,
within120s/4GiB/8GiB constraints. No model/optimizer/checkpoint work or quality
result; prior fixtures, runtime and closed campaigns remain unchanged.

Ledger mechanics are complete for this prerequisite. Next prioritize scored
validation/provenance and inference checks against retained v2 fixtures, then
bind real worker/parent artifacts and externally persisted heads. Do not extend
metadata-only scaffolding absent a concrete failure. Actual-corpus pilots, full
retention-aware capacity, scientific seeds and finite launch freeze remain gated.
Evidence: `campaign-ledger-plan-001.json`, `campaign-ledger-001-review.json`,
`campaign-ledger-analysis.json` and `campaign-ledger-decision.json` in the
block-norm preparation evidence directory.

## Cache capacity restored and retained-fixture validation passed, October31,2026

The human authorized unused shared dataset-cache deletion. The exact manifested
cleanup removed9.32GiB allocated, with17.87GiB free afterward. Research weights,
archives, model caches and immutable evidence were verified unchanged. The
remaining checkpoint audit found no additional verified deletion allowlist; historical
numerical payloads remain pending any specific archival/dependency audit.
This clears the current15,170,221,264B full-panel projection with about3.74GiB
headroom; recheck before dispatch. Capacity is no longer the current blocker.
See `tokenizer-evidence/dataset-cache-cleanup-001/completed.json` and checkpoint review.

Read-only validation attempt001 failed: the new scorer's combined MPS-to-CPU
float64 conversion returned zeros. A separate tiny-tensor diagnostic reproduced
it for literal values and crossentropy. The corrected path transfers FP32 to CPU
first and converts there. Failed001 and its exact source are retained. Original
`src/research/coverage_eval.py:54` already uses this safe MPS path; historical
quality results and source remain unchanged. No tolerance was relaxed.

New attempt002 passed34 checks using both retained generated-text v2 H200
checkpoints: all1598 validation targets,13 context windows including62-target
tail, independent CPU FP64 logsumexp oracle, exact MPS replay, RNG preservation,
artifact/hash and malformed-report rejection. Maximum per-window mean NLL
errors were3.786403e-7 RMS and3.410218e-7 LayerNorm (limit3e-5). A separate
8327-target uniform-logit analytic oracle checked66 windows and5 blocks, with
short tails7/135 and mean error2.758057e-7 (limit2e-6). No test split was scored.
Guard7.41663s, RSS808,304,640B, peak sampled MPS driver1,103,872,000B.

This is evaluation correctness on two-step fixtures, not a quality comparison.
No optimizer update, new binary checkpoint, cleanup of research evidence or
model/runtime change occurred. Next bind real worker/parent artifact acceptance,
external ledger heads and retention receipts; complete inference measurement
acceptance, seed census, resource pilots and final finite freeze. Avoid further
metadata-only expansion absent a concrete failure.

## Actual retained artifact acceptance passed, October31,2026

Parent-side acceptance passed52 guarded checks on both retained generated-text
v2 output bundles. Each binds actual config/raw/H200/metrics/summary/export files
and379 source files. Raw optimizer/RNG/EMA resume preflight passes without a step;
all88 RMS or96 LayerNorm averaged parameters exactly match parent EMA, and
nonlearned buffers match the raw parent. Summary, log and checkpoint agree on
updates, tokens, LR schedule, evaluation settings, timing and averaging lineage.

The parent independently recomputes full validation. Its scores exactly match
the earlier CPU-oracle-checked reports. A fabricated score with internally
consistent sums/CE/byte arithmetic is rejected by recomputation: structural
validation alone cannot authenticate a worker's loss. Corrupt configurations,
metadata, signatures, tensors, lineage and each of six file hashes are rejected.
No saved artifact is edited, and CPU/MPS RNG state is preserved.

Guard11.85870s, RSS1,820,884,992B, peak sampled MPS driver1,163,706,368B;
no training, optimizer update, new binary or holdout evaluation. This proves
acceptance of actual existing two-step fixtures, not a new worker dispatch or
quality claim. Receipts still prohibit launch and adoption.

Next connect a thin serial worker/parent path, external head persistence and
audited retention boundaries to these tested primitives, complete inference
measurement acceptance, reserve fresh scientific seeds and freeze the finite
pilot/quality plans. Capacity currently fits after authorized cache cleanup;
recheck at dispatch. Avoid additional metadata-only expansion absent a concrete
failure. Evidence: artifact-acceptance-plan-001/review/analysis/decision and source
snapshot in the block-norm preparation evidence directory.

## Real readonly worker dispatch passed, October31,2026

Two actual child processes sequentially loaded the retained v2 RMS/LayerNorm
fixtures, scored all validation targets, exited and were reaped. Only then did
the parent audit each six-file bundle and independently recompute its score.
Both exactly match the previous CPU-oracle-checked reports. All22 checks passed,
including duplicate-attempt and STOP rejection,17 corrupt process/report/guard
variants, and rejection of parent impersonation of a child. Sources (382 files),
configuration, requests, environment, argv, parent PID and inherited locks bind
the handoff. The frozen bounded subprocess supervisor is reused unchanged.

Child wall times were4.132s/2.620s; peak child RSS653,885,440B and sampled
MPS driver1,125,711,872B. Parent peak RSS1,222,098,944B and driver1,155,317,760B.
A live-child parent uses synchronous guard checks so the supervisor owns cleanup;
the parent's strict MPS audit guard begins after child exit. No training, new
binary, deletion or holdout scoring occurred. Attempt001 failed test collection
before starting a worker; its source/log/plan are preserved. Attempt002 changes
only the test's relative import to an absolute namespace.

This closes the real readonly worker-to-parent transport prerequisite. It does
not freeze or launch the scientific campaign. Next finish inference measurement
acceptance and connect durable external heads and audited retention receipts to
the actual campaign path, then census seeds, freeze the finite pilots and run
the actual-corpus resource pilots. Ledger mechanics need no further expansion.
Evidence: fixture-dispatch-plan-001/002, original source snapshots,002-campaign,
reviews, analysis and decision in the block-norm preparation evidence directory.

## Cached inference passed; scientific seeds reserved, October31,2026

The retained v2 pair passed21 guarded inference checks. Before any warmup, both
arms at batch1/8 passed full96 versus cached32+64 teacher-forced calls and cache
reset/replay; maximum absolute logit difference1.43052e-6. Both cache sizes are
196,608B at batch1 and1,572,864B at batch8. LayerNorm adds8,192B of learned
parameter storage. Two warmups and eight alternating paired repeats per batch
produce eight warmup/32 measured rows. Loading/cache allocation stay outside
synchronized timing intervals; saved-value/provenance review and independent
median/throughput arithmetic pass.19 corrupted reports and a swapped-arm request
are rejected. CPU/MPS RNG and all retained artifacts remain unchanged.

The22.577s guard observed peak RSS869,646,336B and MPS driver1,154,236,416B.
These generated two-step fixture timings are implementation checks only: no
quality or speed winner, deployment peak or cost adoption conclusion. Real
campaign inference still needs its worker/process provenance and fresh trained
pairs. No optimizer updates, new binary files, deletion or holdout scoring.

A separate120s-guarded freshness census scanned40,314 recorded research/config/
source/log text files (3,701,570,300B), finding no whole-word occurrence of the
seven proposed seeds. Reserve pilot374001, screen374002, fresh374011/374012/374013,
evaluation374003 and bootstrap374004. Both arms share paired training seeds.
The census excludes only its own new output directory, bytecode/binary artifacts,
raw corpus and external files; it claims freshness in recorded repository work.
The exact file/hash inventory and audit source are retained.

Next build/freeze the minimal finite pilot entry using these reserved seeds and
tested config/worker/acceptance primitives, then run the two actual-corpus20-step
resource pilots. Keep their weights until the required audits and dependent
prerequisites finish. Before cleanup or quality dispatch, connect durable external
heads and audited retention receipts; do not expand already-passed ledger mechanics.
Final quality freeze remains separate. Evidence: inference-measurement-plan001/
review/result/analysis/decision and scientific-seed-census-001.
