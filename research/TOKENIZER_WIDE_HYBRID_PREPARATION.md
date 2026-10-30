# Wider hybrid: next finite preparation

October27,2026 Los Angeles. The direct width512 MLA/GQA round is complete: MLA
passes the scoped quality and memory gates, with slower decode. The next proposed
question is whether a wider3:1 KDA/gated-MLA hybrid benefits from broader bounded
tuning versus homogeneous controls. This responds to the human's width and HP
steering. It is a new experiment, not a reopening of closed hybrid001.

## Decision and alternatives

Prioritize preparing width512 hybrid/all-KDA/all-gated-MLA contracts and resources.
The earlier width64 hybrid received only two LR trials per family and cannot be
called an optimum. New RoPE MLA evidence does not establish NoPE gated-MLA or
hybrid quality. Keep the newly adopted RoPE MLA recipe separately scoped.

MoE adds capacity and storage, activation/norm/residual comparisons are useful
controlled alternatives, and standalone schedule/regularization tuning remains
valid. They are deferred for this preparation to address the explicit broader
hybrid question; attention refinements must not indefinitely displace them. No
new quality launch is authorized by this preparation note alone: executable
contracts/tests, full projection, pilot freeze and quality freeze are required.

## Proposed bounded stages, not a frozen plan

Use width512 four layers, GPT-2 legacy untied tables, existing dense backbone and
same closed-holdout boundaries. Declare NoPE for all three new families, K3-style
bounded-decay/full-rank output gates and complete layerwise recurrent/convolution
and latent/shared-key cache semantics. This is distinct from Kimi Linear's
low-rank output gates. No claim to reproduce a large Kimi model or long context.

Proposal: three retained resource pilots, then at most24 quality trajectories:
9 independent LR screens (three per family),6 bounded second-stage probes
(two per family),9 fresh confirmations (three seeds for all three families).
Second-stage hypotheses may address KDA decay/gate initialization and MLA latent
rank; exact candidates, fair budgets, screening seeds and deterministic selection
must be declared before any quality data. Do not adaptively enlarge a grid.
A shared horizon/schedule study is an alternative second-stage design; choose
once from prior evidence before freezing, not after viewing new screening scores.
Fresh confirmation compares the selected hybrid with BOTH selected homogeneous
controls. Freeze quality neutrality/strict and actual memory benefit criteria,
including multiplicity/conditional-interval limitations and honest latency costs.
No claim that any finite grid finds the global optimum.

## Resource gate and next useful work

Current free disk was17,551,732,736bytes (~16.35GiB). A deliberately optimistic
lower bound for27 retained trajectories is40,617,872,799bytes (~37.83GiB), including
8GiB reserve. It counts ONLY the two vocabulary tables with raw weights, Adam
moments, saved EMA and H200 export, then2MiB/trajectory plus15% allowance and4MiB
round metadata. Attention/FFN/norm weights and optimizer state are excluded, so
the actual complete projection will be larger. The panel cannot fit locally.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/storage-lower-bound.json`.
This lower bound is accounting, not a policy-validated construction or launch plan.

Do not lower reserve, silently shrink width, delete more checkpoints, reuse
closed controllers or launch a partial panel. Request an external checkpoint
location or additional space; roughly60GiB free is a provisional planning target,
not a measured sufficient budget. Full model/pilot accounting must determine it.
Meanwhile useful bounded work is explicit wider hybrid geometry/optimizer/lineage
support, CPU equation/gradient/causal-cache/resume/EMA tests and exact meta-device
accounting. Preserve all failures. Only after resources fit should a finite pilot
source/plan be frozen; actual pilot costs then gate every remaining quality run.
Single worker, strict MPS, sampled4GiB RSS/6GiB driver and8GiB reserve remain.

## Anchor implementation and accounting

A separate `tokenizer_wide_k3_hybrid_nope_v1` policy now admits only widths512/1024
with head32 and the three declared layer layouts. The old policy still admits
only64/256. The new anchor keeps D/4 decay and query/KV bottlenecks, convolution4,
bounded log-decay−5, full-rank output gates, NoPE and the inherited dense backbone.
This anchor policy admits no optional rank/decay/gate overrides. The config helper is not
a dispatcher or frozen experiment; its default seed/rate are preparation values.

Meta construction gives width512 parameter counts65,037,792 hybrid,
63,474,816 all-gated-MLA and65,558,784 all-KDA. At FP32/context128/batch1 their
allocated cache/state sizes are333,824/327,680/335,872bytes respectively. The hybrid
is only0.61% smaller than all-KDA and1.875% larger than all-gated-MLA at this anchor.
Therefore a meaningful memory-saving route against both controls cannot be
assumed. Quality and actual latency remain unknown; no trained hybrid comparison
or long-context claim follows from these shapes.

The complete anchor model/optimizer/EMA accounting and conservative27-trajectory
projection are in `tokenizer-evidence/wide-hybrid-preparation-001/anchor-storage-projection.json`.
This substitutes the anchor geometry for every stage2 probe; changed tuning
geometries must be accounted for again before freeze. It is not a sufficient
resource guarantee, measured checkpoint size or measured MPS training peak.
Width1024 has meta accounting only. Numerical prerequisites run at512 on CPU.

The corrected conservative anchor panel requires56,866,579,251bytes (52.96GiB)
at512 and124,702,393,587bytes (116.14GiB) at1024, including reserve. Exact saved
model/optimizer/EMA tensor counts are separated from conservative allowances.
An initial accounting draft included nonpersistent causal masks as saved buffers;
the draft and correction are retained. No launch decision used the draft.

Thirty-five geometry/equation/cache/optimizer/lineage and historical-policy checks
passed. A separate combined CPU continuation process passed the hybrid's exact
model/optimizer/RNG/EMA resume check, then exited90 at its resource guard. The
gated-MLA and all-KDA continuation checks are unverified. The guard diagnostic
was lost inside pytest capture on hard exit, so the exact trigger sample is not
available. Last observed RSS was4,018,368KiB, and post-exit disk remained above
reserve; an RSS crossing is likely, not a proven sampled peak. This is fixture
process memory, not an MPS training measurement. Preserve
`cpu-resource-failure-001.json`; no numerical retry or MPS launch followed.
Review fixture lifetime and a separately recorded validation procedure before
further numerical work. The wider anchor is not launch-ready.

The failure review now identifies overlapping serialized checkpoint, deserialized
checkpoint and EMA lifetimes as hypotheses to investigate, without asserting a
measured cause. A separate opt-in CPU guard preserves observations outside pytest
capture;11 lightweight guard checks passed. See
`tokenizer-evidence/wide-hybrid-preparation-001/cpu-failure-review-001.md`.
This diagnostic work does not complete the two missing continuation checks or
authorize rerunning the failed fixture. No further numerical attempt has run.

The [staged tuning draft](TOKENIZER_WIDE_HYBRID_DESIGN.md) now selects bottleneck
rank D/8 and D/2 as the second stage following three-rate anchor screening. It
declares equal per-family budgets, prospective order/selection, fresh triple
comparisons and proposed quality/cost gates. It is not frozen: remaining
continuation prerequisites, measured pilot resources and serial runners still
need implementation/review. A four-check small synthetic serialization
test verifies that fully deserialized data remains usable after closing its
scratch BytesIO; it does not demonstrate lower model RSS or authorize retry.

## Explicit rank variants and complete draft panel accounting

The separate `tokenizer_wide_k3_hybrid_rank_nope_v1` policy now requires an
integer divisor2/4/8. It binds both KDA decay and gated-MLA query/KV ranks toD/b
in model, optimizer, cache and checkpoint contracts. Both earlier quarter-rank
policies still reject the new field. Width1024 remains meta-only.

Seventy-two selected CPU/meta checks passed in21.42s under a600s/4GiB-RSS/
8GiB-reserve guard, including37 rank checks and35 anchor/historical regressions.
Peak recorded process RSS was1,400,913,920bytes. All continuation tests were
excluded; the failed combined fixture was not retried. These checks cover rank
geometry, independent parameter-count formulas, equations/gradients, cached/full
parity, causality/reset, optimizer grouping/scaling, lineage and invalid policies.
They are not optimizer-resume/EMA or training-quality results for the new ranks.

Complete constructed meta accounting for18 width/family/rank combinations is in
`tokenizer-evidence/wide-hybrid-preparation-001/rank-storage-projection.json`.
For27 retained trajectories, reserving D/2 resource pilots and the most costly
possible confirmations, the width512 draft requires57,224,879,357bytes
(53.2948GiB), including8GiB reserve. Width1024 requires126,135,276,067bytes
(117.4726GiB). This replaces the anchor-only projection for the draft design.
It still needs actual pilot export/runtime/memory validation after full fit and
freeze; current local disk does not fit. No MPS pilot or quality launch occurred.

At512/context128/batch1, cache/state bytes by rank divisor are:

| Family | D/8 | D/4 | D/2 |
|---|---:|---:|---:|
| hybrid | 301,056 | 333,824 | 399,360 |
| all-gated-MLA | 196,608 | 327,680 | 589,824 |
| all-KDA | 335,872 | 335,872 | 335,872 |

KDA's rank changes parameterization, not recurrent state size. Thus the hybrid's
memory tradeoff depends on independently selected control ranks. These geometry
facts cannot select ranks or establish quality, latency or adoption.

## Draft selection and conditional-gate arithmetic

Twenty-nine synthetic tests passed for the pure staged-selection and statistics
module. Complete nine-LR/six-rank panels are required; exact ties select lower LR
then larger divisor at the respective stages. Confirmations use the declared
three fresh seeds and both homogeneous controls. Conditional bounds use common
paired resamples, token-weighted tails and the predeclared one-sided99.1667%
linear quantile, distinct from descriptive95% intervals. Memory gates require
every seed/batch/control cell and use integer arithmetic for the exact10% limit.
An independent scalar bootstrap reference and JSON persistence check passed.

Evidence is `tokenizer-evidence/wide-hybrid-preparation-001/rules-review.json`.
This is arithmetic preparation only: no quality result, source freeze, model
continuation retry or launch. Controller dry-runs, saved-score/artifact identity
integration, model optimizer/resume/EMA and actual pilot resource prerequisites
remain outstanding. The storage gate and all closed campaigns remain unchanged.

## Saved-score metadata adapter

The new read-only `src/research/tokenizer_wide_hybrid_scores.py` consumes an
independently expected request, file hashes and training/tokenizer contract.
It regenerates the explicit rank config, checks result identity/completion,
and validates exact target/block coverage and saved CE/BPB/perplexity arithmetic.
It hashes distinct local raw/H200 files before and after inspection, constructs
the expected model on the meta device, and checks checkpoint parameter shapes,
dtypes/names, config, tokenizer and raw-to-export EMA metadata/parent lineage.
It then supplies the selection row and aligned H200 block sums to the pure rules.

Thirty-six meta-only synthetic adapter checks and the29 arithmetic checks passed
(65 total,1.51s; peak process RSS306,724,864bytes). Fixtures carry only meta tensors,
are each below512KiB and load no corpus or weight values. Mutation tests include
updated file hashes so that lineage/shape/arithmetic checks are tested separately
from hash rejection. See `tokenizer-evidence/wide-hybrid-preparation-001/score-adapter-review.json`.

This verifies the adapter's metadata path, not real new trial artifacts, tensor
finiteness, exact EMA values, optimizer state correctness or continuation. The
controller still must bind expected inputs to frozen source/runtime/data and
persisted selections, serialize worker dispatch, and require separate tensor and
serving audits before adoption. No MPS launch or failed-guard retry occurred.

## Serial stage and persistence simulation

Twenty-one new controller-stage simulations plus29 rule regressions passed
(50 checks,0.27s; peak process RSS67,518,464bytes). The simulation executes24
synthetic scalar quality records in the prospective order, persists LR/rank
choices before subsequent callbacks and checks the exact completed prefix.
Exclusive global/local locks, changed provenance/selection, incomplete or
duplicate records, boundary STOP, interruptions and simulated timeouts at each
stage boundary are covered. Existing run directories are never resumed or
overwritten; failure preserves the active trial and completed prefix. State and
selection writes flush/fsync, with selections published exclusively.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/controller-simulation-review.json`.
This is a small isolated synthetic driver, not a tested MPS launcher. It starts
no subprocess, loads no corpus/model and produces no checkpoint. Resource pilots,
production subprocess/watchdog integration, frozen provenance/score adapter
binding, actual filesystem fit and unresolved numerical prerequisites remain.
All failures and closed campaigns remain preserved; no launch gate changed.

## Bounded subprocess primitive

Seventeen isolated subprocess tests passed in2.73s for the fresh-only parent
runner (`src/research/tokenizer_serial_process.py`). Tiny Python children tested
inherited global/run lock descriptors, success/nonzero exits, reserve and full
remaining-allocation rejection, timeout termination including SIGTERM refusal,
post-exit output checks, observer failures and parent interruption. Cleanup
errors retain a separate diagnostic without hiding the original failure. No
model/corpus/checkpoint workload ran; peak test-parent RSS was53,493,760bytes.

The primitive dispatches one explicit argv without a shell, starts a new child
session, preserves exclusive output directories/logs and records terminal errors
without retry. It samples output/log bytes, wall time and the8GiB reserve and
checks the full remaining disk allocation before dispatch. It does not replace
the worker's4GiB RSS/6GiB MPS-driver watchdog or validate frozen prerequisites.
Production pilot/quality/inference integration still needs source/runtime/data
binding, the score adapter, all remaining-arm accounting and finite freeze.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/process-runner-review.json`.
No storage gate, historical controller or failed continuation fixture changed.

## Prospective quality-controller integration

The new quality driver connects exact-prefix scheduling, immutable LR/rank
selections, a shared worker/run lock, full remaining-arm accounting, the bounded
subprocess primitive and the saved-score metadata adapter. It has no prepare or
freeze command. Drafts, changed source/runtime/protocol/projection/corpus/recipe,
missing numerical/pilot reviews or unfrozen underlying prerequisite artifacts
are rejected before any dispatch. Every previous reviewed request/result and
checkpoint hash is rechecked before further work. Completion remains pending
tensor-value and serving audits and never automatically adopts a recipe.

Thirty-eight checks passed in3.44s:20 controller integration cases with mocked
dispatch/score review and18 tiny-process runner checks. Peak test-parent RSS was
230,064,128bytes. Synthetic plan reviews are test fixtures, not campaign evidence.
The integration caught two engineering details: process output guards now count
the entire trial directory including sibling checkpoints, and the new controller
closes lock descriptors without an explicit unlock so a child descriptor retains
exclusion if cleanup is uncertain. A descriptor-inheritance test verifies this.
Historical controllers and their source files remain unchanged.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/quality-integration-review.json`.
This does not establish MPS end-to-end correctness. No frozen new campaign plan,
passing full numerical review or resource pilot review exists; local storage
still fails the complete panel projection. Pilot/inference controllers, those
reviews, actual resources and final freeze remain required. No training or failed
continuation retry ran, and no new quality result exists.

## Inference measurement contract preparation

Twenty-five synthetic callback/tiny-storage tests passed in0.42s; peak process
RSS226,017,280bytes. The new timing helper checks the full correctness barrier,
exact rotated execution order, canonical input-token hash, positive finite
durations, integer memory observations and stable state allocation across
warmups/repeats. A small CPU view/alias fixture verifies unique backing storage
rather than logical view elements. Per-seed repeat positions are intentionally
3/3/2; the declared seed offsets balance each family/position across all three
seeds. Callback failures propagate immediately without retry or completed report.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/timing-contract-review.json`.
No real model, corpus, checkpoint or MPS timing was used. The helper trusts its
callbacks to implement actual numerical checks/measurements and cannot certify
checkpoint, tokenizer or device identity. Production model loading, correctness
callbacks, guarded serial dispatch and persisted-result audit remain required;
the returned status explicitly remains pending artifact audit. No numerical
continuation retry, pilot launch, cleanup or gate change occurred. Local space
remains below the complete53.2948GiB projection; the test holdout stays closed.

## Saved inference-result audit preparation

Twenty-eight new synthetic saved-result checks and25 timing-contract regressions
passed (53 total,0.47s; peak process RSS226,279,424bytes). The new audit binds
bounded saved JSON bytes to an independently expected hash, rejects duplicate
JSON keys, checks every seed/selection/token identity and exact correctness/
warmup/measurement sequence, and recomputes throughput and latency summaries.
Corruption tests cover missing, duplicate and reordered records; changed ranks,
inputs, state allocation and throughput; failed cache flags; nonfinite timings;
integer/boolean identity aliases; and incomplete fresh-seed panels.

Summary tests use deliberately extreme synthetic warmups to verify their
exclusion. The prospective aggregation is now explicit: eight-repeat medians
per seed/family/batch, seed-paired hybrid/control ratios, then the median of the
three ratios. All three seed ratios are retained. The exact10% state-saving
rule requires both controls in every fresh seed/batch; one failing cell rejects
that memory gate. Output remains pending provenance review and never adopts.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/timing-result-review.json`.
No trained checkpoint, corpus or real timing was used, and these tests do not
prove the numerical correctness callbacks or MPS behavior. Provenance-verified
model loading, real cache/timing callbacks, guarded serial wrappers, numerical
prerequisites, resource pilots, storage fit and final freeze remain open. No
closed controller, failure record, reserve or test-holdout boundary changed.

## Cached correctness and timing callback preparation

Sixteen new tiny deterministic recurrence/adapter-rejection checks and53 timing
contract/audit regressions passed (69 total,0.70s; peak process RSS275,578,880bytes).
The synthetic recurrence has two output logits and no trained parameters; it is
not a campaign architecture or a retry of the failed continuation fixture.
Tests verify all96 cached positions, full-forward equality, reset/replay, late
cached drift, nonfinite logits, fresh cache allocation per timing call, exact
synchronization/clock boundaries, failure propagation and strict-MPS rejection.

The new production callback adapter checks the explicit width512 rank policy,
family layout, matching model/cache configs, eval mode and MPS FP32 tensors.
Its kernels compare96 full logits against32prefill+64decode calls, then reset and
repeat correctness; measured timing uses a fresh cache per call. Unique actual
cache backing storage and allocator readings are taken after the timed regions.
No MPS path or real KDA/gated-MLA model was exercised by these synthetic tests.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/measurement-kernel-review.json`.
Checkpoint/tokenizer provenance loading, guarded serial inference integration,
resource pilots, full numerical prerequisites, storage fit and freeze remain
required. No research training, real benchmark, failed-guard retry, cleanup or
closed-controller execution occurred. The held-out test remains closed.

## Selected-export loading integration

Twenty new mocked loading checks and36 saved-score metadata regressions passed
(56 total,1.73s; peak process RSS306,954,240bytes). The new loading helper binds
all three fresh selected family requests to independently supplied selection,
training/tokenizer contracts and request/result hashes. It checks every family
request/config/corpus before tensor loading, invokes the existing raw/H200
metadata adapter, hash-checks exports around CPU deserialization, rejects
nonfinite or non-CPU export tensors, and rebuilds tokenizer/split identity and
first96 training tokens before constructing each model. The complete request,
result, corpus and raw/H200 set is rechecked after all three loads.

Tests use tiny two-element tensors, synthetic corpus bytes, mocked model/device
construction and a mocked score adapter in the integration layer. Separate
metadata regressions use meta tensors; neither set loads a real trained model
or executes MPS. Cases include altered selections/seeds/configs/tokenizers,
changed corpus/input tokens, nonfinite/meta/empty tensors, audit failure,
mutation during deserialization and earlier-artifact changes during later loads.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/export-loading-review.json`.
The helper has no CLI, controller, lock or watchdog and cannot authorize launch.
The future worker must bind frozen source/runtime and complete quality/tensor
reviews, hold the global/run locks, enforce resource limits, and integrate the
measurement/result-audit components. Export finiteness is not exact raw-to-EMA
or optimizer auditing. Resource pilots, numerical prerequisites, full storage
fit and final freeze remain unresolved; all closed campaigns/test boundaries
and retained failure evidence remain unchanged.

## Complete-panel inference prerequisite gate

Thirty-two new synthetic gate checks plus20 quality-driver regressions passed
(52 final checks,1.26s; peak process RSS230,309,888bytes). An initial50-check run
also passed and is retained. The new read-only gate requires a frozen quality
plan with the prospective inference workload/input hashes, the complete24-run
quality prefix, unchanged persisted LR/rank selections and all96 retained
request/result/raw/H200 file bindings. The selected three-family inference
bindings are derived from that reviewed prefix, never from caller-selected
checkpoint paths alone.

A separately hash-bound tensor review must cover the exact24 raw/H200 pairs,
source/runtime, plan and completed state, with positive finite-tensor,
exact-EMA and optimizer-state checks plus hash-verified underlying evidence.
These are required review fields, not a newly executed tensor audit. Tests use
synthetic review documents; no real passing campaign review was created.
The gate rebuilds training/tokenizer contracts only after preliminary checks,
then rechecks source/runtime, STOP and all bound evidence before returning.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/inference-gate-review.json`.
The prospective finite inference contract specifies600s per seed triple,4MiB
output,8GiB reserve,4GiB RSS/6GiB driver and the already declared measurement
sequence. The parent process wrapper must enforce these limits and locks; the
gate itself is read-only and does not dispatch or run a watchdog. Guarded worker
integration, resource pilots, numerical prerequisites, complete storage fit and
final freeze remain outstanding. No real model/corpus was loaded in these tests,
and no training, failed-continuation retry or closed-controller launch occurred.

## Serial inference worker/controller integration

Seventeen new integration checks plus18 bounded-process regressions passed
(35 final checks,3.56s; peak test-parent RSS234,881,024bytes). The initial33-check
run also passed and remains retained. Integration tests invoke the worker inline
with mocked loading, device callbacks, prerequisite binding and memory watchdogs;
separate process tests use tiny Python children. No model, corpus, actual MPS
worker or trained checkpoint was loaded. All test children exited.

The new worker/controller connects the complete-panel gate, selected-export
loader, numerical/timing callbacks and saved-result audit. It checks inherited
global/study lock file identities, never re-enters an existing attempt, preserves
failure state and stops later seed dispatch after any failure. Each child has a
600s parent timeout plus worker RSS/disk/time guard and MPS driver watchdog.
The entire three-triple inference directory, including logs and metadata, shares
a4MiB output cap; this is stricter than assigning4MiB to each child. Free space
must cover the cap above the unchanged8GiB reserve before dispatch.

The parent validates result/model receipts and successful guard observations,
rechecks earlier inference artifacts and quality provenance before later seeds,
and requires all three triples before publishing a non-adopting summary.
STOP at a seed boundary pauses further work; STOP after the last child also
prevents final summary publication. Tests cover each-seed load failure, corrupt
results/model receipts/guard records, changed previous evidence, full allocation
and output-cap rejection, lock conflict, missing prerequisites and no re-entry.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/inference-integration-review.json`.
The real MPS loading/timing path and model-metadata audit remain unexercised in
this integration suite. No complete numerical/pilot/tensor reviews or frozen
campaign plan exist, so this code does not authorize a current launch. Next
preparation is the finite resource-pilot wrapper and its audits, plus reviewing
remaining numerical-prerequisite procedure and storage fit. The failed combined
continuation fixture remains closed; no retry, cleanup or holdout access occurred.

## Finite resource-pilot configuration and retained-panel accounting

Twenty-eight new pure configuration/accounting and two-scalar EMA checks plus20
mocked quality-controller regressions passed (48 checks,1.70s; peak test-parent
RSS272,957,440bytes). The new helper specifies three prospective width512 NoPE
D/2 pilots, ordered hybrid/gated_mla/kda, seed3491, LR.0025 and20updates. The
highest grid LR is chosen prospectively as a short numerical stress check; no
result supports stability or quality claims. Cooldown starts13, H200 snapshots
are13–20 at interval1, and validation cadence is20. Pilots cannot select recipes.

The full initial reservation exactly matches the retained53.2948GiB projection.
After each audited pilot, only its allocation is removed from required remaining
free space; all24 quality runs remain reserved, with worst-case confirmation
ranks and unchanged metadata/reserve. Tests reject partial-pilot-only budgets,
invalid ordered prefixes, insufficient space by one byte and corrupt projections.
The two-scalar EMA test agrees with an independent scalar recurrence and eight
snapshots. It does not instantiate GPT, perform optimizer continuation, read
corpus/checkpoints or retry the failed combined fixture. Controller regressions
use mocked workers. All tests ran under a180s/4GiB RSS/8GiB reserve CPU guard.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/pilot-config-review.json`.
No MPS worker or training launched, no checkpoints were deleted, and the holdout
remains closed. The helper has no dispatch/freeze API. Numerical prerequisites,
resource-pilot worker/audits, actual resource measurements, complete storage fit
and final frozen plans remain unresolved. Next preparation is the guarded pilot
wrapper and artifact review, without reopening closed controllers or the failed
combined continuation fixture.

## Pilot saved-artifact metadata audit

The prospective adapter now has a separate `review_pilot` entry point sharing
raw/H200 metadata checks with the quality adapter. It requires the exact declared
width512/D2/seed3491/LR.0025/20-update request and regenerated short schedule,
independently expected tokenizer/training contract and immutable file hashes.
Both checkpoints must match meta-model tensor geometry; the raw endpoint must
carry online optimizer/EMA metadata and the H200 export must carry matching
parent hash/tokenizer, inference-only purpose,20 completed updates and eight
snapshots. Full-validation coverage and diagnostic score arithmetic are checked.
Pilot receipts have no selection row and explicitly remain pending tensor, cache
and resource audits. Quality records cannot be relabeled as pilots, and pilot
records cannot enter the quality adapter. No selection or quality thresholds changed.

Thirty-nine new pilot cases plus84 regressions passed (123 checks,2.52s; peak
RSS308,461,568bytes) under the180s/4GiB RSS/8GiB reserve CPU guard. Tests construct
meta-device geometry and tiny metadata-only checkpoint files, not real tensors,
trained checkpoints or corpus data. They cover each family, identity/schedule
changes, invalid scores, incomplete trajectories, parent/tokenizer mismatch,
tensor shape/dtype metadata and wrong export purpose. Existing quality cases,
pilot configuration checks and mocked controller regressions also passed.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/pilot-artifact-review.json`.
This is not a finite-tensor, exact-EMA, optimizer-continuation or numerical-cache
audit. A guarded pilot worker/controller, full prerequisite binding and actual
resource/tensor/cache reviews remain needed. No training/MPS launch, failed-guard
retry, deletion or holdout access occurred. Next work is explicit bounded tensor
value/export audit preparation and guarded pilot integration, retaining the
complete-panel storage gate and unresolved numerical prerequisites.

## Bounded tensor-value and stored-EMA export audit preparation

The new read-only `tokenizer_wide_hybrid_tensors.py` audit first invokes the
independently bound pilot or quality metadata adapter, then maps one raw/H200
pair on CPU. It checks finite raw model, stored EMA, optimizer tensor/scalar and
export values. Every exported parameter must exactly equal the corresponding
stored H200 tensor; persistent buffers must exactly equal the raw endpoint.
Temporary finiteness/equality operations use chunks of262,144 elements and do
not construct another model or explicit file-sized byte buffer. All request,
result and checkpoint hashes are rechecked after inspection. Mapping is not a
measured peak-memory guarantee: the future caller must hold locks and enforce
time/RSS/disk limits. No dispatcher, freeze or retry API was added.

Forty-five new tiny-tensor/file checks plus75 metadata-adapter regressions passed
(120 checks,2.45s; peak test RSS309,346,304bytes) under the180s/4GiB RSS/8GiB
reserve guard. New cases cover NaN/infinities in each tensor group, exact export
and buffer mismatches, geometry/name errors, missing optimizer tensors, malformed
optimizer scalars, meta/sparse/noncontiguous tensors, chunk tails, before/after
hash changes and a missing metadata prerequisite before loading. File integration
uses synthetic checkpoint pairs under16KiB with a mocked metadata reviewer;
separate metadata tests construct only meta geometry. No real research weights,
corpus or MPS model were loaded, and no optimizer step or continuation was run.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/tensor-export-review.json`.
The report explicitly distinguishes exact export-to-stored-EMA equality from
correctness of the EMA recurrence. Finite optimizer values do not establish
optimizer structure or continuation correctness. Those numerical prerequisites,
cache checks, actual resource observations and guarded pilot integration remain
required; the helper cannot produce a complete passing numerical/pilot review.
No failed fixture was retried, no evidence deleted and no holdout accessed.
Next preparation is guarded pilot integration and its prerequisite/resource
review contracts. Complete storage fit and a final frozen plan remain necessary.

## Frozen pilot prerequisite gate preparation

The new read-only `tokenizer_wide_hybrid_pilot_gate.py` binds each prospective
pilot to a frozen source/runtime, exact three requests and full conditional
27-trajectory design. The design names all nine LR screens, six conditional
rank probes, nine fresh confirmations, selection/tie rules, both controls and
quality/state/bootstrap thresholds. It does not choose outcome-dependent LRs
or ranks. The frozen storage contract must equal the complete retained-panel
projection, not a pilot-only allocation. Corpus, inherited recipe, design,
projection, numerical review and underlying numerical evidence must be hash-bound.

Numerical review must cover every width512 family at divisors8/4/2 with passing
geometry, cache, optimizer, resume and EMA checks, under matching source/runtime
and corpus identity. These are required review fields; no such real complete
review was produced. Source/runtime and all file hashes are rechecked after
review, with STOP checked before and after and MPS fallback rejected.

Forty-three new synthetic-gate checks plus28 configuration/accounting regressions
passed (71 checks,.98s; peak RSS272,891,904bytes) under the180s/4GiB RSS/8GiB
reserve CPU guard. Tests use synthetic frozen documents, mocked source/runtime
and synthetic corpus identity, never actual training data or a real numerical
review. They reject altered budgets/thresholds/seeds/order, missing or rehashed
incomplete numerical cases, unbound evidence, source drift, STOP and fallback.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/pilot-gate-review.json`.
The gate neither freezes a plan nor acquires worker locks, measures free space
or launches anything. The future controller must independently establish the
audited retained prefix, full remaining storage fit, exclusive attempt/locks
and resource watchdogs before dispatch. No real plan is frozen, numerical
prerequisites remain incomplete, and the full panel still does not fit locally.
Next preparation is the guarded serial pilot worker/controller and its result
review, with no failed-fixture retry, evidence deletion or closed-controller run.

## Raw/H200 pilot cache audit integration preparation

The new pilot cache helper first checks the finite pilot/config/tokenizer/corpus
bindings and invokes the tensor/export audit. It loads raw then H200 checkpoints
one at a time on MPS, releasing the model and CPU mapping before the next
endpoint. Each endpoint rebuilds the tokenizer/splits and must recover the frozen
first96 training token IDs. The pilot prerequisite gate now requires an explicit
nonempty tokenizer identity, canonical token hash and the cache workload contract
before any future dispatch. No actual tokenizer input or plan was frozen here.

At batches1/8 the shared numerical kernel compares full96-token logits with a
32-token prefill plus64 one-token cached calls, then resets all caches and replays
the same sequence. Existing rtol3e-4/atol3e-5 and nonfinite rejection remain.
The shared single-model validator retains the explicit width512 NoPE rank policy,
family layout, model/cache config identity, eval mode and MPS FP32 checks. Actual
parameter storage/count and unique backing cache storage must match the D/2
constructed projection. Allocator samples remain observations, not peak claims.

Eighteen new mocked cache-integration checks and three new gate checks plus104
regressions passed (125 checks,1.54s; peak RSS276,480,000bytes) under the180s/4GiB
RSS/8GiB reserve guard. New integration tests use tiny checkpoint tensors, mocked
GPT/tokenizer/MPS/numerical callbacks and mocked tensor review. Separate retained
regressions exercise the numerical recurrence on a tiny CPU fixture. New checks
cover serial model release, either-endpoint failures, changed corpus/tokenizer/
config/tokens/files, nonfinite payloads and projected state/parameter mismatch.
No real GPT weights, research corpus, MPS worker or training trajectory ran.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/pilot-cache-review.json`.
The helper has no worker/CLI and requires caller-held locks, frozen prerequisites,
full-panel fit and resource watchdogs. Actual MPS cache correctness and peak
memory, numerical prerequisites and resource reviews remain unresolved. Next
preparation is the serial pilot worker/controller joining these components.
No failed-guard retry, cleanup, closed-controller run or holdout access occurred.

## Guarded serial pilot integration preparation

The new pilot worker/controller connects the frozen prerequisite gate, exact
three requests, common training/export/validation worker, raw/H200 tensor/cache
audit and saved-result review. It requires inherited global/study lock identities
and a matching active parent prefix, rejects existing attempts, checks frozen
tokenizer/training IDs before training and preserves failures. Every dispatch
reserves all remaining quality trajectories plus remaining pilots and metadata
above8GiB. Per-pilot output caps and the aggregate pilot allocation are checked.
A600s parent timeout covers worker startup, training, validation/export and audits;
a worker CPU RSS/disk/time guard and the common training worker's4GiB RSS/6GiB
MPS-driver watchdog apply. Sampled guards do not establish instantaneous peaks.

Before the next family, the parent rechecks raw/H200 metadata, tensor/export
receipts, both endpoint/batch cache checks, projected parameters/state, observed
allocator limits and successful guard records. It retains hash-bound requests,
results, checkpoints, audit reports, worker/guard/process journals and logs.
Earlier evidence and frozen provenance are checked again before later dispatch.
STOP at a boundary prevents later pilots; STOP after the last child prevents
final completion. Completion remains pending resource and numerical review,
never a passing pilot prerequisite or automatic quality launch. No prepare/freeze
or resume CLI was added; only the guarded worker entry exists.

Twenty new mocked integration checks plus18 tiny-process regressions passed
(38 final checks,3.59s; peak test-parent RSS230,768,640bytes). An initial35-check
run also passed and is retained. Integration invokes workers inline with real
lock files and mocked training/data/prerequisite/tensor/cache/guard components;
separate subprocess tests use tiny Python children. Checks cover each-pilot
failure, corrupted guard/cache/token/geometry/EMA/file receipts, prior-evidence
changes, full-panel space, metadata cap, missing prerequisites, global lock
contention, inherited-lock identity, input mismatch before training and no reentry.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/pilot-integration-review.json`.
No actual model, research corpus, checkpoint or MPS worker ran. All test children
exited. Real integrated training/export/cache behavior and peak resource costs
remain unverified; no complete nine-case numerical review or fitted frozen plan
exists. The failed combined continuation fixture remains closed. Next work is
reviewing the integrated path and exact outstanding numerical/resource evidence
without failed-guard retry, partial launch, cleanup or holdout access.

## Integrated readiness review and projected memory-route constraint

A read-only review now distinguishes retained numerical evidence from simulated
controller coverage. All nine width512 family/rank configurations have retained
meta geometry/optimizer-group checks and seven-token CPU full/cache/reset/
causality/lineage checks at batches1/8. Those meta checks do not execute optimizer
updates or reload optimizer state. The one completed hybrid continuation in the
failed combined process used the older quarter-rank WIDE_POLICY; it is not a
complete RANK_POLICY continuation result. No nine-case rank-policy optimizer/
resume/EMA review exists, and the96-token MPS workload remains unmeasured.
The failed combined fixture stays closed. Mocked worker success cannot fill
these gaps or establish real training/export/cache memory feasibility.

The same review enumerated all27 possible selected rank triples using retained
meta-device cache counts, independently checked the layer-state formula, and
cross-checked exact rational savings against the existing integer memory gate.
At width512/context128/FP32, only hybrid D/8 with all-gated-MLA D/2 and any KDA
rank can meet the current10% reduction against BOTH controls:3 of27 triples.
Batch1 projected state is301,056bytes for that hybrid,589,824 for gated MLA and
335,872 for KDA, giving48.9583% and10.3659% savings respectively. Batch8 scales
all three equally. Same-rank selections cannot pass this memory threshold.
These are architecture projections, not fresh measured memory or quality results;
other workloads can have different tradeoffs.

Keep validation-CE rank selection unchanged. Do not select a worse-quality rank
to manufacture a memory pass, weaken the threshold after results, or treat this
projection as adoption evidence. The strict-quality route remains available for
other selected triples if its full fresh gates pass. The10% threshold, like the
+.01-nat margin, is a prospective agent engineering convention, not a human
numeric tolerance. No threshold, rank, rate, seed, order or workload changed.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/readiness-review-001.json`
and its reproducible `review_readiness.py`. This iteration performed source/
retained-evidence review and27 synthetic arithmetic checks only; no model,
corpus, checkpoint or MPS worker was loaded and no numerical fixture was retried.
Full accounting remains57,224,879,357bytes including the8GiB reserve, versus
about16.40GiB observed free. No plan was frozen or trial launched. Actual
numerical prerequisites, full storage fit, measured pilot resource review and
final provenance/fresh-seed freeze remain required before the quality panel.

## October 28 human-authorized storage policy change

The human explicitly authorized cleaning non-critical checkpoint weights moving
forward instead of silently remaining blocked. The earlier prohibition on further
cleanup/external-storage-only resolution is superseded by
[TOKENIZER_CHECKPOINT_RETENTION.md](TOKENIZER_CHECKPOINT_RETENTION.md). Other
stopping, numerical, quality, data and resource constraints remain unchanged.
Cleanup002 removes successful closed pilots and unselected table/wide screens;
all selected screens and all fresh confirmation raw/H200 pairs remain protected.

A new prospective full-panel accounting retains three final selected screen pairs
and all nine fresh confirmation pairs, while pruning audited pilot weights and
losing screens only at completed stage boundaries. It enumerates all 27 possible
final rank selections, uses the original per-trajectory allowances, increases the
round metadata/log allocation to 256 MiB, and preserves the 8 GiB reserve. See
`tokenizer-evidence/checkpoint-cleanup-002/prospective-retention-projection.json`.
Its complete campaign high-water requirement is 30,631,281,412 bytes (28.5276 GiB),
compared with the prior all-retained requirement of 57,224,879,357 bytes.

This changes retention, not the 27 trajectories, rates, seeds, order, rank/CE
selection or adoption gates. The current v1 runners still require all weights to
exist and cannot execute this retention design. Next work is a versioned,
manifest-aware retention contract with tested stage-boundary audits/pruning and
full future high-water checks; no fake passing reviews or bypass of current guards.
Nine-case numerical prerequisites and actual pilot resource reviews still remain,
and the failed combined continuation fixture remains closed. No training has been
launched by this cleanup or accounting change.

## October 28 v2 retention planning and receipt components

Added separate `wide_hybrid_retention_v2` planning and read-only provenance
components; existing v1 runners and guards are unchanged. Boundary candidates
are recomputed from the exact ordered pilot/LR/rank prefix and the original CE
selection rules. Every member of a stage requires audit, including winners.
All 27 rank choices retain exactly three final screen pairs; every fresh
confirmation remains ineligible for cleanup. The complete high-water envelope
reconciles to the independent cleanup002 projection: 30,631,281,412 bytes,
including the unchanged 8 GiB reserve and 256 MiB metadata allowance.

The receipt primitive requires an independently reconstructed exact manifest,
bound manifest/receipt hashes, a complete deletion journal, unchanged bound
non-checkpoint evidence and unchanged retained screen pairs. It rejects partial
or failed receipts, changed allowlists, reappearing deleted files, altered/missing
protected files and symlink artifacts/ancestors. It never deletes anything or
approves dispatch. Its caller must establish full audit semantics, evidence
inventory completeness and persisted selection provenance; those production
integrations remain outstanding. A caller-supplied manifest is not scientific
review evidence by itself.

Validation: 99 planning/selection/simulation tests and 24 tiny-file receipt tests
passed in separate bounded CPU processes (180 seconds, 4 GiB RSS, 8 GiB disk
reserve). Observed peak RSS was 68,091,904 and 62,914,560 bytes respectively.
Only disposable synthetic test files were deleted. No corpus, real checkpoint,
model update, MPS worker or failed continuation fixture was used. Evidence and
source hashes: `tokenizer-evidence/wide-hybrid-preparation-001/retention-v2-component-review.json`.

Next: versioned full-stage pilot/quality/inference audit bindings, guarded actual
cleanup and protected inventory verification, complete serial lifecycle tests,
and future storage checks based on verified retained/deleted state. Nine real
rank-policy numerical continuation/EMA cases and measured pilot resource checks
are still unresolved. No plan is frozen or launch enabled by these components.

## October 28 v2 quality-stage audit binding

The new read-only quality-stage binder requires every LR/rank stage member,
including selected winners, to have matching request/result, tensor-value,
scalar validation and resource evidence before deriving a removal manifest.
It rechecks the saved CE decisions, independent frozen-context hashes and all
nine rank-policy numerical review cases and trace files. Exact stored-EMA export
agreement is explicitly kept separate from optimizer-resume/EMA-recurrence
correctness; the former cannot substitute for the missing numerical work.

The binder verifies checkpoint hashes, immutable metadata, all files inside the
stage trial directories, and the existing full-validation coverage/arithmetic.
It rejects omitted files, symlink paths, changed/partial evidence and resource
limit violations. Selected pairs are protected. It only returns a manifest;
it neither deletes files nor approves dispatch. Campaign-wide inventory and the
independent frozen-plan gate are still required. The prospective v2 resource
report format must be produced by a guarded wrapper; existing v1 outputs do not
satisfy it. Pilot cache/resource binding and controller integration remain open.

Validation: 39 new synthetic checks, 112 passing checks with planning/receipt
regressions, in 1.12 seconds. CPU guard: 180 seconds, 4 GiB RSS, 8 GiB reserve;
observed peak RSS 229,703,680 bytes. An earlier 34-check run passed before adding
stage-directory inventory coverage; both logs are retained. These are JSON and
tiny byte fixtures, not numerical/model-quality evidence. No real checkpoints
were removed and no model/corpus/MPS workload or failed fixture was executed.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/retention-v2-quality-audit-review.json`.

Next work remains v2 resource production, pilot/full-panel provenance gates,
locked cleanup with durable deletion journals and protected postchecks, full
future high-water accounting and serial lifecycle tests. Nine numerical cases
and measured pilot resources remain prerequisites. No plan is frozen.

## October 28 v2 resource recording component

Added a fresh-attempt quality resource recorder with its own bounded, flushed and
fsynced event file. It samples RSS/MPS-driver/free-disk/elapsed observations and
preserves a terminal failure independently of captured stdout/stderr. It keeps
4 GiB RSS, 6 GiB MPS driver, 8 GiB reserve and 7,200-second limits. A completed
report requires explicit success plus unchanged request/result hashes and a
completed closed-holdout result. Missing success, worker exceptions, observer
errors and exceeded limits cannot produce an accepted completed report.

The quality-stage binder now requires exact event-log/summary agreement and
binds both hashes into retained evidence. This supplements the existing memory
watchdog and serial parent; those remain in place. The older watchdog may exit
first, leaving an incomplete trace, which must block acceptance. Sampling does
not guarantee instantaneous peaks or complete diagnostics after an OS kill.

Event log and report are each bounded to 2 MiB, with at most 7,203 samples.
Worst-case resource pairs plus 4 MiB child logs across 27 trajectories consume
216 MiB of the prospective 256 MiB metadata allowance. The remaining 40 MiB
must cover the complete other metadata inventory and be enforced by the future
controller; this is not yet a full observed inventory or a freeze authorization.

Validation: 62 synthetic tests passed in 2.98 seconds, including captured-output
hard exits, a background driver-limit failure, event/sample caps, partial traces,
immutable result bindings and stage-audit regressions. The CPU guard observed
231,260,160 bytes peak RSS under 180s/4GiB/8GiB limits. The earlier 59-test pass is
also retained. No MPS device, model, corpus or real checkpoint was loaded; no
training or real cleanup occurred. Evidence:
`tokenizer-evidence/wide-hybrid-preparation-001/resource-v2-component-review.json`.

The recorder is not yet wired to a v2 training controller. Pilot/full-panel
provenance, campaign inventory, guarded boundary deletion and full serial
lifecycle integration remain. All nine numerical prerequisites and measured
pilot resources are still unresolved; the failed combined CPU fixture stays
closed. No scientific plan or resource threshold changed.

## October 28 guarded boundary cleanup component

Added an internal v2 cleanup executor with no CLI and no production caller yet.
It requires inherited global/campaign lock identities and trusted controller
callbacks for independent authorization and repeated frozen-context checks.
It verifies original audited hashes and the complete immutable evidence inventory
before removal, records/fsyncs each intent before unlink, fsyncs the directory,
then records/fsyncs completion. All protected files are rehashed afterward.
A completed receipt alone is insufficient: the final execution marker also binds
the full journal, guard, inventory and manifest. Any existing attempt is closed
to automatic re-entry, including an interruption between unlink and its journal
completion. This never authorizes restoring or silently regenerating weights.

Validation: 20 new cleanup tests and 24 receipt regressions passed (44 total,
0.50 seconds). Tests cover altered hashes, omitted inventory, wrong locks,
symlinks, context changes, unlink errors, protected postcheck failures, metadata
caps and a real subprocess hard exit immediately after a **synthetic** unlink.
The durable intent survived that exit and no completed marker appeared. Only
disposable synthetic bytes were removed; no retained model weight was touched.
The outer CPU guard observed64,323,584bytes peak RSS under180s/4GiB/8GiB limits.
Earlier incremental passes are retained as well.

Executor limits are600seconds,4GiB RSS,8GiB reserve,1MiB per document/journal,
5MiB per completed cleanup attempt and2,048 inventoried files. Three cleanup
budgets plus resource/log caps consume231MiB of the256MiB prospective metadata
allowance, leaving25MiB for the remaining complete metadata inventory. That
inventory and cumulative controller enforcement still precede freeze.

One integration issue is explicit: an LR anchor can legitimately lose the later
rank comparison. The current execution rechecker is boundary-local and expects
its then-protected files. Final provenance must instead validate a cumulative
chain of separately authorized cleanup receipts for such later retirements;
never add an arbitrary missing-file exception or permanently protect a losing
anchor to work around this. The immutable evidence tree must also be separated
from mutable controller state in the future frozen v2 layout.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/cleanup-v2-component-review.json`.
Full plan/stage-audit/idle-worker callbacks, cumulative receipts, pilot/full-panel
gates and runner/storage integration remain. No production cleanup or training
was launched. Nine numerical cases and measured pilot resources remain missing;
the failed combined fixture and all closed campaigns remain closed.

## October 28 cumulative cleanup provenance

The read-only v2 chain verifier now supports an LR winner legitimately losing
the later rank comparison. It requires the exact pilot/LR/rank prefix, every
existing cleanup attempt, the completed execution markers and unchanged stage
inventories. Rank cleanup must refer to the identical LR anchor pairs by index,
role, path, size and original hash. Every other earlier protected file remains
protected. The final complete inventory is checked against actual files; deleted
weights cannot reappear. Even an empty interrupted later attempt blocks accepting
an earlier prefix. Existing boundary-local validation remains strict.

Validation: 28 new chain tests plus 44 cleanup/receipt regressions passed, 72
total in 2.27 seconds. These include legitimate anchor replacement and retention,
missing or reordered boundaries, protected-pair substitution, altered original
inventories, resurrected weights, omitted evidence and interrupted attempts.
The CPU guard recorded 67,026,944 bytes peak RSS under 180s/4GiB/8GiB limits.
Both the initial 68-test pass and final pass are retained in
`tokenizer-evidence/wide-hybrid-preparation-001/retention-chain-v2-review.json`.
Only disposable synthetic bytes were deleted; no production cleanup or training.

This component still relies on independently reconstructed stage manifests and
controller-bound current inventory/context, under the worker locks and resource
guard. It returns no dispatch authorization. Versioned pilot/full-panel gates,
immutable evidence layout, remaining metadata accounting, complete future storage
gates and runner integration remain before freeze. Nine numerical cases and
measured pilot reviews are unresolved; the failed combined CPU fixture stays
closed. No quality threshold, scientific trajectory or reserve changed.

## October 28 complete prospective metadata accounting

Review found that the earlier216MiB trajectory resource/log budget plus15MiB
cleanup budget had not explicitly allocated the three4MiB inference logs.
The new v2 layout allocates the complete existing256MiB allowance:

| Output category | Maximum MiB |
|---|---:|
| 27 trajectories: logs, resource traces/reports and other metadata | 222.75 |
| Three cleanup attempts | 15 |
| Three inference comparisons: logs and other metadata | 12.75 |
| Immutable campaign plans/reviews | 2 |
| Mutable controller state/logs, including temporary overlap | 3 |
| Emergency failure evidence | 0.5 |
| Total | 256 |

Each trajectory retains the4MiB log and2MiB each resource trace/report limits,
plus a prospective256KiB combined allowance for request/result, metrics,
training summary, process, tensor/cache and other audit metadata. Each inference
comparison has4MiB log plus256KiB other metadata. These are new prospective v2
output caps, not changes to historical evidence or scientific/memory/time rules.
A read-only observation of14 closed wide/pilot trajectories found at most25,217
bytes of v1 trial metadata; that excludes new v2 outputs and is not proof of
future sizes or transient compliance.

The layout separates `evidence/` from mutable `control/` and `emergency/`, with
three pilot slots,24 quality slots and three fixed inference seed slots. The
census rejects undeclared paths, symlinks, nonregular files and unexpected hard
links. Distinct atomic temporary files count at full size; the publisher's exact
final/temporary hard-link pair counts once. Actual checkpoint `.pt.tmp` writes
are recognized and included in separate weight totals, without cleanup credit.
The write-capacity helper reserves new temporary payload space without subtracting
the old file. Group caps cannot borrow unused space from another category.

Validation:59 synthetic tests passed in0.29seconds, peak RSS61,898,752bytes under
180s/4GiB/8GiB guards. The initial56-test pass is retained. Evidence:
`tokenizer-evidence/wide-hybrid-preparation-001/metadata-v2-component-review.json`,
`metadata-layout-v2.json` and `metadata-v2-historical-size-observation.json`.
The full campaign projection remains30,631,281,412bytes including8GiB reserve;
actual free space must be checked again before any freeze/launch.

No production writer or controller uses this layout yet. Idle census checks do
not establish active-producer transient limits: bounded writers, worker monitoring,
full future high-water gates and versioned plan/runner integration remain required.
No model, corpus or MPS worker was used; no real weights were deleted. All nine
numerical cases and measured pilots remain outstanding; the failed CPU fixture
and closed campaigns remain closed.

## October 28 v2 complete campaign identity gate

Added a read-only `wide_hybrid_campaign_v2` plan binder at the prospective
`evidence/meta/plan.json` location. It binds the existing scientific design,
retention contract, complete metadata layout, storage projection, pilot/quality
limits, cache contract and serving workload to frozen source/runtime and
corpus/recipe/protocol/projection hashes. The gate has no prepare, freeze or
dispatch command; no real v2 plan exists yet.

The plan includes three pilot requests and108 conditional quality-request
options:9 LR requests,18 rank requests and81 fresh requests. These are possible
requests, not extra trajectories. Execution remains3 pilots plus9 LR screens,
6 rank probes and9 fresh confirmations. Each future selected family LR/rank
maps to exactly one request at its existing slot, seed and order.

Nine numerical cases with optimizer/resume/EMA and all other required checks,
plus their trace files, are mandatory. The v2 numerical review uses paths relative
to the immutable evidence root, shared with stage-audit binding. A separate
training-input review binds96 training IDs, their canonical digest, tokenizer,
corpus and source/runtime identity. Neither a syntactically valid token digest
nor unbound pass flags suffice. Actual evidence producers still need to perform
and record those checks; synthetic fixture reviews are not research passes.

Validation:51 tests passed in1.71seconds, including all729 family LR/rank choice
combinations, changed scientific/resource/configuration fields, missing numerical
cases/traces, changed input IDs, STOP and source changes. Peak RSS233,488,384bytes
under180s/4GiB/8GiB guards. Initial48-test evidence is retained. See
`tokenizer-evidence/wide-hybrid-preparation-001/plan-v2-component-review.json`.

The returned report explicitly leaves current-storage fit, measured-pilot review,
producer enforcement and dispatch authorization false. Measured-pilot acceptance,
bounded producers, complete future storage checks, controller/stage/cleanup
integration and v2 inference acceptance remain required. No model, corpus or
MPS worker was used in these synthetic tests. The failed combined CPU fixture
and closed campaigns remain closed; no numerical retry or launch occurred.

## October 28 v2 pilot resource and complete-review components

Added a separate600-second pilot resource recorder and read-only three-pilot
acceptance binder. The quality recorder retains its7,200-second contract. Both
retain4GiB RSS,6GiB MPS driver and8GiB reserve limits; sampled observations are
not instantaneous limits. The pilot terminal event and resource summary now bind
both the completed result and completed tensor/cache audit hashes. A missing or
changed audit cannot be declared complete after the resource record has closed.

The pilot boundary requires all three families, raw/H200 pairs, full validation
metadata, exact stored-EMA export audit, cache checks at batches1/8, expected
parameter/state sizes, numerical prerequisite traces and resource/process/log
provenance. Every file under each pilot slot must be hash-bound or an explicitly
listed checkpoint; altered/aliased weights and unbound artifacts block acceptance.
It compiles a candidate six-file pilot cleanup manifest without deleting anything
or granting dispatch permission. All audit execution, full campaign binding,
inventory/locks, storage and producer gates remain necessary.

Validation:204 tests passed in4.74seconds, peak RSS239,026,176bytes under
180s/4GiB/8GiB outer guards. Earlier143/180-test passes are retained. Tests include
all three family failures, incomplete/changed numerical and cache evidence,
resource-profile substitution, missing terminal traces, changed result/audit
hashes and a tiny subprocess exiting90 at the600-second synthetic clock boundary
while retaining its independent diagnostic stream. No real model, corpus or MPS
worker was used. Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/pilot-audit-v2-component-review.json`.

The v2 cache wrapper must bind its immutable campaign projection copy; the closed
v1 producer/controllers are unchanged. Bounded producers, complete current/future
storage gates, controller/audit/cleanup integration and v2 inference acceptance
are still required. All nine real numerical cases and measured pilot checks remain
outstanding; the failed CPU fixture is closed and was not retried. No real plan
was frozen or launched. The latest free-space observation was30,657,261,568bytes,
only25,980,156bytes above the unchanged30,631,281,412-byte full-panel requirement;
this transient observation grants no launch permission.

## October 28 bounded v2 metadata writers

Added an internal metadata writer for prospective v2 producers. Immutable JSON
uses exclusive publication, and append streams must start in a new file. Only
mutable `control/` JSON can be atomically replaced. Each write checks its shared
group budget and8GiB disk reserve before writing payload bytes. Atomic replacement
includes old/new overlap; newline bytes and transient file names also count.
A failed serialization or I/O operation leaves its partial file for review and
blocks reentry. Failed control replacement preserves the previous completed file.
Checkpoint writes are explicitly excluded and need their own full trajectory gate.

The writer verifies inherited global/campaign lock identities. Because parent and
child share those lock descriptors, a separate empty `control/.metadata.lock`
serializes metadata operations, with a bounded2-second acquisition wait. This
file uses the existing control budget; the256MiB total and30,631,281,412-byte full
campaign projection remain unchanged. It rejects unexpected symlinks/hardlinks,
changed append-file identity/size, undeclared/reserved paths and file-count overflow.

Validation:113 tests passed in0.66seconds, peak RSS71,008,256bytes under
180s/4GiB/8GiB outer guards. The initial90-test pass is retained. Two tiny child
processes sharing inherited worker locks attempted to exceed one group together;
only one write succeeded, and the other was rejected before payload output.
Other tests cover exact boundaries, missing newline capacity, disk reserve,
partial I/O, serialization failure, lock timeout and atomic old/new overlap.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/metadata-writer-v2-component-review.json`.

This producer is not yet connected to the runner or resource recorder. Unconverted
writers cannot claim its bounds, and active checkpoint export/rename interactions
with the fail-closed census still need integration tests. Full future storage
gates, v2 controller/audit/cleanup/inference integration and all real numerical,
input and measured-pilot prerequisites remain outstanding. No training, real
checkpoint deletion, failed-fixture retry or plan freeze occurred.

## October 28 resource-recorder / bounded-writer integration

The prospective campaign pilot and quality resource recorders now route their
independent event streams and final reports through the bounded metadata writer
at the exact declared trajectory slots. Existing600s/7,200s profiles, completed
result/audit bindings, failure codes and no-reentry rules remain intact. These
adapters are concrete producer integration; no training controller invokes them yet.

A namespace interaction test interleaved resource samples with20 tiny checkpoint
temporary-file creation/publication cycles. Coordinating these transitions under
the same metadata lock kept the census complete and counted each final weight
once. An uncoordinated disappearing artifact fails closed. The future real
checkpoint producer must implement that coordination and retain its own full
trajectory/output/storage guards; this test is not torch.save feasibility evidence.

Validation:205 tests passed in3.25seconds, peak RSS242,630,656bytes under
180s/4GiB/8GiB outer guards. Initial165-test evidence is retained. Two tiny child
processes exercised synthetic600-second and6GiB-driver violations: both exited90
and preserved bounded resource diagnostics independently of captured stdio.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/campaign-resources-v2-component-review.json`.

Complete current/future high-water storage gates, bounded checkpoint/log/state/cache
producers, controller/stage/cleanup-chain integration and v2 inference acceptance
remain required. All real numerical/input and measured-pilot prerequisites are
still outstanding. No model/corpus/MPS worker, real checkpoint deletion, plan
freeze or failed-fixture retry occurred; closed campaigns remain closed.

## October 28 complete remaining-campaign storage gate

Added a read-only v2 storage gate for idle, audited boundaries. Initial full-panel
required free space remains30,631,281,412bytes. Later checks retain8GiB reserve and
the full256MiB metadata budget, plus the larger of:

- all outstanding trajectory allowances in the current stage; and
- the largest remaining phase's full checkpoint allowance, less verified capacity
  already occupied by live campaign checkpoints.

This second quantity accounts for existing disk occupancy; it never increases the
observed free bytes or treats planned deletion as completed. Sparse/compressed
logical bytes receive credit only up to allocated bytes. The current-stage check
still requires enough actual free space to finish that stage. Cleanup release is
not guaranteed by logical sizes or receipts; actual free space must pass again
after each completed boundary. Final persisted rank choices may narrow the future
confirmation peak without changing scientific selection or removing any of the
three selected screens or nine fresh pairs.

The gate requires all original completed raw/H200 pairs, including previously
deleted ones; independently reconstructs each cleanup manifest from stage records
and saved selection; and verifies the complete execution chain and current evidence.
Unexpected or unreviewed output, missing receipts, aliased/changed weights,
over-budget pairs, STOP, changed free space and same-size evidence changes block
acceptance. File identity, size, timestamps and allocation stamps are rechecked
at the end. Returned storage checks do not authorize dispatch or claim that the
separate frozen-plan/numerical/stage audits were executed.

Validation:191 tests passed in3.00seconds, peak RSS67,862,528bytes under
180s/4GiB/8GiB guards. The initial128-test pass is retained. Tests include all27
final-rank combinations, a complete synthetic27-trajectory sequence with three
real cleanup executions over disposable tiny bytes, exact disk threshold/one-byte
shortfall, sparse allocation, missing/changed receipts and same-size mutation.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/storage-v2-component-review.json`.

No real training, retained research checkpoint deletion, plan freeze or failed
fixture retry occurred. Bounded checkpoint/log/state/cache producers and complete
v2 controller/stage/inference integration remain required, together with all real
numerical/input reviews and measured pilots. Closed campaigns remain closed.

## October 28 bounded v2 checkpoint producer

The new explicit checkpoint producer streams `torch.save` into an exclusive
trajectory temporary file. Raw, H200 export and temporary weights share the
original per-trajectory byte allowance; current free space must preserve8GiB
before each bounded chunk. Creation, writes and exclusive publication share the
metadata namespace lock with the resource recorder. The lock is released between
chunks. Completed checkpoints are never overwritten; partial failures remain and
block another checkpoint attempt in that slot. Identity, size and timestamp changes,
hard links, STOP, reserve shortfall and failed publication fail closed.

Initial synthetic tests found that this PyTorch runtime passes memoryviews to its
file-like writer; the bytes-only implementation was incorrect. Buffer support now
copies at most1MiB at a time so emitted bytes and their digest agree, without
staging a whole archive in a new RAM buffer. The single worker must keep source
payloads stable throughout serialization. One separate test assertion needed a
Path conversion. The original6-failure/86-pass log and subsequent98-test pass are
retained. Final validation:275 tests passed in5.41seconds, peak RSS243,646,464bytes
under180s/4GiB/8GiB outer guards. Ten tiny real torch archives were saved while20
resource samples ran, with complete census and matching archive hashes.

This is producer integration preparation, not wide-model save feasibility or
numerical optimizer/resume/EMA validation. PyTorch may itself allocate additional
memory, and large-write/fsync latency still needs measured-pilot review. No real
model/corpus/MPS run, research checkpoint deletion, freeze or failed-fixture retry
occurred. The v1 saver and closed controllers remain unchanged. The future v2
worker must bind the exact allowance to the frozen plan and pass the independent
full campaign storage/time/memory/audit gates. Bounded log/state/cache producers,
complete controller/stage/cleanup-chain/inference integration, actual nine numerical
cases/input review and measured pilots remain outstanding.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/checkpoint-writer-v2-component-review.json`.

## October 28 bounded v2 subprocess capture

The prospective v2 subprocess helper captures merged binary stdout/stderr through
a pipe in at most64KiB reads. It writes only the remaining declared log prefix
through the bounded metadata writer, then fails and terminates the owned process
group on overflow. This closes the direct-file overshoot behavior of the existing
sampled v1 logger without modifying closed v1 controllers. Exact-cap output remains
valid; an extra byte makes the attempt fail and cannot be silently truncated into
a successful result.

Launch and terminal reports are immutable exclusive evidence. Live process state
uses bounded atomic replacement under control/. Any prior launch/log/report/control
attempt blocks relaunch. Parent census and child checkpoint/metadata producers use
the same namespace lock. Terminal output accounting includes the report's own
bytes, with16KiB reserved before completion. Remaining campaign space is rechecked
immediately before spawn; runtime checks preserve8GiB reserve, declared output/time
bounds and STOP. Independent child RSS/MPS monitoring and frozen scientific gates
remain mandatory. Arbitrary unconverted child writes are only sampled, so all
intended worker producers must be integrated before a real launch.

Validation:85 initial tests and279 broader tests passed (9.63seconds,
peak RSS241,057,792bytes for the broader run). A subsequent PID-recording hardening
for failure immediately after spawn passed all42 process tests in3.30seconds,
peak RSS64,356,352bytes. All runs had180s/4GiB/8GiB outer guards. Tests include
fast overflow, binary merged logs, inherited locks, EOF before exit, ignored
SIGTERM, orphan descendants, parent interruption, failed diagnostics and no retry.
A real tiny child wrote640 synthetic checkpoint bytes and metadata while the
parent captured20 progress lines and observed a reconciled complete census.

No trained model/corpus/MPS operation, research checkpoint deletion, freeze or
failed numerical-fixture retry occurred. The helper is ready for future campaign
controller integration; it is not a dispatcher. State/cache producers, complete
controller/stage/cleanup-chain/inference wiring and all actual nine numerical/input
reviews and measured pilots remain required.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/process-v2-component-review.json`.

## October 28 v2 pilot cache-audit producer

The v2 producer now uses the immutable campaign projection copy and its bound
SHA256, including the exact D/2 geometry for the declared family. The shared
raw/H200 tensor/load/cache/reset implementation accepts that explicit projection;
the public v1 default retains its source-projection behavior. This closes the
previous producer/binder mismatch without changing storage or scientific rules.

Before expensive audit work, the bounded writer publishes an exclusive attempt
record binding slot, request/result hashes, frozen projection, tokenizer, training
tokens and cache contract. Success and failure reports use the same metadata
budgets. Existing attempt, failure, partial or completed audit files prohibit
retry. Projection, STOP and attempt identity are checked again during/before
completion. The v2 three-pilot stage binder requires the producer version and
matching attempt record, and rejects failure/partial audit files even when they
are listed in the supplied evidence inventory.

Validation:130 initial tests and261 final tests passed. The final run took4.54s
with peak RSS250,036,224bytes under180s/4GiB/8GiB outer guards. Tests used tiny
synthetic checkpoints and mocked MPS/model/cache APIs, including serial endpoint
release, frozen-copy use with an inaccessible source projection, changed inputs,
STOP, failure publication, runtime/fallback rejection and no retry. These are
integration checks, not actual MPS measurements, nine-case numerical evidence,
model-quality results or wider-model feasibility.

Next connect the prepared metadata/checkpoint/resource/process/cache producers to
an explicit v2 pilot worker, including training/export output hooks and frozen
identity binding, then complete controller/stage/cleanup-chain/inference wiring.
All actual numerical/input and measured-pilot prerequisites remain required before
launch. No research weights were deleted, no plan was frozen and no closed run or
failed numerical fixture was retried.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/pilot-cache-v2-component-review.json`.

## October 28 bounded training and export outputs

Training and H200 export now accept explicit output hooks. The v2 sink binds the
exact configuration and canonical slot paths before loader/model creation, allows
only fresh fixed-budget final-only training, and routes checkpoint serialization,
metrics and summary through the bounded writers. It requires exact raw-parent
lineage for H200, matching summary/metrics, and a complete raw/export sequence.
Attempt/failure records prevent retry, including when a caller catches a producer
exception. Existing default output behavior and training math remain unchanged.
The explicit export path hashes in1MiB chunks rather than reading the entire
archive solely to compute its digest; full wider export memory is still unproven.

Validation:106 initial tests passed. Expanded tests had173 passes and one fixture
error (the test supplied `optimizer` instead of the actual `optimizer_state`);
the failed log is retained. After correcting that test,174 checks passed in3.86s,
peak RSS330,317,824bytes under180s/4GiB/8GiB outer guards. Two-update width8
synthetic CPU training produced exactly matching default/opt-in model, optimizer,
RNG, stored EMA and exported model tensors. Failure tests cover changed lineage,
partial serialization, inconsistent summaries, invalid configs, no fallback and
no retry. These checks do not satisfy the nine wider numerical prerequisites.

Next compose the explicit v2 pilot worker with frozen identity, resource recording,
validation/result/sample writers and cache review, then complete controller,
stage/cleanup-chain and inference acceptance. Actual numerical/input reviews and
measured MPS pilots remain outstanding. No corpus training, research checkpoint
deletion, plan freeze or failed numerical-fixture retry occurred.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/training-outputs-v2-component-review.json`.

## October 28 bounded validation and result producer

The v2 evaluation producer now consumes a hash-bound completed training/export
pair for a declared slot. It checks request/config, summary, corpus, tokenizer,
96 training IDs and raw/H200 lineage, then evaluates one endpoint at a time.
Payload, loader and model are released between endpoints. Validation retains the
128-context/batch8/2048-block workload and exact34278 targets/111536 bytes; the
H200 descriptive sample retains the existing settings. No test split is scored.

Attempt, result and failure publication use bounded immutable metadata writes.
Input hashes and STOP are checked around endpoint work and after result
publication. An attempt or failure blocks retry. Pilot and quality stage review
now rejects failed/partial training or evaluation outputs even when those files
are included in the evidence inventory. Successful evaluation-attempt provenance
must still be bound by the complete future worker/controller alongside the
independent result, tensor, cache and resource audits.

Validation:49 initial and255 final checks passed, the final suite in5.40s with
peak RSS361,070,592bytes under180s/4GiB/8GiB outer guards. Tests used tiny synthetic
checkpoint files and mocked MPS/model/scoring/sampling operations, with serial
release checks, invalid provenance/coverage/lineage, changed inputs, failed
sampling, output-cap overflow and retry rejection. No actual wider validation,
quality measurement or feasibility conclusion follows from these tests.

Next compose the explicit v2 pilot worker from training/export, evaluation,
cache and resource producers with full campaign/storage/prerequisite binding;
then complete controller/stage/cleanup-chain/inference wiring. The actual nine
numerical/input reviews and measured pilots remain outstanding. No research
weights were deleted, no plan was frozen and no failed fixture was retried.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/evaluation-v2-component-review.json`.

## October 28 composed v2 pilot output body

The pilot body now composes training/export, validation/sample/result, tensor/cache
review and resource recording. It independently requires the full frozen campaign
and numerical/input gate, exact request/order and bound projection copy before
scientific work. The600s resource interval encloses actual input reconstruction
through completed cache review; completion binds both result and cache hashes.
All checkpoint and metadata output routes use the bounded producers.

The successful evaluation-attempt identity is checked against the request,
training-output digest, reconstructed training contract and frozen96 training IDs.
Worker attempt and terminal references are immutable. Frozen evidence, producer
references and STOP are rechecked between phases; the full campaign/source/runtime
gate runs again after the resource thread closes. A late failure is retained even
if the resource record completed. Pilot and quality stages reject worker failure
or partial-worker evidence, including when bound in an inventory.

Integration identified a controller-ordering requirement: the idle full-future
storage gate must precede creation of the new slot request/process directory.
It correctly rejects unreviewed slot artifacts. The next startup transition must
bind the idle storage/audit receipt, recheck actual disk before dispatch and then
require the successful worker-output provenance before accepting a pilot. The
storage gate was not bypassed or weakened. This body has no CLI or dispatcher.

Validation:18 initial and272 final tests passed. The final run took8.66s with
peak RSS352,223,232bytes under180s/4GiB/8GiB outer guards. Composition tests used
mocked scientific phases, MPS observations and frozen-plan gate, with real bounded
checkpoint/metadata writers and resource recording. They cover phase ordering,
every phase failing, failure after resource completion, changed inputs, incomplete
provenance, no fallback and no retry. These are not real numerical, MPS pilot,
quality or performance results.

Next implement the controller startup transition and require successful worker
provenance for per-pilot/stage acceptance, then complete cleanup-chain and inference
wiring. Actual nine-case numerical/input reviews and measured pilots remain
outstanding. No research weights were deleted or real trials launched.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/pilot-execution-v2-component-review.json`.

## October 28 per-pilot acceptance and successful worker provenance

Pilot acceptance now requires the successful worker report and exact references
to its attempt, training outputs, evaluation attempt, result, cache audit and
resource report. The review checks plan/request/slot/projection/allowance identity,
training raw/H200 size/hash/path receipts, summary/metrics agreement, evaluation
input-file bindings and the frozen descriptive sample settings. Missing producer
records cannot be replaced by a completed result or resource flag alone.

A new read-only prefix review applies the same full per-pilot audits to the first
one, two or three completed pilots, returning checkpoint-pair/evidence bindings
for idle storage acceptance. Prefix review produces no cleanup manifest and no
stage, dispatch or deletion authorization. The full three-pilot boundary remains
required for a candidate pilot cleanup manifest. A later unreviewed pilot slot
blocks prefix acceptance instead of being ignored.

Validation:108 initial and265 final tests passed. The final suite took7.70s with
peak RSS241,385,472bytes under180s/4GiB/8GiB outer guards. Semantic mutation tests
refresh downstream hashes so that changed worker identities, claims, checkpoint
receipts and evaluation bindings must fail on their contents. All evidence is
synthetic; no model, numerical or actual pilot success is claimed.

Next implement the immutable startup transition using prefix acceptance plus the
idle full-campaign storage gate before creating the next request/process slot,
then recheck actual disk immediately before dispatch. Complete controller,
cleanup-chain and inference wiring and actual nine-case numerical/input/measured
pilot prerequisites remain outstanding. No real trial or cleanup was launched.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/pilot-prefix-v2-component-review.json`.


## October 28 v2 idle-to-request startup transition

The new startup component checks frozen campaign identity, full completed-pilot
prefix provenance and the unchanged full-remaining-campaign storage gate before
materializing the next pilot request. An immutable receipt binds the prior file
inventory, reviewed checkpoint pairs, projection and storage report. Predispatch
verification rehashes that inventory and recomputes occupied checkpoint capacity
and the complete future requirement from the frozen projection. Unexpected
empty pilot/process directories also block progress. Failed startup/predispatch
checks leave permanent receipts or partial files, preventing silent retries.

The bounded process helper receives the verified full requirement and checks
space again immediately before creating a child. Synthetic integration confirms
success with enough space and refusal before child creation at one byte below
the full requirement. No next-pilot-only accounting or v1 guard bypass is used.

Validation:37 initial startup tests and333 combined startup/audit/storage/metadata/
process/execution/plan tests passed. The combined suite took10.23s with peak RSS
249,397,248bytes under180s/4GiB/8GiB guards. Two later process-handoff cases were
added; the first run had44 passes and one test-only error-text mismatch, with the
correct disk rejection already occurring. That log is retained. After correcting
the assertion, all45 startup/integration cases passed in1.79s. These fixtures use
synthetic records and tiny files; they are not actual numerical or pilot evidence.

Next wire this transition into a versioned bounded pilot controller/worker entry
point, then complete stage cleanup, quality and inference orchestration. Actual
nine-case numerical/input and measured-pilot prerequisites remain unresolved.
The plan is unfrozen; no MPS launch, failed numerical-fixture retry, research
checkpoint cleanup or scientific-rule change occurred. Recheck current storage
at every real transition; the observed fit is narrow, not a durable guarantee.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/pilot-startup-v2-component-review.json`.


## October 28 v2 single-pilot controller and child entry

A new single-pilot function composes the audited startup transition, exact bounded
process launch, gated child worker, independent completed-prefix review and full
remaining-storage acceptance after the run. It requires previous successful
pilot dispatch receipts and preserves failures. It never freezes a plan, retries
a slot, deletes weights or authorizes the next stage. No parent campaign run CLI
has been added. The caller must inspect actual workers and own both shared locks.

The child validates its parent PID, exact reconstructed module arguments,
inherited locks, immutable startup receipt and full storage/launch limits before
entering the training body. Only the fresh launch record and bounded process log
may appear between idle startup and child verification. Checkpoint hashes run
outside the metadata namespace lock so the parent can continue monitoring; all
other evidence remains exact. Frozen-plan metadata censuses are serialized with
parent publication to avoid races with atomic control-file updates.

Bounded sampled CPU guards cover controller preparation and child entry. The
controller cap1800s includes preparation, the unchanged600s child process and
post-review. Child entry retains600s/4GiB/8GiB limits; the existing scientific
resource recorder enforces MPS driver6GiB and measurement traces. Final dispatch
receipts require a completed bounded child guard observation and a fresh frozen
provenance check after the independent audits.

Validation:120 initial tests passed and one failed because changed arguments were
rejected before the test's expected guard record. Moving argument validation
inside the entry guard made that failure durable; the failed log is retained.
The expanded combined suite passed372 tests in13.56s, then87 completion tests
passed, and105 targeted tests passed after the final metadata locking change.
Final peak RSS was238,370,816bytes under180s/4GiB/8GiB outer guards. Three serial
synthetic pilot transitions retained their six tiny weights; a separate tiny
child exited90 on a synthetic sampled RSS violation. No actual model, numerical,
cache or pilot-resource success is claimed.

Next implement the complete pilot boundary cleanup transition and integrate
quality/selection/cleanup-chain/inference orchestration. The actual nine-case
numerical/input and measured-pilot prerequisites remain unresolved, and the plan
remains unfrozen. Storage readings fluctuated around the unchanged initial
full-panel threshold; every real transition must recheck the complete gate.
No research weights were removed, no MPS launched and no failed numerical fixture
or historical closed controller was rerun.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/pilot-dispatch-v2-component-review.json`.


## October 28 v2 full pilot cleanup boundary

The pilot boundary now connects all three full pilot audits and successful
startup/dispatch/child/guard provenance to the existing guarded cleanup executor.
A complete immutable pre-deletion review binds the exact candidate pairs and
protected file inventory. Controller callbacks recheck frozen identity, STOP,
protected metadata and dispatch guards before every unlink. Only the six fully
audited pilot weights are eligible. Failures remain permanent, including a
failure after some or all candidate weights have already been removed.

Completion requires the full journal, cleanup guard and receipt chain, unchanged
protected evidence, full remaining storage and a finished outer controller guard.
A read-only pre-quality verifier reconstructs the manifest from retained original
records and verifies the chain without reopening deleted tensors. Neither API
launches training or authorizes the next stage; there is no CLI or retry path.

Validation:309 combined tests passed in23.26s. Initial integration failures caught
a synthetic fixture rewriting immutable request bytes and retaining nominal
process limits; the fixture now preserves those exact inputs. One subsequent
failure showed that the mocked frozen gate hid STOP. The controller now checks
STOP directly in its per-removal context callback, independently of the real
frozen gate. All failed logs remain retained. Final interruption tests stop after
one deletion and preserve the intent/deleted/failure records. Tests also reject
incomplete pilots, changed audit/dispatch/guard evidence, unknown weights,
changed protected metadata, incomplete journals and recreated deleted files.

Successful cleanup removed only six disposable tiny fixture weights and preserved
all other fixture evidence hashes. No research checkpoint was deleted or actual
numerical/MPS success claimed. The actual nine-case numerical/input/measured-pilot
prerequisites, quality/selection/LR-rank-cleanup/inference integration and full
campaign freeze remain outstanding. The old failed numerical fixture and closed
campaigns stay closed. Current storage must be checked again at real transitions.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/pilot-boundary-v2-component-review.json`.


## October 28 v2 quality producers and stage provenance

The new quality body composes bounded training, raw/H200 export, full validation
and successful resource reporting inside the quality7200-second interval. It
requires the complete frozen numerical/input campaign gate, an exact frozen
conditional request option and the rank-specific checkpoint allowance. Immutable
attempt/completion records bind every producer. Failures remain terminal. The
parent must independently bind saved LR/rank selection, verified pilot receipts,
current full storage and the process guard before dispatch; this body has no CLI.

The read-only v2 LR/rank wrapper now requires those producer records for every
selected and losing member, in addition to the existing full stage audits. It
checks canonical paths, unique checkpoint inodes, raw/export receipt hashes and
sizes, exact training summary/metrics, evaluation input provenance, training-token
digest, descriptive sample settings and actual corpus hash. The common producer
binder preserves the pilot contract. Completion makes no adoption or launch claim.

Validation:245 initial tests passed;407 combined tests passed in34.59s after
adding producer-to-audit compatibility coverage. The final guard observed peak
RSS357,810,176bytes under180s/4GiB/8GiB limits. Synthetic body cases cover LR,
rank and fresh slots0/9/15/23; scientific training/validation/MPS observations and
full numerical gating are mocked, while bounded writers, recorder lifecycle and
provenance checks run. No research weights were deleted or real quality claimed.

Next integrate quality startup/selection-aware dispatch, bounded independent
tensor-audit production, LR/rank cleanup and inference orchestration. The wrapper
does not yet accept the fresh confirmation stage. Actual nine-case numerical/input
and measured-pilot prerequisites remain unresolved. The failed numerical fixture
and historical controllers remain closed. The27-trajectory plan stays unfrozen;
full storage must be rechecked at real transitions.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/quality-producers-v2-component-review.json`.


## October 28 v2 bounded quality tensor audit

The quality body now runs the existing independent tensor/export checker through
a bounded publisher before marking its resource interval successful. The
publisher requires the active matching quality watchdog and exact request,
training/export receipts, evaluation contract/token lineage and projection copy.
It preserves an immutable attempt and all input hashes, checks the actual mapped
raw/H200 tensors, rechecks STOP and inputs, and publishes the report under the
existing metadata cap. The7200s/4GiB RSS/6GiB driver/8GiB reserve limits are unchanged.
No checkpoint weights are rewritten. Failures and partial reports remain terminal.

Worker completion binds the tensor report SHA. The v2 LR/rank stage wrapper now
requires its producer provenance and rejects missing inputs, changed plan or
contract, overstated numerical claims, failure markers and partial tensor audits.
Exact export equality establishes agreement with stored EMA; it does not prove
the EMA recurrence or optimizer continuation. Those independent prerequisites
remain required. Large checkpoint hashing does not hold the metadata lock.

Validation: the initial suite had230 passes and one missing-evidence exception
contract failure. The missing file was rejected, but an explicit input check now
raises the intended validation error; the failed log remains retained. The final
combined suite passed481 tests in34.30s, peak RSS247,119,872bytes under the180s CPU
guard. Tiny real mmap tensor fixtures reject nonfinite values, wrong EMA and
changed buffers, and preserve raw inputs. Further tests cover STOP/tampering,
inactive or mismatched guards, metadata caps, producer binding and existing pilot
controller paths. Wide metadata and MPS observations are mocked; no actual wide
numerical/MPS success or model-quality result is claimed.

Next integrate quality startup/selection-aware dispatch, LR/rank cleanup and
inference orchestration. Actual nine-case numerical/input and measured-pilot
prerequisites remain unresolved; the failed numerical fixture and closed
controllers remain closed. No research checkpoint deletion or training launch
occurred. The27-trajectory campaign remains unfrozen with full storage gates.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/quality-tensors-v2-component-review.json`.


## October 28 first quality request and historical pilot receipts

A separate historical pilot verifier now accepts declared quality additions
while checking the original frozen context, complete pilot boundary receipt,
protected file hashes, dispatch/guard provenance and cumulative cleanup chain.
It does not reload deleted weights or grant quality-audit, storage or dispatch
credit. The strict immediate-post-cleanup verifier retains its original scope;
new metadata still fails that stricter API.

The bounded first-quality transition handles only LR slot0. After verifying
pilot history it runs the unchanged complete remaining-panel storage gate before
creating a quality request directory. Its immutable attempt/result/failure records
and1800s/4GiB RSS/8GiB reserve preparation guard forbid retries. The read-only
predispatch check permits exactly the first request, rejects any process files,
new weights or later slots, rechecks all history and reconstructs the full
required-free amount independently. No child entry or launch is implemented.

Validation:57 transition/pilot-boundary tests passed in35.64s;190 surrounding
dispatch/startup/storage/chain/cleanup tests passed in8.32s. These use synthetic
pilot dispatch/numerical gates with actual bounded files, locks, receipt-chain
checks and cleanup of tiny disposable fixtures. Exact30,631,281,412-byte free
space passes and one byte less rejects before request creation. That full-panel
requirement remains unchanged after pilot deletion; next-trial-only space is
insufficient. Original protected metadata survives, and recreated deleted weights,
changed journals, failed/partial appends and tampered storage totals are rejected.

Next implement later quality prefix review, selection-aware startup and process
handoff, LR/rank boundary cleanup and inference orchestration. The real nine-case
numerical/input and measured-pilot prerequisites remain unresolved. No research
checkpoint deletion, training/MPS launch or failed-numerical retry occurred. The
27-trajectory campaign remains unfrozen, with closed campaigns and test holdout.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/quality-startup-v2-component-review.json`.


## October 28 complete LR-prefix evidence review

A read-only v2 verifier now reviews each ordered completed LR prefix, from one
through nine screens, against independently supplied frozen requests. It binds
every member's producer and tensor provenance, full validation arithmetic,
1200-step/41-snapshot endpoint, resource trace/process limits, numerical/source/
runtime/corpus prerequisites and complete stage-local file inventory. Full
LR/rank boundaries reuse these same member checks. All weights remain intact.

The output supplies reviewed records and checkpoint pairs for later storage
accounting. It produces no rate selection or cleanup manifest, including at nine
screens, and grants no frozen-plan, storage, deletion or dispatch authorization.
The caller still needs worker locks, an outer guard, complete pilot history,
full remaining-panel storage and selection-aware startup/process handoff.

Validation: the initial suite had287 passes and two missing-resource exception
contract failures. Missing evidence was rejected, but an explicit producer check
now raises the intended validation error. The failed log is retained. The final
289 tests passed in5.23s, peak RSS237,715,456bytes under the180s/4GiB/8GiB CPU guard.
Tests cover every prefix length, reordered or missing slots, changed independent
requests, missing evidence in first/last members, numerical/source/token/corpus
tampering, failure/partial files, checkpoint aliases and immutable inputs, plus
full LR/rank, pilot-audit and storage regressions. Fixtures are synthetic; no
actual wide numerical or model-quality success is implied.

Next connect reviewed LR prefixes to later-trial startup and guarded process
handoff, then persisted LR/rank selections, boundary cleanup and inference.
Real numerical/input and measured-pilot prerequisites remain unresolved. No
research training/MPS launch, checkpoint deletion or failed-numerical retry
occurred. The27-trajectory campaign remains unfrozen with closed test holdout.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/quality-prefix-v2-component-review.json`.


## October 28 later LR request preparation and output allowlist

The later-LR preparation component now covers slots1 through8. Before creating
the next request, it rebinds frozen requests and pilot cleanup history, reviews
every completed LR member, and verifies the entire remaining-panel storage
requirement. Its immutable receipt retains the complete prefix, checkpoint
pairs, inventory and finished1800s/4GiB RSS/8GiB reserve guard. Existing or failed
attempts cannot retry. Preparation grants no process, selection or deletion
authority; rank and fresh stages are outside this component.

Integration found that hashing all current files could turn an unknown output
into supplied evidence. The v2 quality producer binder now requires an explicit
artifact/directory allowlist as well as matching hashes. Unknown files are
rejected even when included in the inventory. A missing tensor file mapping
also receives an explicit validation error. The initial164-pass/two-failure log
is preserved; the failure that accepted unknown output occurred only in a
disposable synthetic fixture, with no real request or checkpoint affected.

An intermediate335-test run passed but shared storage-test mocks affected some
outer-guard free-space readings. The storage fixture now uses a module-local
namespace, and the final harness captures the original disk function before
pytest. The final337 tests passed in42.05s, peak RSS244,203,520bytes, under the
180s/4GiB/8GiB CPU guard with actual disk observations. Exact remaining-capacity
and one-byte-short tests preserve all live weights; no speculative deletion
credit is used. Tests include prefix/full-stage, tensor producer, execution and
storage regressions. Scientific/model/MPS prerequisites remain mocked.

Next implement read-only later-LR predispatch and exact guarded process handoff,
then saved LR/rank selection transitions, boundary cleanup and inference. Real
nine-case numerical/input and measured-pilot prerequisites remain unresolved.
The campaign stays unfrozen; no research training, MPS launch, checkpoint
deletion, failed-numerical retry or holdout access occurred.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/quality-lr-startup-v2-component-review.json`.


## October 28 later LR predispatch revalidation

A read-only predispatch verifier now checks the later-LR preparation receipt,
request, immutable attempt, completed guard, frozen source/context, pilot cleanup
history and all completed LR member audits again. The prefix API can explicitly
permit the exact next frozen request as its only pending-slot file; its default
remains strict. The pending slot gains no completion or checkpoint credit. Any
new process files, weights, later slots, failed/partial artifacts or changed
evidence block handoff.

Storage is reconstructed from current, unique checkpoint inodes and allocated
capacity plus the complete remaining-panel projection. Claimed storage totals
and selected rates cannot substitute for this calculation. Actual free bytes are
checked twice, including after the final source/inventory review. Shared worker
lock descriptors are checked on entry and immediately before return. This idle
check does not authorize dispatch or implement child entry.

Validation:207 tests passed initially; after adding explicit lock rechecks, all
209 tests passed in70.45s. The final guard observed peak RSS239,828,992bytes under
180s/4GiB/8GiB limits, with the original disk-reading function captured before
pytest. Tests preserve inputs on success/failure and reject altered storage
claims, changed records/pairs/guards, active pending slots, exact-threshold
shortfalls, late space loss and invalid locks. Scientific/model/MPS evidence is
synthetic; this is controller preparation only.

Next implement exact guarded child handoff and parent dispatch/postprocess
acceptance, then persisted LR/rank selection transitions, boundary cleanup and
inference. Actual numerical/input and measured-pilot prerequisites remain
unresolved. No research launch, deletion, failed-numerical retry or holdout use
occurred. The27-trajectory campaign remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/quality-lr-predispatch-v2-component-review.json`.


## October 28 live quality handoff transport checks

The read-only LR handoff component binds exact child arguments, startup/request
references, actual parent PID, inherited worker locks and the declared7200s,
8GiB reserve,4MiB log and supplied full-storage/output limits. A pending slot may
contain only its request plus fresh launch/log files. Prelaunch evidence remains
immutable. The same bounded log inode may grow while hashes are checked, but
previously observed log bytes cannot change. Hashing does not hold the metadata
namespace lock, allowing the parent to continue bounded control/log writes.

This component verifies transport only. Inventory, full remaining-space needs
and trajectory allowance must still come from independent frozen/audited gates;
it grants no numerical, current-storage or dispatch credit. The reserved child
module contract has no runnable worker entry or launcher. The next composition
must combine this check with full plan, historical/prefix, startup and storage
validation before invoking the already bounded quality producer.

All120 handoff/process/writer tests passed in4.66s, peak RSS237,846,528bytes under
the180s/4GiB/8GiB CPU guard with actual disk observations. Tests reject parent,
argument and limit changes; failed/partial/new process evidence; log rewriting,
replacement, truncation and overflow; protected-file changes; STOP and invalid
locks. Tests explicitly permit parent control writes and bounded log appends
during hashing. Fixtures are synthetic and establish no model-quality result.

The pending prior commit push was retried with a60-second hard network timeout
and remained unconfirmed; local commits and all evidence were preserved. No
research worker, MPS trial, checkpoint deletion, failed numerical retry or test
holdout access occurred. The27-trajectory campaign remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/quality-handoff-v2-component-review.json`.

## October 28 immutable quality readiness seal

The new readiness component seals the result of the complete idle predispatch
review for LR slots0–8. It invokes the existing full pilot-history/completed-LR
reviews and full remaining-storage calculation, independently rebinds the frozen
request and original trajectory allowance, then records hashes of the complete
prelaunch evidence inventory, frozen source files and existing control receipts.
Its own finished guard is included. Evidence, control or source changes and late
space loss block publication. Failed and completed seals cannot be retried.

The seal must be created before a separate running parent dispatcher: every
existing control file except the seal's own guard must remain unchanged. Hashing
does not hold the metadata namespace lock. The future child must receive an
exact hash anchor for this seal, verify unchanged inputs and current allocated
checkpoint capacity/free bytes, and compose the live parent transport check
before invoking the bounded quality producer. There is no consumer or dispatch
entry yet; the seal explicitly grants no dispatch or deletion authorization.

All134 readiness, live-handoff and first/later startup tests passed in84.11s,
peak RSS242,024,448bytes under the180s/4GiB/8GiB outer CPU guard using actual disk
observations. Fixtures use synthetic semantic evidence and tiny checkpoint bytes.
They reject altered evidence/controls/source, missing audit credit, invalid locks,
STOP, active process files and exact-threshold/late space shortfalls. Existing
inputs remain unchanged, and failure artifacts are preserved.

No research training, MPS launch, research checkpoint deletion, failed numerical
retry or holdout access occurred. Real numerical/input and measured-pilot gates
remain unresolved; the27-trajectory campaign remains unfrozen. The preceding
5f19697 push succeeded and includes ac49ffc; the earlier network issue is resolved.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/quality-readiness-v2-component-review.json`.

## October 28 readiness-bound quality child composition

The child contract now includes an exact readiness SHA in addition to startup
SHA. The consumer requires the same live parent as the sealed completed guard,
verifies unchanged startup/request/evidence/control files, and independently
rebinds frozen source/runtime/numerical prerequisites. It reconstructs remaining
storage from original pilot records and the sealed completed LR prefix, measures
current unique checkpoint allocation and logical bytes, and rejects changes,
extra weights or a current free-space shortfall. The reserve and full metadata
allowance are unchanged. Parent idle audit semantics are inherited only through
the exact parent-supplied seal and unchanged inputs; the hash is not an external
authenticity signature.

Fresh process launch/log files are the only evidence additions. Existing log
bytes and inode remain unchanged through final storage/source checks; bounded
appends and parent control writes remain possible. The quality producer no longer
holds the metadata namespace lock while performing full frozen-source checks.
Regression tests exercise parent control writes during both source checks.

A callable7200s child body performs these gates before lazily importing the
bounded quality producer. Failed/completed entry guards cannot be retried. No
CLI or parent dispatcher is implemented yet. Next add the CLI and tested parent
dispatch/postprocess acceptance, then saved LR/rank selections, boundary cleanup
and inference. The parent must seal readiness before starting mutable control
updates that would invalidate existing sealed controls.

Initial167 tests passed. After extending the log checks,171 passed and one
append-positive fixture failed because it attempted to reopen the exclusive
new-file writer. The fixture was corrected to simulate an existing parent stream;
all172 tests then passed in61.19s under240s/4GiB/8GiB CPU/disk limits. Every run's
logs and guard observations, including the failure, are retained. Scientific
metadata and parent PID relation are synthetic; producer calls are mocked or use
tiny CPU payloads. No research training/MPS launch, checkpoint deletion, failed
combined numerical retry, closed-controller rerun or holdout access occurred.
Real numerical/input and measured-pilot gates remain unresolved and the campaign
remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/quality-entry-v2-component-review.json`.

## October 28 child CLI and independent parent acceptance

The worker-only CLI now requires the exact canonical command, startup/readiness
hashes and inherited locks. It rejects abbreviations, duplicate/reordered options,
unsupported slots, malformed hashes and lock aliases before the child body.
No parent launch loop is implemented yet.

The read-only acceptance component requires the caller's shared locks, outer
review guard and reaped child. It binds same-parent readiness, original launch,
returned/immutable terminal identity and actual child PID to the completed7200s
entry guard. It verifies the self-inclusive output/log census and process limits,
then independently replays all completed LR producer/tensor/resource/validation
audits, pilot cleanup history and complete remaining storage. Frozen provenance
and evidence are rechecked. Even after all9 LR trials it grants no saved selection,
cleanup or next-stage authorization. Hashing leaves the metadata namespace free.

Initial54 tests passed and33 failed: one fixture corpus mismatch was correctly
rejected, and32 fixture assertions attempted to reconstruct intermediate reports
for previously completed trials. Preserving existing fixture bytes removed those
assertions. A subsequent86 passed/1 failed run exposed a reused first-slot fixture
closure variable. After that fixture correction, all87 CLI/acceptance/child-entry
tests passed in91.23s, peak RSS242,139,136bytes under240s/4GiB/8GiB actual CPU/disk
guards. Both failed logs and all guard observations remain intact. Production
acceptance checks were not weakened to accommodate the fixtures.

Next implement the bounded serial parent transition, child reaping, immutable
success/failure receipts and checks of prior accepted prefixes, then saved LR/rank
selection transitions, cleanup and inference. Parent guard handling must preserve
owned child termination and the existing7200s child limit. Real numerical/input
and measured-pilot prerequisites remain unresolved. Fixtures are synthetic; no
research training/MPS worker, checkpoint deletion, failed combined numerical
retry, closed-controller rerun or holdout evaluation occurred. The27-trajectory
campaign remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/quality-acceptance-v2-component-review.json`.


## October 28 serial LR parent transition

The new callable parent controller prepares, seals, dispatches and independently
accepts exactly one LR slot. Tests exercise two-slot and all-nine-slot sequences,
retain prior evidence unchanged, and stop before selection or cleanup. Historical
completion receipts bind ordered predecessors, saved parent/child identities,
startup/readiness/launch/terminal/worker hashes, finished parent and child guards,
original limits and actual trial/log byte counts. A later parent can have a
different PID or inherited descriptors; the next startup still independently
replays the complete scientific prefix, pilot cleanup history and future storage.

Preparation and post-child review have separate1800s guards. Preparation finishes
before sealing. During the child, a cooperative parent resource callback raises
through the7200s process supervisor, which terminates and reaps its owned group.
No hard-exit parent watchdog surrounds a live child. Tiny real subprocess tests
verify this failure path. The child time/RSS/MPS, full storage and8GiB reserve
gates are unchanged. Failures and partial/completed attempts cannot retry.

Initial99 tests passed; one sequential fixture failed because its later-slot
module still referenced the real projection instead of the synthetic frozen
copy. Correcting that fixture yielded70 passing controller/process tests in40.88s.
After strengthening historical terminal-limit/census checks, all30 controller
tests passed in43.67s. The33 independent acceptance checks passed in the initial
run. All logs and240s/4GiB/8GiB actual CPU/disk guard observations, including the
fixture failure, are retained. Scientific child execution and PIDs are mocked;
this establishes controller behavior, not model correctness or quality.

Next implement saved LR/rank selection transitions and their audited cleanup,
rank/fresh dispatch and inference orchestration. Actual nine numerical/input
cases and measured pilots remain unresolved; the failed combined fixture stays
closed. No research training/MPS launch, checkpoint deletion, closed-controller
rerun or holdout evaluation occurred. The27-trajectory campaign remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/quality-dispatch-v2-component-review.json`.


## October 28 audited LR selection transition

The new versioned selection component requires all nine completed parent/child
receipts and independently replays frozen requests, every producer/tensor/resource/
validation audit, pilot cleanup history and complete future storage before saving
rates. It preserves a preselection review, writes the minimum-H200-CE choice with
the existing lower-LR tie break, then repeats full stage and storage audits before
publishing a finished-guard-bound completion receipt. A partial choice publication
blocks continuation and cannot be retried.

The read-only pre-cleanup verifier recomputes the complete scientific stage,
choice and remaining storage, binds original attempt/review/guard provenance and
checks the saved manifest. The exact idle inventory is required; later deleted
weights will need a separate reviewed history consumer. The synthetic choice
identifies12 candidate losing-screen files, but all18 screening weights remain
unchanged. Selection grants no deletion, rank dispatch, adoption or next-stage
credit.

All14 negative tests passed. The positive test reached its final assertion and
failed on a test-only storage field name. After correcting that assertion, the
positive case and rule/stage-audit dependencies passed66 tests in15.79s. The initial
176.15s run and focused rerun retain their raw logs and240s/120s,4GiB,8GiB actual
CPU/disk guard observations. Production code did not change between runs. These
fixtures use synthetic scientific evidence and mocked children; they are not
model-quality or numerical-correctness results.

Observed synthetic metadata occupancy was941073bytes against the2MiB group cap.
Full frozen configs/source maps/receipts still require exact metadata accounting;
the simplified fixture cannot establish their fit. Next integrate audited LR
cleanup and post-cleanup history, then rank/fresh transitions and inference.
Actual nine numerical/input cases and measured pilots remain unresolved. No
research training/MPS launch, checkpoint deletion, failed combined numerical
retry, closed-controller rerun or holdout evaluation occurred. The27-trajectory
campaign remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/lr-selection-v2-component-review.json`.


## October 28 LR cleanup and idle post-cleanup verification

A versioned LR boundary controller now requires the complete saved selection,
all nine parent/child receipts and independent scientific audits before invoking
the existing bounded deletion executor. It captures original checkpoint pairs,
full metadata inventory and all prior control hashes, then repeats the full
stage and future-storage audit immediately before deletion. Only the12 selected
losing-screen files enter the exact path/size/SHA manifest. The executor journals
each deletion and verifies every protected file; the parent then checks the
complete pilot/LR chain and actual remaining storage before publishing completion.

The post-cleanup verifier rebinds frozen source/runtime/plan, original selection,
pre-deletion audits and guards, surviving weights, preserved metadata and the
full deletion journal chain. It reconstructs the manifest without reloading
intentionally removed tensors. This is an exact idle-boundary verifier before
any rank request; an append-aware history consumer remains to be implemented.

All10 integration tests passed in152.39s under240s/4GiB/8GiB actual CPU/disk
guards. The successful synthetic case deletes exactly12 temporary byte files,
keeps all6 selected raw/H200 files and preserves every prior non-checkpoint byte.
Corrupted preconditions block deletion; changed journals, restored losers,
missing winners and altered controls block verification. An injected second
unlink failure retains one confirmed deletion, an intent/deleted/intent journal,
17 remaining files and permanent failure markers; it cannot retry or publish a
successful receipt. These are disposable fixtures, not research checkpoints.

Synthetic post-cleanup metadata occupied1082303bytes of the2MiB group allowance.
The full frozen configuration/source/receipt footprint still needs exact
accounting. Next add append-aware LR history for rank preparation, rank/fresh
transitions and final selection/cleanup/inference. Actual nine numerical/input
cases and measured pilots remain unresolved. No research training/MPS launch,
research checkpoint deletion, failed combined numerical retry, closed-controller
rerun or holdout evaluation occurred. The27-trajectory campaign remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/lr-boundary-v2-component-review.json`.


## October 28 LR history across rank preparation

A separate read-only history consumer now verifies completed LR cleanup after
declared rank files are appended. It keeps the original cleanup context identity
and checks saved selection, pre-deletion audits, finished guards, frozen sources,
protected controls, surviving weights and the entire pilot/LR deletion chain.
The complete supplied inventory is checked before and after verification. The
strict idle-boundary verifier still rejects appended files.

Only slots09 through14, declared producer outputs and explicit prospective rank
controller/selection metadata may be appended. Raw names match the production
final_001200_timestamp format. Temporary/failure files, unknown paths, fresh
confirmation slots and rank-cleanup attempts are rejected. All selected LR
anchors must remain present. New rank bytes receive no scientific, selection,
storage or dispatch credit; independent rank audits are still mandatory.

All44 tests passed in207.43s under300s/4GiB/8GiB actual CPU/disk guards, including
the10 existing cleanup integration tests. Tests cover unchanged protected
history, stale inventories, changed metadata/controls/journals, recreated losers,
unauthorized appends and mutation during verification. Deliberately invalid
scientific rank payloads demonstrate that history verification cannot grant
rank-audit credit. All fixtures are disposable and no research checkpoint was
used or deleted.

Next implement rank-prefix audits using independent conditional requests, all
three surviving LR anchors and completed rank members, then connect preparation
to the complete future storage gate. Fresh dispatch, final selection/cleanup and
inference remain unfinished. Full frozen metadata accounting and actual nine
numerical/input/measured-pilot prerequisites remain unresolved. No MPS/training,
failed combined fixture retry, closed-controller rerun or holdout evaluation
occurred; the27-trajectory campaign remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/lr-history-v2-component-review.json`.


## October 28 live LR-anchor and rank-prefix audits

The rank-prefix auditor now rechecks all three retained LR anchors plus zero
through six completed rank members. It requires independently supplied frozen
requests, exact conditional rank order, nine historical LR records and persisted
LR choices. Every live member must pass producer/tensor/export, training/EMA,
resource/process, validation and corpus/input checks. An optional next request
must be the only file in its exact pending slot and receives no completion credit.

The report distinguishes live audited indices from the six historical losers.
It grants no LR-history, frozen-plan, global-inventory, storage, final-selection,
deletion or dispatch credit. The controller must still bind historical rows,
rates and original pairs through the complete LR cleanup receipt chain, then
independently reconstruct conditional options and verify the full remaining
storage requirement before rank preparation or dispatch.

All173 tests passed in7.36s under180s/4GiB/8GiB actual CPU/disk guards:87 new
rank-prefix cases and86 existing LR-prefix/quality-stage dependency cases. They
cover all prefix/pending positions, malformed independent requests and choices,
missing live-member provenance, changed/absent/aliased weights, failed/partial
outputs, EMA/CE/holdout changes and numerical/input/corpus tampering. These are
synthetic fixtures, not model-quality or numerical results.

Next integrate bounded rank request preparation with verified LR history, frozen
conditional options and full future storage. Rank/fresh orchestration, final
selection/cleanup/inference, full metadata accounting and actual nine numerical/
input/measured-pilot prerequisites remain unfinished. No research training/MPS,
checkpoint deletion, failed combined fixture retry, closed-controller rerun or
holdout evaluation occurred. The27-trajectory campaign remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/rank-prefix-v2-component-review.json`.


## October 28 rank request preparation with verified history and storage

A new bounded API prepares rank slots9 through14 only after complete pilot/LR
cleanup history, unique independently frozen conditional requests, all live LR
anchor/rank audits and full future storage pass. Original deleted pairs remain
bound to their receipts; actual live occupied capacity is counted. No next-slot
directory exists until those gates succeed. Immutable attempts, requests and
finished-guard-bound reports retain the evidence; failures keep any partial
request and permanently block retry. Frozen inputs, full current inventory,
historical integrity, locks/STOP and free space are rechecked before completion.

The initial CPU run passed100 tests and failed3 positive cases in182.98s. One
production comparison incorrectly required the dynamic current-inventory digest
to stay identical after intentionally adding a request. It now compares unchanged
historical fields while still verifying the entire new inventory and chain. Two
later-prefix fixtures had preserved an intermediate synthetic tensor record;
the helper now preserves only files that existed before fixture construction.
The three repaired positives and source-change-after-publication case passed4
focused tests in76.18s. Both runs retain logs and successful300s/180s,4GiB,8GiB
actual CPU/disk guards. Previously passing cases were not broadly repeated.

Preparation grants no process/dispatch, selection or deletion approval. Next
implement read-only rank predispatch, then readiness/child entry and serial parent
dispatch with all prior transport receipts. Fresh transitions, final selection/
cleanup/inference, full metadata accounting and actual nine numerical/input/
measured-pilot prerequisites remain unfinished. Only synthetic temporary byte
fixtures were used; no research training/MPS, checkpoint deletion, failed combined
numerical retry, closed-controller rerun or holdout evaluation occurred. The
27-trajectory campaign remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/rank-startup-v2-component-review.json`.


## October 28 read-only rank predispatch verification

Prepared rank slots9 through14 now have a read-only predispatch check. It binds
startup/attempt/request/finished guard, verifies complete pilot and LR deletion
history, and re-audits every live LR anchor and completed rank member against
independently frozen requests. The pending request receives no completed-trial
or checkpoint credit. Original deleted pairs remain traceable to their receipts.

The check independently reconstructs complete future storage using actual live
occupied capacity, the unchanged8GiB reserve and full256MiB metadata allowance.
It rejects unexpected process/weight/future/failure/partial artifacts, then repeats
frozen inputs, historical evidence, locks, checkpoint stamps, census and actual
free space before returning. Its report explicitly leaves dispatch unauthorized.

All12 synthetic tests passed in139.85s, including28 saved-report mutations,
16 actual evidence/active/failure mutations, exact-capacity and late-space-loss
cases, late history/weight/lock changes, and first/last rank-slot positives. The
360s/4GiB/8GiB actual CPU/disk guard finished normally; peak RSS241,418,240 bytes.
Raw logs and guard evidence are retained. No scientific worker or numerical
fixture was run, and no research weights or closed-holdout data were touched.

Next integrate rank readiness/child entry and serial parent transport verification.
Fresh transitions, final rank selection/cleanup/inference, full metadata accounting
and actual nine numerical/input/measured-pilot prerequisites remain unresolved.
The27-trajectory campaign remains unfrozen and the failed combined numerical
fixture remains closed.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/rank-predispatch-v2-component-review.json`.


## October 28 rank readiness receipts

The v2 readiness seal now supports rank slots9 through14 as well as the existing
LR slots. It binds the unique conditional frozen request and output allowance,
requires complete pilot/LR history and live rank-prefix predispatch verification,
and records the entire immutable evidence/control inventory, frozen context and
full remaining storage requirement. Post-review and post-guard checks preserve
source, controls, locks and free-space requirements. A failed or partially written
seal remains permanent evidence and cannot be retried. No dispatch or deletion
approval is granted; the child CLI still accepts LR slots only.

An initial eager import produced two collection errors before any test executed:
rank history already depends on readiness helpers. Rank-specific imports now
occur after module initialization when binding/verifying rank requests. The
original failure log and guard remain preserved. The repaired full suite passed
37 tests in202.28s:15 new rank tests and22 existing LR readiness tests. Both420s,
4GiB,8GiB actual CPU/disk guards finished normally; repaired peak RSS242,548,736
bytes. This import repair did not retry the closed combined numerical fixture.

Next implement rank child live-handoff/storage/entry and parent acceptance/serial
dispatch with every prior rank transport receipt. Fresh transitions, final rank
selection/cleanup/inference, full metadata accounting and actual nine numerical/
input/measured-pilot gates remain unresolved. No research training/MPS, checkpoint
deletion, closed-controller rerun or holdout evaluation occurred. The27-trajectory
campaign remains unfrozen and all scientific choices/guards remain unchanged.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/rank-readiness-v2-component-review.json`.


## October 28 rank child handoff and entry

Canonical sealed child argv and CLI now accept rank slots9 through14, requiring
the exact rank-startup and readiness references. Unsealed rank transport and
fresh confirmation slots remain rejected. Child entry inherits the live parent's
full idle audit only through its exact immutable seal hash, then verifies all
sealed evidence/control bytes, frozen numerical/source/request bindings, parent
identity, locks and bounded append-only worker log before producer import.

Rank storage rebinds the original pilot/LR cleanup receipts, original deleted and
retained pairs, and saved LR choice. It derives the three live LR anchors plus
completed rank members and counts only actual live occupied capacity. Deleted
weights remain historical evidence and cannot supply capacity credit. Complete
future storage,8GiB reserve,256MiB metadata allowance and the7200s child guard
remain unchanged; final source/evidence/control/log/parent/STOP/free-space checks
still run. Entry failure permanently blocks retry.

All94 tests passed in197.50s:10 new rank tests,40 LR entry regressions and44
transport regressions. The420s/4GiB/8GiB actual CPU/disk guard finished normally,
with peak RSS242,876,416 bytes. Tests used disposable byte fixtures and a
synthetic producer; no research worker, checkpoint deletion, failed combined
numerical retry, closed-controller rerun or holdout evaluation occurred.

Next implement rank parent acceptance and serial dispatch with every prior rank
transport receipt. Existing LR controllers are not rank launchers. Fresh
transitions, final rank selection/cleanup/inference, exact full metadata
accounting and actual nine numerical/input/measured-pilot gates remain unresolved.
The27-trajectory campaign remains unfrozen; the child CLI grants no campaign
launch authorization or quality claim.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/rank-entry-v2-component-review.json`.


## October 28 parent acceptance of completed rank children

A separate exact-slot rank parent acceptance API now shares the strict completed
process checks with LR acceptance. It verifies the complete same-parent sealed
predispatch review, immutable evidence/controls, frozen request/output allowance,
canonical launch and terminal records, exact output/log census and successful
7200s child guard. A successful exit alone cannot establish acceptance.

Rank review independently reconstructs all frozen LR/conditional rank requests,
verifies original pilot/LR cleanup history and saved rates, and re-audits every
live selected LR anchor and completed rank member including the new child.
Original records and anchor pairs must agree with preserved history. Full storage
receives all original pilot/LR pairs, including deleted losers, and live rank
pairs with both verified boundaries. Final source/inventory/guard/control/lock/
free-space checks remain. Even after the last rank screen, selection staysNone;
adoption, dispatch, deletion, cleanup and next-stage permission all stay false.

All50 tests passed in175.15s:17 new rank acceptance tests and33 LR regressions.
The420s/4GiB/8GiB actual CPU/disk guard finished normally with peak RSS241,729,536
bytes. First/middle/last rank cases, altered evidence/process records, independent
full audits/storage and late failures were tested with disposable synthetic
fixtures. No research child/model/MPS, checkpoint deletion, failed combined
numerical retry, closed-controller rerun or holdout evaluation occurred.

Next implement serial rank dispatch with complete verification of all earlier
rank parent/child transport receipts. Fresh transitions, final rank selection/
cleanup/inference, exact full metadata accounting and actual nine numerical/
input/measured-pilot gates remain unresolved. The27-trajectory campaign remains
unfrozen and all scientific choices and resource guards remain unchanged.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/rank-acceptance-v2-component-review.json`.


## October 28 serial rank dispatch and metadata-cap finding

The v2 controller now has separate exact-slot LR and rank dispatch APIs. Each
rank transition verifies original LR/pilot cleanup history, every earlier rank
parent/child receipt and guard, and all controls bound by earlier readiness
seals. It then prepares one frozen conditional request, supervises one child and
independently re-audits the live scientific prefix and full remaining storage.
Failures remain terminal. The final rank screen grants no selection, cleanup,
adoption or fresh-confirmation permission.

The first six-screen synthetic integration test exposed a metadata-cap failure:
four rank screens completed, then slot13 readiness publication reached the
unchanged2MiB evidence/meta cap. The run finished with46 passes and1 failure in
275.89s; its CPU resource guard finished normally. Raw failure output and the
complete synthetic file-size/hash census are preserved. This is separate from
the closed failed numerical fixture.

New metadata publications now use compact sorted ASCII JSON, preserving all
fields, streaming checks and the trailing newline. Existing evidence is never
rewritten. No cap, per-trajectory allowance,256MiB total or8GiB reserve changed.
With this source fix, all82 rank/LR/metadata-writer tests passed in287.28s.
A final historical-control tightening passed four focused regressions in89.83s,
including all six rank screens and all nine LR screens. Peak RSS was245,760,000
bytes across the passing runs; all outer CPU/disk guards finished normally.

The successful synthetic six-screen prefix uses1,985,851 metadata bytes,
leaving111,301 bytes under the2MiB sub-cap. This is **not** proof that final
selection, cleanup, nine fresh confirmations and inference metadata fit. Next
resolve their complete bounded representation/accounting and implement final
rank selection and cleanup with full reviews. Fresh transitions, inference and
actual nine numerical/input/measured-pilot prerequisites remain unresolved.
No research training/MPS, real checkpoint deletion or holdout evaluation occurred.
The27-trajectory campaign remains unfrozen and no research worker is active.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/rank-dispatch-v2-component-review.json`.


## October 28 lossless inventory references and metadata sizing

New readiness and rank-startup receipts use the explicitly versioned
`wide_hybrid_inventory_delta_v1` format. A receipt references an earlier immutable
inventory by document SHA256 and stores its changes plus the complete inventory
count/hash. Readers reconstruct the full map before existing provenance,
scientific, cleanup and storage audits. Old full-inventory documents remain
compatible and are never rewritten.

Decoding checks campaign identity, canonical paths and exact row types, both
physical hashes and the reconstructed digest. It rejects malformed or redundant
deltas, symlinks, traversal and unknown versions, and bounds rows, decoded bytes,
reference depth and total referenced bytes. The prospective plan explicitly
binds these limits. A delta removal only describes a map difference; reviewed
cleanup receipts and protected-evidence checks still govern actual weight removal.

All135 integration/provenance tests passed in470.61s, including the complete six
rank-screen sequence. The subsequent explicit-layout declaration passed75
codec/plan tests in2.04s. Both bounded CPU/disk guards finished normally; maximum
RSS was243,892,224 bytes. The measured synthetic prefix shrank from1,985,851 to
1,403,111 metadata bytes (582,740 bytes,29.34%) under the unchanged2MiB cap.
No per-group cap,256MiB total,8GiB reserve or scientific choice changed.

Actual read-only configuration sizing found355,839 compact bytes for the known
request/source/runtime/layout plan fields alone:3 pilot options and108 conditional
quality options still produce27 selected trajectories. This is a lower bound,
not a complete plan or campaign fit proof. Final rank selection/cleanup, fresh
confirmations, inference and actual numerical/input evidence still need complete
metadata accounting and compact immutable receipts before freeze. No research
worker, MPS training, real checkpoint cleanup or holdout evaluation ran.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/inventory-delta-v1-component-review.json`.


## October 28 final rank selection boundary

The new v2 rank selection component independently reviews the three live LR
anchors and all six rank screens, binds original pilot/LR cleanup history and
serial transport, and persists the prescribed minimum-H200-CE choice with the
larger-divisor tie rule. Complete remaining storage is checked before and after
selection, without credit for an intended rank cleanup. Immutable compact
receipts preserve records and original checkpoint pairs; read-only verification
reconstructs the full stage and manifest against the exact saved inventory.
Source and historical controls are rechecked immediately before completion.

All five initial tests passed in256.06s. After the final publication recheck,
all seven tests passed in375.85s, including tampering, incomplete panels, three
partial-publication failures and two late source/control changes. The driver
then raised an AssertionError in its optional census-export glob because pytest
truncated a temporary directory name. Its output and fixture-exception guard
are retained; this was after all tests passed, with no resource-limit breach.
A separately guarded read-only census captured the completed fixture using
its observed directory, without rerunning tests or changing fixture evidence.

The synthetic campaign uses1,420,508 of2,097,152 metadata bytes; the four new
selection documents total16,405 bytes. All18 live screen checkpoint files remain.
This is not a full real-panel metadata fit proof. Rank cleanup/cumulative history,
fresh transitions, inference and actual numerical/input/measured-pilot gates
remain unresolved. No research training/MPS, real checkpoint deletion or test
holdout evaluation occurred; the campaign remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/rank-selection-v2-component-review.json`.


## October 28 audited rank cleanup boundary

The new v2 boundary repeats the full saved-selection audit before cleanup and
binds an exact losing-weight manifest, original hashes and complete protected
inventory. It uses the locked, fsynced deletion executor. Compact immutable
receipts support strict idle verification of the cumulative pilot/LR/rank chain
without loading intentionally deleted weights. Frozen source, control files,
selected checkpoints, all non-checkpoint evidence and remaining capacity remain
mandatory. Canonical rank-cleanup review names are now admitted by the earlier
LR history reader as bytes only, with no deletion or dispatch credit.

All five full-boundary tests passed in319.35s and all36 affected LR-history
regressions passed in57.64s. Both CPU/disk guards finished normally; maximum
observed RSS was243,744,768 bytes. The equal-CE disposable fixture selected D/8
under the existing tie rule, removed12 losing files including former LR anchors,
and retained six selected files and every original non-checkpoint artifact.
The verifier checked30 cumulative deletions and rejected altered evidence or
recreated weights. Partial-unlink and completion-publication failures retained
their journals/failure evidence and blocked retry. No test/driver failures occurred.

Synthetic metadata is1,436,578/2,097,152 bytes, with111,636 additional bytes in
the separate5MiB rank-retention allowance. This does not prove complete real-panel
fit. Next implement retained rank history for fresh appends and fresh confirmation
preparation, then serial fresh/inference integration and complete metadata
accounting. Actual numerical/input/measured-pilot gates remain unresolved.
No research checkpoints were deleted, no model training/MPS or test-holdout
evaluation ran, and the campaign remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/rank-boundary-v2-component-review.json`.


## October 28 retained rank history and fresh confirmation audits

The new read-only rank-history reader permits declared fresh files while
reconstructing the original selection, execution identity, cumulative cleanup
chain and historical storage observation. It grants no current storage or fresh
scientific/dispatch credit. The existing strict idle verifier remains strict.
A separate prefix auditor checks the persisted LR/rank choices and independently
frozen requests, then reaudits all three selected screen anchors and every
completed fresh trial through producer, tensor, resource, validation and
numerical/input/corpus evidence. Pending requests receive no completed-trial
credit. Neither component selects again or adopts a recipe.

All156 confirmation/shared-fixture tests passed in8.25s; all40 history and
strict-boundary tests passed in160.47s. Both guards finished normally, with
maximum observed RSS245,481,472 bytes and no test/driver failures. The full
disposable chain retained all24 selected/fresh checkpoint files, kept30 reviewed
fixture deletions absent and preserved all original non-checkpoint evidence.
Corruption, undeclared files, resurrected weights and concurrent byte changes
were rejected. All zero-through-nine fresh prefixes were checked.

Synthetic metadata totals2,111,145 bytes, including1,436,485/2,097,152 bytes in
the separately capped meta group. Fresh parent-controller receipts and inference
are absent; this is not complete real-panel fit evidence. Next implement fresh
startup and its complete history/live-audit/transport/storage gates, then serial
fresh and inference integration and full actual metadata accounting. Actual
numerical/input/measured-pilot prerequisites remain unresolved. No research
weights were removed, no training/MPS or test-holdout evaluation ran, and the
campaign remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/confirmation-history-v2-component-review.json`.


## October 28 first fresh-confirmation startup

A new v2 component prepares only the first fresh request, slot15, after full
retained-history, selected-screen producer/tensor/resource/validation, frozen
request, numerical/input and complete remaining-storage checks. Its read-only
verifier reconstructs the pending request without completed-trial or checkpoint
credit, rechecks original identities and current capacity, and rejects changed
evidence or forged receipts. It launches no child and grants no dispatch credit.
Later fresh slots require earlier fresh transport receipts and remain unsupported
by this first-slot component.

All14 initial tests passed in295.65s, including insufficient space and source
changes before/after request creation. Failed attempts retain their evidence,
publish no successful receipt and cannot retry. The first synthetic receipt
was91,589 bytes because it repeated history. Replacing that repetition with
immutable reference-bound hashes reduced the receipt to5,073 bytes; all11
targeted post-compaction checks passed in89.63s. Verification reconstructs the
records, pairs, cleanup stages and frozen files and rejects forged hashes.
Both CPU/time/disk guards finished normally, maximum observed RSS243,744,768
bytes, with no test or driver failures. Both census versions are preserved.

The final disposable fixture has1,442,712/2,097,152 bytes in the meta group,
1,986,094 total metadata bytes and511 files. Six selected checkpoint files
remain; zero fresh trials completed. This is not a full real-panel fit proof.
Next integrate fresh readiness/child/parent transport and serial dispatch,
then inference and full actual metadata accounting. Actual numerical/input/
measured-pilot prerequisites remain unresolved. No research checkpoints were
deleted and no training/MPS or test-holdout evaluation ran; the campaign remains
unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/first-confirmation-startup-v2-component-review.json`.


## October 28 first fresh-confirmation readiness

Readiness sealing now accepts first-fresh slot15 through its exact startup
reference and frozen conditional request. It requires the complete idle audit,
original rank history and selected-screen checks, and independently reconstructed
remaining capacity. The seal binds the full evidence/control inventory, frozen
inputs, finished guard and output allowance. These are rechecked after the guard
finishes and before publication. No child transport or dispatch permission is
granted; later fresh slots remain unsupported.

All25 tests passed in417.23s, covering the new seal and failure cases plus
affected LR0/1/4/8 and rank9/14 readiness paths. Missing audit claims, late source
changes, late space loss and partial publication preserve failure/partial
evidence and block retry. The guard finished normally, maximum observed
RSS243,761,152 bytes, with no test or driver failure. Six selected fixture
checkpoints remain, with zero fresh trials completed; child transport still
rejects the new first-fresh seal.

The synthetic seal is16,358 bytes. The fixture uses1,458,831/2,097,152 bytes
in the meta group,2,001,503 total metadata bytes and513 files. This is not a
full real-panel fit proof. Next integrate first-fresh child entry, independent
parent acceptance and serial dispatch, then later fresh transport, inference
and full actual metadata accounting. Actual numerical/input/measured-pilot
prerequisites remain unresolved. No research checkpoint deletion, training/MPS
or test-holdout evaluation occurred; the campaign remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/first-confirmation-readiness-v2-component-review.json`.


## October 28 continuation: first fresh child and serial parent integration

The old `tokenizer-quality-research` heartbeat is paused. The new five-minute
`tokenizer-quality-research-continuation` heartbeat belongs to the continuation
chat. Authorized process checks succeeded at entry and completion, finding no
research training/test workers; none were killed. The clean starting HEAD was
`e007cfa`, two commits ahead; both pending commits were pushed without force.
The earlier process-tool execution blocker is resolved.

Slot15 now has a readiness-bound child CLI and guarded producer entry. It
reconstructs the compact startup digests from original rank receipts and all
three cleanup stages, verifies the exact parent/argv/locks and immutable evidence,
and counts only the three selected live screen pairs toward occupied capacity.
Original deleted weights remain historical evidence. Frozen numerical/input
bindings, source, controls, log prefix and full remaining storage are rechecked.

The independent parent acceptance path re-audits all three selected screens and
the completed fresh child, including producer/tensor/resource/validation evidence,
original cleanup history, exact process/entry-guard identity and output/log census.
A new single-slot dispatcher prepares, seals, supervises and accepts only slot15.
It retains saved LR/rank choices and stops before slot16. Late source/evidence
changes and process/review failures preserve evidence and prohibit retry.

The first synthetic completion receipt was 86,122 bytes. Eight more receipts of
that size would take its 1,540,001-byte meta group to 2,228,977 bytes, exceeding
the unchanged 2,097,152-byte cap before later startup/readiness overhead. The
explicit `wide_hybrid_first_confirmation_acceptance_digest_v1` representation
stores complete audit digests for repeated records, checkpoint pairs and cleanup
stages, anchored to their original rank and fresh producer references. The compact
receipt is 4,397 bytes. Later-fresh readers must reconstruct and validate these
values; a digest alone grants no audit or dispatch credit.

All 7 child-entry and 15 parent tests passed. After compaction, all 97 integration/
child regressions and 96 parent/CLI regressions passed, including complete serial
LR and rank sequences. Peak observed RSS was 245,170,176 bytes. The initial
seven-case harness had legacy disk mocks in intermediate outer observations;
that log remains intact. All subsequent runs captured the actual system disk
reader before fixtures, retained the 4GiB RSS/8GiB reserve guards and finished
normally. The final integration run repeated the first-fresh child positive.
No test or driver failure occurred. A stale CLI rejection fixture now targets
unsupported slot16 rather than the already-supported rank slot9.

The final synthetic transition retains eight selected/fresh checkpoint files,
uses 1,458,935/2,097,152 meta bytes and 2,000,195 total metadata bytes. This is
not complete real-panel fit evidence. Actual free space at review was
56,529,055,744 bytes, above the unchanged 30,631,281,412-byte initial projection;
current capacity must still be checked at every real transition.

Next implement strict compact prior-fresh transport verification and slots16–23
startup/readiness/child/acceptance/serial transitions, then v2 inference and full
actual metadata accounting. Actual nine-case numerical/input/measured-pilot
prerequisites remain unresolved. The failed combined numerical fixture and all
closed campaigns remain closed. No research checkpoint deletion, MPS/training,
holdout evaluation, paid compute, subagents or plan freeze occurred.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/first-confirmation-integration-v2-component-review.json`.


## October 28 strict first-fresh historical transport replay

A new read-only idle verifier reconstructs the complete audit behind the compact
slot15 dispatch receipt: screening/fresh records, four live checkpoint pairs and
all three cleanup stages. It binds the original startup attempt and full sealed
inventory, historical parent/descriptor identity, every parent phase guard and
child launch/terminal/worker reference. Full acceptance replay independently
rechecks selected/fresh scientific evidence, frozen inputs, output/log census,
child/readiness guards, original retention history and complete remaining storage.

A subsequent controller may have a different PID and lock descriptor numbers.
Only the private first-fresh replay path accepts an explicit historical identity;
public immediate acceptance still requires the current parent. Historical storage
is compared to reconstructed capacity while allowing only observation-time free
bytes, metadata bytes and the current-inventory digest to differ. The original
post-child inventory excludes only the completed dispatch receipt. Later fresh
requests/attempts remain rejected until an explicit append-aware contract exists.

All 8 new history tests and 51 affected parent acceptance regressions passed in
305.99s and 235.43s. Both actual CPU/disk guards finished normally, with maximum
RSS 244,187,136 bytes. Tests cover new caller PID/FDs, rehashed false claims,
changed guards/weights/scientific evidence, later appends, STOP, and late source,
control, receipt and capacity changes. All eight selected/fresh fixture weights
and prior evidence remain unchanged; the reader adds no campaign metadata.
No test or driver failure occurred.

Next integrate explicit later-fresh startup/history appends and slots16–23,
then v2 inference and full real metadata accounting. Actual numerical/input/
measured-pilot prerequisites remain unresolved. These are synthetic controller
checks, not new model-quality evidence. No research training/MPS, checkpoint
deletion, holdout evaluation, closed-fixture retry or plan freeze occurred.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/first-confirmation-history-v2-component-review.json`.


## October 28 real supervisor integration corrections

A realistic supervisor control snapshot exposed a launch-blocking mismatch:
`control/process-quality-15.json` was rejected by the fresh-history allowlist,
although the real process runner always writes it. The synthetic completed-child
fixture had omitted that file. Fresh history now admits only the declared
quality supervisor names for slots15–23 as bound bytes, with no scientific or
dispatch credit. The completed-child fixture includes the pre-terminal snapshot.

Real tiny-process checks then exposed a second mismatch: terminal output counts
used pretty JSON size while the metadata writer published compact JSON. Two
existing process census assertions were 100 bytes too high. Publication and
self-inclusive census now share one encoder. No old evidence is re-encoded.

Parent acceptance independently checks the completed supervisor identity and
limits against the immutable terminal. Its earlier output count must exclude
exactly the terminal document's bytes; its elapsed/free observations remain
bounded. The supervisor hash is saved in acceptance, checked again after the
full storage audit and before receipt publication, and verified during LR/rank
transport and first-fresh historical replay. Late changes fail closed.

The original failing dispatch and subsequent three census failures are retained
as development evidence. An unquoted pytest node selector was rejected by zsh
before a test process launched; the quoted invocation ran normally. These are
new controller/process checks, not retries of the closed numerical fixture.

All 84 final focused checks and 72 affected regressions passed under actual
CPU/disk guards (218.30s and 292.92s; peak RSS 251,166,720 bytes). The final
process inspection found no research worker. The synthetic compact receipt is
4,533 bytes; its meta group uses 1,478,212/2,097,152 bytes, with 2,030,725 total
metadata bytes. These observations do not prove full real-panel fit.

Validation details and source/log hashes are in
`tokenizer-evidence/wide-hybrid-preparation-001/supervisor-control-v2-component-review.json`.
Later-fresh append-aware startup and serial slots16–23 remain the next integration
step, followed by v2 inference and full real metadata accounting. Actual numerical,
input and measured-pilot prerequisites still precede freeze. No research training,
MPS run, checkpoint cleanup, holdout evaluation or plan freeze occurred.


## October 28 explicit slot16 preparation append contract

A separate read-only consumer now replays the completed compact slot15 receipt
while permitting exactly a current-parent slot16 preparation attempt, its bounded
startup guard and, optionally, its independently frozen pending request. It
accepts no startup receipt, readiness, child/process artifact, result, weight or
later-slot append. The original public idle verifier retains its strict evidence
contract and now also requires the exact original control census.

The explicit path reconstructs the original post-child inventory by removing only
the referenced attempt and pending request, then replays the full slot15 audit
under its historical parent identity. The slot16 request is matched against the
independent frozen conditional choice before the scientific prefix audit. Original
selection, selected screen anchors, the fresh pair, all three cleanup stages,
transport and source evidence remain bound. Current evidence and control bytes,
STOP, locks and complete remaining storage are checked again before return.

Storage retains its default refusal of unfinished trajectories. The new explicit
allowance is limited to the hash-bound `quality/16/request.json` after exactly
16 completed quality records and three verified cleanup boundaries. Its spec and
result path must match the next selected fresh member. It grants no completed
record, checkpoint, occupied-capacity or reduced future-space credit; any other
output in that or a later slot remains forbidden.

The initial integration test preserved one failure at the existing strict storage
gate. The final contract adds the narrow allowance and strengthens late-mutation
tests to prove they reach and pass the storage audit before injecting changes.
The 60 focused append/storage checks passed in 232.25s. A combined regression
batch reached its unchanged 600-second wall limit after 39 pass markers, without
a test failure marker. That includes the historical-replay/startup groups and
part of ordinary parent acceptance. Its incomplete verdict and failed guard are
retained; all 34 checks in the remaining parent-acceptance group passed separately
in 37.80s under a 300-second limit. This does not retry the closed numerical fixture or raise any
resource guard. Validation details are in
`tokenizer-evidence/wide-hybrid-preparation-001/confirmation-append-v2-component-review.json`.

Next implement exclusive slot16 preparation and a compact startup receipt with
an independent predispatch verifier using this contract. Startup receipt appends
need their own exact binding; do not relax the original idle reader. Later
readiness/entry/acceptance/serial dispatch and slots17–23 remain unfinished, as do
v2 inference, full real metadata accounting and actual numerical/input/measured
pilots. No training, plan freeze, checkpoint deletion, holdout evaluation or
closed numerical-fixture retry occurred.


## October 28 slot16 startup receipt and independent predispatch review

Slot16 now has an exclusive preparation writer and a separate read-only receipt
consumer. The writer creates one bounded attempt, reconstructs completed slot15
history before creating the independently frozen next request, finishes its
startup guard, then fully replays the pending-request audit before publishing a
compact receipt. Failure evidence is retained and an existing attempt cannot be
retried. No readiness or child launch is implemented by this component.

The receipt binds the current parent, finished guard, attempt, request, prior
dispatch and rank cleanup references, frozen context/files, full evidence and
control inventory digests, completed-record/live-pair/stage digests, selections
and complete remaining storage. The consumer permits only this exact additional
receipt and reconstructs its claims against full independent historical replay.
Its original evidence inventory excludes only the explicitly bound preparation
artifacts. The old idle and attempt/request readers continue rejecting startup
receipt appends. All launch, deletion, adoption and next-stage flags remain false.

Publication pins the finished-control inventory before the final replay and
checks it again after replay and before writing the receipt. A direct injected
post-replay control append must leave failure evidence and no successful receipt.
The independent consumer also rejects rehashed false claims, unknown output,
changed guards, source/evidence mutations, STOP and lost capacity after the
storage audit. Pending slot16 still has no completed-trial/checkpoint credit; all
eight selected/fresh fixture checkpoint files remain present.

All 9 initial core and 4 initial failure/late-change checks passed. After the
final control pin, 4 publication-boundary and compatibility checks passed. The
three bounded batches finished normally in 154.99s, 303.12s and 299.20s, with
peak RSS 244,563,968 bytes; no test or guard failure occurred. Source phases are
recorded separately. The final synthetic receipt is 3,081 bytes, its meta group
uses 1,481,322/2,097,152 bytes and total metadata is 2,031,887 bytes. This does
not establish full real-panel metadata fit. Evidence:
`tokenizer-evidence/wide-hybrid-preparation-001/slot16-startup-v2-component-review.json`.

Next integrate slot16 readiness and child entry using this compact startup
contract, then parent acceptance/serial dispatch and later fresh slots17–23.
Actual numerical/input/measured-pilot prerequisites, inference and full real
metadata accounting remain before plan freeze. No research training, MPS work,
checkpoint deletion, holdout evaluation or closed numerical-fixture retry occurred.


## October 28 slot16 readiness and bounded child entry

Slot16 readiness now performs the full independent compact-startup replay while
allowing exactly its own explicitly referenced, schema-checked, bounded current
parent guard. The ordinary idle/startup and attempt readers remain strict about
additional artifacts. The seal binds the complete evidence/control inventory,
frozen request and source, full remaining storage and finished readiness guard.
Its lossless inventory delta uses the original slot15 readiness seal as an anchor;
the compact startup receipt has no expanded inventory to use as a delta base.

Child entry binds canonical argv, inherited locks and the exact live parent,
launch and seal hashes. It reconstructs all16 completed records and the four live
checkpoint pairs from original rank receipts and the sealed first fresh result,
checks their compact digests and three cleanup boundaries, and independently
recomputes current allocation and full future capacity. Selected screens and the
completed fresh pair remain protected. No pending-trial credit is taken. The
7200-second entry guard precedes the producer, and entry cannot be retried.

The synthetic readiness seal is21,376bytes, with meta-group usage1,503,560 of
2,097,152bytes and total metadata2,058,140bytes. These disposable fixtures do not
establish full real-panel metadata fit, numerical correctness or MPS feasibility.
The exact tests, guards, source hashes, preserved command-collection error and
verdicts are recorded in
`tokenizer-evidence/wide-hybrid-preparation-001/slot16-readiness-entry-v2-component-review.json`.

All20 checks passed in two serial batches of171.84s and386.05s. Peak RSS was
243,941,376bytes, with no resource-guard or test-assertion failure. An intervening
command exited4 during collection because of an incorrect test selector; no
test or fixture ran in that attempt, and its raw evidence is retained separately.
The corrected command used a new evidence name and the same unchanged limits.

Next integrate slot16 parent acceptance and serial dispatch, then verified
transport for later fresh slots17–23. Inference, full real metadata accounting
and actual nine-case numerical/input/measured-pilot prerequisites remain before
finite-plan freeze. No training, MPS run, protected checkpoint deletion, holdout
evaluation or retry of the closed failed numerical fixture occurred.


## October 28 slot16 parent acceptance and sealed serial dispatch

Slot16 parent acceptance now independently audits17 completed quality records,
all three selected screen pairs and both fresh raw/H200 pairs, original retention
proofs and complete remaining storage. A shared read-only transport checker binds
both the current child and original slot15 launch/terminal/supervisor/entry-guard
records. The prior parent's PID/descriptors are historical evidence; current
acceptance still requires the current parent. Slot15's public API stays limited
to slot15. Entry and acceptance share the compact startup inventory binding.

The new `confirmation_dispatch_v2.run_sealed_confirmation_once` consumes an
existing same-parent slot16 startup and readiness seal. It deliberately does not
recreate preparation artifacts. Under its1800-second prepare guard it replays the
full idle history, permitting only the explicit finished readiness seal/guard
and its exact current prepare guard. The old public idle/startup readers retain
their strict append contracts. A second prelaunch check pins all evidence,
controls, frozen inputs, STOP and complete capacity after prepare completion.

The controller supervises exactly one7200-second child with the existing process
supervisor, bounded pipe/log/output, shared descriptors and cooperative parent
monitor, then runs independently guarded acceptance. Only after finished parent
guards, frozen provenance, complete evidence/control inventories, STOP and full
future storage pass does it publish a compact receipt. The receipt preserves
completed-record, live-pair and cleanup-stage digests plus the exact prior
slot15 receipt. No slot17 work, retry, selection, cleanup or adoption follows.

The disposable slot16 dispatch receipt is4,798bytes; its fixture has17 completed
records,10 live checkpoint files, meta-group usage1,507,469 of2,097,152bytes and
total metadata2,075,184bytes. The acceptance fixture's remaining full-panel
requirement is30,153,547,769bytes. Fixture sizes and capacities are synthetic and
do not prove the real panel fits. Full source/test/guard/verdict evidence is in
`tokenizer-evidence/wide-hybrid-preparation-001/slot16-parent-integration-v2-component-review.json`.

All65 checks passed:46 initial acceptance/LR/entry compatibility checks and19
final dispatch/mutation/failure checks. The four serial batches completed in
193.12s,258.70s,243.13s and246.21s under unchanged600-second,4GiB RSS and8GiB
reserve limits. Peak observed RSS was244,432,896bytes. There were no test,
driver or resource-guard failures. Initial and final source phases are recorded
separately; the initial46 checks are not claimed as rerun after every extension.

Next reconstruct and verify the new compact slot16 transport receipt before
preparing slot17, then integrate the bounded remaining fresh slots17–23. V2
inference, full real metadata accounting and actual nine-case numerical/input/
measured-pilot prerequisites still precede finite-plan freeze. These controller
checks launch no model or MPS work, delete no protected checkpoint, consume no
test holdout and do not retry the closed failed combined numerical fixture.


## October 28 independent compact slot16 history replay

A new strict idle consumer reconstructs the compact slot16 dispatch receipt
before any slot17 artifact exists. It verifies the historical parent's exact
startup attempt, completed startup/readiness/controller guards, sealed evidence
and control inventories, original slot15 lineage and canonical child transport.
Current-parent immediate acceptance remains unchanged; historical PID/descriptors
are used only by the separate read-only replay path for supported fresh slots.

The consumer re-runs the complete scientific/transport/retention/storage audit,
reconstructs all17 completed records, five live selected/fresh pairs and three
cleanup boundaries, and compares every compact acceptance claim. It separately
recomputes the pre-slot16 storage requirement from the16-record prefix and four
then-live pairs. The slot15 receipt and slot16 startup therefore cannot merely
validate each other's copied capacity claims. Newly completed slot16 weights
receive no retroactive prechild capacity credit.

Only the exact current dispatch receipt is excluded when reconstructing its
original post-child inventory. Unexpected later requests, attempts or controls,
partial/failure artifacts, changed protected evidence and STOP block replay.
Final transport/control/source/census and capacity checks repeat before returning;
all dispatch, deletion, adoption and next-stage claims remain false.

The disposable fixture retains10 checkpoint files,17 completed records and three
cleanup boundaries. Its meta group uses1,507,341 of2,097,152bytes; total metadata
is2,074,487bytes and reconstructed remaining capacity is30,153,547,769bytes. This
is synthetic controller evidence, not real panel-fit or model-quality evidence.
Exact test verdicts, source hashes and guards are in
`tokenizer-evidence/wide-hybrid-preparation-001/slot16-history-v2-component-review.json`.

All18 checks passed on unchanged source in two serial batches of154.69s and
302.42s, with peak RSS245,137,408bytes. Both batches retained the600-second,
4GiB RSS and8GiB reserve limits; no test, driver or resource-guard failure
occurred. Compatibility includes original slot15 replay and rejection of
immediate slot16 acceptance by a different simulated parent.

Next implement an explicit later-fresh preparation append contract, starting at
slot17, and integrate the remaining finite slots17–23 with bounded inventory
references and independently verified prior transport. Inference, full real
metadata accounting and actual nine-case numerical/input/measured-pilot
prerequisites remain before freeze. No model training, MPS work, protected
checkpoint cleanup, closed-fixture retry or test-holdout evaluation occurred.


## October 28 slot17 preparation append contract

A separate read-only contract now accepts exactly slot17's current-parent bounded
startup guard, immutable attempt and optional independently frozen pending request
while fully replaying completed slot16 history. Historical parent/descriptors
continue to identify original evidence; they do not substitute for the new
attempt's current parent. Starting and finished guards require the unchanged
1800-second,4GiB RSS and8GiB reserve contract.

The public slot16 idle reader remains strict. Only the explicit preparation API
may remove those exact declared append bindings when reconstructing the original
post-child inventory. All selected/fresh science, original transport, retention
proofs, compact claims, prechild/current/future capacity and final integrity
checks still run. Unsupported current outputs, startup/readiness/process
artifacts, later slots and undeclared controls cannot acquire historical credit.

The pending slot17 request must equal the independent frozen plan option. Both
the scientific-prefix and storage readers recognize it only through explicit
caller bindings; ordinary storage still rejects the extra request. It creates
no completed record or checkpoint pair and changes neither the occupied
checkpoint capacity nor the remaining full-panel requirement. Only17 completed
records and10 live checkpoint files remain, with required free space
30,153,547,769bytes in the synthetic fixture.

The fixture's meta group uses1,507,860 of2,097,152bytes and total metadata is
2,075,132bytes. This does not prove real-panel metadata fit. Source/test hashes,
guards and exact verdicts are retained in
`tokenizer-evidence/wide-hybrid-preparation-001/slot17-append-v2-component-review.json`.

All73 checks passed on unchanged source in two serial batches of167.91s and
267.70s. Peak RSS was245,104,640bytes; both batches kept the600-second,4GiB RSS
and8GiB reserve limits. No test, driver or resource-guard failure occurred.
Coverage includes the full existing storage module, old slot16 append behavior
and strict idle slot16 replay, as well as the new preparation/rejection cases.

Next publish the exclusive compact slot17 startup receipt and independently
verify it, then integrate readiness/child/parent serial transitions and the
remaining finite fresh slots. Inference, full real metadata accounting and
actual nine-case numerical/input/measured-pilot prerequisites still precede
freeze. No research training, MPS work, protected checkpoint cleanup, closed
numerical-fixture retry or test-holdout evaluation occurred.

## October 28 slot17 compact startup receipt

The separate slot17 startup writer publishes one attempt, the independently
frozen pending request and a compact receipt under the current parent. It
replays completed slot16 history before and after request publication, preserves
failures and refuses retries. The receipt binds the exact prior dispatch,
original retention lineage, source, completed-record/checkpoint/stage digests,
evidence and controls, and complete remaining storage. Controls are pinned
before final replay and checked again before publication.

Its independent consumer reconstructs the entire retained history and original
transport rather than trusting the compact claims. Only the explicitly bound
receipt is added to the slot17 preparation contract; public idle and append-only
readers remain strict. No readiness, child dispatch or later slot is enabled.

The disposable fixture retains17 completed records and10 checkpoint files. The
pending slot17 request earns no completed-trial or capacity credit. Its receipt
is3,084bytes, meta group1,511,070 of2,097,152bytes and total metadata2,078,917bytes.
This synthetic census does not establish real-panel metadata fit. Evidence:
`tokenizer-evidence/wide-hybrid-preparation-001/slot17-startup-v2-component-review.json`.

All17 cases passed on unchanged source in serial batches of189.16s,355.02s and
279.54s, with peak RSS245,071,872bytes. Each retained the600-second,4GiB RSS and
8GiB reserve limits. No test, driver or resource-guard failure occurred. Checks
covered publication/replay, rehashed false claims, unsupported outputs/slots,
preserved failures/no retries, late evidence/control/source/space/STOP changes,
post-replay control additions and old append/idle compatibility. The final
process inspection found no research training/test workers.

Next integrate slot17 readiness and child entry, then parent acceptance/serial
dispatch and remaining finite fresh slots18–23. V2 inference, full real metadata
accounting and actual nine-case numerical/input/measured-pilot gates remain
before finite-plan freeze. Training remains unlaunched; the closed failed
combined numerical fixture and test holdout remain untouched.

## Slot17 readiness and child-entry preparation

The readiness sealer now admits slot17 through an explicit, bounded,
current-parent readiness-guard capability. It reconstructs complete compact
slot16 history and then seals the exact evidence/control inventory, frozen
request/source and full remaining storage. Ordinary idle and append-only readers
remain strict. The immutable seal uses the prior slot16 readiness inventory as
its delta anchor and grants no dispatch or deletion permission.

Child entry reconstructs17 completed quality records, all three retention
boundaries, three selected screen pairs and both completed fresh pairs. It
binds original result/request/checkpoint identities and compact history digests,
counts only the five live pairs, and checks exact argv, live parent, inherited
locks, sealed bytes/controls, numerical provenance, capacity, STOP and append-only
logs. The7200-second entry guard precedes the separately bounded producer.
Tests use synthetic producer callbacks; no actual model or MPS work is enabled
by the preparation tests, and parent acceptance/dispatch for slot17 remains pending.

The synthetic seal is22,275bytes, meta group1,534,361 of2,097,152bytes and total
metadata2,106,990bytes, retaining all10 checkpoint files. Its inventory reference
chain has11 documents and228,139 physical bytes within the16-document/8MiB
bounds. Adding one link per remaining fresh slot would exceed the document cap
at slot23, so later slots must use a stable earlier readiness anchor. This is
a preparation constraint, not a reason to relax the cap. Real-panel metadata
fit remains unproven. Evidence:
`tokenizer-evidence/wide-hybrid-preparation-001/slot17-readiness-entry-v2-component-review.json`.

All17 checks passed on unchanged source in serial batches of207.25s,384.66s and
244.28s. Peak RSS was244,776,960bytes; all batches retained the600-second,4GiB
RSS and8GiB reserve limits. There were no test, driver or resource-guard failures.
Coverage includes failure preservation/no retry, malformed guards and unsupported
slots, changed history/checkpoints/transport, late integrity/capacity checks,
and existing slot15/16 readiness/entry/CLI compatibility. Final process inspection
found no research training/test workers.

Next integrate slot17 parent acceptance and serial dispatch, then remaining fresh
slots18–23 with bounded inventory anchors and independent compact prior transport
verification. V2 inference and real numerical/input/measured-pilot and metadata
prerequisites still precede finite-plan freeze. No research training, protected
checkpoint cleanup, closed-campaign reopening or test-holdout use occurred.

## Slot17 parent acceptance and serial-dispatch preparation

A separate slot17 parent API now audits the third completed fresh child. It
rechecks both prior fresh transports against their original process, entry,
supervisor and parent guards; reconstructs the exact original slot16 controls
from the current same-parent seal; and audits all selected/fresh scientific
evidence, three retention boundaries and complete remaining storage. Existing
slot15/16 acceptance APIs remain strict. Historical replay of a completed slot17
receipt under a new parent is a separate remaining prerequisite.

The new single-transition controller consumes an existing slot17 startup and
readiness seal. An explicit readiness-receipt/current-prepare-guard capability
permits full slot16 replay before the one supervised child. The controller pins
evidence, controls, source, locks, STOP and capacity before launch and again
before publishing success. Its compact receipt retains both earlier dispatch
references. Failures preserve partial evidence and refuse retries; success
grants no adoption, cleanup or next-slot permission. No slot18 is prepared.

The synthetic success has18 completed records and12 live checkpoint files.
The compact parent receipt is4,918bytes, meta group1,538,100 of2,097,152bytes and
total metadata2,122,478bytes. Real-panel metadata fit remains unproven. Evidence:
`tokenizer-evidence/wide-hybrid-preparation-001/slot17-parent-integration-v2-component-review.json`.

All44 checks passed on unchanged tested source in four serial batches of192.65s,
321.41s,294.08s and362.25s. Peak RSS was245,006,336bytes; each batch retained
the600-second,4GiB RSS and8GiB reserve limits. No collection, test, driver or
resource-guard failure occurred. Coverage includes exact-slot boundaries,
current/prior integrity and late-change rejection, all five injected dispatch
failure phases, no retries, and existing slot16 acceptance/dispatch compatibility.
Final process inspection found no research training/test workers.

Next independently replay the completed compact slot17 receipt, including both
earlier transports and original control/evidence inventories, before integrating
the finite remaining fresh slots18–23. Later seals need a stable earlier inventory
anchor: chaining each slot would exceed the16-document cap at slot23. V2 inference,
full real metadata accounting and actual nine-case numerical/input/measured-pilot
gates remain before finite-plan freeze. No research training, protected checkpoint
cleanup, closed failed numerical-fixture retry or test-holdout use occurred.

## Slot17 independent compact-history replay

A strict idle reader now reconstructs the completed slot17 dispatch under its
original parent/descriptor identities. It rechecks exact original attempt,
startup, readiness, parent phases, terminal/entry/supervisor controls and both
earlier fresh transports. It reconstructs all18 completed records, six live
checkpoint pairs, three retention boundaries and the complete current/future
storage requirement before comparing the compact claims. Unknown evidence,
later controls, partial/failure artifacts and STOP are rejected.

The reader also independently reconstructs capacity before slots16 and17, with
only the checkpoints that existed at each boundary. Neither later checkpoints
nor matching false claims in two old receipts can substitute for that calculation.
Original control/evidence sets and frozen source are checked again after the
full audit. Historical identities never resume a process or grant dispatch,
cleanup, adoption or continuation permission; the public slot17 parent acceptance
API still requires its live parent.

The disposable replay fixture retains12 checkpoint files and18 completed records,
uses1,538,232 of2,097,152bytes in the meta group and2,123,196 total metadata bytes.
Its required free space is30,153,547,736bytes. These are synthetic census results,
not proof of real-panel metadata fit. Evidence:
`tokenizer-evidence/wide-hybrid-preparation-001/slot17-history-v2-component-review.json`.

All19 checks passed on unchanged source in serial batches of240.59s and323.56s.
Peak RSS was246,693,888bytes; both retained the600-second,4GiB RSS and8GiB reserve
limits. No test, driver or resource-guard failure occurred. Coverage includes
changed-parent/descriptor replay, false compact and nested prechild-capacity
claims, malformed identities/slots, protected evidence and late-change rejection,
and existing slot16 replay compatibility. Final process inspection found no
research training/test workers.

Next prepare bounded shared support for the remaining finite fresh slots18–23,
starting with an explicit slot18 attempt/guard/pending-request append around the
strict slot17 replay. Keep idle readers strict and use a stable earlier readiness
inventory anchor; one new reference link per slot would exceed the cap at23.
V2 inference, full real metadata accounting and actual nine-case numerical/input/
measured-pilot gates remain before finite-plan freeze. No research training,
protected checkpoint cleanup, closed numerical-fixture retry or holdout use occurred.

## Finite remaining-fresh append and storage contracts

The shared append validator now declares exact paths and immediate-prior lineage
for slots18–23. Full preparation replay is integrated only for slot18 around the
completed slot17 history; public requests for slots19–23 still fail before
mutation. The explicit capability permits one bounded current-parent startup
guard, its exact attempt and an optional independently frozen pending request.
It permits no compact startup receipt, readiness, child output or next-slot append.
Existing public idle readers remain strict.

The storage contract now supports explicitly bound pending requests at each
remaining finite position through23, with tests for every supported position
16–23 and rejection outside that range. Pending spec/order/result path and
inventory must match; original checkpoint and retention audits, full future
capacity, reserve and final integrity checks remain. A pending request earns
no completed-record, checkpoint, occupied-capacity or future-storage credit.
Historical acceptance permits the slot18 request only through the explicit
slot17 replay path; immediate parent acceptance does not acquire that capability.

The slot18 synthetic preparation retains18 completed records and12 live checkpoint
files, with required free space30,153,547,736bytes unchanged. Its meta group uses
1,539,139 of2,097,152bytes and total metadata is2,125,961bytes. Full real-panel
metadata fit remains unproven. Evidence:
`tokenizer-evidence/wide-hybrid-preparation-001/remaining-append-v2-component-review.json`.

All101 checks passed on unchanged source in serial batches of208.76s and332.46s.
Peak RSS was247,496,704bytes; both retained the600-second,4GiB RSS and8GiB reserve
limits. No test, driver or resource-guard failure occurred. Coverage includes the
full65-case storage module, every pending position16–23, explicit slot18 replay,
shared finite boundaries, old append/idle compatibility and integrity/late-change
rejection. Final process inspection found no research training/test workers.

Next publish and independently verify the compact slot18 startup receipt, using
shared finite-slot validation, then complete remaining readiness/entry and serial
transitions. Later readiness seals must use a stable earlier inventory anchor;
one link per slot would exceed the16-document cap at23. Inference, full real
metadata accounting and actual numerical/input/measured-pilot gates remain before
finite-plan freeze. No research training, protected checkpoint deletion, closed
numerical-fixture retry or test-holdout use occurred.


## Compact slot18 startup publication and independent replay

The remaining-fresh startup module publishes only the slot18 attempt, independently
frozen request and compact receipt. It uses the shared finite naming contract,
binds the current parent and exact slot17 dispatch, and independently reconstructs
all18 completed records, six retained raw/H200 pairs and three cleanup boundaries.
The receipt binds complete evidence/control digests, frozen source/context and
full remaining capacity. Receipt publication repeats source, evidence, control,
locks, STOP and capacity checks after the guarded preparation replay. An explicit
receipt reader validates the whole payload and repeats historical and current
storage audits; the ordinary idle and append-only readers stay strict.

The disposable synthetic receipt is3,087bytes. Its meta group uses1,541,963 of
2,097,152bytes, total metadata is2,127,619bytes and the full remaining free-space
requirement remains30,153,547,736bytes. A pending request earns no completed-record
or checkpoint capacity credit. Full real-panel metadata fit is still unproven.
Failures preserve their evidence and prevent retries. No readiness, child or
later-slot capability is granted by this receipt.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot18-startup-v2-component-review.json`.

All20 checks passed on unchanged source in three serial batches of237.01s,
318.31s and344.14s. Peak RSS was246,562,816bytes; all retained the600-second,
4GiB RSS and8GiB reserve limits. Coverage includes compact publication and
read-only replay, rehashed false claims, unsupported slots, three preserved
failure/no-retry cases, late integrity/capacity/STOP changes, post-replay control
append and prior slot17 changed-parent/descriptors compatibility. No test, driver
or resource guard failed. Final process inspection found no research workers.

Next integrate slot18 readiness and child entry using a stable earlier inventory
anchor such as the original slot16 seal, then complete parent acceptance/serial
dispatch and the finite remainder19–23. No chain cap is relaxed. V2 inference,
full real metadata fit and actual numerical/input/measured-pilot gates remain
before finite-plan freeze. No research training, protected checkpoint deletion,
closed numerical-fixture retry or test-holdout use occurred.


## Slot18 readiness and child entry with a stable inventory anchor

Slot18 readiness explicitly validates its own bounded current-parent guard while
replaying the complete startup/history audit. Ordinary idle/startup readers stay
strict. The new readiness seal encodes its inventory against the original slot16
readiness reference from the authenticated slot17 dispatch lineage. This keeps
the reference chain at11 documents and231,748bytes, below the unchanged16-document
and8MiB caps, rather than adding another recursive predecessor link.

The synthetic readiness receipt is25,884bytes. The meta group uses1,568,896 of
2,097,152bytes; total metadata is2,159,558bytes. All18 completed records and six
retained raw/H200 pairs remain bound. Full real-panel metadata fit is unproven.

The slot18 child entry reconstructs the original three-stage retention history,
all selected and fresh records/pairs, frozen request/source/context, and current
remaining capacity. It binds the same live parent, inherited locks, exact argv,
launch evidence, immutable readiness seal and append-only worker log before the
separately guarded producer. Exact slot18 CLI support is added; slots19–23 stay
closed. No parent dispatcher is enabled by this change. Tests use only disposable
synthetic evidence and a stub producer, with no model execution.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot18-readiness-entry-v2-component-review.json`.

All26 checks passed on unchanged source in serial batches of253.25s,347.30s and
341.31s. Peak RSS was246,431,744bytes; each batch retained the600-second,4GiB RSS
and8GiB reserve limits. Coverage includes stable-anchor seal publication, explicit
guard validation, child reconstruction/guard-before-producer, canonical CLI,
unsupported slots/types, three preserved failures/no retries, evidence/transport/
protected-pair/capacity/late-log integrity rejection, and prior slot17 entry and
slot18 startup compatibility. No test, driver or resource guard failed. Final
process inspection found no research workers.

Next implement slot18 parent acceptance and one guarded serial transition with
independent prior transport and full result/pair audits, then the finite remaining
fresh integrations19–23. Keep stable inventory anchors and all existing caps.
Inference, real metadata fit and actual numerical/input/measured-pilot gates
remain before freeze. No research training, protected checkpoint deletion,
closed numerical-fixture retry or test-holdout use occurred.


## Slot18 parent acceptance and one serial transition

A separate same-parent slot18 acceptance API now reconstructs19 completed records
and seven retained selected/fresh raw/H200 pairs. It rechecks original slot15–17
transport, supervision, entry guards and sealed controls before and after the
scientific and full remaining-storage audit. The original slot17 idle reader
remains strict; the new parent supplies an exact authenticated historical control
map while its current child controls receive separate audits.

The slot18 controller consumes an existing startup/readiness seal. It validates
the exact original slot16 inventory wire anchor, replays full prior history under
a bounded preparation guard, supervises one child and independently accepts its
result and both checkpoint files. Final source, evidence, controls, locks, STOP
and capacity checks precede the compact parent receipt. No parent CLI, retry,
cleanup, slot19 request or automatic next launch is enabled.

The synthetic receipt is5,037bytes and binds all three prior fresh dispatches.
The completed fixture retains19 records and14 checkpoint files. Its meta group
uses1,572,718 of2,097,152bytes and total metadata is2,174,878bytes. Acceptance's
full remaining free-space requirement is30,153,547,703bytes. These disposable
fixtures do not establish real-panel metadata fit, numerical correctness, model
quality or MPS feasibility. The process callback is synthetic; no real training
child was launched.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot18-parent-integration-v2-component-review.json`.

All42 checks passed on unchanged source in five serial batches of241.93s,
388.43s,367.44s,369.99s and101.71s. Peak RSS was246,792,192bytes; every batch
retained the600-second,4GiB RSS and8GiB reserve limits. Coverage includes acceptance
and one synthetic transition, exact API boundaries, alternate valid wire-anchor
rejection, prior/current science/transport/capacity/late-change rejection, all
five dispatch failure paths, no retries, changed prelaunch seal blocking the
child, and prior slot17 acceptance/dispatch compatibility. No test, driver or
resource guard failed. Final inspection found no research workers.

Next independently replay the completed compact slot18 receipt under verified
historical parent/descriptors, reconstructing original transport, selected/fresh
science, retained pairs, stable inventory anchor and pre-child/current capacity.
Then integrate finite remaining slots19–23 with the shared bounded contracts.
Inference, real metadata fit and actual numerical/input/measured-pilot gates
remain before freeze. No research training, protected checkpoint deletion,
closed numerical-fixture retry or test-holdout use occurred.


## Independent completed-slot18 history replay

A strict slot18 reader now reconstructs the compact parent receipt under a
verified historical parent and descriptor identity. It independently binds the
original slot18 startup, stable slot16 readiness inventory anchor, exact prior
slot15–17 dispatch/control lineage and complete current control inventory. It
reaudits all19 completed records, seven selected/fresh pairs and three original
cleanup boundaries, comparing the fully reconstructed audit to every compact
receipt claim. Historical identities never resume a process or authorize launch.

Current and pre-child storage are reconstructed separately, including pre-slot18,
pre-slot17 and pre-slot16 capacity. A later checkpoint never earns credit against
an earlier requirement. Final frozen-source, evidence/control, census, owned-lock,
STOP and full remaining-capacity checks remain. Ordinary parent acceptance still
requires the current parent. This idle reader rejects all later artifacts and
controls; even an explicit pending slot19 request remains unsupported.

The disposable synthetic history retains19 records and14 checkpoint files. Its
meta group uses1,572,797 of2,097,152bytes and total metadata is2,175,398bytes; full
remaining free space is30,153,547,703bytes. This does not prove real-panel metadata
fit or model/numerical/MPS correctness.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot18-history-v2-component-review.json`.

All24 checks passed on unchanged source in serial batches of312.83s and416.03s.
Peak RSS was249,069,568bytes; both retained the600-second,4GiB RSS and8GiB reserve
limits. Coverage includes changed-parent/descriptors replay, rehashed false
claims, capacity rejection at three historical levels, unsupported slots and
identities, explicit pending-slot19 rejection, protected/later evidence/control
rejection, late source/transport/space/STOP changes and slot17 compatibility.
No test, driver or resource guard failed. Final inspection found no research
workers.

Next integrate explicitly bounded slot19 preparation around this strict reader,
then compact startup and remaining finite transitions19–23, retaining stable
original slot16 inventory anchors. Inference, full real metadata fit and actual
numerical/input/measured-pilot gates remain before finite-plan freeze. No research
training, protected checkpoint deletion, closed numerical-fixture retry or
test-holdout use occurred.


## Explicit slot19 preparation replay

The shared remaining-fresh preparation verifier now supports exact slots18 and19.
Slot19 uses the independent completed-slot18 reader with only an explicitly
validated current-parent bounded startup guard, exact attempt and optional
independently frozen pending request. Immediate-predecessor identity, original
slot15–17 transport/retention lineage, full source/context/request bindings,
19 completed records, seven retained pairs and all current/historical capacity
checks remain. The ordinary slot18 idle reader stays strict; no other later
artifact or control is allowed.

A pending slot19 request earns no completed-record, checkpoint or occupied-capacity
credit. Immediate-parent acceptance still rejects pending requests; the extra
request is allowed only through explicitly bound historical replay. Shared paths
and storage remain finite, while full preparation for slots20–23 stays closed.
This iteration does not publish a production startup receipt, enable child entry
or launch a training worker.

The synthetic pending state retains19 completed records and14 checkpoint files.
Its meta group uses1,573,363 of2,097,152bytes and total metadata is2,176,195bytes;
full remaining free space is30,153,547,703bytes, unchanged by the pending request.
Full real-panel fit remains unproven.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot19-append-v2-component-review.json`.

All37 checks passed on unchanged source in serial batches of275.14s and403.54s.
Peak RSS was249,266,176bytes; both retained the600-second,4GiB RSS and8GiB reserve
limits. Coverage includes different-parent attempt/pending replay, no capacity
credit, strict idle and unbound-request rejection, finite paths/types/predecessor
boundaries, immediate-parent restrictions, false claims and undeclared artifacts,
protected evidence and late mutations/STOP, plus slot18 preparation/history
compatibility. No test, driver or resource guard failed. Final process inspection
found no research workers.

Next publish and independently consume compact slot19 startup while preserving
slot18 compatibility, then finish remaining readiness/entry/parent transitions.
Keep stable slot16 inventory anchors and existing caps. Inference, full real
metadata fit and actual numerical/input/measured-pilot gates remain before freeze.
No research training, protected checkpoint deletion, closed numerical-fixture
retry or test-holdout use occurred.

Staged whitespace review removed only an extra EOF blank line in the new test.
The exact tested snapshot and original hash manifest are preserved; the Python
AST is identical. No production source changed or behavioral tests were repeated.


## Compact fifth-fresh startup receipt

The shared startup writer now publishes and independently consumes exact slot19
receipts while preserving slot18 compatibility. The compact receipt binds the
current parent, immediate slot18 dispatch, original slot15–17 transport/retention
lineage, all19 completed records, seven protected pairs and full remaining
capacity. Ordinary history and preparation readers remain strict. Publication
failures are preserved without retries; slot19 readiness and dispatch stay closed.

All21 guarded checks passed on unchanged source in four serial batches. Coverage
includes rehashed false claims, undeclared outputs/controls, preserved failures,
late source/evidence/control/free-space/STOP changes, exact API boundaries and
slot18 startup/history compatibility. No test, driver or resource guard failed;
final process inspection found no research workers.

The synthetic receipt is3,089bytes. Its meta group uses
1,576,587 of2,097,152bytes; total metadata is
2,180,056bytes. All14 protected checkpoint files remain;
required remaining free space is30,153,547,703bytes. Full real
metadata fit remains unproven.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot19-startup-v2-component-review.json`.

Next complete slot19 readiness/entry using the stable original slot16 inventory
anchor and unchanged chain caps, then parent acceptance/serial dispatch and
finite remaining integration. Inference, real metadata fit and actual prerequisite
gates still precede freeze. No research training, quality result, protected
checkpoint deletion, closed numerical-fixture retry or test-holdout use occurred.


## Fifth-fresh readiness and child entry

Slot19 readiness now independently verifies the compact startup and complete
prior history with only an explicitly validated current-parent readiness guard.
Ordinary idle readers remain strict. The immutable seal uses the original slot16
inventory anchor; the synthetic chain has 11 documents and
235,357 bytes under unchanged 16-document/8MiB limits.

Child entry reconstructs all 19 completed records and seven protected pairs,
binds original transport/retention history and frozen inputs, and verifies actual
retained allocation and full remaining capacity before the bounded producer.
The finite shared reconstruction supports slots18/19 while preserving slot18
APIs. Slot19 parent acceptance and serial dispatch remain closed.

All 38 guarded checks passed on unchanged source in four serial batches. They
cover explicit seal/guard bindings, canonical CLI and guard-before-producer order,
protected evidence/checkpoint integrity, live parent/locks/argv/STOP, capacity
boundaries, late evidence/log changes, preserved failures/no retry, exact slot
boundaries, slot18 compatibility and slot19 idle-startup compatibility. No test,
driver or resource guard failed. Final inspection found no research workers.

The synthetic readiness receipt is 29,493 bytes; its meta group
uses 1,607,160 of 2,097,152 bytes and total metadata is
2,215,856 bytes. All 14 protected checkpoint files remain.
Full real-panel metadata fit remains unproven.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot19-readiness-entry-v2-component-review.json`.

Next finish slot19 parent acceptance and bounded serial dispatch, then independent
completed-history replay and the finite remainder. Inference, real metadata fit
and actual numerical/input/measured-pilot gates still precede freeze. No research
training, model-quality result, protected deletion, closed numerical-fixture retry
or test-holdout use occurred.


## Fifth-fresh parent acceptance and serial dispatch

Same-parent slot19 acceptance now reaudits the current child, all four original
fresh transports, selected screens and retention/scientific history. It binds
20 completed records, eight protected pairs and full remaining capacity. The
shared controller supports exact slots18/19 from existing sealed readiness, with
bounded prepare replay, one supervised child and bounded acceptance. Its compact
receipt binds the complete prior-dispatch chain and grants no automatic slot20
launch, adoption, retry or cleanup.

The exact original slot16 inventory wire anchor remains mandatory, including
against a valid re-encoding using slot18. Ordinary idle readers stay strict;
original sealed control maps are used only inside full current-child acceptance.
Independent historical replay of completed slot19 remains pending.

All 40 guarded checks passed on unchanged source in five serial batches. Coverage
includes full acceptance and serial success, exact API boundaries, current/prior
science/checkpoint/transport/guard integrity, rehashed terminals, pending outputs,
late source/control/receipt/STOP/capacity changes, preserved process/review/receipt
failures, no retry, changed-seal refusal before child, and slot18 compatibility.
No test, driver or resource guard failed; final inspection found no research workers.

The synthetic completed state retains all 16 protected checkpoint files. Its
compact parent receipt is 5,157 bytes, meta group uses
1,611,066 of 2,097,152 bytes and total metadata is
2,231,013 bytes. Read-only acceptance requires
30,153,547,670 bytes of remaining free space. Full real-panel fit
remains unproven.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot19-parent-integration-v2-component-review.json`.

Next independently replay completed slot19 under verified historical identities
without allowing pending slot20, then finish finite remaining integration. V2
inference, real metadata fit and actual numerical/input/measured-pilot gates still
precede freeze. No research training, model-quality result, protected deletion,
closed numerical-fixture retry or test-holdout use occurred.


## Independent fifth-fresh history replay

The compact completed-slot19 receipt now supports strict independent replay under
verified historical parent/descriptors. The reader reconstructs original slot19
attempt/startup/readiness, all four prior fresh transports, retention lineage,
canonical argv and bounded guards. Full acceptance reaudits 20 completed records
and eight protected selected/fresh pairs while preserving every evidence byte.

Current capacity and original pre-child capacities at slots19,18,17,16 are
reconstructed from sealed prefixes and actual retained allocation. The exact
slot16 inventory anchor and existing chain caps remain. Ordinary replay rejects
later artifacts and controls; pending slot20 is closed even with a valid
historical identity. The same-parent acceptance APIs keep their existing behavior.

All 25 guarded checks passed on unchanged source in three serial batches. They
cover different-parent/descriptor replay, exact API boundaries, false rehashed
claims, forged historical capacities, protected evidence, later appends, late
source/control/receipt/free-space/STOP changes and slot18 compatibility. No guarded
test, driver or resource guard failed; final inspection found no research workers.
A draft signature syntax error was corrected before source freeze and validation.

The synthetic census retains all 16 checkpoint files; meta group usage is
1,611,204 of 2,097,152 bytes and total metadata is
2,231,789 bytes. Full remaining free space is
30,153,547,670 bytes. Full real-panel metadata fit remains unproven.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot19-history-v2-component-review.json`.

Next extend explicit preparation to slot20 while keeping ordinary idle replay
strict, then integrate its compact startup and the finite remainder. Inference,
real metadata fit and actual numerical/input/measured-pilot gates still precede
freeze. No research training, model-quality result, protected deletion, closed
numerical-fixture retry or test-holdout use occurred.


## Explicit sixth-fresh preparation replay

The shared remaining-fresh preparation verifier now supports exact slots18..20.
Slot20 uses the independent completed-slot19 reader with only an explicitly
validated current-parent bounded startup guard, exact attempt and optional
independently frozen pending request. Original transport/retention lineage, all
20 completed records, eight protected pairs and full current/historical capacity
remain bound. The ordinary slot19 idle reader stays strict.

Pending slot20 earns no completed-record, checkpoint or occupied-capacity credit.
It is admitted only through explicit historical replay; immediate-parent
acceptance still rejects pending requests. Full preparation21..23, compact
slot20 startup publication, readiness and dispatch remain closed.

All 37 guarded checks passed on unchanged source in two serial batches. Coverage
includes different-parent preparation, guard finalization, no pending credit,
strict/unbound-request rejection, finite paths/types/predecessor boundaries,
immediate-parent restrictions, false claims and undeclared outputs/controls,
protected evidence, late mutations/STOP and slot19 preparation/history compatibility.
No test, driver or resource guard failed; final inspection found no research workers.

The synthetic pending state retains 20 completed records and all 16 checkpoint
files. Its meta group uses 1,611,711 of 2,097,152 bytes; total
metadata is 2,232,337 bytes and remaining required free space is
30,153,547,670 bytes, unchanged by the pending request. Full
real-panel fit remains unproven.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot20-append-v2-component-review.json`.

Next publish and independently consume compact slot20 startup while preserving
slot18/19 compatibility, then complete the finite remaining transitions. Keep
stable slot16 inventory anchors and existing caps. Inference, real metadata fit
and actual prerequisite gates still precede freeze. No research training,
protected deletion, closed numerical-fixture retry or test-holdout use occurred.


## Compact sixth-fresh startup receipt

The shared startup writer now supports exact slots18..20. Slot20 exclusively
publishes its attempt, bounded current-parent startup guard, independently frozen
request and compact receipt. Independent consumption fully replays completed
slot19, original slot15..18 transport/retention lineage, all20 completed records,
eight protected pairs and full current/historical capacity. Ordinary history
and preparation readers remain strict. Publication failures are preserved without
retries; slot20 readiness and dispatch stay closed.

All43 guarded checks passed on unchanged source in five serial batches. Coverage
includes full publication/replay, false claims and undeclared outputs/controls,
preserved failures/no retry, late mutations/STOP, exact slot/type boundaries,
slot18/19 startup and slot19 history compatibility. No test, driver or resource
guard failed; final inspection found no research workers.

The synthetic receipt is 3,092 bytes. Its state retains20
completed records and all16 checkpoint files. The meta group uses
1,614,941 of2,097,152 bytes; total metadata is
2,236,237 bytes and remaining required free space is
30,153,547,670 bytes. Full real-panel fit remains unproven.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot20-startup-v2-component-review.json`.

Next integrate slot20 readiness/child entry with the stable original slot16
inventory anchor and unchanged caps, then parent acceptance and remaining finite
transitions. Inference, real metadata fit and actual prerequisite gates still
precede freeze. No research training, quality result, protected deletion, closed
numerical-fixture retry or test-holdout use occurred.


## Sixth-fresh readiness and child entry

Slot20 readiness independently replays the completed history with only its
explicit current-parent readiness guard. The immutable seal binds the startup
receipt, full frozen source/request/evidence/control identities and complete
remaining capacity. Ordinary startup and history readers stay strict.

The inventory encoding keeps the original slot16 anchor. The synthetic seal uses
11 documents and 238,966 bytes in its
chain, below the unchanged16-document/8MiB caps. Its readiness receipt is
33,102 bytes; the meta group uses 1,649,156
of2,097,152 bytes and total metadata is 2,275,904 bytes.
All16 checkpoint files and20 completed records remain. Real-panel fit is unproven.

Slot20 child entry reconstructs the original selected/fresh history and actual
retained capacity before invoking the separately guarded producer. Shared finite
reconstruction now supports18..20 recursively while preserving earlier APIs.
Slot20 parent acceptance and serial dispatch remain closed.

All58 checks passed on unchanged source in five serial guarded batches. Coverage
includes seal/guard bindings, canonical CLI and guard-before-producer order,
protected evidence and transport integrity, both free-space boundaries, late
evidence/log changes, preserved failures/no retry, finite APIs, and slot18/19
entry plus slot20 startup compatibility. No test, driver or resource guard
failed; final process inspection found no research workers.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot20-readiness-entry-v2-component-review.json`.

Next implement slot20 parent acceptance and one serial transition, preserving
all original history and capacity checks, then independent completed-slot20
replay and the remaining finite transitions. Inference, real metadata fit and
actual prerequisite gates still precede freeze. No research training, quality
result, protected deletion, closed fixture retry or holdout use occurred.


## Sixth-fresh parent acceptance and serial dispatch

Slot20 same-parent acceptance reaudits all five prior fresh transports and sealed
control histories, the full confirmation prefix, frozen inputs, protected pairs
and retention lineage. Actual current and complete remaining capacity are checked
again. No pending request or historical slot20 acceptance is enabled.

The finite dispatcher consumes one existing same-parent startup/readiness seal.
A bounded preparation replay admits only explicitly verified readiness and
controller artifacts, followed by one supervised child and independently bounded
acceptance. The receipt is compact and grants no retry, cleanup, adoption or
automatic later launch. The stable original slot16 inventory anchor and existing
16-document/8MiB caps remain unchanged. Ordinary history/startup readers stay strict.

All 90 checks passed on unchanged source in six serial guarded batches. Coverage
includes all protected science/checkpoint/transport/guard identities, false
rehashed terminals, pending outputs, exact API boundaries, late changes and STOP,
preserved failures/no retry, pre-child seal refusal, slot19 compatibility and
slot18/19 boundary checks. No test, driver or resource guard failed; final process
inspection found no research workers.

The synthetic completed state retains21 records, nine protected pairs and18
checkpoint files. Its dispatch receipt is 5,277 bytes; the meta
group uses 1,653,146 of2,097,152 bytes and total metadata is
2,290,881 bytes. Required free space after acceptance is
30,153,547,637 bytes. Full real-panel fit remains unproven.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot20-parent-integration-v2-component-review.json`.

Next add independent completed-slot20 replay, then explicit preparation for the
remaining finite slots21..23. Inference, real metadata fit and actual prerequisite
gates still precede freeze. No research training, quality result, protected
deletion, closed numerical-fixture retry or test-holdout use occurred.


## Independent sixth-fresh history replay

Completed slot20 now has strict independent replay under verified historical
parent/descriptors. It reconstructs original startup/readiness, all five prior
fresh transports, exact sealed/current controls, canonical argv and bounded
guards. Full acceptance reaudits 21 completed records and nine protected pairs
with frozen source/request/evidence and retention lineage. No process is resumed.

Current and original pre-child capacity are independently reconstructed through
slots 20, 19, 18, 17, 16; later weights earn no earlier-capacity credit. Stable original
slot16 inventory anchoring and existing caps remain unchanged. Ordinary replay
rejects later artifacts and controls; pending slot21 is closed even with a valid
historical slot20 identity. No launch, cleanup, retry or adoption is granted.

All 25 checks passed on unchanged source in four serial guarded batches. They
cover different-parent/descriptor replay and byte preservation, exact API/identity
boundaries, receipt and five-depth historical-capacity forgery rejection,
protected/late-change checks, failure/partial/later-artifact rejection and slot19
compatibility. No test, driver or resource guard failed; final process inspection
found no research workers.

The synthetic completed state retains 21 records and 18 checkpoint files. Its meta
group uses 1,653,224 of 2,097,152 bytes; total metadata is
2,291,455 bytes and required remaining free space is
30,153,547,637 bytes. Full real-panel fit remains unproven.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot20-history-v2-component-review.json`.

Next add explicit slot21 preparation, then the remaining finite transitions.
Inference, real metadata fit and actual prerequisite gates still precede freeze.
No research training, model-quality result, protected deletion, closed numerical
fixture retry or test-holdout use occurred.


## Explicit seventh-fresh preparation

Slot21 preparation now permits an exact current-parent attempt, bounded startup
guard and optional independently frozen request, using completed-slot20 replay.
It reaudits all 21 completed records, nine protected pairs, original transport
and retention lineage, and current plus historical pre-child capacity. Adding
the pending request changes no completed records, pairs, stages, rates, selection,
audited indices, remaining projection, occupied capacity or free-space requirement.

Ordinary idle readers remain strict. Immediate-parent acceptance cannot accept
a pending request; unintegrated historical slot21 acceptance remains closed.
Undeclared child outputs, weights, startup/readiness receipts and later controls
are rejected. No dispatch, cleanup, retry or adoption is granted.

All 39 checks passed on unchanged source in four serial guarded batches. They
cover different-parent preparation, guard finalization, no pending capacity
credit, strict idle behavior, finite paths/types/predecessors, false rehashed
claims, protected evidence, undeclared artifacts, late mutations/free-space/STOP
changes and slot20 preparation/history compatibility. No test, driver or resource
guard failed. Final process inspection found no research workers.

The synthetic pending state retains 21 completed records and 18 checkpoint files.
Its meta group uses 1,653,791 of 2,097,152 bytes; total metadata
is 2,292,213 bytes. Required remaining free space is
30,153,547,637 bytes. Full real-panel fit remains unproven.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot21-append-v2-component-review.json`.

Next implement compact slot21 startup publication and independent reading, then
readiness/entry/parent dispatch and the remaining finite slots22..23. Inference,
real metadata fit and actual prerequisite gates still precede finite-plan freeze.
No research training, model-quality result, protected deletion, closed numerical
fixture retry or test-holdout use occurred.


## Compact seventh-fresh startup receipt

The shared startup writer now supports exact slots18..21. Slot21 publishes an
exclusive current-parent attempt, guarded frozen request and compact receipt.
The independent reader reconstructs completed slot20, original slot15..19
transport and retention lineage, all 21 completed records, nine protected pairs
and current plus historical pre-child capacity. Ordinary history and preparation
readers remain strict; the startup receipt grants no child launch or cleanup.

Publication rechecks frozen requests, evidence, controls, locks, STOP and complete
remaining free space after replay. Failures are retained and retries rejected.
Slot21 readiness stays closed until separately integrated. Existing slot18..20
readiness behavior and the stable original slot16 inventory anchor are unchanged.

All 46 checks passed on unchanged source in six serial guarded batches. They
cover compact publication/replay, false rehashed receipts, undeclared outputs and
controls, failure preservation, current-parent identity, late source/evidence/
control/free-space/STOP changes, exact slot/type boundaries and slot20 startup/
history compatibility. No test, driver or resource guard failed. Final process
inspection found no research workers.

The synthetic startup receipt occupies 3,095 bytes. The
state retains 21 completed records and 18 checkpoint files; its meta group uses
1,657,029 of 2,097,152 bytes and total metadata is
2,296,155 bytes. Required remaining free space is
30,153,547,637 bytes. Full real-panel fit remains unproven.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot21-startup-v2-component-review.json`.

Next integrate slot21 readiness/entry, then parent acceptance/serial dispatch and
remaining finite slots22..23. Inference, real metadata fit and actual prerequisite
gates still precede finite-plan freeze. No research training, model-quality
result, protected deletion, closed numerical-fixture retry or test-holdout use
occurred.


## Seventh-fresh readiness and guarded child entry

Slot21 readiness now validates an explicit current-parent guard and fully
reconstructs completed-slot20 history: 21 completed records, nine protected pairs,
original transport/retention lineage and current plus historical capacity.
Ordinary idle startup/history readers remain strict. The immutable seal reuses
the original slot16 inventory anchor and unchanged 16-document/8MiB chain caps.

Child entry reconstructs the prior records and live pairs, binds exact parent,
inherited locks, canonical argv, frozen source/request, all sealed evidence and
controls, and complete remaining capacity. Its resource guard starts before the
synthetic producer. Retry, changed history and unbound controls are rejected;
bounded log growth is allowed only with the existing prefix unchanged.

All 50 checks passed on unchanged source in eight serial guarded batches. They
cover seal/guard/chain integrity, guarded entry, exact CLI/API boundaries,
protected raw/H200 pairs and prior transports, false compact claims, capacity
boundaries and late mutations, parent/locks/argv/STOP, preserved seal/partial
failures and earlier-slot compatibility. No test, driver or resource guard
failed. Final process inspection found no research workers.

The synthetic readiness receipt occupies 36,719 bytes;
its inventory chain has 11 documents and
242,625 bytes. The state preserves 21 completed
records and 18 checkpoint files. Its meta group uses 1,695,176
of 2,097,152 bytes; total metadata is 2,341,311 bytes. Full
real-panel fit remains unproven.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot21-readiness-entry-v2-component-review.json`.

Next integrate slot21 parent acceptance/serial dispatch, then independent history
and remaining finite slots22..23. Review consolidation of repeated slot wiring
and fixture setup while retaining every audit and guard. Inference, real metadata
fit and actual prerequisite gates still precede freeze. No research training,
model-quality result, protected deletion, closed numerical-fixture retry or
test-holdout use occurred.


## Seventh-fresh parent integration and shared transport checks

Slot21 same-parent acceptance and single-child dispatch now bind all prior
transport, retained science, retention lineage and full capacity before a compact
success receipt. Three duplicated prior-transport checks now share one bounded
implementation for slots18..21; production audits remain independent and uncached.

All 87 pytest cases passed in two serial guarded batches, including
79 sequential negative scenarios on one disposable prefix, partial-publication
failure preservation, exact API/type gates and different-parent completed20 replay.
No resource guard failed. Synthetic completed state has 22 records and 20 protected
checkpoint files; its meta group uses 1,699,986 of 2,097,152 bytes.
Full real-panel metadata fit remains unproven.

No research training or new quality result. Next consolidate independent completed21
history and finite slots22..23, then inference and actual prerequisites before freeze.
Closed campaigns, failed numerical fixture and holdout remain closed.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot21-parent-integration-v2-component-review.json`.


## Complete finite confirmation integration through slot23

Slots22..23 now share startup, readiness, child entry and parent dispatch, with
one bounded history reader for completed21..23. It reconstructs all original
transport, retained science, retention lineage and current/historical capacity.
Different-parent replay is tested; immediate dispatch remains same-parent and
single-child. Slot23 rejects all later quality appends. The original16 inventory
anchor, chain caps, resource guards and every fresh pair remain protected.

All 181 checks passed in three serial bounded batches, including
20 sequential history mutations, new entry/API/CLI boundaries and legacy
compatibility. One initial combined sequence hit its600s wall-time guard; its
source and evidence are preserved. Removing a redundant test-only inventory
rescan allowed the fresh bounded run to pass without changing production audits
or limits. A separate negative-test path typo was corrected with its evidence
retained. Production code and the completed A2 checks were unchanged. No research
worker remains.

The synthetic final state has24 quality records and24 checkpoint files: three
selected screen pairs and all nine fresh pairs. Meta-group usage is
1,801,738/2,097,152 bytes; total metadata is 2,492,850 bytes.
This does not establish real-panel fit. No training or model-quality result.
Next complete v2 inference, real numerical/input/resource prerequisites and finite
pilot/quality freezes; the closed numerical fixture and holdout remain closed.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/final-confirmation-v2-component-review.json`.


### Retention-aware inference provenance for all three fresh seeds

The v2 idle inference gate now replays completed slot23 and binds all 24 quality
records, the three selected screening pairs, all nine fresh raw/H200 pairs and
original transport/retention lineage. All three seed triples are constructed in
one call with exact frozen requests, result hashes, tokenizer/input identity and
the existing export loader's binding schema. Deleted losing checkpoints are
proven through retention history rather than required to exist again.

Validation passed **5 tests plus six sequential adapter-negative scenarios** in
527.08 seconds under the unchanged 600-second, 4-GiB RSS and 8-GiB reserve guard.
A single completed synthetic prefix served all nine bindings. Production history
was not cached or bypassed; frozen scientific inputs and training contracts were
mocked as stated in `inference-binding-v2-component-review.json`. All
331 source/test hashes remained unchanged during validation.

This gate is read-only and returns no dispatch, adoption or deletion authority.
The strict idle final-history API remains unchanged. No receipt containing the
large live inventory is written; bindings remain in memory. Next implement one
shared, finite inference append/serial controller for seeds3511/3512/3513 using
compact references, exact output/control namespaces and existing loading/cache/
timing audits. Then audit real metadata/high-water fit, complete the separate
nine-case numerical/input prerequisites, freeze and measure resource pilots,
and freeze quality training. Training and inference have not launched; closed
campaigns, the closed numerical fixture and the test holdout remain closed.


### Real request-table metadata overflow and compact identity

The real 3-pilot/108-conditional-request table exposed a concrete metadata issue:
substituting only that table for the tiny synthetic plan raises the completed
meta group to **2,119,268 bytes**, above its **2,097,152-byte** cap by22,116 bytes.
This lower bound omits the rest of the real plan and prerequisite trace growth.
Evidence: `inference-plan-metadata-lower-bound.json`.

The plan gate now also accepts an exact, versioned request-table SHA256 plus
pilot and per-slot counts. It reconstructs and checks every request under frozen
source/runtime and returns the same full request table to downstream components.
Full-document compatibility remains tested. This reduces the request field from
317,940 to **229 bytes**, saving **317711 bytes**, without changing the
finite grid, resource caps or required audits. The real complete metadata and
transient high-water census remains a prerequisite; this is not a fit claim.

All **59 plan tests passed** in1.84 seconds, including compact/full equality,
729 selection combinations and tampered identity/reconstructed-config rejection.
Together with the earlier five inference binding tests, this iteration passed64
tests plus six sequential adapter negatives. Only the plan implementation/test
changed after the full inference-prefix run; both source manifests and the scope
of that validation are preserved in `inference-compact-plan-component-review.json`.
Use the compact request identity in the future frozen plan. Continue the shared
v2 inference controller, then real metadata/numerical/input/pilot prerequisites;
training, inference, protected deletion and closed-fixture retry remain unlaunched.


### Shared inference preparation and reusable synthetic replay

One immutable request now binds the complete fresh inference panel (seeds3511,
3512,3513). Preparation performs the full idle audit, appends only the request
and its bounded guard, then repeats the retained science/transport audit through
an explicit scope. Final quality, rank and pilot history remain strict by default;
only the independently validated inference reference admits these two files.
No worker dispatch, measurement, adoption or deletion authority is granted.

The initial integration test failed because the scoped reference was not yet
propagated to rank/pilot history. Its request, failed guard, failure receipt,
raw log and source versions remain intact. The correction was validated on
fresh attempts in two other completed disposable synthetic prefixes; the failed
prefix and closed numerical fixture were not retried. No resource guard failed.

The final guarded batch passed **29 tests**, including **19 sequential negative
scenarios**, completed replay from a different process/new lock descriptors,
late STOP/capacity failures and strict API compatibility. The reusable test
fixture pins every original file and reinstalls only synthetic input mocks;
all production audits execute again. Preparation plus negative scenarios took
60.57s instead of rebuilding the entire synthetic sequence for aboutnine minutes.
The final combined batch took143.29s under180s,4-GiB RSS and8-GiB reserve limits.

The request lives in the existing inference3511 metadata allowance and adds no
file to the2-MiB meta group. Every original fixture byte and all24 selected/fresh
checkpoint files were preserved. Real-panel metadata/high-water fit remains
unproven. Next integrate the shared serial child controller, entry/measurement
and completed-result acceptance against this request, including the active
control/log lifecycle; keep ordinary idle history strict. Then complete real
numerical/input/resource prerequisites and finite pilot/quality freezes before
training. Evidence: `inference-preparation-v2-component-review.json`.


### Bounded inference worker body and independent artifact review

A shared worker body now supports all three fresh seeds using the existing
export loader, loaded-model identity check, cache callbacks, timing sequence and
saved-result audit. It requires an independently implemented live entry verifier
at entry, after loading, after measurement and after publication. The controller
must still provide that verifier and supervise exactly one child at a time.
There is no new launch CLI or dispatch authority.

Independent saved-output review binds the attempt, selected raw/H200 pairs,
model metadata, timing arithmetic and finished worker guard. It rejects rehashed
corruption, failed/partial artifacts and changed resource limits. It explicitly
grants no transport, full-quality-history or next-seed acceptance; those remain
mandatory controller work. The600s worker,4-GiB RSS,6-GiB MPS and8-GiB reserve
limits remain, and the4-MiB output cap applies to the whole inference panel.

All **58 final tests passed in1.66s**, using mocked loading/MPS/live entry with
real bounded metadata producers and independent artifact/timing review. A
refactor regression that called entry verification before the initial panel-cap
check was corrected; its raw failure and source are preserved. No loading or
measurement ran in that failing case. No resource guard failed.

Next implement the actual live entry verifier, readiness/active-output scope,
supervisor transport acceptance and serial controller against these contracts.
Then complete real metadata/numerical/input/resource prerequisites and finite
freezes before training. No real inference/training, protected deletion,
closed-fixture retry or holdout use occurred.
Evidence: `inference-outputs-v2-component-review.json`.


### Compact first inference readiness and live entry verification

The first fresh inference triple now has a real live entry verifier composed
with the bounded worker body. Before readiness publication, the parent reruns
the complete retained quality, cleanup and historical transport audits. The
child reconstructs all nine fresh bindings from the existing immutable evidence
and validates their digest against the shared request, frozen source/runtime,
numerical/input contracts, original controls and selected/fresh checkpoints.
The readiness record is 996 bytes; no copied history manifest consumes the
per-seed metadata allowance.

The live verifier checks canonical argv and inherited lock descriptors, current
parent/child/process-group identity, exact phase-specific files, bounded parent
supervisor and worker guards. It permits the supervisor and log to advance
without treating those mutable files as frozen quality evidence. It rejects
replaced log prefixes, extra/future/partial outputs, changed protected bytes,
source/input bindings, STOP and resource-limit changes. Existing ordinary idle
and prepared-history scopes remain strict.

The final guarded batch passed **55 tests in32.55s**, including a full reused
completed synthetic history and all four composed worker boundaries. All 785
original prefix files and24 retained checkpoint files stayed unchanged. The
only append to that prefix was the small readiness record. One earlier test
incorrectly expected a new failure marker for an already-existing partial
output; the existing no-retry gate correctly refuses before creating a worker.
The test was corrected; failed raw evidence and original test source remain.

There is no launch CLI or serial controller yet. Real process creation, model
loading and MPS inference were not exercised; unit process identity and model
work remain synthetic. The first seed's live entry is implemented, while
seeds3512/3513 deliberately remain blocked pending completed transport review
and full idle-history replays between seeds. Implement that shared serial
integration next, then complete actual metadata/high-water, numerical/input and
resource prerequisites and finite freezes before training. No protected cleanup,
closed fixture retry or holdout access occurred. Evidence:
`inference-entry-v2-component-review.json`.


### Three-seed idle inference acceptance and full-history replay

A shared acceptance component now validates exact readiness, completed-output
and accepted-result states for seeds3511/3512/3513. Each accepted seed binds
its worker artifacts, original launch, canonical historical descriptors,
self-inclusive terminal output bytes and final supervisor observations. New
acceptance requires the same current parent and descriptors and a reaped owned
process group. Historical read-only replay supports new lock descriptors.

Acceptance and next readiness independently replay the complete retained
quality, cleanup and original transport history. Later readiness binds every
earlier accepted inference file/control without copying those histories. Only
explicit, fully checked idle inference references admit appended files into
lower quality-history readers; ordinary idle/prepared readers stay strict.
No process-success flag alone grants next-seed credit.

**85 tests passed** across83 focused cases, one isolated fixture-adapter test
and one full three-seed integration. The final integration executed ten full
history replays in328.76s under420s,4-GiB RSS and8-GiB reserve guards; observed
peak RSS was257,835,008bytes. All783 original successful-prefix files and24
retained checkpoint files remained unchanged. Its complete synthetic inference
metadata occupied71544bytes; this is not real-panel or transient high-water proof.

Two test-fixture integration failures are preserved. The old completed fixture
uses a placeholder token digest, so a range(96) input assertion first stopped
before any append. A second attempt exposed a missing adapter in the independent
timing reviewer, which correctly rejected that synthetic report and preserved
a failed worker. Both prefixes are closed to further worker appends. The exact
test-only token adapter was independently checked before a third, untouched
successful prefix completed the panel. Production token hashing and scientific
gates were unchanged; no resource guard failed or closed numerical fixture ran.

OS process identity, loading and timings in the panel integration remain
synthetic. The live entry is still intentionally first-seed-only and no launch
CLI or serial controller is exposed. Next integrate the shared later-seed live
namespace and actual serial supervision/cancellation, then finish real metadata,
numerical/input/resource prerequisites and finite freezes before training.
Evidence: `inference-acceptance-v2-component-review.json`.


### Shared three-seed inference CLI and guarded serial controller

The live entry now supports all three seeds through one finite namespace and a
canonical inherited-FD child CLI. Each later readiness binds the prior accepted
inference files and controller record. Metadata censuses during live entry
serialize against atomic supervisor writes; retained training weights are
hashed outside that lock.

The controller creates only one fresh child at a time. Preparation/review have
separate1800s CPU watchdogs, and the live600s phase uses cooperative parent
resource checks so STOP, time/RSS/disk/output failures and interrupts terminate
and reap the owned process group. Parent phases retain4-GiB RSS and8-GiB reserve;
the worker retains its6-GiB MPS guard and the whole inference panel retains its
4-MiB cap. SIGINT/SIGTERM record STOP/failure and restore prior handlers. No
failed or already-attempted seed is resumed. Exact successful parent guard
records append after acceptance and are independently checked on later replay.

**99 distinct tests passed** across112 executions, with no unexpected failure.
The final batch passed16 tests in47.70s and includes12 real child processes:
three canonical serial CLI children, controlled cancellation/resource/signal
failures, a second-seed loading failure and idle watchdog hard exits. Every
owned process group was reaped, prior successful evidence was retained, and no
third seed launched after the second failed. Loading, MPS and the scientific
inputs in these process tests are synthetic; CPU guards, argv, descriptors,
PIDs/groups, writers and transport checks are real. These are implementation
tests, not model-quality or inference-performance evidence.

A separate read-only test reran the complete retained science, cleanup and
transport history of the previously completed synthetic three-seed panel. All
818 original files and24 retained checkpoint files stayed unchanged. Its old
receipts remain readable; the new controller advances only from a preceding
guarded controller completion.

The controller returns three reviewed seed receipts pending the frozen
quality/cost decision; it does not adopt a recipe. Next audit actual complete
metadata/transient high-water fit, finish the separately reviewed numerical
and exact-input prerequisites, and freeze/measure resource pilots before the
quality-plan freeze and training. Real research inference/training remain
unlaunched; no protected cleanup, closed numerical retry or holdout access
occurred. Evidence: `inference-dispatch-v2-component-review.json` and
`inference-dispatch-v2-process-review.json`.

The final consolidated controller check passed **159 tests in65.98s** under the
180s guard. A source review narrowed the live frozen-identity lock to the
metadata census: hashing source/prerequisite files must not delay the
supervisor's bounded pipe/control publication. An independent lock test confirms
that snapshot reading is exclusive, file hashing leaves the lock available,
and only the exact campaign's MetadataWriter may request this live path.
Default idle identity verification is unchanged. The consolidated batch reran
real subprocess/cancellation tests and the complete old-panel replay; all818
historical files and24 checkpoint files remain unchanged. Final evidence:
`inference-dispatch-v2-final-review.json` and
`inference-dispatch-v2-process-review-final.json`.


### Actual training-input provenance and remaining metadata capacity

The first96 GPT-2 training tokens are now verified against the real frozen
Shakespeare corpus and inherited tokenizer contract. Only the80% training
text was tokenized; validation/test text was not encoded, and no model or
checkpoint was loaded. Ordinary-text encoding matched the independent
`encode_ordinary` entry point and round-tripped the entire training split.
The canonical uint32 little-endian input digest is
`0e77bdca8368d30a983e0b1ef033bd141411c35ff00c8f9e7cc1e05c6a276812`.
The three immutable input artifacts occupy27,838bytes and bind current
source/runtime. A separate review independently recomputed the input digest
and rechecked all artifact hashes. The180s/4GiB/8GiB guard finished in0.19s,
with262,733,824bytes peak RSS. No numerical optimizer test ran.

The initial input-only attempt failed before encoding because the default
GPT-2 cache was absent and sandbox DNS blocked the implicit download. Its
guard and original script remain preserved and closed. The two public assets
were restored into ignored `research_state/tiktoken-cache`, verified against
the installed tiktoken SHA256 pins, and a new fresh input-only attempt passed.
The revised procedure refuses implicit downloads and requires a verified cache.
Future workers must receive that explicit cache path. This did not retry the
closed failed combined continuation fixture.

A guarded read-only census verified all818 retained synthetic-prefix files,
including24 checkpoint files, unchanged. Substituting the known real plan/input
fields and propagating the known frozen-file maps through34 occurrences gives
a **partial** meta-group projection of1,900,578bytes, leaving196,574bytes of
the2,097,152-byte cap. Nine additional numerical-artifact references leave
134,987bytes;18 leave73,400bytes, before their payloads and other unmodeled
growth. These figures are sensitivity calculations, not a complete upper bound
or fit claim. The actual numerical payloads, inventory propagation, full-duration
resource records, final controller/decision receipts and transient overlap
still require a complete audit. Existing metadata/resource limits are unchanged.

Evidence: `training-input-and-metadata-review.json`,
`training-input-v2-002/meta/training-input.json`, and
`prerequisite-metadata-v2-review.json`, under the wider-hybrid preparation
evidence directory. Next implement and review the separate serial nine-case
rank-policy numerical procedure with bounded disk-backed checkpoint lifetimes,
then execute its fresh cases once. Do not rerun the closed combined fixture.
Complete actual metadata and quality/cost aggregation review, freeze/measure
resource pilots, and freeze the finite quality plan before research training.


### Separate frozen nine-case CPU numerical procedure begins

The separately reviewed rank-policy procedure is frozen in
`rank-numerical-v2-frozen-plan.json`; its executable run plan hashes to
`d89c3e501c80f8391da471dafde93884e5f635df5f93048a576c69e9729e32a1`.
Five small streaming-hash/config/freeze/STOP/provenance orchestration tests
passed in0.50s under the180s/4GiB/8GiB CPU guard. The prior four-test batch
also passed and is retained. No closed continuation test is imported or called.

This fresh prerequisite uses nine width512 family/rank cases, with baseline,
partial-save and resume isolated in27 serial CPU processes. The existing trainer,
model, sampler, optimizer, checkpoint reader and EMA math remain unchanged.
The partial checkpoint is written directly to disk; baseline and resumed
end states are streamed into complete hashes without retained serialized
buffers. Each phase has600s/4GiB/8GiB guards, inherited global/run locks and
owned-process cleanup. Each case has a2GiB output cap; the initial complete
reservation is26GiB versus roughly51GiB observed free. Failures stop the panel
and preserve all artifacts. This is four-update correctness validation with
H200 snapshots1–4, not a resource pilot or quality trajectory.

The first complete case, hybrid D/8, passed all three phases. Full model,
optimizer, EMA and torch/loader RNG hashes exactly matched baseline versus
resume. Independent scalar EMA recurrence checks sampled the first/middle/last
values of every parameter; complete EMA state equality complements that sample.
The inference-only EMA load matched every stored parameter, preserved buffers
and correctly rejected optimizer resume. Phase times were74.58/21.31/58.44s;
maximum RSS was3,855,155,200bytes (3.59GiB). Its981,866,923-byte partial checkpoint
is retained. The remaining eight cases are still pending/running serially.

Evidence: `rank-numerical-v2-procedure.md`,
`rank-numerical-v2-procedure-review.json`, and
`rank-numerical-v2-first-case-review.json`. Actual running evidence is under
`research_runs/tokenizer-wide-hybrid-numerical-v2-001`. Monitor this controller;
do not start a competing worker or retry its existing directory. After all27
phases finish, independently review original guards/process/receipt/checkpoint
hashes and export the compact numerical trace/review. No nine-case numerical
pass, MPS pilot, quality-plan freeze or adoption is claimed yet. The closed
failed combined fixture and test holdout remain closed.


### Nine-case rank-policy numerical prerequisite completed

All nine width512 cases (hybrid, gated-MLA and KDA at divisors8/4/2)
completed27 isolated CPU phases under the frozen separate procedure. For each
case, four uninterrupted updates exactly matched one disk-checkpointed update
plus three resumed updates in complete model, optimizer, EMA, torch RNG and
loader RNG hashes. Every saved floating tensor was checked finite. Independent
scalar EMA recurrences sampled first/middle/last values of every parameter;
complete EMA equality, inference-only loading, persistent-buffer preservation
and optimizer-resume rejection also passed. These are four-update correctness
checks, not1,200-step quality, general stability or MPS measurements.

Every600s/4GiB-RSS/8GiB-reserve guard finished successfully. Peak RSS was
3,890,331,648bytes (3.62GiB); the longest phase was87.57s and total worker
time was1,395.45s. All nine partial checkpoints remain retained, totaling
8,858,999,191bytes. No failed artifact was replaced or cleaned up, and the
old failed combined continuation fixture was never called. No worker remains.

The independent terminal reviewer verified all121 original files, all27
process/guard/receipt chains, exact argv/config/plan identities and original
checkpoint hashes before and after review. Eight small reviewer checks passed,
including rejection of failed guards, mismatched resume/checkpoint hashes,
extra/missing files, nonfinite recurrence diagnostics and wrong argv. Together
with the five earlier procedure checks, these are13 distinct small tests; the
27 actual CPU phases are reported separately. The reviewer strengthened exact
file-set/hash/finite checks before this first terminal review; it did not change
the frozen worker. Geometry/cache coverage reuses the unchanged, hash-verified
prior rank evidence, rather than claiming new cache runs.

The compact numerical package contains only two meta files, totaling65,396bytes,
and passes the production `numerical_review` binder. Original metadata for112
non-checkpoint files is also archived byte-for-byte in the portable terminal
review; original checkpoints and run records remain at their recorded paths.
Evidence: `rank-numerical-v2-reviewed-001/meta/numerical.json`,
`rank-numerical-v2-terminal-review.json`, and the terminal-review guard/test logs.

A fresh disposable identity plan then exercised `bind_campaign` without mocks,
using the actual numerical package, the27,838-byte train-only input package,
current source/runtime/corpus and the complete3-pilot/108-conditional-request
table represented by its compact exact identity. It passed with11 bound files,
a40,760-byte plan and142,859 initial metadata bytes. The gate correctly returned
false for dispatch authority, complete storage fit and measured pilots. This
identity-only plan is not the resource-pilot or quality campaign freeze, and
its initial census does not prove future metadata/transient high-water fit.
Evidence: `actual-prerequisite-binding-v2-review.json` and guard.

Next finish the complete real metadata/inventory/transient envelope and final
quality/cost aggregation review, then freeze and measure the serial resource
pilots and freeze the finite quality plan before training. Source/runtime
changes require explicit prerequisite review; do not silently reuse these
source-bound passes. MPS pilots, research inference and quality training remain
unlaunched. All selected/fresh evidence and the test holdout remain protected.


### Finite wider-hybrid freeze and final decision wiring — October 29, 2026

The real numerical/input prerequisites are now frozen for `/Users/junyi/Documents/ChatGPT/LLM-pretraining-RSI/research_runs/tokenizer-wide-hybrid-001`;
plan SHA256 `bb44ea3b55db81b6808072f36d09a9eb8c5d263aa18bc88d06acc88b4ffc19d5`. The plan fixes three resource pilots and
24 quality trajectories (108 conditional request options), with the existing
serial worker, retention, memory, time, storage and closed-holdout constraints.
Quality remains gated on measured and independently accepted pilots.

The first virtual metadata estimate missed552 occurrences of synthetic
`raw.pt` names. Corrected accounting reserves real timestamped filenames and
binds both the preparation and final-decision sources. The numerical trace is
now a lossless gzip attachment:63,746 original bytes become12,576 bytes;
exact decompression and the unmodified production identity gate pass. Original
reports, traces and every numerical checkpoint remain untouched. No numerical
or synthetic training fixture was repeated. Corrected meta bound is
2,058,492 of2,097,152 bytes, leaving
38,660 bytes. Every other group fits its
existing cap; the conservative count is854 of2048 files. Full-duration quality
resource reports/events are bounded at1,282,827/1,106,155 bytes before remaining
fixed-schema padding; metrics, sample text, controller receipts and control
replacement overlap are included. Success-schema estimates remain subject to
the actual bounded writers and watchdogs; unexpected output stops the run.

`review_final_campaign_v2.py` now composes strict quality OR (neutral quality
AND measured memory), after full production acceptance replay of all fresh
triples. It checks its own frozen source hash, requires all three controller
receipts, binds result hashes, and rechecks protected files. It preserves the
conditional-bootstrap caveat and reports latency without a speed gate; it does
not itself modify an adopted recipe. Seventeen new small tests passed across
two bounded runs (14 arithmetic/boundary cases,3 file-binding wrapper cases).
No model or timing worker was used in these tests. The final external review
has a128KiB cap, additionally reserved at freeze.

Evidence: `finite-campaign-freeze-v2-review.json`, its finished guard, the exact
`finite-campaign-v2-frozen/meta/` archive, and the two final-decision test logs.
The earlier `full-metadata-v2-review.json` is retained as an explicitly
superseded estimate. Source/runtime hashes match the completed numerical and
train-only input passes. At freeze no resource pilot or quality worker has
launched. Next: serial resource pilots through the existing production
`run_pilot_once`, then the independent pilot boundary and quality readiness.


### Real MPS resource pilots accepted — October 29, 2026

All three width512/divisor2 resource pilots completed and passed independent
production acceptance: hybrid49.04s, gated MLA41.69s, KDA56.64s. Each used
20 updates. Maximum observed RSS was3,233,792,000bytes (3.01GiB), and maximum
MPS driver allocation was4,112,433,152bytes (3.83GiB), below4/6GiB guards.
Raw/H200 prefill/decode/reset cache checks passed at both batches; maximum
absolute error was4.65e-6. Tensor finiteness and exact EMA-export checks passed.
These establish measured local feasibility, not1200-step stability or quality.

The authorized pilot retention boundary independently re-audited all receipts,
executed its exact six-file pilot-checkpoint manifest, verified the deletion
journal and protected noncheckpoint evidence, and passed remaining full-campaign
storage. Numerical partials, selected checkpoints and fresh confirmations were
outside that manifest. All original pilot metadata bytes plus the subsequent
boundary changes are archived in `resource-pilot-reviewed-v2.json` and
`resource-pilot-boundary-v2-archive.json`. The complete boundary ref is
`meta/pilot-boundary-complete.json`, SHA256
`455b3f962b50b7e9f6bd72fa1e81707ff23cbcdf412ba946893d04564a6c6afc`.

The finite quality grid was fixed before pilots. Next is its first1200-step LR
trial through the existing production readiness/serial-dispatch path, using
this accepted boundary and the same frozen source/runtime. No research quality
result or adoption is claimed by these pilot observations.


### First quality readiness failure and narrow correction — October 29, 2026

Campaign `research_runs/tokenizer-wide-hybrid-001` is now closed after its first
quality request failed before a process launch. The error was
`Readiness inputs changed after guard completion`; quality slot00 contains only
its request, with no process directory or training output. All pilot successes,
audited retention receipts and the failed readiness/dispatch evidence are
preserved. This campaign and failed attempt must never be retried.

The production binder returns live `metadata_bytes` alongside frozen identity.
Readiness compared the entire returned object across its own guard's completion,
although the legitimate final guard write can change byte length. Prior small
fixtures omitted that live census field. Read-only forensics reproduced the
comparison sensitivity using the real binder result; the exact starting-guard
bytes were not retained, so no exact historical byte delta is claimed.

The correction excludes only that census counter from the post-guard frozen
comparison, checks its range, and retains current census enforcement, finished
guard validation, immutable inventory, every other control, full storage and
lock checks. All26 focused readiness tests passed, including actual byte-count
growth from a valid finished guard and rejection of changed frozen files,
source context and other controls. No closed fixture or campaign was rerun.

`readiness-fix-source-applicability-v2-review.json` verifies that this is the only
changed source file and runtime is identical. Trainer/model/optimizer/EMA/cache
and input/tokenizer code are unchanged, so the nine-case numerical and exact
train-input evidence remain applicable by explicit review, without claiming
those old runs executed under the new hash. A new package and finite freeze
must bind both source identities, that applicability review and original
evidence. No quality training has started; replacement campaign gates remain.


### Replacement campaign002 frozen — October29,2026

The pre-freeze audit found the live-census comparison in child entry as well
as readiness. Entry now excludes only that counter while preserving all
append-only log, immutable inventory, control, transport and storage checks.
All40 distinct entry cases pass:39 passed initially, with an overflow error
message mismatch because the real census rejected it earlier; all5 log cases
passed after updating that test expectation in a fresh small fixture. Initial
log/source preserved. The unchanged readiness fix retains its26-case pass.

The explicit source-applicability review verifies only these two control files
changed; the other226 source hashes and runtime match the actually executed
numerical/input evidence. Campaign002 binds a lossless attachment containing
the original reports and applicability review. No numerical/input rerun and
no transplant of old pilot/process/cleanup receipts occurred.

Plan SHA256 `dba216b378680ecd56641051c00c5077cb99006f45858ddc03bb23f338cfc769` fixes3 pilots,24 quality trajectories and108
conditional requests under the same scientific/resource/retention rules.
Meta envelope: 2,073,259 of2,097,152bytes; 23,893bytes
remaining. Other groups fit; file peak855. The unmodified production binder
passes. The freeze/evidence commit is e04f4a0. Campaign001 remains closed.
The serial starter will run3 new guarded pilots, verify their retention
boundary, then admit only quality slot0 through production readiness.


### Campaign002 resource acceptance and first quality training — October29,2026

All3 new resource pilots passed independent acceptance with raw/H200 cache
checks and tensor audits. Durations were47.69s, 41.76s, 56.53s
for hybrid, gated MLA and KDA. Peak RSS was3,233,333,248
bytes; peak MPS driver memory was4,112,433,152
bytes, within the original4/6GiB limits. The exact authorized pilot-retention
manifest completed and its boundary was independently verified. Original
pilot metadata and boundary files are archived; no unrelated or protected
weights were targeted.

Quality slot0 has passed the corrected readiness and live-child handoff and
started real MPS training: width512 hybrid, divisor4, LR0.000625, seed3501,
1200steps. Parent PID3352, child PID3463 at this observation.
Only one training worker is active. The starter stops after this one quality
trial; future LR slots require their production gates and explicit next
continuation. No quality result, selection or adoption is claimed yet.
Follow actual process/events and `campaign002-terminal-v2.json`; never launch
a duplicate while it runs, and preserve/close any failure without retry.


### Campaign002 first quality result independently verified — October29,2026

Slot0 completed1200 MPS updates at width512: hybrid, divisor4,
LR0.000625, seed3501. Full-validation H200 CE is4.961644521
and raw endpoint CE is4.988754896, over34278 targets
and111536 target bytes. This is one LR-screening cell, not a family comparison,
selection or adoption. Sampled online validation did not choose the checkpoint;
the closed test remains unused.

Child wall time was789.85s. Sampled peak RSS was2702704640bytes
and MPS driver memory was3969826816bytes, below the unchanged4/6GiB
limits. The completed entry guard also passed. Independent read-only replay
verified the original dispatch transport, all scientific LR-prefix artifacts,
raw/H200 hashes, pilot-retention lineage, frozen source/input bindings and
remaining full-panel storage. About43.30GB was free against29.38GB required.
The exact non-weight campaign bytes and all file hashes are preserved in
`tokenizer-evidence/wide-hybrid-preparation-001/campaign002-lr-prefix-01-v2-review.json`;
the independent review guard finished successfully. Both slot0 checkpoints remain.

The one-slot continuation wrapper calls the unchanged production `run_lr_once`
for a reviewed prefix and stops after that slot. Next is frozen slot1:
all-gated-MLA, divisor4, LR0.000625, seed3501,1200updates. All9 LR
cells must complete before rate selection or eligible screen cleanup. No
training implementation, frozen plan, guard or statistical rule changed.

Slot0 review/evidence was committed and pushed as `cd46ebc`. Frozen LR slot1
started through production gates with parent4486 and child4497. The wrapper
stops after this slot; later slots require fresh prefix review and serial dispatch.


### Campaign002 LR slot1 independently verified — October29,2026

All-gated-MLA at width512/divisor4, LR0.000625, seed3501 completed1200updates.
Full-validation H200 CE is4.953339260; raw endpoint CE is4.920094199.
Hybrid minus gated-MLA H200 CE at this one LR/seed is+0.008305262;
this incomplete screening comparison supports no selection or adoption.
H200 remains the declared selector even when the raw endpoint scores lower.
Child wall time was467.00s, sampled peak RSS3,169,337,344bytes and
MPS driver3,567,042,560bytes. All guards passed; test holdout stays closed.

Independent replay of both completed LR cells passed, including original
transport/control receipts, checkpoint hashes, scientific artifacts, pilot
retention history and frozen bindings. Exact non-weight bytes and file hashes
are archived in `tokenizer-evidence/wide-hybrid-preparation-001/campaign002-lr-prefix-02-v2-review.json`;
its guard finished successfully. Both raw/H200 pairs remain protected during
the incomplete screen. Remaining full-panel storage requires28,162,784,220bytes
against42,054,352,896bytes free. Next is frozen slot2: all-KDA at the same
LR/divisor/seed/width/budget, through the unchanged one-slot production wrapper.

The two-cell review was committed and pushed as `ce4d9e7`. Frozen LR slot2
started through production gates with parent5171 and child5184. Exactly one
training worker is active; the wrapper stops after this all-KDA trial.


### Campaign002 first LR trio independently verified — October29,2026

All-KDA slot2 completed1200updates at width512/divisor4, LR0.000625,
seed3501. Full-validation H200 CE is4.947694237; raw CE is4.975362476.
The first LR trio is hybrid4.961644521, gated-MLA4.953339260 and
KDA4.947694237. These single-seed screening differences do not determine
family superiority, rate selection or adoption; the remaining six LR cells,
rank stage and fresh confirmations retain their frozen rules.

Child wall time was909.66s; sampled peak RSS2,600,484,864bytes and MPS
driver4,104,044,544bytes passed the unchanged guards. Independent replay
of all three completed cells passed the transport/scientific/checkpoint/input
and retention-history checks. Lossless non-weight bytes and all hashes are
archived in `tokenizer-evidence/wide-hybrid-preparation-001/campaign002-lr-prefix-03-v2-review.json`;
its review guard finished. All three raw/H200 pairs remain. Full remaining
storage requires26,907,661,636bytes against40,776,089,600bytes free.
Test holdout is unused. Next is frozen slot3: all-KDA, LR0.00125,
divisor4, seed3501, width512 and1200updates, through production serial gates.

The three-cell review was committed and pushed as `f04fcfc`. Frozen LR slot3
started through production gates with parent6128 and child6139. The sole
training worker runs all-KDA at LR0.00125; the wrapper stops after this trial.


### Campaign002 second-rate KDA result — October29,2026

Slot3 all-KDA at LR0.00125, width512/divisor4, seed3501 completed1200updates.
Full-validation H200 CE is4.980911690, worse by0.033217454 than its
LR0.000625 screen; raw endpoint CE is5.084563022. Faster early sampled
validation improvement did not predict the final H200 ranking. This is a
single-seed LR comparison, not final rate selection or evidence of a general
architecture disadvantage. The frozen grid is unchanged.

Child wall time was911.94s; sampled peak RSS2,629,910,528bytes and
MPS driver4,104,044,544bytes passed the original guards. Independent replay
of all four completed cells verified transport, scientific records, checkpoint
hashes, frozen source/input and retention history. The lossless archive is
`tokenizer-evidence/wide-hybrid-preparation-001/campaign002-lr-prefix-04-v2-review.json`;
its guard finished. Remaining full-panel storage requires25,652,539,052bytes
against39,483,129,856bytes free. All four raw/H200 pairs remain; no test use,
cleanup, selection or source change occurred. Next is slot4: hybrid at
LR0.00125, with the same divisor4/seed3501/width512/1200update budget.

The four-cell review was committed and pushed as `1a531c5`. Frozen LR slot4
started through production gates with parent7031 and child7063. The sole
training worker runs hybrid at LR0.00125; the wrapper stops after this trial.


### Campaign002 second-rate hybrid result — October29,2026

Slot4 hybrid at LR0.00125, width512/divisor4, seed3501 completed1200updates.
H200 full-validation CE is4.981121139, worse by0.019476618 than its
LR0.000625 screen; raw endpoint CE is5.083567067. Both hybrid and KDA
currently score worse at the middle rate than the lower rate on this screening
seed. The higher-rate cells, gated-MLA middle-rate cell and fresh confirmation
remain necessary; no grid change, rate selection or adoption follows.

Child wall time was806.65s; sampled peak RSS2,612,396,032bytes and MPS
driver3,969,826,816bytes passed the unchanged guards. Independent review of
all five completed cells passed original transport, scientific/checkpoint,
frozen-input and retention-history checks. Exact non-weight bytes and hashes
are archived in `tokenizer-evidence/wide-hybrid-preparation-001/campaign002-lr-prefix-05-v2-review.json`;
its guard finished. Full remaining storage requires24,405,766,076bytes
against38,221,434,880bytes free. All five raw/H200 pairs remain and test
holdout is closed. Next is frozen slot5: gated-MLA at LR0.00125,
divisor4, seed3501, width512 and1200updates.

The five-cell review was committed and pushed as `a1b0bf1`. Frozen LR slot5
started through production gates with parent7791 and child7818. The sole
training worker runs gated-MLA at LR0.00125; the wrapper stops after this trial.


### Campaign002 closed on slot5 checkpoint-lock timeout — October29,2026

Slot5 gated-MLA at LR0.00125 logged all1200updates but failed during raw
checkpoint output. The parent reported `Metadata producer lock timeout; no retry`
and terminated/reaped child7818 with returncode−15 after456.04s. No research
worker remained on process inspection. There is no complete raw/H200 pair,
training summary, full-validation result, tensor/resource audit or acceptance
for slot5. Its sampled training metrics cannot be used for selection. The failed
761,768,510-byte raw `.pt.tmp` is protected, SHA256
`2744ab5d95a972152be379d6adf49fc2125fecf8b942499b227a881a4c67ebb3`.

The forensic review verifies every file from the accepted five-cell archive is
unchanged and preserves all current non-weight bytes, hashes and relevant source
bytes in `tokenizer-evidence/wide-hybrid-preparation-001/campaign002-lock-failure-v2-review.json`.
Its independent guard finished. No cleanup or retry occurred. Last sampled RSS
and MPS peaks were616,054,784 and3,567,042,560bytes, respectively; sampled
resource limits did not trigger. These samples do not establish an instantaneous
peak after the final sample. Test holdout stays closed. Campaign002 is now closed;
slot6 must not launch, and the unsuccessful slot5 cannot be resumed or retried.

The shared metadata lock has a2s nonblocking acquisition deadline. Checkpoint
writes release it between1MiB chunks and immediately compete to reacquire it;
this provides no fairness guarantee. The evidence is consistent with starvation
or a slow critical section, but no lock trace distinguishes them. Existing tests
cover tiny chunk boundaries and a640-byte child producer with explicit sleeps;
they do not establish progress for unpaced sustained writes with a larger
campaign census. Do not relabel this infrastructure failure as a numerical or
quality regression.

Next prerequisite is a new isolated, bounded producer/supervisor contention
fixture: no model/corpus/training, one child, fresh paths, explicit source hashes,
wall/RSS/disk/output caps, actual lock-wait/hold durations, sustained unpaced
checkpoint chunks and simultaneous supervisor/resource publication. First retain
baseline evidence without changing production source; then design and test a
versioned progress fix, preserving fail-closed behavior, metadata/inode/byte
integrity, storage checks and independent resource supervision. No arbitrary
lock-timeout increase or disabled guard is justified by this failure. A fresh
source/evidence applicability review and finite campaign plan are required before
any later training; neither002 nor any earlier closed fixture is reopened.


### Checkpoint namespace handoff mitigation — October29,2026

A separately frozen nontraining baseline used one child, an800-file synthetic
metadata census and56MiB of unpaced checkpoint bytes. It preserved the original
2s lock deadline and1MiB chunks,180s parent/150s child limits,4GiB RSS,
8GiB reserve and64MiB total output cap. It completed in1.88s; the parent
waited up to0.40315s while the longest measured hold was0.02817s. Only one
resource sample occurred. This did not reproduce the original failure; its
larger census and much smaller checkpoint differ from002's250files/761MiB
failed partial. Instrumentation can also affect scheduling. Baseline plan,
script, original source and lossless metadata are retained.

Only the production checkpoint writer changes: after each accumulated MiB it
yields20ms outside the namespace lock, including across small serialization
callbacks. This gives other publishers an interval to acquire the lock before
more checkpoint output. It preserves the2s deadline and all byte/inode/hash,
capacity, disk, memory and supervision checks. It adds intentional output time;
it is a cooperative mitigation, not a guarantee of scheduler fairness.

All134 checkpoint/metadata/process tests pass under the bounded test guard.
New tests verify both supervisor and resource publishers finish during a
namespace-free handoff, and small callbacks accumulate yields without changing
bytes. Existing lock-timeout, partial retention, mutation, storage and process
failure checks still pass. A distinct changed-source fixture with the same
56MiB/800file workload completed in3.91s: parent maximum wait0.05603s,
maximum hold0.03078s and three resource samples. Both fixtures verified exact
output hashes; these single observations do not establish general performance
or prove the original failure is eliminated.

The independent report `tokenizer-evidence/wide-hybrid-preparation-001/checkpoint-handoff-v1-review.json`
archives the tests and changed-source fixture and verifies every closed002
file remains unchanged, including five accepted pairs and the failed partial.
No training, numerical panel, closed fixture or failed trial was rerun. An
initial report command used system Python and failed importing a dependency
before any review mutation; the report then completed under the repository
venv. Both original fixture attempts remain immutable.

Before further training: explicitly review source applicability of this
scheduling-only change, preserve original executed-source provenance, and
re-audit a fresh finite plan's complete storage/metadata envelope including
all002 protected weights. Old001/002 remain closed. Do not use the passed
synthetic fixture as a pilot, numerical or quality result, or reuse002 process
receipts as a new campaign's transport provenance.


### Campaign003 source applicability and finite freeze — October29,2026

The explicit applicability review compares against the original executed
numerical/input source. Exactly three production files differ: the two already
reviewed census checks and checkpoint-writer scheduling. The other225 source
hashes and runtime match. The CPU numerical procedure uses the unchanged
checkpoint save/read helpers; input encoding has no checkpoint-output dependency.
Serialization/guard coverage includes the134 passed tests. Original numerical
and input reports, hashes and executed-source identity remain in the lossless
applicability attachment; neither prerequisite was rerun. New real pilots must
still validate this source's actual raw/H200/export/resource behavior.

Campaign003 is a fresh complete panel under the handoff source, not a continuation
of002's failed attempt. The original candidate grid, seeds, budgets and selection
rules remain unchanged:3 pilots,9 LR cells,6 rank cells,9 fresh confirmations,
then3 inference seeds. The reused screening seed3501 is not independent
replication;002's five accepted cells remain descriptive and are excluded from
003 selection. No old process, pilot or cleanup receipts are transplanted.
Old001/002 stay closed and every retained/failed checkpoint remains protected.

Plan SHA256 `4a08476fe21b92ae8debc7d7b004dad8b0f082637e76e65ad7cdc33f22f85102`.
The unmodified production binder accepts the complete numerical/input bindings.
The conservative meta envelope is2,075,524 of2,097,152bytes
(21,628 remaining), all other groups fit, and the file peak is855.
Full future storage plus the external review allowance requires30,631,412,484bytes
against37,209,874,432bytes actually free. That free-space reading already includes
all6,987,283,846bytes of protected002 weights/partial; no speculative cleanup
credit is used. Reserve and all time/memory/metadata guards stay unchanged.
The freeze and applicability guards finished successfully.

After committing this freeze, the serial starter may run only the3 new resource
pilots, verify their exact authorized retention boundary, then admit quality
slot0 through production gates and stop. Any failure closes that new attempt;
no automatic retry, source change or later quality dispatch occurs.

Campaign003 freeze/evidence was committed and pushed as `bc8e49b`. The starter
is active with parent9293 and initial pilot0 child9294. Only one local MPS
worker is active. No quality trial has launched at this observation; actual
pilot acceptance and retention gates precede the starter’s sole quality slot0.


### Campaign003 pilot acceptance and first quality dispatch (2026-10-29)

All three fresh resource pilots completed and were accepted: hybrid89.5785s,
gatedMLA83.7171s, KDA100.3218s. These are20-step resource/export checks at
width512, divisor2, LR0.0025 and seed3491, not quality comparisons.
The immutable pilot archive contains82 file hashes and76 exact noncheckpoint
byte records. Archive byte hashes and the three dispatch references were checked.
The verified retention boundary removed exactly six authorized pilot weight
files totaling3,769,780,224bytes; their archived hashes match the deletion manifest.
No selected, confirmation, numerical or failed-attempt checkpoint was removed.

Boundary reference `meta/pilot-boundary-complete.json` SHA256
`711f3f95cb61c692bf27257e901cca26b625592e5043afe9984a6865a369c8b8`.
The standalone boundary/storage review is evidence only; its dispatch-authorized
flags remain false. The production quality gates separately admitted slot0,
hybrid/divisor4/LR0.000625/seed3501, with1200 planned updates.
At observation parent9293 and sole worker9618 were active, with no process error;
no quality acceptance or new quality finding is claimed yet. The starter stops
after this slot. Independently review its accepted prefix before any later slot.
Source and plan remain frozen. Campaigns001/002 remain closed, test holdout closed.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/campaign003-pilots-v2-archive.json`,
`campaign003-pilot-boundary-v2.json`, and `campaign003-quality0-start-v2.json`.
Read-only archive inspection initially resolved evidence-relative paths against
the campaign root and stopped with missing-path/key errors; resolving against
`evidence/` completed verification. No artifact or worker was changed by these checks.


### Campaign003 first LR cell independently reviewed (2026-10-29)

Slot0 completed1200 updates: hybrid, width512, divisor4, LR0.000625,
seed3501. Full-validation H200 CE is4.957330844412886 and raw CE
4.985172284648273, on34278 targets/111536 bytes across17 blocks.
The child completed successfully in832.5694s; its raw/H200 pair totaling
1,246,772,976bytes remains retained. This single screening cell establishes
no family comparison or adoption decision; closed002 results are not pooled.

The independent prefix1 review verified production prior transport, frozen
identity, pilot history, result/checkpoint bindings, all115 file hashes and
remaining storage, and archived exact nonweight bytes. Its CPU guard finished
in4.5116s with no reason/error and peak RSS220,086,272bytes. Available space
35,935,940,608bytes exceeded required29,384,508,436bytes. No cleanup occurred.
The external reviewer and one-slot continuation were adapted from002 solely
by changing campaign identity/path strings; source equivalence and syntax were
checked. Production source, finite plan and guards remain unchanged.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/campaign003-lr-prefix-01-v2-review.json`,
its guard, and `campaign003-terminal-v2.json`. After committing this evidence,
the sole next dispatch is slot1: gatedMLA, divisor4, LR0.000625, seed3501.
The continuation stops after that slot, preserves failures and never retries.
No selection, fresh confirmation, inference or closed-test result is claimed.


Prefix1 evidence and serial wrappers were committed/pushed as `5a3f5c0`.
The sandbox-only launch preflight could not see MPS and stopped at its availability
assertion before any launch receipt, campaign mutation or child. Host execution
then passed that same preflight with CPU fallback disabled. Slot1 controller
PID10515/session57898 is active through the original production gates; it may
run only this one frozen cell. This is not a retried training attempt.
Launch identity is preserved in `campaign003-lr-01-v2-launch.json`.


### Campaign003 second LR cell independently reviewed (2026-10-29)

Slot1 gatedMLA completed1200 updates at width512/divisor4/LR0.000625,
seed3501. Full-validation H200 CE4.953336207757286; raw CE4.92009252749222.
Child elapsed508.1716s. Relative to slot0 hybrid, H200 CE is lower by
0.003994636655600 on this screening seed; the incomplete adaptive panel
supports no superiority or adoption conclusion. Both raw/H200 pairs are retained,
2,468,497,192bytes total. No closed002 result is pooled.

Independent prefix2 review verified144 file hashes, losslessly archived all
nonweight evidence, and passed production prior/frozen/pilot/checkpoint/storage
bindings. The external guard finished in8.7815s, peak RSS220,069,888bytes,
with no error. Observed free34,695,581,696bytes exceeds required28,162,784,220.
No cleanup, selection or closed-test evaluation occurred. Evidence is in
`tokenizer-evidence/wide-hybrid-preparation-001/campaign003-lr-prefix-02-v2-review.json`,
its guard and `campaign003-lr-01-v2-terminal.json`.
After committing/pushing this review, dispatch only slot2 KDA at the same
LR/divisor/seed through the original one-slot controller. Frozen source and
all resource guards remain unchanged.


Prefix2 review committed/pushed as `ea6f4e9`. The sole slot2 KDA controller
PID11085/session26358 is active through production gates; its immutable
receipt is `campaign003-lr-02-v2-launch.json`. It stops after this cell.


### Campaign003 third LR cell independently reviewed (2026-10-30 UTC)

Slot2 KDA completed1200 updates at width512/divisor4/LR0.000625, seed3501.
Full-validation H200 CE4.941342525602911; raw CE4.968353925571475.
Child elapsed952.9232s. The first LR group is complete: H200 CE hybrid
4.957330844412886, gatedMLA4.953336207757286, KDA4.941342525602911.
KDA is lowest at this setting on one screening seed; the full nine-cell LR
panel must finish before family-specific LR selection. No superiority,
confirmation or adoption claim follows from this partial panel.

Independent prefix3 review verified173 file hashes and losslessly archived
nonweight evidence. Production prior/frozen/pilot/checkpoint/storage bindings
passed; external guard finished13.0889s with peak RSS224,231,424bytes and no
error. All three raw/H200 pairs remain retained,3,723,619,776bytes total.
Observed free33,422,503,936bytes exceeds required26,907,661,636bytes.
No cleanup or closed-test evaluation occurred; closed002 results are not pooled.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/campaign003-lr-prefix-03-v2-review.json`,
its guard and `campaign003-lr-02-v2-terminal.json`.
After commit/push, the sole next cell is slot3 KDA at LR0.00125 with unchanged
width/divisor/seed/budget and original production gates. Frozen source and
resource guards remain unchanged.


Prefix3 review committed/pushed as `427f1d1`. Slot3 controllerPID12027
(session88858) is active through production gates for the single frozen KDA
LR0.00125 cell. Launch receipt: `campaign003-lr-03-v2-launch.json`.


### Campaign003 fourth LR cell independently reviewed (2026-10-30 UTC)

Slot3 KDA completed1200 updates at width512/divisor4/LR0.00125, seed3501.
Full-validation H200 CE4.978608155903051; raw CE5.076564651378714.
Child elapsed954.1204s. Increasing KDA LR from0.000625 to0.00125 raised
H200 CE by0.037265630300140 on this screening seed. This descriptive result
favors the lower of these two rates, but selection remains deferred until all
nine frozen LR cells complete. No closed002 data are pooled and no fresh-seed
or adoption claim follows.

Independent prefix4 review verified202 file hashes and losslessly archived
nonweight evidence, with production prior/frozen/pilot/checkpoint/storage
bindings intact. Guard finished17.3942s, peak RSS224,575,488bytes, no error.
All four raw/H200 pairs remain retained,4,978,742,360bytes total. Observed
free32,149,258,240bytes exceeds required25,652,539,052bytes. No cleanup or
closed-test evaluation occurred. Evidence:
`tokenizer-evidence/wide-hybrid-preparation-001/campaign003-lr-prefix-04-v2-review.json`,
its guard and `campaign003-lr-03-v2-terminal.json`.
After commit/push, the next sole cell is slot4 hybrid at LR0.00125 with the
same width/divisor/seed/budget. Production source, plan and guards remain frozen.
Old heartbeat is confirmed paused and this continuation heartbeat active.


Prefix4 review committed/pushed as `3da4671`. Slot4 controllerPID12854
(session74741) is active through production gates for the single frozen hybrid
LR0.00125 cell. Launch receipt: `campaign003-lr-04-v2-launch.json`.


### Campaign003 fifth LR cell independently reviewed (2026-10-30 UTC)

Slot4 hybrid completed1200 updates at width512/divisor4/LR0.00125,
seed3501. Full-validation H200 CE4.975304379509017; raw CE5.07451207831503.
Child elapsed850.2786s. Hybrid H200 CE is0.017973535096131 higher than
its LR0.000625 cell on this screening seed. Five of nine LR cells are complete;
family-specific LR selection remains deferred. Closed002 results are not
pooled; no fresh-seed or adoption claim follows from this partial panel.

Independent prefix5 review verified231 file hashes and losslessly archived
nonweight evidence. Production prior/frozen/pilot/checkpoint/storage bindings
passed; guard finished21.7198s, peak RSS224,280,576bytes, no error. All five
raw/H200 pairs remain retained,6,225,515,336bytes total. Observed free
30,871,764,992bytes exceeds required24,405,766,076bytes. No cleanup or
closed-test evaluation occurred. Evidence:
`tokenizer-evidence/wide-hybrid-preparation-001/campaign003-lr-prefix-05-v2-review.json`,
its guard and `campaign003-lr-04-v2-terminal.json`.
After commit/push, the next sole cell is slot5 gatedMLA at LR0.00125 with
unchanged width/divisor/seed/budget. This is the originally frozen003 panel;
closed002's failed slot5 remains preserved and is not resumed or retried.
Production source, plan and guards remain frozen.


Prefix5 review committed/pushed as `714fd96`. Slot5 controllerPID13822
(session26123) is active through production gates for the single frozen gatedMLA
LR0.00125 cell. Launch receipt: `campaign003-lr-05-v2-launch.json`.


### Campaign003 sixth LR cell independently reviewed (2026-10-30 UTC)

Slot5 gatedMLA completed1200 updates at width512/divisor4/LR0.00125,
seed3501. Full-validation H200 CE4.821469271057781; raw CE4.824475110326953.
Child elapsed529.3563s. H200 CE improved by0.131866936699505 versus gatedMLA
at LR0.000625; hybrid and KDA instead worsened at this higher rate on the same
screening seed. GatedMLA is0.153835108451236 lower than hybrid at LR0.00125.
These descriptive comparisons motivate completing equal family-specific tuning;
six of nine frozen cells are complete, with no LR selection or adoption yet.
Fresh confirmations and final conditional criteria remain necessary.

The checkpoint/export/validation path completed successfully under the frozen003
handoff source. This does not prove the prior metadata timeout impossible;
closed002's failed cell and partial checkpoint remain preserved, not resumed.
Independent prefix6 review verified260 file hashes and losslessly archived
nonweight evidence; production prior/frozen/pilot/checkpoint/storage bindings
passed. Guard finished29.4563s, peak RSS224,624,640bytes, no error. All six
raw/H200 pairs remain retained,7,447,239,552bytes total. Observed free
29,623,996,416bytes exceeds required23,184,041,860bytes. No cleanup or
closed-test evaluation occurred; closed002 results are not pooled.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/campaign003-lr-prefix-06-v2-review.json`,
its guard and `campaign003-lr-05-v2-terminal.json`.
After commit/push, next is solely slot6 gatedMLA at LR0.0025, unchanged
width/divisor/seed/budget, original production gates, frozen source and guards.


Prefix6 review committed/pushed as `eb6c9f3`. Slot6 controllerPID14687
(session77627) is active through production gates for the single frozen gatedMLA
LR0.0025 cell. Launch receipt: `campaign003-lr-06-v2-launch.json`.


### Campaign003 seventh LR cell independently reviewed (2026-10-30 UTC)

Slot6 gatedMLA completed1200 updates at width512/divisor4/LR0.0025,
seed3501. Full-validation H200 CE4.85658606647173; raw CE4.919580532579774.
Child elapsed535.4267s. H200 CE is0.035116795413949 higher than gatedMLA
at LR0.00125. Its three-rate screen is complete:0.000625/0.00125/0.0025
produce4.953336207757286/4.821469271057781/4.85658606647173 respectively.
The middle rate is descriptively lowest; formal selection remains deferred
until the two remaining frozen hybrid/KDA cells complete. No adoption claim.

Independent prefix7 review verified289 file hashes and losslessly archived
nonweight evidence; production prior/frozen/pilot/checkpoint/storage bindings
passed. Guard finished35.0949s, peak RSS224,526,336bytes, no error. All seven
raw/H200 pairs remain retained,8,668,963,768bytes total. Observed free
28,387,024,896bytes exceeds required21,962,317,644bytes. No cleanup or
closed-test evaluation occurred; closed002 results are not pooled.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/campaign003-lr-prefix-07-v2-review.json`,
its guard and `campaign003-lr-06-v2-terminal.json`.
After commit/push next is solely slot7 KDA at LR0.0025 with unchanged
width/divisor/seed/budget, original production gates, frozen source and guards.


Prefix7 review committed/pushed as `4ad370f`. Slot7 controllerPID15590
(session89694) is active through production gates for the single frozen KDA
LR0.0025 cell. Launch receipt: `campaign003-lr-07-v2-launch.json`.


### Slot7 reporting-label correction (2026-10-30 UTC)

Prior prose incorrectly named slot7 hybrid. The unchanged frozen plan and actual
request specify slot7 KDA at LR0.0025; slot8 is hybrid. The production controller
correctly dispatched KDA, with four KDA layers, divisor4/width512/seed3501.
A read-only check matched the actual request spec to frozen panel slot7, verified
the original plan digest and unchanged LR-rule source hash. No execution or
selection change was needed. The erroneous label affected commentary and these
notes only; numerical results for completed slots0..6 remain unchanged.
The active KDA worker remains sole worker15757 under controller15590.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/campaign003-slot7-label-correction-v1.json`.


### Campaign003 eighth LR cell independently reviewed (2026-10-30 UTC)

Slot7 KDA completed1200 updates at width512/divisor4/LR0.0025, seed3501.
Full-validation H200 CE5.114296068829509; raw CE5.279939868603322.
Child elapsed975.6345s. H200 CE is0.172953543226598 above KDA at LR0.000625.
KDA's three-rate screen gives4.941342525602911/4.978608155903051/5.114296068829509
for ascending rates. The lowest rate is descriptively best on this seed;
formal family-specific selection still waits for the final hybrid cell.
Eight of nine LR cells are complete; no adoption or fresh-seed claim.

Independent prefix8 review verified318 file hashes and losslessly archived
nonweight evidence. Production prior/frozen/pilot/checkpoint/storage bindings
passed. Guard finished40.8564s, peak RSS224,641,024bytes, no error. All eight
raw/H200 pairs remain retained,9,924,086,352bytes total. Observed free
27,098,804,224bytes exceeds required20,707,195,060bytes. No cleanup or
closed-test evaluation occurred; closed002 results are not pooled.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/campaign003-lr-prefix-08-v2-review.json`,
its guard and `campaign003-lr-07-v2-terminal.json`.
The actual frozen panel identifies final slot8 as hybrid at LR0.0025,
width512/divisor4/seed3501/1200steps. Dispatch only that cell after commit/push.
After acceptance, use the original full-panel LR selection and retention
components with independent archive before authorized cleanup. The prefix
reviewer deliberately stops at8 completed cells; do not alter production gates.


Prefix8 review committed/pushed as `1b50d3c`. Slot8 controllerPID16764
(session51551) is active through production gates for the final frozen hybrid
LR0.0025 cell. Launch receipt: `campaign003-lr-08-v2-launch.json`.


### Campaign003 closed at final LR slot8: metadata lock timeout (2026-10-30 UTC)

Final hybrid/LR0.0025 slot8 failed with
`TimeoutError('Metadata producer lock timeout; no retry')`. Controller16764
stopped/reaped worker16918 (returncode-15), child elapsed811.4618s, wrapper
893.9771s. Actual process inspection confirms no remaining research worker.
The surfaced traceback places the parent at `run_process -> save_control ->
MetadataWriter.publish_json -> _serialized`. All1200 updates were logged, but
raw checkpoint publication did not finish. There is no completed raw/H200 pair,
training summary, training-outputs, result, tensor/resource audit, worker-outputs
or parent acceptance for slot8. Online metrics are not selection evidence.

The protected partial is
`research_runs/tokenizer-wide-hybrid-003/evidence/quality/08/checkpoints/final_001200_1793325864452219000.pt.tmp`,
780,502,912bytes, SHA256
`6a572d5c2a72198e56de2e188940e24161f2cbcea367af55743d0bcaef546f64`.
Independent failure audit verified all337 campaign file hashes twice and every
previously archived prefix8 file unchanged; nonweight bytes and relevant source
were archived losslessly. All eight accepted pairs plus partial remain protected,
10,704,589,264bytes total. Failure-review guard finished8.8228s with no error,
peak RSS228,900,864bytes. Failed worker sampled peaks were RSS634,847,232bytes
and MPS driver3,969,826,816bytes, under original caps; samples do not bound the
instant of failure. No cleanup, LR selection, rank dispatch or test evaluation
occurred. Campaign003 is closed; no retry/resume/replacement cell is authorized
by this evidence. Closed001/002 and the failed numerical fixture stay closed.

The20ms-per-MiB cooperative handoff did not eliminate the timeout. No recorded
lock-acquisition/hold trace distinguishes starvation, a slow filesystem operation,
a long census or other critical-section delay. The earlier56MiB synthetic checks
were not a reproduction of this failure and cannot establish safety at this
checkpoint size. Do not simply extend the2s deadline or repeat training.
Next prepare a separate finite nontraining diagnostic with actual-scale synthetic
checkpoint writes and bounded per-process/thread lock-wait/hold and operation
traces. Preserve original guards, freeze its script/source/workload and allow
one fresh attempt; no model/corpus/checkpoint load and no rerun of closed fixtures.
Only diagnose before choosing another versioned implementation change.

At this audit26,295,799,808bytes were free. The unchanged full-panel initial
requirement30,631,412,484bytes would exceed that by4,335,612,676bytes, before
new diagnostic outputs. Protected evidence gets no speculative cleanup credit.
A new full training panel therefore needs a separately verified storage solution
as well as an evidence-backed fix and new finite plan; it must not auto-launch.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/campaign003-lr-08-v2-terminal.json`,
`campaign003-lock-failure-v2-review.json`, its guard and
`review_campaign003_lock_failure_v2.py`. Previous eight cells remain descriptive
screening evidence, not a completed panel or adoption result.


### Prospective actual-scale contention trace v2 (2026-10-30 UTC)

Freeze one separate nontraining diagnostic after closed003's repeated timeout:
`tokenizer-evidence/wide-hybrid-preparation-001/run_metadata_contention_trace_v2.py`
and `metadata-contention-trace-v2-plan.json`. The new output root is
`research_runs/tokenizer-metadata-contention-trace-v2-001`; no existing fixture
or campaign is resumed. All228 production/runtime-input source hashes match003.
The synthetic CPU payload has four240MiB float32 tensors plus128 small tensors,
serialized through actual torch.save and the unchanged bounded checkpoint writer.
337 flat synthetic control files approximate the failed campaign's file count.

One child only, four CPU threads,150s child/180s parent limits, original4GiB RSS,
8GiB reserve,2s namespace-lock deadline and20ms-per-MiB handoff. Checkpoint cap
964MiB, total output cap992MiB, each external JSON cap4MiB, at most16384 trace
records per process. Both worker locks and original process supervisor remain.
Process-local observational wrappers time the original lock scopes, census,
write and fsync operations with monotonic clocks/PIDs/thread names. No source
mutation, model/corpus/training, existing checkpoint read or test-holdout use.

This is a new scale/serialization diagnostic, not a rerun of the closed56MiB
checks or failed numerical fixture. Instrumentation changes scheduling; flat
synthetic files and CPU tensors do not reproduce nested topology, tensor layout,
prior MPS/dirty-page state or machine load. A successful run cannot prove the
failure eliminated or authorize training. Preserve every outcome, independently
review traces/hashes and unchanged closed003 evidence, then choose further work
from observations. New full-panel storage remains blocked. Syntax and source
pins checked before execution; commit/push this finite plan before its sole run.


### Actual-scale trace v2: timeout reproduced and small-write starvation observed

Frozen plan committed/pushed as `010ff39`; exactly one synthetic attempt ran.
It failed closed with the same 2 s metadata producer-lock timeout. No worker
remains, no retry occurred, and no production source was changed. The new
synthetic partial is 1,006,654,080 bytes, SHA 256
`1c4b101ebf2e022be9035a6662b165bcb05b75e71cafd6972aaf571aa6c8ccfb`.
All 350 diagnostic files are preserved (1,011,820,960 bytes total, below 992 MiB).

During the parent's 2.002416167 s failed acquisition, the child made 154 checkpoint
writes totaling 11,504 bytes, with write sizes 4..256 bytes. 310 overlapping child
lock scopes occupied 1.961031242 s (97.9333%) of that interval. Maximum overlapping
hold was 0.012483916 s; maximum child operation starting in the interval was
0.007236292 s. Across the whole saved trace, parent/child maximum holds were
0.024327625/0.023709708 s. The last 1 MiB write preceded this small-write tail.
The child recorder completed 44 samples with no resource-recorder error.

This supports starvation in this instrumented synthetic run: many small
callbacks repeatedly acquire the namespace lock for census/write checks, while
only 11.5 KiB is accumulated during the parent's deadline. The byte-only 1 MiB
handoff therefore fails to schedule a gap in that interval. No individual long
critical section appears in the saved trace. This narrows a credible mechanism
but does not prove the uninstrumented 003 failure had the identical cause.

Instrumentation has a separate failure: parent trace reached its 16384-record
cap during later failure publication/shutdown. The original lock timeout is
record 16305; later records continue after its end. Per-JSON-piece metadata
writes consumed 15206 trace records. The final traced fsync raised the cap error,
and a partial process-control JSON remains. Child termination caused a secondary
torch.save unexpected-position error. Both fixture guards report exceptions;
this is a failed diagnostic with an incomplete final trace tail, not a passed
test. Saved trace sizes 4,074,367/1,023,669 bytes remain below their 4 MiB caps.

Independent read-only review verified frozen source/script pins, all diagnostic
hashes, original output limits and every one of closed 003's 337 files unchanged.
It archived all noncheckpoint bytes losslessly and hashed the partial without
loading it. Review guard finished 5.4562 s, peak RSS 264,224,768 bytes, no error.
Worker/parent peaks were 1,232,322,560/236,929,024 bytes, below 4 GiB. No cleanup,
training, corpus access or test-holdout use. Evidence:
`tokenizer-evidence/wide-hybrid-preparation-001/metadata-contention-trace-v2-review.json`,
its guard and `review_metadata_contention_trace_v2.py`.

Next design a handoff bounded by elapsed work as well as bytes, so small
callbacks cannot indefinitely delay the cooperative gap; retain the 2 s deadline,
all integrity checks and resource caps. Add targeted regression coverage for
sub-MiB small-call sequences and competing producers. Bound diagnostic operation
tracing by aggregation/thresholds so JSON piece counts cannot consume the event
budget; retain every lock wait/failure and meaningful operation maximum. A
changed-source comparison needs its own frozen plan and new single-attempt root;
never rerun this closed attempt. New training remains blocked pending a validated
fix, fresh evidence contract/finite plan and verified storage solution.


### Prospective elapsed-time checkpoint handoff and trace-v3 comparison

The checkpoint writer now yields outside the namespace lock after either 1 MiB
or 20 ms since its previous handoff. Checks run after preflight, each bounded
write and flush, including empty callbacks. The clock resets after the gap;
existing accumulated-byte behavior remains. The 20 ms gap, two-second namespace
deadline, integrity checks and resource caps are unchanged. A single operation
is not preempted: this is cooperative scheduling, not a hard latency guarantee.

140 checkpoint/metadata/process tests passed in 6.73 s under the unchanged
180 s CPU guard (peak RSS 240,599,040 bytes). Six added cases cover sub-MiB
callbacks with competing publishers, time spent in preflight/write/flush,
STOP during a gap and interrupted handoff preservation. The first command named
a nonexistent process test file, exited 4 and ran no tests; its log and guard
remain, alongside the corrected invocation. No numerical fixture was retried.
Three recorder tests also passed: per-JSON-piece aggregation, explicit overflow
accounting and preservation of original failures when the detail cap is reached.

Freeze one new nontraining trace-v3 comparison in
`research_runs/tokenizer-metadata-contention-trace-v3-001`. It uses the same
four 240 MiB tensors, 128 small tensors, 337 synthetic files, one CPU child,
four threads, 150/180 s child/parent bounds, 4 GiB RSS, 8 GiB reserve and
992 MiB total output allowance as trace-v2. All 228 source pins were compared;
only the checkpoint writer changed. The new recorder keeps detailed locks and
checkpoint writes, aggregates five fixed operation labels and retains bounded
failure records. Overflow marks the trace incomplete without masking an original
exception. All external traces remain capped at 4 MiB. Python version is checked.

The handoff and instrumentation overhead both change, so a pass cannot isolate
their effects or prove training reliability. Commit and push this finite plan
before its sole execution. Review hashes, guards, trace completeness and
unchanged closed003/trace-v2 evidence afterward. Preserve every outcome, no
cleanup or retry, no training or holdout use. Full training-panel storage remains
blocked. Plan, test logs/guards and component review are under
`tokenizer-evidence/wide-hybrid-preparation-001/` with `trace-v3` and
`checkpoint-time-handoff-v2` names.


### Elapsed-time handoff: actual-scale trace-v3 completed and independently reviewed

The prospective source/plan was committed and pushed as `19f65b1` before its
single run. The synthetic checkpoint completed: 1,006,700,215 bytes, SHA256
`853ba84e95525d097d26f2ab9aad58af4bca86ccc182c9cf7eb7cf4f660552db`.
Parent/child guards finished in 73.3748/71.9172 s; peak RSS was
220,463,104/1,230,225,408 bytes. All time, memory, disk and output caps held.
No research worker remains and no retry occurred.

Maximum parent lock wait was 0.0453204 s; child main/resource maxima were
0.0605946/0.0602781 s. Maximum parent/child holds were 0.0233675/0.0222068 s.
The child completed 1,650 checkpoint writes, including 690 sub-MiB writes,
and 70 resource samples with no recorder errors. The trace retained all 629
parent and 2,419 child lock scopes. There were no failures or dropped records.
Aggregation handled 25,337 parent JSON writes without spending detail capacity;
parent/child trace sizes were 161,451/1,048,335 bytes, below 4 MiB each.

Independent guarded review rehashed the output, verified source/plan pins and
all 337 closed003 plus 350 closed trace-v2 files unchanged, and archived every
noncheckpoint byte. Review guard finished in 5.8864 s, peak RSS 240,238,592 bytes.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/metadata-contention-trace-v3-review.json`,
its guard and `review_metadata_contention_trace_v3.py`. The synthetic checkpoint
and all failed evidence remain preserved; no cleanup, model/corpus access or
holdout use occurred.

This supports the revised handoff under one actual-scale synthetic workload.
It does not prove that future training cannot time out, or isolate the handoff's
effect from the changed tracing overhead. Flat synthetic topology, CPU tensors
and absence of preceding MPS training remain limitations. Next audit source
applicability against the completed numerical/input prerequisites and prepare
an explicit new evidence contract. Never reopen campaigns001/002/003 or rerun
the closed numerical/trace fixtures. A new full panel still requires
30,631,412,484 bytes including reserve versus roughly 22.68 GB currently free;
resolve storage prospectively under retention policy or a justified new finite
scientific design before any training. Do not promote the incomplete old panel
or delete protected weights to fit it.


### Time-handoff applicability and retention audit: capacity is now the blocker

A read-only guarded audit compared all 228 source entries: 227 are unchanged
since campaign003, with only the checkpoint writer's scheduling changed. Relative
to the completed original numerical execution, 225 entries remain unchanged;
the two earlier census fixes and checkpoint scheduling are explicitly accounted
for. Runtime identity is unchanged. Original numerical/input reports, trace
artifacts and all numerical phase files/checkpoints were hash-verified without
reexecuting or loading them. The mathematical/state and input evidence remains
applicable, with original executed-source identity retained. New MPS pilot and
process/retention evidence will still be required for a new campaign.

The retention inventory verified all 189 tokenizer `.pt`/`.pt.tmp` files,
62,800,809,984 logical bytes: 148 previously protected selected/fresh pairs' files
(34,119,142,876 bytes), 28 incomplete-campaign/failure files (17,691,873,110),
nine numerical-prerequisite checkpoints (8,858,999,191) and four preserved
synthetic diagnostic weights (2,130,794,807). All 42 prior cleanup paths remain
absent. Closed002, closed003 and both actual-scale diagnostic inventories matched
their saved reviews. No eligible deletion was identified and nothing was removed.
Incomplete screens have no completed, persisted selection permitting cleanup;
synthetic diagnostic saves are not successful resource pilots or losing screens.

The unchanged full-panel requirement is 30,631,412,484 bytes including the 8 GiB
reserve. Observed free space was 22,849,273,856 bytes, a 7,782,138,628-byte shortfall.
For this design, provide additional local capacity or a new output volume with
at least 30.64 GB free (with practical headroom). Do not shrink a frozen guard,
promote an incomplete old panel, delete protected evidence or repeatedly poll
unchanged free space. No new campaign directory, plan freeze or worker was created.

Independent preparation produced six losslessly verified prerequisite files
(64,513 bytes) in `tokenizer-evidence/wide-hybrid-preparation-001/time-handoff-prerequisite-transport-v2/`.
The original reports and applicability review are embedded with hashes and exact
bytes; numerical traces decompress exactly, and the 96 train IDs are unchanged.
The numerical gate accepted the transported evidence. No holdout was accessed.
Replacing the six prior payloads adds 14,501 bytes to the last metadata envelope,
leaving only 7,127 bytes of its former 21,628-byte margin before other plan/path
changes. This is a static delta, not a new complete envelope proof. Once storage
is available, recheck the final destination, source and full metadata envelope,
then freeze the finite plan before any new pilot or training launch.

Evidence: `checkpoint-time-handoff-source-applicability-v2-review.json`,
`time-handoff-prerequisite-transport-v2-review.json`, their guarded procedures
and guard journals under `tokenizer-evidence/wide-hybrid-preparation-001/`.
The old heartbeat remains paused and this continuation heartbeat active.


### October 29: verified lossless archival recovers the required capacity

The human explicitly requested proactive archival/removal while preserving
selected models, fresh confirmation pairs and failure evidence. The earlier
finding of no eligible irreversible deletion still holds; cold archival retains
all checkpoint bytes and resolves storage without discarding those records.
The committed prospective plan (`f6f544b`) covered only completed numerical and
non-selected/non-fresh screen weights. No closed campaign was reopened.

35 loose files totaling 25,008,600,879 bytes were replaced by three protected local
Zstandard archives totaling 13,454,327,571 bytes. Net logical reclamation is
**11,554,273,308 bytes (11.55 GB)**. Each archive was fully decompressed and every
member's path, size and original SHA256 verified before any loose copy was
removed. Removal was journaled and archive hashes were rechecked afterward.

| Archive group | Files | Original bytes | Archive bytes | Net bytes reclaimed |
|---|---:|---:|---:|---:|
| Completed numerical prerequisites | 9 | 8,858,999,191 | 3,625,971,340 | 5,233,027,851 |
| Completed campaign002 screens | 10 | 6,225,515,336 | 3,791,160,683 | 2,434,354,653 |
| Completed campaign003 screens | 16 | 9,924,086,352 | 6,037,195,548 | 3,886,890,804 |

All **154 excluded loose checkpoints** were hash-verified unchanged, including
selected models, all fresh confirmation pairs and every failed partial. All
**3,740 non-checkpoint files** in tokenizer run roots were verified unchanged.
A separate read-only review reconciled the archives with all 250 original
campaign002, 337 campaign003 and 121 numerical-prerequisite file identities.
The original reports retain their executed-source hashes and byte contents;
there was no model load, numerical rerun, training or holdout access.

Observed free space after the operation was 33,493,016,576 bytes, exceeding the
unchanged 30,631,412,484-byte panel requirement by 2,861,604,092 bytes. Other system
activity can affect observed free space; attributable savings are the exact
logical figure above. One archival worker finished within the guarded bounds.
Initial filesystem-compression/sparse-copy probes saved no allocated blocks;
only their independently verified scratch duplicates were removed.

Archives remain protected under `research_archives/checkpoint-compaction-001/`
and are excluded from Git. Text manifests, archive/member hashes, removal
journal, restoration instructions and guard/review evidence are committed under
`tokenizer-evidence/checkpoint-compaction-001/`. Do not delete these archives as
caches. Restore exact bytes before invoking an original-path-only audit, or use
the explicit archive resolver; absent loose paths are intentional archival,
not lost evidence or permission to regenerate checkpoints.

Capacity is now sufficient at the observed snapshot. Next incorporate the
archival proof into prospective prerequisite provenance, audit the complete
metadata envelope at the final new destination, and freeze the finite plan.
Recheck actual capacity before dispatch and run new resource pilots first.
No training has launched under the changed source. Continue proactive bounded
storage management without waiting for another human cleanup request.


### Campaign004 finite freeze after elapsed-time handoff and archival

A fresh complete campaign is frozen at `research_runs/tokenizer-wide-hybrid-004`.
Plan SHA256:
`6e42949c51dcfab5766444a293763856d22e2913aa01ebbdc1eb8b7a1f6625c2`.
Source SHA256:
`0fc7f6b3440484700151a99f6a02f1b847914f7c8914612b4d684f54d058ccaa`.
The original nine-case numerical and train-only input reports, applicability
review and explicit archival resolver are carried losslessly, retaining original
executed-source identities. All three protected archive hashes were checked.
No old quality trajectories, process receipts or cleanup chains are reused.

The unchanged finite design has three resource pilots, nine LR cells, six rank
cells and nine fresh confirmation cells, followed by three inference workloads.
All 108 conditional quality request options are bound. Seed3501 is reused for
screening and is not independent replication; old incomplete screen results
remain excluded from selection. Fresh3511/3512/3513 have not been launched by
previous closed campaigns. No holdout use or quality/adoption claim follows
from this engineering change.

The full metadata envelope fits: 2,093,907 of 2,097,152 bytes for evidence/meta,
leaving 3,245 bytes, and 855 conservative files below the 2,048 cap. Other group
caps and producer guards remain unchanged. Full-panel storage requirement remains
30,631,412,484 bytes including reserve; freeze observed 33,491,800,064 bytes free.
The guarded preparation finished in 5.77 s. The actual plan binder accepted
numerical/input provenance and explicitly left dispatch unauthorized until
measured pilot gates pass. Originals and archival member identities are preserved.

`start_campaign004_v2.py` uses the existing production dispatchers to run the
three resource pilots serially, independently audit the completed pilot boundary,
perform only its policy-eligible six-weight cleanup after exact metadata archival,
and dispatch quality slot0 only if every prerequisite passes. It stops after
slot0 or any failure, with no retry or automatic later slots. Commit/push the
finite plan before launching this wrapper with local MPS access. One training
worker, all existing resource limits and the closed test remain enforced.

Campaign004 launch followed committed/pushed freeze `73f2507`: controller23108
started one MPS resource-pilot worker23109. Pilot0 was observed running under
its production process guard; no quality result exists yet. The bounded wrapper
may advance through the remaining serial pilots and qualityslot0 only after
all required reviews succeed. Launch/start receipts are recorded separately;
no other worker, later quality slot or retry was launched.


### Campaign004 resource pilots accepted; first quality cell running

All three 20-update width512 D/2 resource pilots completed under the frozen
elapsed-time-handoff source. Full/cache and raw/H200 audits, resource reviews
and parent acceptance passed for hybrid, gated MLA and KDA. Worker times were
99.94, 100.67 and 120.15 s. Largest sampled RSS was 3,228,827,648 bytes and MPS
driver allocation was 4,112,433,152 bytes, within the unchanged 4/6 GiB limits.
No resource violations were recorded; sampled maxima are not instantaneous bounds.

The complete pilot boundary passed its production audit. Exactly six successful
pilot weights (3,769,780,224 bytes) were removed under the frozen retention rule
after preserving exact non-weight bytes and hashes. The external archive retains
76 non-checkpoint files; a separate read-only check verified the immutable pilot
records, archive bytes, six-file manifest and saved execution-review reference.
Boundary reference SHA256:
`12a766f7c9a301d5176f4556d8512bebfe320af243925d0e9fdf08b1a9fa1f93`.

The existing serial wrapper then started quality slot0 (hybrid, LR0.000625,
seed3501, 1,200 updates) with one MPS worker23466 and controller23108. No quality
result exists yet. This is execution/correctness evidence, not a quality gain or
proof that the full training campaign cannot fail. The wrapper stops after
slot0 acceptance or failure; later slots require independently reviewed progress.
Evidence: `campaign004-pilots-reviewed-v2.json`, pilot archive/boundary receipts
and `campaign004-quality0-start-v2.json` under the preparation evidence directory.


### Campaign004 first quality cell accepted and independently reviewed

Slot0 completed all 1,200 updates: width512 hybrid, D/4, LR0.000625,
seed3501. Full-validation H200 CE is 4.963671529014013; raw endpoint CE is
4.991196894351007 over 34,278 targets / 111,536 bytes. Both checkpoint hashes,
request/result audits, parent acceptance and the completed pilot-cleanup chain
passed independent prefix1 review. The review guard finished in 4.52 s with
220,250,112-byte peak RSS. The controller and child exited; no research worker
remained before continuation.

The review measured 32,232,665,088 bytes free against a remaining-campaign
requirement of 29,384,508,436 bytes, including the unchanged 8 GiB reserve.
Both slot0 checkpoints remain protected. This is one of nine LR screening cells;
there is no family selection, fresh-seed claim or adoption, and the test holdout
remains closed. Closed-round results are not pooled with this panel.

The campaign004 prefix-review and one-slot continuation wrappers differ from the
reviewed campaign003 wrappers only in run-root/receipt identifiers; production
source and frozen plan are unchanged. Evidence: `campaign004-terminal-v2.json`,
`campaign004-lr-prefix-01-v2-review.json` and its guard in
`tokenizer-evidence/wide-hybrid-preparation-001/`. After committing/pushing this
review, the next authorized serial cell is slot1, gated MLA at LR0.000625,
D/4, seed3501, through all original production gates. No retry or automatic
later-slot dispatch is enabled.
