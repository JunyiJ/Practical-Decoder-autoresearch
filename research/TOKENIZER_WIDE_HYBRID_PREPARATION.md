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
