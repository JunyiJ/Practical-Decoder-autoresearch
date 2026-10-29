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
