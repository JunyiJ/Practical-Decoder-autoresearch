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
