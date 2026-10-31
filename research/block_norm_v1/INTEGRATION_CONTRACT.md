# Block-normalization runtime integration contract v1

October31,2026. This is a preparation contract, not a frozen training plan.
The explicit grouped-embedding v2 policy has passed model/cache/checkpoint,
MPS repeat/resume and generated-text batch8/H200 prerequisites. Pure finite rules
and statistics pass67 CPU checks; read-only configuration/source/environment and
train/validation bindings pass66 more. Synthetic append-only ledger mechanics pass42
checks. Real retained raw/H200 bundle acceptance and parent validation recomputation
pass52 checks. Real readonly serial worker/parent transport passes22; actual campaign
integration remains pending. Retained-pair inference passes21 and scientific seeds are reserved. Read-only retained-fixture validation passes34 checks;
authorized dataset-cache cleanup clears the current full-panel storage projection.
Earlier baseline/policy evidence keeps its original scope and source snapshots.

## Source applicability

The original `src/**/*.py` and requirements fingerprint stays unchanged. The
seven v1 and six v2 activation-recovery runtime files stay unchanged. Closed
reviews continue to resolve their original imports and hashes; new research
must not reinterpret those reviews as tests of changed code.

An AST dependency map identifies 43 local modules (213,313 bytes) reachable
conservatively from GPT, cache, optimizer, checkpoint, EMA and training modules.
The copy is under `research.block_norm_v1.runtime`. Relative imports resolve
within that namespace. The map found no absolute local or dynamic import hazards.
The baseline-copy manifest records origin paths/hashes, exact copied hashes and
two empty package initializers. Never rewrite that baseline manifest after edits:
record a new delta manifest against it, and keep the baseline snapshots.

Do not redirect `sys.modules` or `sys.path`, monkeypatch original validators,
remove signatures, temporarily substitute RMS modules to pass production checks,
or run old research controllers on the new policy. The unchanged baseline test
contains a disposable negative replacement test and restores the module afterward;
it does not provide a production replacement path.

All future producers, loaders, optimizers, cache builders and H200 exporters must
import the new namespace explicitly. The copied Hydra CLI has an unresolved
relative `../config` path and is not supported: do not invoke it. A new controller
must supply an explicit configuration and bind its tokenizer/data/training
contract. Importability does not demonstrate an executable copied training CLI.

For every new prerequisite, pin all relevant new source files, verify frozen
original/recovery hashes before and after, preserve exact tested sources and
record actual import locations. At campaign freeze, bind the complete new source
inventory and environment plus original immutable dependencies. A source edit
invalidates earlier applicability to that edited code; preserve the earlier
result and perform targeted new verification. An added file must be included in
the inventory, not omitted by a fixed historical list. Inference/final reviews
must verify this same binding; no implicit latest-code acceptance.

## Named model and configuration policy

Add a new explicit versioned policy for this D256, four-layer RoPE MLA/SiLU
backbone. Old configurations retain their exact behavior and RMS-only rejection.
The new policy supports exactly `rms_incumbent` and `layer_matched_epsilon` from
`spec.py`. It must validate the full field set, not silently project away unknown
or conflicting fields. Bind kind, FP32, width256, epsilon2^-23, affine/bias,
eight pre-attention/pre-FFN placements, final LayerNorm epsilon1e-5, serial unit
residuals, SiLU/hidden672, attention geometry/score0.5 and untied tables.

Construction validates the actual eight native modules and parameter identities,
not config alone. RMS uses native eps=None with an FP32 contract; LayerNorm uses
explicit2^-23 and bias. Constant initialization must preserve RNG and exact shared
weights. Model execution must reject non-FP32/autocast violations. Model metadata
must describe the actual constructed policy; same-shape opposite-arm signatures,
wrong epsilon, missing bias, mixed modules and mutated final norms must fail.

The implementation must adapt copied architecture field validation and construction
explicitly. Do not feed a new configuration to an RMS-only helper and bypass its
failure. Reuse the old geometry checks through a reviewed policy-aware branch
that preserves old rejections, with negative tests for each newly allowed field.

## Integration surfaces and required evidence

| Surface in the new runtime | Required change or invariant | Required validation |
| --- | --- | --- |
| `models/tokenizer_architecture.py`, new block-norm policy module | Explicit named-policy config and actual-module validation; retain old RMS contract | Unknown/missing/conflicting fields, wrong module types/epsilon/dtype/placement rejected; old control unchanged |
| `models/blocks.py`, `models/gpt.py` | Construct eight norms through the specification; record new signature; retain all other operations and initialization | Parameter counts28,492,416/28,494,464, shared state/RNG equality, formulas/gradients, causality and old-control parity |
| `train/optimizers.py` | Validate new policy before grouping; new bias vectors in AdamW at base LR; preserve Muon scaling and independent decay | Unique full name/identity coverage, group order, eight new vectors, gradient/update/decay and optimizer-state resume |
| `utils/checkpoint.py` | Require saved/current new-policy signatures before strict loading; retain raw/H200 purpose distinction | Actual serialized round trips; opposite arm and same-shape signature/epsilon mutations rejected; H200 optimizer-resume rejection |
| `train/train.py` | Propagate full signature/config through training contract and raw checkpoints; future explicit controller only | Interrupted/resumed CPU numerical/RNG/optimizer agreement, batch/data/tokenizer identity and raw payload completeness |
| `train/weight_averaging.py` | H200 reconstruction uses new GPT and loader; average all gains/biases without folding them | Exact controlled EMA oracle, candidate gains/biases included, export/load round trip, resume preserves accumulator state |
| `attention/cache.py` and attention modules | Preserve state layout/math; consistently validate the new model policy | Full vs cached tolerances, offsets/reset/interleaving/causality, sizes and CPU/MPS agreement |

Initial prospective edit allowlist is the seven named existing files above
(two model construction files, architecture, optimizer, checkpoint, training,
weight averaging), plus a new policy module and separate prerequisite/controller
files. Cache/attention code should remain byte-identical unless a demonstrated
policy-validation dependency requires a narrowly documented extension. Any such
extension gets an explicit delta review and targeted tests before acceptance.
The dependency closure includes older policies to preserve their import graph;
the new experiment does not authorize changing their semantics or testing new arms.

## Completed baseline and remaining gates

`runtime-baseline-prerequisite-002-review.json` passed five CPU checks on the
unmodified copy: all43 import locations, exact initial weights/RNG/logits/CE,
old-versus-new chunked cache outputs/state, optimizer group metadata/order/counts,
and old policy's rejection of LayerNorm configs and silent module replacement.
This compares old versus new cached execution; it does not establish full versus
cached numerical agreement. No gradients/optimizer steps were run.

Attempt001 stopped before tests on sandbox PermissionError during the safety
entry; no guard or test log was created. It remains preserved. Attempt002 used
the same sources with authorized process access:0.851s, peakRSS543,424,512 bytes,
four threads,120s cap,4GiB RSS cap and8GiB reserve. Original source, all13 frozen
recovery modules and three closed campaign inventories/nonweight hashes were
unchanged. No checkpoint or corpus loaded, no training, MPS, cleanup or holdout.

Next implement the explicit full-model policy, then validate construction,
optimizer, strict checkpoint/EMA integration and caches with new evidence. Full
serialized resume, numerical/MPS and controller/retention/statistical prerequisites
remain. Only after all prerequisites and sufficient full-panel capacity may the
finite seeds/order/plan be frozen. Recorded storage shortfall remains2,186,024,144
bytes; a baseline CPU guard observation is not a new training capacity approval.


## Construction implementation status

`model-policy-delta-001.json` records three changed model files plus the new
`models/block_norm.py`; all other copied files retain their baseline bytes.
The dedicated `tokenizer_mla_block_norm_v1` policy validates the full field set
directly using fixed backbone constants; it does not disguise LayerNorm as RMS
for validation. `block_normalization` carries the complete standalone signature.
Construction and execution validate actual modules; optimizer entry inherits
these checks from the architecture validator, with no grouping-code change.

Attempt002 passed41 CPU construction, parameter/group, rejection and old-control
regression tests, including direct original versus named-RMS equality. No update,
serialized checkpoint, EMA/resume or candidate cache-numerical claim follows.
Checkpoint save/load and H200 exporter still require dedicated signature
transport and actual serialized tests. Do not treat their config-only checks as
that completed prerequisite. Keep both policy attempts and original baseline
snapshots; subsequent code changes require a new reviewed delta and test scope.


## Serialized transport implementation status

The next reviewed delta (`checkpoint-transport-delta-001.json`) changes only
`models/block_norm.py`, `utils/checkpoint.py` and `train/weight_averaging.py`
relative to the construction implementation.26 targeted CPU tests pass. New
raw/H200 payloads carry `block_normalization_signature` from validated constructed
modules. Loads require exact typed signature/config agreement plus complete
finite tensor inventory/shapes/dtypes before mutation. Old policy payloads retain
their prior format. Both raw and derived real files round-trip successfully.

Two synthetic averaging snapshots confirm all normalizer gains and learned
biases are averaged; final accumulator restore, parent immutability and export
RNG preservation pass. This supersedes the pending signature-transport item,
but not full resume: optimizer state was empty, no optimizer update occurred,
and loader/RNG trajectory replay was not exercised. Review raw resume paths for
missing optimizer/training state acceptance and test populated state/interrupted
replay next. Retain the four fixture files/manifest; no trained artifact is used.


## Raw resume preflight and replay status

`resume-replay-delta-001.json` adds a new policy-specific helper at
`utils/block_norm_resume.py` and changes only checkpoint load and H200 export
call sites. This explicitly extends the new-module allowlist to isolate resume
validation. It checks complete typed/raw metadata, RNG, exact scheduled optimizer
groups, complete finite FP32 state and EMA before model mutation. H200 uses
validated inference weight loading after its explicit raw-parent checks, since
export does not resume optimizer/RNG state. Original policy behavior is preserved.

30 guarded synthetic CPU tests give exact uninterrupted/restored model, populated
optimizer, schedule, EMA, samples/losses and RNG equality. Three read-only export
regressions also pass. Actual `_training_contract`, capture/restore and production
model/optimizer/schedule/EMA primitives were exercised on generated IDs; the
`run_training` dispatcher and real tokenizer/data binding were not invoked. This
is batch1/context2 over three synthetic updates, not an actual research run or
proof of MPS/full-budget stability. Candidate cache/gradient numerical checks and
trainer/controller integration still precede any campaign freeze.


## CPU cache/gradient integration status

`cache-numerical-delta-001.json` records changes only in `models/block_norm.py`,
`models/gpt.py` and `attention/cache.py`. The cache-file allowlist is explicitly
extended to bind new-policy signatures, layer/shape/dtype/device, independent
storage and contiguous positions before writes; reset returns position to zero.
Old-policy behavior/math remains unchanged. No attention projection math changed.

33 synthetic CPU tests pass with predeclared tolerances: full/cached offsets,
reset/interleaving/causality, mutation-free invalid-cache rejection, explicit FP64
normalizer arithmetic through shared attention/FFN, all parameter/norm-input
gradients and exact original RMS gradient equivalence. This is a normalizer
integration oracle, not a second attention implementation. No binary fixture or
optimizer update was needed. Strict MPS numerical/resource and actual tokenizer
trainer/controller integration remain before freezing the quality campaign.


## Strict MPS numerical status

`mps-numerical-plan-001.json` binds the unchanged runtime inventory, prospective
tolerances/cases and strict backend environment.11 tests pass: CPU/reference
comparisons, cached offsets/reset/interleaving/causality, device-mismatch rejection
and original MPS RMS control. The guard records RSS4GiB/driver6GiB/reserve8GiB/
600s and samples driver memory every0.25s. No optimizer update or checkpoint was
used. Treat this as numerical evidence, not an optimizer/resume/resource pilot
or performance result. Preserve fallback=0 and fast_math=0 in future bindings.

Actual tokenizer-bound `_make_loader`/`run_training` entry remains untested for
the new policy. Use a separately identified generated-text fixture with its own
`tokenizer_no_bias_prior` binding and guarded storage projection, without touching
real holdout data or claiming quality. Strict MPS optimizer/resume and finite
controller/retention/selection/statistical integration still precede launch.


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


## October 31 versioned grouped embedding integration: 33 checks passed

The isolated runtime now has an explicit `tokenizer_mla_block_norm_grouped_embedding_v2`
architecture policy and exact `embedding_backward` specification. Its MPS backward
core is AST-identical to the tested standalone prototype: unique-token FP32
membership matmul and distinct-row copy, first order, vocabulary50,257/width256,
maximum1,024 gradient tokens. CPU gradients use the native reference path; no-grad
inference uses native embedding on either device. Padding, max-norm, sparse or
frequency-scaled gradients, autocast and altered shapes/dtypes remain unsupported.
This is a declared FP32 reduction-order change for both future normalization arms,
not a reinterpretation of the old native-embedding policy or prior checkpoints.
Original source/requirements and the13 frozen recovery modules are unchanged.

33 checks passed under the frozen integration plan. Both RMS and matched-epsilon
LayerNorm preserved old-policy initialization, RNG, parameter inventories, CPU
forward and CPU gradients exactly. Native MPS full/cached outputs were also
bit-exact between old/new policies. Full-versus-cached maximum error was8.940697e-7
under inherited atol3e-4/rtol3e-4. Four full-model gradient repeats per arm at shapes
1x128 and8x17 were bit-exact for every parameter (16 MPS backwards total). Maximum
error against native CPU embedding/shared MLA/FFN with independent FP64 block
normalization was5.564652e-8 under inherited gradient atol3e-5/rtol3e-3. This is
scoped repeatability evidence; full8x128 gradients and optimizer replay are pending.

Four untrained raw payloads serialized in RAM and reloaded exactly. Old/new policy
checkpoint and cache interchange was rejected in both directions before mutation;
missing/malformed specifications and actual module changes were rejected. These
raw payloads have no optimizer/training state, and strict-resume loading rejects
them. H200 export/reconstruction and populated optimizer resume are not established
by this transport check. No binary files or optimizer updates were produced.

Guard8.165s; peakRSS1,806,532,608bytes and sampled MPS driver1,244,495,872bytes.
Strict fallback0/fast_math0,4CPU and4GiB/6GiB/8GiB/600s guards held. Frozen sources,
closed campaigns and retained generated fixtures passed unchanged audits. No
quality training, deletion or holdout scoring occurred. SiLU remains retained;
no normalization quality conclusion is available.

Next freeze a new v2 both-arm MPS repeat/resume prerequisite with populated optimizer,
EMA and RNG state, using the original failed gate (atol2e-6/rtol2e-4), then finish
v2 H200/inference transport and full-workload/resource/controller/statistical gates.
Do not rerun historical failed fixtures, widen tolerances or change Muon precision.
Full-panel capacity remains unresolved; no partial panel, seed/control/width trimming
or protected-weight deletion. Freeze exact finite seeds/order before quality launch.


## October 31 v2 MPS repeat/resume: both arms exactly reproduced

The new, separately frozen `grouped-embedding-resume-plan-001.json` exercised the
explicit grouped-embedding v2 policy with new prerequisite seeds373101/373102/373103.
No historical failed v1 fixture or closed campaign was rerun. Runtime files,
original source/requirements and13 recovery modules were unchanged. Installed
Muon precision, including its BF16 orthogonalizer, was not altered.

Both RMS and matched-epsilon LayerNorm passed the original model/optimizer/EMA
acceptance tolerance (atol2e-6/rtol2e-4), with observed maximum errors **zero**.
Each arm ran three reference updates, three fresh-repeat updates and two updates
after restoring the step-one RAM checkpoint:16 synthetic updates total on real
GPT-2 loading of generated text, batch1/context128. Every intermediate gradient
and post-update model/optimizer/EMA tensor hash matched. Initialization, restored
state, schedules/groups/counters, batches, CPU/MPS random draws and all final RNG
states matched exactly. This short prerequisite supports the scoped grouped
embedding workaround; it does not establish global determinism or model quality.

Each arm also rejected five malformed MPS RNG states and three invalid device/
PyTorch-version bindings before any model/optimizer/EMA/RNG mutation. Both `mps`
and `mps:0` aliases passed preflight. Complete one-update raw states serialized to
RAM archives of445,082,791bytes (RMS) and445,128,471bytes (LayerNorm); their hashes
and restored state hashes are retained. Payload bytes were ephemeral, no binary
files were written, and no historical/selected quality checkpoint was loaded.
H200 serialization/export and production controller resume remain separate gates.

The2 parameterized tests passed in15.70s; guard elapsed16.741s, peakRSS2,469,740,544
bytes and sampled MPS driver1,242,398,720bytes. Four CPU threads, strict fallback0/
fast_math0 and4GiB/6GiB/8GiB/600s limits held. Source, closed-root and existing
fixture audits passed unchanged. No quality training, cleanup or holdout scoring.

Next prepare v2 H200 file export/reconstruction and full8x128 paired generated-loader
MPS readiness with a bounded storage projection before writing any new fixture.
Verify exact EMA replacement, parent hashes, tokenizer identity, inference/cache
parity and refusal to resume an online optimizer from H200. Keep raw/H200 fixtures
and failure evidence under the retention policy. Then finish controller, retention,
selection, statistics and inference/resource gates and freeze the finite quality
plan. Full-panel capacity remains unresolved; do not launch a partial panel,
trim controls/seeds/width, delete protected pairs or repeatedly poll unchanged space.
SiLU remains retained; no normalization quality result is available.


## October 31 v2 full-batch trainer and H200 transport passed

The separately frozen `grouped-embedding-h200-plan-001.json` passed9 checks.
Both normalization arms used the actual isolated `run_training` with GPT-2 on
generated Unicode/special-marker text, paired512 sampling, batch8/context128,
strict MPS, two updates and2,048 target tokens each. This closes the prior
full-batch generated-loader gap; it is not an actual-corpus quality/resource pilot.
The paired sampler was checked against independently drawn starts before training.

Both raw/H200 pairs were serialized to exclusive capped files. H200 matched all
saved EMA parameters exactly (88 RMS/96 LayerNorm), copied nonlearned buffers
exactly, and preserved parent SHA, tokenizer/config and versioned model signatures.
Export preserved CPU/MPS RNG. Raw complete resume preflight passed; H200 optimizer
resume was rejected before model mutation. Malformed tokenizer metadata and
re-export to an existing path were rejected. Reconstructed CPU/MPS and cached
H200 inference at1x17 and8x17 passed inherited atol3e-4/rtol3e-4, with maximum
CPU/MPS error7.748604e-7 and full/cached error1.072884e-6.

A prospective storage projection reserved1,216,348,160bytes above8GiB, including
two448MiB raw caps, two112MiB H200 caps, metadata and source/log allowances; no
temporary checkpoint copies. The capped writer rejects overflow before writing.
The completed fixture retains14 files totaling1,118,443,110bytes, including four
binary checkpoints under `research_state/block-norm-v2-h200-fixture-001`.
All files are SHA-bound in `grouped-embedding-h200-fixture-manifest.json`; retain
these pairs for dependent tests. No old checkpoint was deleted or modified.

Guard elapsed11.985s; peakRSS1,628,127,232bytes and sampled MPS driver2,501,722,112
bytes. Preflight free space10,347,212,800bytes; last sampled free space9,034,428,416
bytes, above the8GiB reserve. These are observed filesystem values, not inferred
space savings. Four CPU threads, strict fallback0/fast_math0 and existing time/
memory/disk guards held. Original/copied runtime sources,13 frozen recovery modules,
closed campaigns and prior fixtures passed unchanged audits. No holdout scoring.

Next implement the new isolated controller/config/source bindings and bounded
selection/statistical/retention/serial-dispatch tests, with full-validation and
inference evidence paths. Reuse retained v2 fixtures read-only where possible;
new binary production needs a new reserve projection. Test failure/stop handling,
finite stage/order and exact selection rules before freezing any quality launch.
Actual-corpus pilot cost, full-horizon behavior and full retained-panel capacity
remain unestablished. Do not launch a partial panel, trim controls/seeds/width,
remove protected pairs or poll unchanged capacity. SiLU remains retained; no
normalization quality conclusion follows from this generated-text prerequisite.


## October 31 finite-stage, selection and statistics rules passed

The new pure `campaign_rules.py` passed67 CPU tests under prospective
`campaign-rules-plan-001.json`. It suggests exactly two20-step pilots, six1200-step
screens and six fresh confirmations on the explicit v2 backbone. Both arms have
LRs0.000625/0.00125/0.0025; each independently selects minimum full-validation H200
CE, breaking exact ties toward lower LR. Screens alternate arm order by ascending
LR; confirmations alternate by declared fresh-seed order. All18 suggested actions,
including pilot/screen audit boundaries, selection publication and final review,
were checked. Premature, incomplete, duplicate, reordered, failed or malformed
records are rejected. Persisted selection cannot change at retention/confirmation.

The seed contract requires distinct typed pilot/screen/three-fresh/evaluation/
bootstrap values. Test values373301/373302/373311–313/373303/373304 are synthetic
inputs only, not scientific seed reservations. Final seed freshness/census and
exact launch configuration/source/order remain to be frozen after prerequisites.
Pilot cooldown/EMA and the complete worker configuration still need their explicit
controller binding; these pure row specifications are not a complete training plan.

Statistics use aligned2048-target base sums, grouping to2048/4096/8192 and2000
PCG64 common paired resamples per size. RNG resets to the declared bootstrap seed
for each size. Sampled target counts weight short tails; the same text indices
apply to all three trained pairs. Linear quantile1-0.05/3 supplies each gate upper
bound. An independent Python scalar/per-seed calculation with separate NumPy
linear quantiles agrees to3.469447e-18 under predeclared absolute5e-14. Repeated
arithmetic is exact. Strict gates reject equality at mean-0.02, any seed0 or upper0.
Malformed inputs, nonfinite/overflow arithmetic and inconsistent panels are rejected.

Numeric eligibility always returns `adopted=False`. These functions do not prove
checkpoint/source/config/tokenizer/target identity or full validation coverage.
Boundary flags must later resolve immutable audited receipts; they do not authorize
cleanup. Retention only identifies four losing-screen indices while protecting both
selected screens and all fresh pairs; no paths are removed. The executable ledger
must handle pending requests, locks, STOP/failure, source drift and serial acceptance.

Guard1.793s, peakRSS236,732,416bytes,4CPU/120s/4GiB/8GiB limits. No model, corpus or
checkpoint payload was loaded, no MPS or training ran, and no binary file/cleanup/
holdout access occurred. Prior fixture bytes were hash-audited read-only, including
both retained v2 raw/H200 pairs. Original/runtime/recovery sources and closed-root
inventories remained unchanged. Full-panel capacity remains gated.

Next implement and test the new isolated manifest/config binding and append-only
session ledger before any worker launch: bind exact source inventories/environment,
corpus/tokenizer/split/validation coverage and v2 trial configurations; enforce one
pending immutable request, exact accepted prefixes, persistent terminal failure/STOP,
selection and audited retention receipts. Then connect full-validation/inference
and worker/parent acceptance with retained fixtures. Do not treat this pure rule
pass as controller, cleanup, quality adoption or launch readiness.

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

## Two actual-corpus resource pilots frozen, October31,2026

The finite pilot plan is now frozen for RMS then LayerNorm: seed374001,20updates,
LR0.0025, batch8/context128, cooldown/EMA start13 and H200 interval1. It binds
388 source files, exact configs/data/environment, completed prerequisites and
the15,170,221,264B complete-panel projection; observed free space19,149,811,712B.
Quality dispatch and checkpoint cleanup remain prohibited by this pilot freeze.

A concrete fixture-to-production audit gap was fixed before freeze: the trainer
logs at evaluation indices, not each update. The20step pilot must have rows0/19;
1200step quality runs must have0/300/600/900/1199.52 existing artifact checks and
12 sparse-metric checks pass. The original runtime and logging schedule are
unchanged, and the previous acceptance source remains preserved. The finite
trial acceptance scope additionally binds exact configuration/spec/seeds.

Run the two bounded children serially, independently audit raw/H200, resume state,
full validation and raw/H200 cache/reset at batches1/8, and anchor each accepted
receipt outside the campaign directory only after parent guards close. A partial
attempt or absent external anchor blocks progression; no retry or cleanup.
Keep both pilot pairs until complete audits and dependent prerequisites finish.
Evidence: pilot-acceptance-plan001/review/decision and pilot001-plan/freeze/build.
