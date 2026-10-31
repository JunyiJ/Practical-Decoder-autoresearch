# Block-normalization runtime integration contract v1

October 31, 2026. Prospective implementation contract, not a frozen training
plan. The original baseline was an unmodified control copy. A named candidate
construction policy has now passed41 CPU tests; no training trajectory, serialized
resume test or MPS result exists. Historical baseline applicability is preserved
by snapshots and the original manifest.

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
