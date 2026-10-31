# Tokenizer block-normalization preparation

Status: preparation, October31,2026. Versioned v2 model/cache/checkpoint,
MPS repeat/resume and generated-text full-batch/H200 prerequisites have passed.
Pure finite-stage/selection/statistical rules pass67 CPU checks; exact configuration,
source/environment and train/validation preparation bindings pass66 more. The executable
controller and scored artifact/validation provenance gates remain pending. Synthetic
append-only ledger mechanics pass42 checks; real retained-artifact acceptance now
passes52 checks and real readonly worker/parent transport passes22; actual campaign
integration remains pending; retained-pair inference passes21 and scientific seeds are reserved; no quality
campaign has launched and the finite launch plan is not frozen. Retained-fixture
validation passes34 checks after preserving/correcting its transfer failure. Cache
cleanup now provides sufficient capacity against the current full-panel projection. Historical sections below retain their dated scope.

## Selected question and alternatives

Select a two-arm **block RMSNorm versus affine LayerNorm** comparison on the
adopted width256, four-layer quarter-latent RoPE MLA backbone. Change only the
eight pre-attention/pre-FFN normalizers. Keep SiLU SwiGLU hidden672, serial unit
residual branches, final affine LayerNorm, untied GPT2 tables, attention geometry
and score0.5 fixed. Neither older character conclusions nor the recent gate
comparison predicts a positive gain from this change.

This tests a tokenizer normalization question directly, without adding another
attention or activation sweep. Historical FFN-product RMS/LayerNorm effects were
small and mixed across seeds; neither was adopted. QK normalization reversed
between character widths64/256. Residual0.1 attenuation harmed all fresh seeds
at both character widths. These results lower priority for simply repeating
those exact interventions; they do not establish results on this subword/MLA
backbone. Parallel residual topology remains a separate later hypothesis. MoE
remains deferred because its complete three-arm storage requirement does not fit.
See FFN_HIDDEN_NORM_RESULTS.md, QKNORM_QUALITY_RESULTS.md and RESIDUAL_GAIN_RESULTS.md.

## Exact proposed normalizer semantics

Let x be the per-token width256 input, mu=mean(x), and epsilon=2^-23
(1.1920928955078125e-7). Compute only in FP32 for this proposed contract:

- Control: gamma * x / sqrt(mean(x*x) + epsilon), with learned gamma initialized1,
  no bias. Preserve the actual incumbent `nn.RMSNorm(256, eps=None)` path; its
  FP32 effective epsilon is2^-23. Explicit signature and tests must bind dtype.
- Candidate: gamma * (x-mu) / sqrt(mean((x-mu)^2) + epsilon) + beta, with learned
  gamma initialized1 and beta0; `nn.LayerNorm(256, eps=2^-23, bias=True)`.

Default LayerNorm uses epsilon1e-5. A bare `norm_type` swap would therefore change
epsilon as well as centering and bias. Match FP32 epsilon prospectively to avoid
that extra change. The candidate still combines centering and a learned shift;
do not attribute any result solely to centering. This is not a claim that the
chosen epsilon is optimal. Final LayerNorm remains affine with epsilon1e-5.
No hidden-product/QK norm, branch gains, normalization placement or depth change.

Eight additional width256 bias vectors add2,048 parameters:28,494,464 versus
28,492,416. All block gain/bias vectors use the existing AdamW base-LR grouping
and independent decay0.00006; hidden matrices retain Muon width scaling. Constant
normalizer initialization must consume no RNG draws and preserve shared initial
weights exactly. No normalization parameters may be folded into H200 projections.

## Proposed finite comparison

Two20-update resource/cache pilots, then three equal LR options0.000625/0.00125/
0.0025 per arm on one common screening seed (six screens). Select each arm's
minimum full-validation H200 CE with lower-LR exact tie break. Confirm both fixed
recipes on three fresh paired seeds (six confirmations), alternating arm order.
No historical controls or already accepted seeds substitute for these new pairs.
Reserve exact pilot/screen/fresh seed identities and order before executable
freeze; do not extend the grid after observing its boundary.

All quality trajectories:1200 updates, batch8/context128, cooldown from800 to
0.1 multiplier, H200start800/interval10. GPT2-tokenized Tiny Shakespeare,
unchanged raw80/10/10 boundaries and validation coverage; test remains closed.
Main axis is equal updates/tokens; report measured time/memory separately.

LayerNorm is the only adoption candidate. Require mean fresh H200 delta<-0.02
nat/token versus its equally tuned RMS control, every seed strictly negative,
and simultaneous negative one-sided conditional-text upper bounds for block
sizes2048/4096/8192 using2000 common paired resamples, linear quantile1-0.05/3.
Freeze RNG seed and arithmetic before results. No cost-only fallback, posthoc
control promotion, family change or seed replacement. Reused validation and
conditional intervals cannot establish population-seed or holdout confidence.

## Storage finding and launch gate

The guarded meta-shape audit constructed/validated the original full model and
optimizer. It counted candidate LayerNorm components separately; it did not
bypass the existing architecture policy or claim a valid candidate full model.
Independent parameter-delta arithmetic agrees with optimizer-state accounting:
additional raw/H200 payload40,992bytes per pair (20*2048 plus eight step scalars).

Two selected screening pairs plus every fresh pair remain protected. Pilots and
losing screens receive cleanup credit only after complete applicable audits.
Using the existing24B/parameter plus buffer/2MiB allowance and15% headroom:

| Quantity | Bytes |
| --- | ---: |
| Two-arm complete high-water requirement, including8GiB reserve |15,170,221,264|
| Observed free space |12,984,197,120|
| Shortfall against conservative requirement |2,186,024,144|
| Final tensor payload plus reserve alone |13,061,854,720|
| Shortfall even before headers/RNG/logs/export overlap |77,657,600|

Thus merely tightening conservative headroom cannot make the retained payload fit
at this observation. No candidate pilot or partial panel may start. Do not reduce
width, controls or fresh seeds for storage convenience, delete protected pairs,
or repeatedly poll unchanged capacity. The panel needs sufficient safe local or
external capacity under a new audited plan; no paid compute is authorized.
CPU design and small bounded prerequisites can proceed while training is gated.

Evidence: [shape/storage audit](tokenizer-evidence/block-norm-preparation-001/shape-storage-review.json)
and [guard](tokenizer-evidence/block-norm-preparation-001/shape-storage-guard.json).
The audit took0.483s, peakRSS294,584,320bytes. No corpus split or checkpoint was
loaded, written, rescored, removed or compressed. No runtime source changed.

## Required preparation

1. Introduce a separate versioned normalizer specification/reference outside all
   frozen source sets. Test FP32 formula/input/parameter gradients, constant and
   shifted inputs, nearly zero variance, initialization/RNG and exact control
   equivalence. Reject non-FP32, wrong epsilon, affine/bias/shape or mixed norms.
2. Resolve a new full-model/optimizer/checkpoint contract and source-applicability
   strategy without editing closed runtime artifacts or bypassing validators.
   Existing tokenizer architecture fixes RMSNorm in config and module types;
   older hidden-norm/residual policies are MHA-only. A config toggle cannot launch.
3. Before pilots require constructed-module signature checks, same-shape and
   epsilon/checkpoint mismatch rejection, full/cached parity, causality/reset/
   interleaving, complete optimizer groups, exact resume/RNG/EMA export/load,
   tokenizer binding and real serialized artifact integration. Preserve prior
   failures; never rerun the closed failed numerical fixture.
4. Implement/test a new bounded retention-aware serial controller, resource and
   immutable-evidence checks, source binding, finite selection/statistics and
   final review. Reuse lessons from scoped completed-slot verification without
   modifying the closed activation recovery runtimes. Freeze/commit/push the
   entire plan only after capacity and all prerequisites pass.

Keep one worker, four CPU threads, strict MPS/no fallback for future training,
RSS4GiB/driver6GiB,8GiB reserve, pilots/inference600s and training7200s. No new
training, cleanup or holdout scoring is authorized by this design document alone.


## Standalone FP32 normalizer prerequisite completed October31

The separate [specification](block_norm_v1/README.md) passed66 bounded CPU tests:
independent FP64 formulas and analytic input/gain/bias derivatives; six input
regimes; bit-exact native RMS control forward/backward; eight independent
normalizers with RNG-preserving constant initialization; precise parameter counts;
signature round trips and rejection of wrong epsilon/dtype/shape/affine/bias,
autocast, mixed/shared/aliased modules and unsupported mutation.

Near-constant shifted LayerNorm input has greater rounding sensitivity: observed
maximum output/input-gradient/gain-gradient discrepancies versus FP64 were
0.0002103/0.01536/0.0009631. Ordinary random/shifted output discrepancies were
below6.26e-7. All passed tolerances declared before execution; no tolerance was
relaxed. This does not establish full-model stability, MPS agreement or quality.

Attempt001's66 tests passed, but its enclosing final metadata audit failed on a
string-versus-Path error. That failure, log and exact source snapshot are retained.
Attempt002 fixed only that wrapper conversion; its complete audit passed in0.388s
at285,900,800bytes peakRSS. Original source, all13 frozen recovery modules, three
closed-root inventories and noncheckpoint hashes were unchanged. No models,
corpus/checkpoint data, training, MPS, cleanup or holdout were accessed.

[Applicable review002](tokenizer-evidence/block-norm-preparation-001/standalone-spec-prerequisite-002-review.json),
[preserved failure001](tokenizer-evidence/block-norm-preparation-001/standalone-spec-prerequisite-001-review.json).
Next resolve full model/optimizer/checkpoint/cache and source applicability as a
separate versioned integration contract. Do not patch frozen old code or bypass
its RMS-only validators. Full-model initialization, cache, resume/EMA, serialization
and MPS remain prerequisites. The recorded training-capacity blocker remains.


## Separate runtime baseline completed October31

The [integration contract](block_norm_v1/INTEGRATION_CONTRACT.md) maps the model,
optimizer, checkpoint/H200 reconstruction and cache dependencies into a separate
`research.block_norm_v1.runtime` namespace. Its43-module unmodified copy passed
five guarded CPU tests: exact original-control initialization/RNG/state, logits
and loss, old-new cached outputs/state, optimizer group equivalence and preserved
RMS-only rejection. This is baseline transport evidence, not a candidate quality,
full-versus-cached numerical, gradient/update, serialization or MPS result.

Attempt001 stopped before tests on a sandbox safety-entry PermissionError. The
unchanged sources passed attempt002 with authorized process access in0.851s,
peakRSS543,424,512bytes. Original source,13 frozen recovery modules and three
closed-root inventories/nonweight hashes remained unchanged. No training or
checkpoint/corpus load, cleanup or holdout scoring occurred.

[Review002](tokenizer-evidence/block-norm-preparation-001/runtime-baseline-prerequisite-002-review.json)
and [copy manifest](tokenizer-evidence/block-norm-preparation-001/runtime-baseline-copy-manifest.json).
Next implement the explicit full-model normalization policy only in the new
namespace, preserving old-policy behavior and binding candidate configuration,
actual modules, optimizer, strict checkpoint/EMA and cache contracts. Frozen
source sets remain untouched. The finite plan and capacity gate remain pending.


## Named full-model policy construction completed October31

The isolated runtime now implements `tokenizer_mla_block_norm_v1` with explicit
RMS-incumbent and matched-epsilon LayerNorm signatures. It validates the exact
config field set and eight actual modules, placement, FP32 parameters, final
affine LayerNorm, untied tables, serial unit residuals and SiLU/MLA backbone.
Non-FP32 construction/execution and autocast are rejected. Only three copied
model files changed plus one new policy module; optimizer, checkpoint, EMA,
training and cache implementations remain unchanged. Frozen original source and
all13 recovery runtime files were preserved.

Forty checks passed in attempt001. Attempt002 added the direct new RMS-policy
versus original-source comparison, passing41 tests without changing runtime code.
The named RMS control is bit-exact for initial state/RNG, synthetic full logits/CE
and original-versus-new cached outputs/state. Candidate shared initial weights
and RNG match the control; only eight zero biases are added (2,048 parameters).
Optimizer checks confirm unique complete parameter coverage, unchanged shared
groups and all new bias vectors in AdamW at base LR with independent decay.
Invalid config/signatures, wrong epsilon/placement/dtype, mixed/shared normalizers,
changed residual/gate/final norm and mutable model signatures were rejected.

Guard002:1.280s, peakRSS658,538,496bytes; four CPU threads,120s cap,4GiB RSS cap,
8GiB reserve. Both attempts and exact source snapshots are retained. Three closed
activation-root inventories and nonweight hashes were unchanged. No model
checkpoint/corpus load, training, cleanup, MPS or holdout scoring occurred.

This is construction/contract evidence. No gradients, optimizer updates, candidate
full-versus-cached numerical equivalence, real checkpoint serialization, resume
or H200 transport has been demonstrated. Current checkpoint save/export code
does not yet persist a dedicated constructed normalization signature: implement
and test that transport before any candidate artifact is accepted. In-memory
architecture-config comparison alone is insufficient. Capacity and finite-plan
freeze remain pending.

[Review002](tokenizer-evidence/block-norm-preparation-001/model-policy-prerequisite-002-review.json),
[reviewed source delta](tokenizer-evidence/block-norm-preparation-001/model-policy-delta-001.json).


## Raw and H200 normalization-signature transport completed October31

The copied checkpoint writer/loader and H200 exporter now persist and validate a
dedicated constructed normalization signature, including exact typed version,
backbone/normalizer semantics, placements and parameter names. Saved config must
agree with that signature and the actual receiving model; all saved tensors must
have the complete expected inventory, shapes, dtypes and finite values. Old
policies omit the new signature and reject artifacts carrying it.

Twenty-six guarded CPU tests passed for both arms: actual raw/H200 serialization
and strict loads, H200 optimizer-resume rejection, malformed/missing/cross-arm/
epsilon/signature/dtype/nonfinite rejection before mutation, and final EMA-state
restoration. A synthetic two-snapshot H200 oracle explicitly changes all norm
gains/biases, confirming their inclusion without folding. Maximum gain error
versus independent FP64 arithmetic was4.8312e-8 and LayerNorm bias error8.223e-13.
All unchanged parameters stayed bit-exact; export preserved the raw parent and
CPU RNG. No actual optimizer update or training trajectory was run.

Guard:3.412s, peakRSS1,323,057,152bytes, four threads,120s cap,4GiB RSS cap and
8GiB reserve. Four fresh synthetic artifacts total684,354,160bytes and remain
retained with SHA256/size manifest; no binary checkpoints enter Git. No trained
checkpoint or corpus was accessed, no cleanup/MPS/holdout scoring occurred, and
frozen source plus closed campaign inventories/nonweight hashes were unchanged.

[Review](tokenizer-evidence/block-norm-preparation-001/checkpoint-transport-prerequisite-001-review.json),
[retained fixture manifest](tokenizer-evidence/block-norm-preparation-001/checkpoint-transport-fixture-manifest.json).
These tests restore an empty optimizer's groups and final EMA accumulator only.
They do not establish populated optimizer-state or interrupted trajectory/loader
RNG replay. That is next, alongside strict raw-resume state requirements; candidate
cache/gradient numerical and MPS checks remain. No training readiness or quality
gain follows. Guard-observed free space was12,084,506,624bytes after fixtures;
against the recorded15,170,221,264 full-panel requirement, the capacity gap is now
3,085,714,640bytes. This observation is not a new audited launch approval.


## Populated optimizer and interrupted CPU replay completed October31

The new policy now rejects incomplete raw-resume metadata before loading model
weights. Preflight checks model/device/PyTorch/progress/RNG contracts, exact
optimizer group order/names/rates and scheduled step, complete finite FP32
Muon/AdamW slots, Adam counters and EMA state. This policy-specific helper lives
in the copied runtime; old-policy behavior is unchanged. H200 export restores
raw weights/EMA without pretending to resume the optimizer; existing raw-parent
and layout checks remain mandatory.

Thirty CPU tests passed using fresh synthetic batch1/context2 data: three updates
per arm, saving after update1 and replaying updates2/3. Final model weights,
populated optimizer slots, EMA state, batches, losses, global RNG and loader RNG
were bit-exact between uninterrupted and restored execution. Tests also rejected
missing optimizer/training/RNG state, changed model contracts, group order/rates,
missing/wrong-dtype slots, nonfinite state, wrong Adam counts and invalid EMA.
No original corpus or trained research checkpoint was used. This is ten synthetic
optimizer updates in an implementation fixture, not a quality trajectory.

A separate three-test read-only regression rederived both retained synthetic H200
exports bit-exactly, rejected their deliberately incomplete full-resume state and
preserved legacy low-level resume semantics. Original/recovery source and three
closed campaign inventories/nonweight hashes stayed unchanged. All evidence and
synthetic artifacts remain retained; no cleanup, MPS or holdout scoring occurred.

Replay guard:12.266s, peakRSS2,186,706,944bytes. Export regression:1.900s,
peakRSS958,185,472bytes. Both used four threads,120s cap,4GiB RSS cap and8GiB
reserve. Two new raw files plus synthetic ID files total890,211,626bytes under
the declared1,200MiB quota; their hashes are recorded without pushing binaries.

[Replay review](tokenizer-evidence/block-norm-preparation-001/resume-replay-prerequisite-001-review.json),
[export regression](tokenizer-evidence/block-norm-preparation-001/resume-export-regression-001-review.json),
[fixture manifest](tokenizer-evidence/block-norm-preparation-001/resume-replay-fixture-manifest.json).
Next complete candidate full/cached causality/reset/interleaving and numerical
gradient checks, then strict MPS and tokenizer-bound trainer/controller integration.
This small CPU replay does not establish those or full-budget training stability.
Latest guard free space11,481,591,808bytes leaves a3,688,629,456byte gap against
the recorded15,170,221,264 full-panel requirement. Training remains gated; do not
poll unchanged capacity, narrow the panel or delete protected evidence to fit.


## CPU cache and normalization-gradient integration completed October31

33 guarded CPU tests passed for both block-normalization arms. Full versus cached
execution covers batch1/2/8 and context17, plus batch1/context128, mixed chunk
sizes and nonzero continuation offsets. Largest absolute logit discrepancy was
8.3447e-7. Two interleaved requests remained independent (maximum7.7487e-7 versus
full execution); resetting one cache left the other unchanged. Reset/fresh cached
outputs and same-length causality checks were bit-exact. Cache storage remains
196,608bytes per batch item (1,572,864 at batch8), identical for both arms.

The explicit reference independently computes RMS/Layer normalization in FP64
and casts at FP32 branch boundaries, sharing attention/FFN/final readout with
the tested model. Maximum logit discrepancy was1.0729e-6, CE discrepancy0, all
parameter-gradient discrepancy1.8627e-8 and norm-input-gradient discrepancy
1.8627e-9. This checks normalization math/placement through the model, not an
independent oracle for the shared attention/FFN implementation. The new RMS
control's complete parameter gradients were bit-exact to original source. All
tolerances were declared before attempt001 and stayed unchanged.

The inherited plain MLA cache lacked normalization-policy and position checks.
The isolated new policy now binds cache signature/layer/geometry/dtype/device,
requires distinct storage and contiguous positions, and resets its position.
Validation covers all four caches before any layer writes. Wrong arm, batch,
dtype, layer order, shared/aliased caches, stale/nonzero-empty positions and
overflow were rejected without partial cache mutation. Old-policy cache behavior
is preserved. This narrow cache-file extension is recorded in the source delta.

Guard:1.418s, peakRSS1,201,012,736bytes, four CPU threads,120s cap,4GiB RSS cap
and8GiB reserve. No checkpoints/corpus were loaded, no binary artifacts or
optimizer updates were produced, and no cleanup/MPS/holdout scoring occurred.
Original/recovery source and closed campaign inventories/nonweight hashes stayed
unchanged. [Review](tokenizer-evidence/block-norm-preparation-001/cache-numerical-prerequisite-001-review.json),
[predeclared source delta and tolerances](tokenizer-evidence/block-norm-preparation-001/cache-numerical-delta-001.json).

Next strict MPS numerical/resource validation with no fallback, then actual
tokenizer-bound trainer/controller integration and finite-plan freeze. These
CPU results do not establish MPS agreement, long-context behavior, training
stability or quality improvement. The full-panel capacity gate remains unmet.


## Strict MPS numerical prerequisite completed October31

11 tests passed for both block-normalization arms with MPS fallback and fast math
explicitly disabled. Model parameters and tested gradients were MPS FP32. Cases
cover batch1/8/context17 and batch1/context128 full/cached execution, nonzero
continuation offsets, batch2/context9 interleaving/causality/reset, CPU-cache
device-mismatch rejection and batch1/context5 gradient comparison to the CPU
normalization reference. Original MPS RMS control logits/CE were bit-exact; its
gradients passed the tighter predeclared control tolerance.

Maximum absolute errors: CPU–MPS logits8.9407e-7; MPS full/cached1.0729e-6;
interleaved requests9.5368e-7; reset0; CPU-reference logits9.5368e-7; CE0; all
parameter gradients1.9325e-7. Tolerances were fixed before attempt001 and not
relaxed. The CPU reference independently changes normalization arithmetic only,
sharing attention/FFN/readout; this remains a scoped numerical check.

Guard:3.340s, peakRSS1,154,957,312bytes, peak sampled MPS driver1,159,659,520bytes.
Limits remained RSS4GiB/driver6GiB, reserve8GiB,600s and four CPU threads. Driver
observations were sampled every0.25s, not an instantaneous peak guarantee. Runtime
code was unchanged from the CPU cache prerequisite. No checkpoints/corpus were
loaded, no binary artifacts, optimizer updates, cleanup or holdout scoring were
performed, and frozen source/closed campaign inventories remained unchanged.

[Review](tokenizer-evidence/block-norm-preparation-001/mps-numerical-prerequisite-001-review.json),
[prospective tolerances and resource plan](tokenizer-evidence/block-norm-preparation-001/mps-numerical-plan-001.json).
This is not a throughput benchmark, MPS optimizer/resume run, resource pilot or
quality result. Next validate actual tokenizer/data binding and trainer entry on
a bounded synthetic fixture with its own correctly derived binding, then strict
MPS optimizer/resume and controller/retention/statistical preparation. Do not use
these small numerical checks to bypass capacity or freeze an incomplete panel.


## Generated-text tokenizer and trainer integration completed October31

The actual copied `run_training` entry and GPT-2 raw-text loader passed12 CPU
checks on generated Unicode/special-marker text, with a correctly computed
`tokenizer_no_bias_prior`. Both normalization arms completed two updates at
batch1/context128, using Muon/AdamW, cooldown and H200. Each generated file has
6,408 train tokens and1,599 validation tokens under80/20 raw-text splits; no test
split exists in the fixture, and the real corpus/closed holdout were not read.

The raw training contract, complete resume preflight, tokenizer-bound raw/H200
loading, exact H200 equality to saved EMA tensors, metrics/summary and parent
hashes passed. Self-consistent but wrong corpus, train-ID and tokenizer priors,
saved tokenizer mismatches and changed resume evaluation seed were rejected.
The campaign's paired512 batch8/context128 sampler separately matched independently
sampled starts; the trainer itself used independent batch1. Runtime source was
unchanged. This is integration evidence, not model-quality or production resource
readiness, full-trainer interrupted replay, MPS optimizer/resume or a controller.

Attempt001 failed during setup before optimizer updates: its two-step fixture
set cooldown start1, violating start<total-1. Attempt002 corrected start to0
without changing runtime validators. Both plans, source snapshots, logs, guards
and all generated files are preserved. Attempt002 guard12.437s, peakRSS
1,419,116,544bytes, four threads,120s cap and8GiB reserve. Successful outputs total
1,118,435,545bytes (four checkpoints plus metadata/text); the failed attempt retains
40,902bytes of generated text. Both attempts left frozen source and closed roots
unchanged. No cleanup, historical checkpoint loading or quality training occurred.

[Passing review](tokenizer-evidence/block-norm-preparation-001/trainer-binding-prerequisite-002-review.json),
[corrected prospective plan](tokenizer-evidence/block-norm-preparation-001/trainer-binding-plan-002.json),
[retained fixture manifest](tokenizer-evidence/block-norm-preparation-001/trainer-binding-fixture-manifest.json).
Next predeclare strict MPS optimizer/resume checks with storage accounted before
writes; prefer bounded in-memory replay and existing evidence where applicable.
Controller/retention/selection/statistical readiness and exact finite seed/order
freeze remain pending. Full-panel capacity remains gated; no partial quality
pilot or thinning controls/seeds/protected evidence to fit.


## Strict MPS optimizer/resume prerequisite failed October31

Two immutable in-memory transport attempts used generated GPT-2 text, batch1/
context128, strict MPS FP32, fallback=0 and fast_math=0. No new binary files were
written. Attempt001 completed three uninterrupted RMS updates but rejected resume
because the saved contract says `mps` while actual parameters report `mps:0`.
The isolated new-policy preflight now accepts these device0 aliases, preserving
other-device and PyTorch-version rejection. It also checks MPS RNG vector length
against the current backend before loading any model weights. CPU/legacy branches
and all mathematical operations remain unchanged; explicit plan deltas bind this.

Attempt002 passed all five malformed-MPS-RNG and three device/version rejection
checks without changing model weights or RNG, accepted both device aliases, and
completed the two resumed updates. Its RMS model comparison FAILED the original
atol2e-6/rtol2e-4: one element out of the first failing172,032-element tensor had
absolute error3.050663508e-6 (relative0.002040846). This is a failed numerical gate,
not an accepted approximate replay. No tolerance was widened. Fail-fast stopped
before the LayerNorm arm. Full optimizer/EMA comparisons and the final exact RNG/
batch/draw assertions were not reached, so their agreement is not established.

Both guards completed with frozen original/recovery sources, closed roots and
retained generated fixtures unchanged. Attempt001:3.229s, RSS2,064,138,240bytes,
peak sampled driver1,190,920,192bytes. Attempt002:4.325s, RSS2,183,725,056bytes,
peak sampled driver1,242,398,720bytes. Limits stayed4GiB RSS/6GiB driver/8GiB
reserve/600s/4CPU. Across attempts eight synthetic RMS optimizer updates ran;
no candidate updates, real-corpus quality training, holdout scoring or cleanup.

[Failed numerical review](tokenizer-evidence/block-norm-preparation-001/mps-resume-prerequisite-002-review.json),
[device/RNG delta and unchanged tolerances](tokenizer-evidence/block-norm-preparation-001/mps-resume-plan-002.json).
The generic wrapper mentions archive hashes, but these failed tests stopped before
emitting their final observation properties: no in-memory archive digest or binary
payload was retained. Exact source, plans, failure logs and guard evidence remain.
Future diagnostics must emit intermediate observations before any assertion.

Close both attempts; do not overwrite or rerun them, loosen tolerance or infer a
root cause. Next prepare a separately bounded diagnostic that first verifies the
loaded model/optimizer/EMA and RNG states, then records exact batch/CPU/MPS draws
and per-step losses/gradients/updates with parameter names before final comparison.
Separate a transport/state defect from backend numerical variation using a
prospectively specified control, with no quality training. MPS resume and all
later controller/finite-plan gates remain incomplete; capacity remains gated.


## RMS MPS diagnostic: exact transport, variation without resume

The separately planned trace001 completed eight synthetic updates: a three-step
RMS reference, a separately initialized three-step uninterrupted repeat and a
resume for updates2/3. It did not invoke either closed failed prerequisite. All
original tolerances remain unchanged, and diagnostic completion is not acceptance.

The fresh repeat started with identical state. The checkpoint round trip restored
all recorded model, optimizer and EMA tensor values/layouts plus group, schedule,
EMA-counter and CPU/MPS/loader RNG metadata exactly. Serialization did not change
RNG. Every compared batch, CPU/MPS random draw, pre/post-draw RNG state and scalar
loss matched exactly. First observed numerical differences were confined to the
embedding gradient: fresh-repeat step1 max4.6566e-10; resumed step2 max2.3283e-10.

The fresh uninterrupted repeat subsequently diverged more than the resumed run.
After step3, fresh-repeat max parameter difference4.902482033e-5 with5,472elements
outside the unchanged2e-6+2e-4*abs(reference) threshold; resumed max3.956258297e-6
with3elements outside. Named intermediate gradients, model/optimizer/EMA hashes,
layouts and all comparisons are retained in the event trace. The observed pattern
supports backend numerical variation independent of checkpoint transport. It does
not prove the precise kernel cause, justify relaxed tolerance or establish MPS
resume readiness. Evidence is one RMS seed, one fresh-repeat control, batch1/
context128; no LayerNorm or full trainer interruption was tested.

Read-only inspection of installed torch.optim._muon confirms that its Newton-
Schulz orthogonalizer casts updates to bfloat16. FP32 parameters, gradients and
saved optimizer slots therefore do not imply FP32 optimizer intermediates. BF16
quantization/nonlinear orthogonalization is a plausible amplification mechanism,
not a causal result from this trace. Installed module/version/function hashes and
line locations are recorded; no optimizer math or backend configuration changed.

Guard7.896s, peakRSS2,297,741,312bytes, peak sampled MPS driver1,191,608,320bytes;
strict fallback0/fast_math0,4CPU,4GiB RSS/6GiB driver/8GiB reserve/600s. No new binary
files or cleanup. The445,082,087-byte in-memory archive's SHA was emitted before
comparisons and verified unchanged; payload bytes were not persisted. Original
source,13 recovery modules, closed roots and generated fixtures stayed unchanged.

[Trace analysis](tokenizer-evidence/block-norm-preparation-001/mps-resume-trace-001-analysis.json),
[full events](tokenizer-evidence/block-norm-preparation-001/mps-resume-trace-001-events.jsonl),
[installed Muon precision](tokenizer-evidence/block-norm-preparation-001/installed-muon-precision-review.json).
Next prospectively isolate repeated embedding backward under default and explicit
strict deterministic controls, then identical-gradient Muon repeat/perturbation
sensitivity as warranted. Use separate bounded diagnostics and disclose backend
support/errors; do not silently change precision, fallback or accept a new threshold.
MPS resume, controller and finite quality-plan readiness remain gated.


## Isolated MPS embedding and Muon diagnostics completed October31

A fresh embedding-only diagnostic used the real generated-text first batch,
FP32 upstream gradients, eight identical backward passes per mode and a unique-
ID control. Repeated token IDs produced8distinct gradient hashes in8passes both
with deterministic algorithms off and with strict deterministic mode on
(warn_only=False). No warning/error or fallback occurred. Maximum repeat delta
was1.862645149e-9; maximum error versus sequential FP64 analytic sums1.513399184e-9.
Unique-ID controls were bit-exact to the oracle and across passes in both modes.
This isolates observed variation to repeated-index embedding backward in this
local case; the strict flag alone does not solve it. It does not identify the
specific kernel implementation or establish universal backend behavior.

A separately frozen Muon sensitivity probe used fixed synthetic FP32 weights,
gradients and populated momentum at shapes256x64 and672x256, with four repeats
per input and deterministic mode off/on. Identical inputs gave bit-exact updated
parameters in every case. Seeded gradient perturbations near1.86e-9 produced
max parameter differences9.765848517e-6 and1.265760511e-5, respectively; results
were the same across deterministic modes. BF16 orthogonalizer inputs reconstructed
from the installed formula changed51/416elements. These reconstructed intermediates
were not instrumented inside the optimizer, and the in-place/out-of-place momentum
paths were not separately compared. Actual parameter outputs were measured.

Together with the earlier exact-restoration trace, this supports small embedding-
backward variation followed by optimizer sensitivity as a plausible explanation.
The synthetic perturbations are not the full-model trace's actual gradient pairs;
there is no complete causal reconstruction or accepted numerical resume result.
No tolerance changed, failed attempt reopened, runtime/math edited, precision
changed or persistent deterministic setting promoted. Global settings were restored.

Embedding guard2.776s/RSS599,015,424bytes/peak sampled MPS driver1,082,621,952bytes;
Muon guard1.972s/RSS454,410,240bytes/driver8,880,128bytes.32embedding backwards and
32synthetic single-matrix optimizer updates, no GPT/model training. Strict MPS
fallback0/fast_math0 and all4CPU/4GiB/6GiB/8GiB/600s guards remained. No new binary
files, cleanup, quality/holdout scoring or changes to frozen source/closed roots.

[Analysis](tokenizer-evidence/block-norm-preparation-001/mps-kernel-diagnostics-analysis.json),
[embedding events](tokenizer-evidence/block-norm-preparation-001/mps-embedding-diagnostic-001-events.jsonl),
[Muon events](tokenizer-evidence/block-norm-preparation-001/mps-muon-diagnostic-001-events.jsonl).
Next design and prospectively test a separate deterministic MPS embedding-gradient
prototype before any runtime adoption. One candidate is unique-token grouping plus
a FP32 membership-matrix product and unique-index output copy, entirely on MPS;
first verify backend support, analytic gradients, repeated-call determinism and
bounded cost/memory without changing production. Preserve native forward behavior,
no CPU fallback, no new quality claim and no tolerance widening. Any eventual
runtime change needs explicit source delta, both-arm equivalence/replay/resource
checks and a fresh finite-plan binding. Quality/controller readiness stays gated.


## Standalone grouped MPS embedding-gradient prototype passed

The prototype preserves native functional.embedding forward and computes backward
entirely on MPS: sorted unique IDs, an FP32 membership matrix times upstream rows,
and a copy to distinct rows of the dense gradient. It introduces no CPU fallback,
Muon precision change, padding, sparse/frequency-scaled semantics or runtime edits.
Its scoped contract is first-order FP32, vocabulary50,257/width256, rank2int64 IDs,
and1..1,024tokens. Higher derivatives and other embedding options are unsupported.

11 tests passed: one numerical/repeat test spanning11cases and ten unsupported-
contract rejections.88backwards covered generated batch1 and paired batch8/context128,
unique IDs, one repeated ID, alternating IDs, zero/constant upstream, noncontiguous
IDs and noncontiguous upstream. Native forward outputs were bit-exact; all gradient
hashes matched within and across deterministic-off/strict-on modes. Unused rows
were exact zero. Maximum absolute gradient error versus independent sequential
FP64 sums was1.002636907e-8, below prospective atol1e-7/rtol2e-5. Unique/zero/constant
cases matched the oracle exactly. These are standalone bounds, not a changed
full-model resume tolerance or a general determinism guarantee.

Guard5.613s, peakRSS624,197,632bytes and peak sampled MPS driver1,082,884,096bytes;
fallback0/fast_math0,4CPU and all4GiB/6GiB/8GiB/600s limits preserved. Native forward
and backward observations took roughly3–7ms after initial cases, but first-use
outliers reached121ms and timings include checks/synchronization; this is not a
native-versus-prototype benchmark or campaign resource pilot. No optimizer steps,
model/quality training, new binary artifacts, cleanup or holdout use occurred.
Global deterministic settings were restored; frozen sources/closed roots and
prior generated fixtures stayed unchanged. All source/evidence snapshots retained.

[Analysis](tokenizer-evidence/block-norm-preparation-001/grouped-embedding-prototype-analysis.json),
[prospective contract and tolerances](tokenizer-evidence/block-norm-preparation-001/grouped-embedding-prototype-plan-001.json),
[review](tokenizer-evidence/block-norm-preparation-001/grouped-embedding-prototype-001-review.json).
The prototype remains outside runtime imports. Next design explicit versioned
embedding-backward config/model/checkpoint/source bindings before integration;
never silently reinterpret prior checkpoints or old model-policy signatures.
Bind all-MPS reduction semantics, CPU inference/reference behavior, token limits
and first-order scope. Both prospective normalization arms must share that policy,
with fresh controls and declared FP32 reduction-order change. Verify initialization/
forward/cache equivalence, both-arm gradients, serialized transport and full-model
MPS repeat/resume under the unchanged original gate before resource/controller
readiness or finite quality-plan freeze. Earlier failed attempts remain closed.


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

## Pilot001 closed after validation-adapter failure, October31,2026

The actual-corpus RMS pilot completed20updates/20,480targets in4.073884 measured
training seconds and wrote its raw and H200 checkpoints (eight EMA snapshots).
The child then failed during post-training validation: `validation.load` required
exactly train/val split keys, while the inherited80/10/10 checkpoint loader
exposes train/val/test. The earlier generated80/20 fixtures did not exercise
this boundary. The failure occurred before full-validation scoring or cache
audits. No parent acceptance exists, and the LayerNorm pilot did not start.

The child exited and was reaped; the root is now closed with STOP, never retried.
Both checkpoint files, frozen source, configuration, logs and failure evidence
are protected. The failed32.737s child guard observed peak RSS1,480,212,480B
and MPS driver2,510,094,336B, below limits; this is a postprocessing error rather
than a resource-limit breach. It is not a passed pilot or a quality result.

The inherited loader does encode/materialize the held-out split, but the frozen
trainer uses only training batches and sampled validation. No test loss, test
training batch or test-based decision was computed. Do not claim the holdout
was never tokenized. Next correct the read-only scorer/inference interface with
an explicit train/val-only view and actual80/10/10 boundary tests, keeping model
math and raw boundaries fixed. Audit the saved RMS outputs in separate recovery
evidence without retraining or relabeling the failed process. A separately frozen
continuation may carry verified RMS evidence and run the unstarted LayerNorm
pilot. Rebuild remaining capacity without cleanup credit for these failed weights.
Evidence: pilot001-failure.json, failure-evidence, close-failure.py and original
pilot001 freeze/source. Free space after closure:18,575,335,424B.

## Actual-corpus validation recovery passed, October 31, 2026

The concrete split-interface failure is fixed with an explicit read-only
train/validation view. It reuses the frozen raw split boundaries, encodes only
train and validation, and verifies tokenizer and training-prior identity.
The training runtime and model math are unchanged. The actual 80/10/10 boundary
test observes exactly two encoding calls and no exposed test array or sampler.

All 62 guarded checks passed: 34 validation checks, 21 inference checks and
seven actual-corpus recovery checks. The saved RMS pilot raw/H200 pair passed
independent parent artifact acceptance, optimizer/RNG/EMA preflight, full
validation and raw/H200 cached-versus-full/reset checks at batches 1 and 8.
All 34,278 validation targets are accounted for. H200 CE is 9.611318003763374;
the independent CPU FP64 oracle is 9.611318312664661, with maximum window-mean
error 4.108e-7. This 20-update pilot score is diagnostic, not a quality comparison.
The audit confirmed 20 updates, 20,480 tokens and eight EMA snapshots; it made
zero optimizer updates and preserved checkpoint bytes and RNG state.

The recovery guard finished in 46.615 seconds, peak RSS 1,380,466,688 bytes and
MPS driver 1,162,608,640 bytes. The failed pilot001 process remains failed and
closed, with its source, STOP and all artifacts unchanged. This separate
read-only acceptance does not relabel that process. Next freeze a new finite
continuation carrying these audited RMS components and run only the unstarted
LayerNorm resource pilot. Recheck remaining capacity with the failed RMS pair
protected. Quality training remains unfrozen; SiLU remains adopted.
Evidence: validation-recovery-plan-001.json, validation-recovery-001-review.json,
validation-recovery-001-rms-result.json and validation-recovery-decision.json.

## LayerNorm-only continuation frozen, October 31, 2026

Pilot002 permits only the unstarted LayerNorm slot1, paired seed374001, LR0.0025,
20 updates. It carries the separately audited RMS raw/H200 components while
keeping pilot001 failed and closed. All recovery sources remain unchanged; only
the new continuation dispatcher is added. Six entry/carry rejection checks and
actual frozen-plan binding passed. Disk free18,519,027,712 bytes exceeds the
unchanged conservative15,170,221,264-byte full projection; failed weights receive
no cleanup credit. One worker/four threads and all original guards apply.
No quality dispatch or checkpoint deletion is authorized by this pilot freeze.
Evidence: pilot002-plan.json, pilot002-freeze.json and pilot002-preflight.json.

## LayerNorm resource pilot accepted, October 31, 2026

Pilot002 slot1 completed20 updates/20,480 tokens with eight EMA snapshots, and
passed independent parent artifact acceptance, full-validation replay and all
four raw/H200 batch1/8 cache/reset checks. The child exited cleanly and was reaped;
its46.375-second guard observed peak RSS1,544,667,136B and MPS driver2,501,705,728B.
The parent audit completed in10.132 seconds. Measured training time was3.509446s.
Full-validation H200 CE9.567695740230638 is a short-pilot diagnostic only, not
evidence for adoption; no quality ranking or test loss was computed.

RMS resource observations and separately audited checkpoint components are carried
explicitly; the original RMS process remains failed and closed. Both pairs are
retained, and the failed RMS pair is protected from cleanup. About17.95GB remains
free. Next measure the actual trained pair's descriptive inference, then connect
the finite quality sequence to real parent acceptance and audited retention
boundaries before final quality freeze. The original15.17GB projection assumes
stage-boundary retention; do not reinterpret it as a retain-all projection.
No additional ledger-only scaffolding is needed. Evidence: pilot002-result.json,
pilot002-accepted-01.json and pilot002-completed-evidence/.

## Trained pilot-pair inference completed, October 31, 2026

The actual20-update RMS and LayerNorm H200 checkpoints passed all four full/cache/
reset comparisons before timing. With two warmups and eight alternating repeats,
median32-token prefill/64-token decode times were8.477/291.908ms for RMS and
8.952/278.811ms for LayerNorm at batch1; at batch8 they were13.713/297.073ms
and14.222/305.529ms respectively. LayerNorm decode time was4.49% lower at batch1
and2.85% higher at batch8, with identical cache allocations196,608/1,572,864B.
This mixed single-pilot result supports no consistent speed advantage. It is
descriptive only; two models were resident, cache allocation/loading excluded,
and timing values are not independently reproducible hashes or adoption gates.

The24.224-second guarded run peaked at RSS881,606,656B and driver1,155,842,048B.
Both original pilot directories, weights and failed-process status remained
unchanged. The new RMS measurement request explicitly rebinds only the additive
continuation source; original requests and recovery evidence remain immutable.
No training, test scoring or deletion occurred.

The remaining quality projection keeps both existing pilot pairs without cleanup
credit, retains all six screens through saved LR selection, then permits only
exactly audited losing-screen weight removal. Two selected screen pairs plus six
fresh pairs define the later peak. Original per-trajectory allowances,256MiB
metadata and8GiB reserve require15,170,221,264 additional free bytes, versus
approximately17.95GB available. This projection fits, but is not launch authority:
real quality dispatch, external acceptance anchors and the exact retention
boundary must be connected and checked before final finite quality freeze.
Evidence: pilot-pair-inference-001-plan/result/review/guard.json and
quality-remaining-capacity-001.json. SiLU remains adopted; no norm quality result.

## Finite normalization quality plan frozen, October 31, 2026

The real quality dispatcher now connects the accepted pilot worker/parent path
to1,200-update trials, a7,200-second total worker/parent deadline, independent
full-validation/cache replay, and external acceptance anchors. Requests and
partial attempts cannot be retried. Six screens span both arms and three LRs;
per-arm minimum H200 CE (lower-LR tie-break) is saved before three fresh pairs.
The original quality thresholds and simultaneous conditional-text bootstrap
remain unchanged; adoption still needs a separate final review.

The exact retention boundary re-audits all six screens, persists selection, then
allows only four losing raw/H200 pairs with a size/SHA/path manifest, fsynced
intent/deletion journal and complete protected-evidence verification. Two chosen
screen pairs, all six fresh pairs, both pilots and all non-weight evidence remain.
The failed RMS pilot remains permanently protected; no cleanup credit is assumed.

Integration001 failed during collection due to a relative test import; no test,
training or deletion ran. Its source and log are preserved. Corrected002 passed
nine checks including disposable-file cleanup/corruption/re-entry handling and
actual parent acceptance of both trained pilot models. No scientific checkpoint
was deleted. The final frozen entry bound successfully with395 source inventory
entries and17,937,055,744B free versus the
15,170,221,264B initial requirement. All CPU/thread/memory/disk/holdout guards remain.
Evidence: quality-integration-001/002-plan/review/source, quality001-plan/freeze/
preflight and quality-retention-disposable-002. The first quality screen is next.

Quality001 slot00 accepted (2026-10-31T22:54:47.339243+00:00): rms_incumbent, seed374002, LR0.000625, 1,200 updates/1,228,800 targets, H200 full-validation CE4.883680153820. Independent parent tensor/optimizer/EMA/validation/cache audits passed and the child exited cleanly. This is one endpoint; no adoption or partial-screen selection. Evidence: quality001-slot00-result.json and accepted-00.json.

Quality001 slot01 accepted (2026-10-31T23:00:01.867456+00:00): layer_matched_epsilon, seed374002, LR0.000625, 1,200 updates/1,228,800 targets, H200 full-validation CE4.880529115287. Independent parent tensor/optimizer/EMA/validation/cache audits passed and the child exited cleanly. This is one endpoint; no adoption or partial-screen selection. Evidence: quality001-slot01-result.json and accepted-01.json.

Quality001 slot02 accepted (2026-10-31T23:05:48.222426+00:00): layer_matched_epsilon, seed374002, LR0.00125, 1,200 updates/1,228,800 targets, H200 full-validation CE4.690080241450. Independent parent tensor/optimizer/EMA/validation/cache audits passed and the child exited cleanly. This is one endpoint; no adoption or partial-screen selection. Evidence: quality001-slot02-result.json and accepted-02.json.

Quality001 slot03 accepted (2026-10-31T23:11:15.403533+00:00): rms_incumbent, seed374002, LR0.00125, 1,200 updates/1,228,800 targets, H200 full-validation CE4.694150685727. Independent parent tensor/optimizer/EMA/validation/cache audits passed and the child exited cleanly. This is one endpoint; no adoption or partial-screen selection. Evidence: quality001-slot03-result.json and accepted-03.json.

Quality001 slot04 accepted (2026-10-31T23:16:46.542625+00:00): rms_incumbent, seed374002, LR0.0025, 1,200 updates/1,228,800 targets, H200 full-validation CE4.772951029132. Independent parent tensor/optimizer/EMA/validation/cache audits passed and the child exited cleanly. This is one endpoint; no adoption or partial-screen selection. Evidence: quality001-slot04-result.json and accepted-04.json.

Quality001 slot05 accepted (2026-10-31T23:22:21.601112+00:00): layer_matched_epsilon, seed374002, LR0.0025, 1,200 updates/1,228,800 targets, H200 full-validation CE4.784594486091. Independent parent tensor/optimizer/EMA/validation/cache audits passed and the child exited cleanly. This is one endpoint; no adoption or partial-screen selection. Evidence: quality001-slot05-result.json and accepted-05.json.

## Six-screen selection and audited retention complete

All six1,200-update endpoints passed independent parent acceptance. Full-validation
H200 CE for RMS/LayerNorm was4.883680/4.880529 at LR0.000625,
4.694151/4.690080 at0.00125, and4.772951/4.784594 at0.0025. The frozen per-arm
minimum rule selects0.00125 for both. LayerNorm's small selected-rate screening
improvement (-0.004070nat/token) is not a fresh confirmation or adoption result.

The boundary independently re-audited all six checkpoint pairs, saved selection,
then removed exactly eight manifested losing-screen files (slots0,1,4,5), totaling
2,236,771,928 logical bytes. The two selected pairs, all non-weight evidence and
both original pilot directories verified unchanged; the failed RMS pair remains
protected. Exact manifest, intent/deletion journal, guard and receipt are retained.
The deleted weights cannot be reloaded; all prior audit evidence remains.

Post-cleanup source/receipt/retained-file verification passed; approximately16.77GB
is free and the remaining requirement is14.05GB. Next run the six frozen fresh
endpoints at LR0.00125, seeds374011/374012/374013, with every raw/H200 pair protected.
All three descriptive inference pairs and final quality review remain required.
Evidence: quality001-selection.json, quality001-screen-retention/, and
quality001-boundary-result/next/guard.json. No test scoring or adoption occurred.

Quality001 slot06 accepted (2026-10-31T23:30:51.552785+00:00): rms_incumbent, seed374011, LR0.00125, 1,200 updates/1,228,800 targets, H200 full-validation CE4.709659289211. Independent parent tensor/optimizer/EMA/validation/cache audits passed and the child exited cleanly. This is one endpoint; no adoption or partial-screen selection. Evidence: quality001-slot06-result.json and accepted-06.json.

Quality001 slot07 accepted (2026-10-31T23:36:36.371349+00:00): layer_matched_epsilon, seed374011, LR0.00125, 1,200 updates/1,228,800 targets, H200 full-validation CE4.706465472198. Independent parent tensor/optimizer/EMA/validation/cache audits passed and the child exited cleanly. This is one endpoint; no adoption or partial-screen selection. Evidence: quality001-slot07-result.json and accepted-07.json.
