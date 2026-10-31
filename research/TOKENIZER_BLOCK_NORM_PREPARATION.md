# Tokenizer block-normalization preparation

Status: prospective design with candidate model construction tested, October31,2026.
No candidate gradient/cache numerical integration, pilot or training run exists. This is not a
frozen launch plan. The completed activation comparison retains SiLU; all closed
campaigns and protected selected/fresh/failure evidence remain unchanged.

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
