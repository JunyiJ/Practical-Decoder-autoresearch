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
