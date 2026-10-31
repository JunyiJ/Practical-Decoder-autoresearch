# Tokenizer block-normalization preparation

Status: prospective design and read-only feasibility, October31,2026. No candidate
full model, numerical integration, pilot or training run exists. This is not a
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
