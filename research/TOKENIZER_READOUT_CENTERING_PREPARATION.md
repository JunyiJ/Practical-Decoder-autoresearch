# Tokenizer final-readout centering preparation

Current status: standalone component002 PASS24 CPU FP64/FP32 and MPS FP32 cases.
All695 source bindings unchanged, including691 preceding campaign sources and
original component001. No candidate full-model integration, active worker,
scientific seed reservation, pilot/quality freeze or quality result.

The accepted incumbent has affine LayerNorm at the final readout, epsilon1e-5,
256 learned scale and256 bias parameters. Read-only inspection of accepted
clip1 H200 and raw optimizer confirms28,492,416 total parameters and44 AdamW
groups with betas(.9,.999). Earlier deferred wording claiming final RMSNorm was
incorrect; the original decision remains immutable, and completed campaigns
always validated the actual LayerNorm. Eight pre-branch RMSNorms stay unchanged.

Select a centering-only final-readout comparison:
- Control: (x-mean(x))/sqrt(mean((x-mean(x))^2)+1e-5)*weight+bias.
- Candidate: x/sqrt(mean(x^2)+1e-5)*weight+bias.

Both retain512 learned affine parameters, scale initialized1 and bias0, no RNG
changes. This candidate deliberately retains bias, unlike a default bias-free
RMSNorm. No gain assumed. Earlier block LayerNorm failures are an adverse prior
at a different placement, not a reopened trial or pooled quality evidence.
Both architecture and bounded HP reviewed: defer AdamW beta2 .95-versus-.999,
keeping beta1 .9, Muon and independent decay unchanged.

Component001 passed CPU FP64 arithmetic/finite differences, then failed an
incumbent FP32 width7 batch3 gradient elementwise bound: absolute9.8711e-5 versus
5e-5+relative5e-5 at one derivative. Its source/plan/result/log/guard remain closed.
Separate component002 keeps the normalization source and every candidate tolerance
unchanged. It adds exact native nn.LayerNorm forward and all-gradient parity and
uses a documented tensorwise max-norm comparison for the FP32 incumbent analytic
oracle; CPU FP64 finite differences and forward bounds remain unchanged. This is
a harness acceptance change, not a claim that component001 passed. All resource
limits remain unchanged. See component002-correction.json for the precise rule.

Component002 passed all24 cases at widths7/256, batches1/3 on CPU FP64/FP32 and
MPS FP32. Exact native LayerNorm forward/all-gradient parity, identical affine
initialization/construction RNG, explicit independent CPU FP64 equations, finite
differences for input/scale/bias, suffix independence and exact future-gradient
zero passed. Maximum absolute forward/gradient errors3.5333644197166336e-6 /
9.871093956093091e-5; maximum gradient normwise error2.9709798822580305e-5.
Native control parity is exact; candidate retains original elementwise5e-5 checks.
These are standalone normalization and read-only incumbent observations, not
candidate MLA/full-model/optimizer/cache validation or scientific quality results.

Next executable action: narrow additive explicit full-model final-normalization
policy, unchanged optimizer groups/shared initialization, checkpoint/cache identity
and CPU/MPS raw-model/optimizer/EMA/RNG repeat/resume/final-H200/full-cache-reset
integration outside frozen namespaces. Do not bypass existing final-LayerNorm or
MHA-only readout guards. Then actual bounded worker/independent parent validation/
inference, fresh scientific census, complete measured retained-output high-water
including serialization overlap/metadata/8GiBreserve, exact authorized recovery
if needed, finite pilots and final quality freeze. Prospective exactly THREE
fresh pairs at common LR0.00125, equal zero LR-search, original strict gates;
no screens/rescue/selection/retry/partial panels. Standard-recipe scope only.

Retain clipping1.0/unit gains/D256 MLA score0.5/SwiGLU672/SiLU/blockRMS/serial/no
product norm/local mixing/warmup, decay0.00006/cooldown800/H200 and final LayerNorm.
Clipping quality002 closed with mean+0.004133149591824823, all seeds worse/all gates
failed. All earlier campaigns/fixtures stay closed. Preserve every selected/fresh
model, original failure evidence and114 protected lossless RAW archives. Archived
RAWs are recoverable, not retired; all H200 original paths and storage004 intact.
No permanently retired tensor restoration or closed-fixture reruns. One worker/
four threads/all resource guards, no paid compute/subagents, closed holdout.
Current free space is not panel readiness; small source/numerical work proceeds.
