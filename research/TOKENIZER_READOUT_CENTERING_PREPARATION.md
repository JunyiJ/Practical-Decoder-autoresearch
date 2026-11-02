# Tokenizer final-readout centering preparation

Current status: full-model model001 and serialized model-resume001 PASS both
layer_affine/rms_affine_bias arms. All710 source/config bindings unchanged,
695 predecessors preserved. No scientific quality result, seed reservation,
pilot/quality freeze or active worker. Next actual bounded worker/independent
parent validation/inference integration; read capacity001 for the concrete recovery
route after successful resume tensor dependencies close.

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

Historical next action (completed below): narrow additive explicit full-model final-normalization
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


## Explicit full-model integration and serialized resume complete

A new policy in research/readout_centering_model_v1 binds final kind/epsilon/bias
to construction, architecture, optimizer, raw/H200 checkpoints and request caches.
Existing final-LayerNorm and MHA-only contracts still reject the new fields. No
frozen namespace or guard was bypassed. Both arms have28,492,416 parameters and
88 EMA tensors with exactly identical shared initialization/RNG and Muon/AdamW
groups; LayerNorm initial logits exactly match the incumbent. RMS-with-bias
intentionally changes the function. Explicit actual MLA readout equations, CPU/MPS
forward/all-gradient checks, full/cache/reset/causality and foreign readout
checkpoint/cache rejection passed. Cache196,608B per item remains unchanged.

Both generated four-update arms repeated and resumed from serialized step2 with
exact raw-model/optimizer/EMA/RNG/clipping-history state (maximum error0). Step2
was serialized only in capped memory; final-step4 raw/H200 exports have3 snapshots.
H200 CPU/MPS and full/cache/reset checks passed batches1/8; maximum CPU/MPS logits
error8.940696716308594e-7 and cached error9.5367431640625e-7. These are generated
prerequisites, not scientific quality or long-run stability results. Actual
run_training/output/worker and independent parent full validation remain untested.

Storage012 resolved the complete resume-pair shortfall by losslessly archiving
exactly six closed branch-scale fresh RAWs:654,730,043B saved,50,525 other files
verified unchanged, exact restoration PASS. Original paths resolve through
storage-recovery-012/raw-completed.json and protected-raw-storage-012 archives.
All RAW bytes remain protected/recoverable, NOT retired; H200 originals unchanged.
Together storage006–012 preserve120 RAWs. After recovery, actualfree10,287,386,624B
exceeded the complete9,781,116,928B resume requirement, including8GiBreserve.

Next executable action: minimal additive actual run_training/output/worker and
independent parent full-validation/inference adapters outside710 frozen sources,
with explicit readout-policy data routing and complete clipping-history acceptance.
Re-audit actual retained resume raw/H200/optimizer/EMA/RNG/history/cache and close
tensor dependencies. Then independent exact successful numerical retirement if
needed and actual full worker-pair capacity recheck. Four successful generated
weights total1,118,317,844B and remain retained.
Capacity001 observedfree8,974,565,376B versus complete9,814,671,360B;
conditional post-retirement10,092,883,220B is a concrete recovery route after
dependencies close and exact audit, not dispatch readiness or missing consent.
Source work proceeds. Scientific-panel projection is separate and requires fresh
census, measured complete retained-output high-water, finite pilots and quality freeze.
