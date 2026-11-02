# Final LayerNorm bias preparation

Current status: component001 passed24 generated CPU FP64/FP32 and MPS FP32
cases. All777 source bindings unchanged,774 predecessors preserved. No active
worker, scientific seed reservation, pilot/quality freeze or quality result.

Select final LayerNorm learned bias versus fixed zero bias. Keep centering,
learned scale, epsilon1e-5 and all eight pre-branch RMSNorms unchanged. Both arms
start with zero bias and identical shared scale/function/RNG. The candidate
removes256 trainable parameters: prospective28,492,160 versus28,492,416. This is
an intentional bias ablation, not a parameter-matched comparison. No benefit is
assumed. Defer AdamW epsilon1e-6 versus1e-8 with beta1 .9/beta2 .999 fixed.

The accepted incumbent inspection confirms final affine LayerNorm,256 scale and
256 bias parameters, epsilon1e-5 and28,492,416 total parameters. The standalone
component has an explicit learned/fixed-zero policy; the latter has no bias
parameter. Checks cover batches1/3 and widths7/256, explicit CPU FP64
forward/all-gradient equations, exact same-backend native forward/all-gradient
parity, bias-gradient sum, CPU finite differences, exact suffix independence and
zero future gradient. Maximum absolute forward/gradient errors are
4.462275904337787e-7/2.8988415756714403e-6 under the frozen normwise oracle.
This does not establish candidate full-model, optimizer, EMA/H200 or cache behavior.

Next implement the minimal additive full-model final-bias policy, optimizer-group,
checkpoint and cache identity outside frozen namespaces. Preserve shared model
initialization/logits/RNG; remove only the candidate final bias from optimizer and
EMA inventories. Explicitly reject foreign-bias checkpoints/caches. Never bypass
existing learned-bias/final-LayerNorm guards. Then check CPU/MPS gradients and
causality, actual raw-model/optimizer/EMA/RNG/clipping-history repeat and serialized
resume, final H200 export and full/cache/reset inference. Follow with bounded
worker/independent parent validation, fresh census and complete measured retained
output high-water including serialization overlap, metadata and8GiB reserve;
exact authorized recovery if needed, then finite pilots and final quality freeze.

Prospective quality design: exactly three fresh pairs at commonLR .00125, equal
zero LR-search budgets, original strict gates, no screens/selection/rescue/retry/
partial panel. Standard-recipe scope; no old score or seed pooling. No new
scientific seeds or pilot/quality plans have been reserved.

AdamW beta2 quality001 is CLOSED_NOT_ADOPTED: mean .95-minus-.999 CE
+0.11006817407836185, all three seeds worse, all strict gates failed. Six accepted
endpoints and three paired inference reviews passed; terminal STOP closes all
quality commands. Retain beta2 .999/final affine LayerNorm/clip1/unit gains/D256
MLA score.5/SwiGLU672/SiLU/blockRMS/serial/no product norm/local mix/warmup/
independent decay.00006/cooldown800/H200. Preserve all six fresh RAW/H200 pairs
3,355,577,712B and182 historical lossless RAW archives, all H200 originals and
storage004. Original failures remain immutable; no closed reruns or permanently
retired tensor restoration. Closure free9,203,707,904B is not new-panel readiness.
Source/small numerical work proceeds without a stale capacity wait.

Evidence: `tokenizer-evidence/final-ln-bias-preparation-001/direction-decision.json`,
`component001-plan.json`, `component001-result.json`, `component001-guard.json`;
`tokenizer-evidence/adamw-beta2-preparation-001/quality001-final-review.json` and
`quality001-closure.json`.
