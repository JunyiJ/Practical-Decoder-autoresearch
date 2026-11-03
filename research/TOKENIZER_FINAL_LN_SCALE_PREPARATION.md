# Final LayerNorm scale ablation preparation

Use the newly adopted AdamW epsilon1e-6 baseline.

## AdamW epsilon1e-6 adopted after original strict gates passed

All six1200-update fresh endpoints and all three paired inference reviews passed.
Independent eps6-minus-eps8 deltas:-0.022838512805131144,-0.034333479511555944,
-0.028255318629028782; mean-0.028475770315238624 nats/token. The original mean
improvement>0.02,every-seed and all conditional text-bound gates passed. Upper
bounds2048/4096/8192:-0.021995713638369736,-0.019390242512917873,
-0.020417804237483028. Separate adoption audit confirmed paired config equality
except epsilon/output paths,saved optimizer identities and all original gates.
Adopt epsilon1e-6 with betas(.9,.999),Muons,width rates,decay.00006,model/schedule
fixed. First frozen candidate is representative,not loss-selected. All six
fresh raw/H200 pairs3,355,576,176B remain protected. No test scoring; conditional
text intervals on reused validation are not seed-population/holdout confidence.

New current recipe: adamw-epsilon-preparation-001/adopted-width256-mla-epsilon-recipe.json.
Historical baseline files,frozen defaults and guards remain immutable; future
code must explicitly bind this adopted recipe in an additive namespace. Terminal
STOP closes all epsilon quality training,inference and final commands; no reruns.

Boundary review selects final LayerNorm learned scale versus fixed unit scale,
keeping learned bias,centering,epsilon1e-5 and eight pre-branch RMSNorms fixed.
AdamW beta1 .9-versus-.95 is deferred,with beta2 .999/adopted epsilon1e-6 fixed.
Final-scale component001 passed24 CPU FP64/FP32/MPS FP32 generated cases:explicit
forward/all-gradient equations,exact native parity,initial-function/shared-bias/
RNG equality,bias-gradient sum,finite differences and exact suffix/future-gradient
checks. Maxforward4.462275904337787e-7,maxgradient2.8988415756714403e-6;all863
bindings unchanged. No full-model candidate or quality evidence. Candidate removes
256 parameters,not parameter-matched. Next explicit full-model scale/group/
checkpoint/cache policy integration on adopted epsilon1e-6,without bypassing
existing learned-scale guards. Source/small numerical work proceeds.
Evidence:quality001-final-review.json,quality001-closure.json,adopted recipe,
final-ln-scale-preparation-001/direction-decision.json,component001-result.json.

The full-model next step must preserve shared initialization/logits/RNG and all
unrelated optimizer groups. Fixed-unit removes only final scale:28,492,160 versus
28,492,416 parameters,87/88EMA tensors,43/44AdamW groups,44Muon both. Keep learned
final bias. Require explicit foreign scale-policy checkpoint/cache rejection and
CPU/MPS raw optimizer/EMA/RNG/history repeat/resume/H200/full-cache-reset checks.
No new scientific seeds or pilot/quality freeze. One worker/four threads and all
original guards. Do not restore retired tensors or rerun closed fixtures.
