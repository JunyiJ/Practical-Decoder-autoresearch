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

## Full-model scale integration passed; serialized resume reserve failure closed

Model001 passed both learned/fixed-unit arms on adopted epsilon1e-6:identical
shared initialization/logits/RNG and unrelated groups,28,492,416/28,492,160
parameters,88/87parameter tensors,44/43AdamW and44Muon groups. Explicit readout
equations,CPU/MPS gradients/cache/reset/causality and foreign scale checkpoints/
caches passed. All878bindings unchanged. Only final scale removed; learned bias
retained. No serialized H200/EMA acceptance or scientific quality claim yet.

Original resume001 closed failed:disk_reserve guard exited90 at7.51s with
8,491,757,568B free below8GiB. RSS2,812,198,912B/MPSpeak2,508,668,928B stayed
within limits. Learned fresh repeat exact0 recorded; full resume/H200/both-arm
acceptance was NOT completed. Initial free10,204,749,824B minus445,087,744B
allocated payload implies1,267,904,512B extra filesystem consumption at guard.
1GiB OS swap observed afterward; cause not proved. Source/config/log/guard/trace
and failure receipt immutable; never retry closed001. New exact failed-retirement
removed only445,084,647B obsolete generated payload;54976survivors unchanged.

Separate resume002 source/plan is prepared,UNEXECUTED. It avoids retaining two
full reference tensor trees by comparing dtype/shape/SHA256 byte identities and
releases serialized bytes after decoding. No measured benefit claimed. Original
numeric/RSS/MPS/time/8GiB guards unchanged; complete capacity forecast strengthened
to11,049,021,440B with observed extra filesystem headroom. Measured
9207844864B leaves1841176576B deficit after recovery. Check free hourly,
quiet/no records when unchanged. Wake on actualfree>=11,049,021,440B or supplied
external volume for verified protected-archive relocation. Then execute separate
resume002,followed by actual worker/parent validation/inference; later full-panel
capacity/census/pilots and original three-pair gates. New independent human work
need not wait; no partial panel or relaxed guard. No worker active.

Storage028 losslessly archived all6 newly closed ADOPTED epsilon fresh RAWs,
reclaimed651099194B,54921survivors unchanged,exact restoration PASS.
Resolve originals via storage-recovery-028/raw-completed.json and protected-raw-
storage-028 archives. All203 archived RAWs are protected/recoverable,NOT retired;
all H100/H200 originals/storage004 intact. Original027 stopped before archive
creation/deletion when unrelated maintenance added a README; separate028 excludes
named unrelated maintenance and reuses unchanged-input copy proof. Do not repeat
closed recovery or numerical fixtures. Current adopted epsilon1e-6 remains.
