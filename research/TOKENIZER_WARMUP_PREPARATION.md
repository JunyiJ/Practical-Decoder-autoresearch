# Tokenizer warmup and peak-LR preparation

Question: does100-update warmup improve the adopted D256 tokenizer MLA recipe
after equally bounded peak-LR tuning? This is prerequisite work only; no new
scientific seeds, resource pilot or quality panel are frozen.

Retain score0.5/SwiGLU672/SiLU/block RMSNorm/no FFN-product norm/final affine
LayerNorm/grouped-embeddingv2/H200. Proposed candidate starts at0.1 of each group's
base LR on zero-based update0 and reaches peak at99; control starts at peak.
Both keep1200updates, cooldown start800 and final ratio0.1, with H200 start800
and interval10. Proposed equal LR set is0.000625/0.00125/0.0025. All values remain
prospective until the real worker/pilot prerequisites and finite freezes.

Character warmup100 failed adoption: its width256 mean gain0.000805 was mixed
and used a common inherited peak LR. It is a weak prior, not pooled evidence.
The independently tuned tokenizer/MLA transfer question remains distinct. Recent
tokenizer screens demonstrate LR sensitivity, not an instability mechanism.
The proposed warmup lowers the multiplier integral from1020 to975 and changes
update/independent-decay balance; it is not equal integrated step-size training.

Architecture review considered causal local mixing, which needs explicit placement
and cache-history contracts before testing. Residual attenuation has unfavorable
historical all-seed evidence; MoE still needs a complete active/total-capacity
resource design. They are deferred, not rejected tokenizer results. Closed FFN,
score, block-norm, EMA, activation, wider-hybrid and character campaigns stay closed.

The separate schedule-component001 plan/source/result passed all1200 absolute
update ratios for both arms against a rational oracle. Width-scaled group rates
remained exact and repeated application did not compound. Actual float64 toy SGD
with momentum serialized/restored after updates0/98/99/799/800 reproduced final
parameters, group states and momentum exactly. This used generated tensors only,
no research corpus or scientific seed. Guard:0.733s, peak RSS297025536B, four CPU
threads and the unchanged8GiB reserve. This does not establish full-model optimizer,
EMA, raw-checkpoint or scientific quality behavior.

Next extend explicit full-model policy outside all frozen namespaces. Verify
shared initialization, exact Muon/AdamW group rate scaling through warmup/cooldown,
raw optimizer/RNG/EMA resume and H200/cache, then bounded actual worker and
independent parent full validation/inference. Follow with fresh seed census,
complete actual retained-output serialization/metadata/transient projection,
finite resource pilots and quality freeze. The11.2GB closure observation is not
a new-panel capacity guarantee. Small source/numerical work proceeds; audit exact
now-obsolete numerical archives under the renewed human permission when recovery
is needed. Never delete selected/fresh/failed weights, relax guards or run a
partial panel. Use current manifests; retired archives cannot be restored.

Evidence: tokenizer-evidence/warmup-preparation-001/direction-decision.json and
schedule-component001-plan.json, executable source, result and guard.


## Warmup full-model and real-worker prerequisites

The inherited runtime already binds the LR schedule into its training contract,
captures base group rates before optimizer restoration, and rejects a changed
schedule on resume. No architecture or frozen source change was necessary.
Both1,200-update schedules now pass exhaustive checks on the actual full-model
Muon/AdamW groups, including width scaling and noncompounding application. Both
arms have identical initialization,28,492,416 parameters and88 EMA tensors.

A separate four-update generated fixture (warmup2 versus none, H200 start2)
passed repeated and serialized-resumed model/optimizer/EMA/RNG checks with zero
difference, plus CPU/MPS full/cache/reset inference at batches1/8. This shortened
fixture is explicitly not a completed warmup100 training trajectory. Original
model-resume001's final receipt publication failed on optimizer tuple metadata
after all numerical checks had passed. Its source, plan, failure and every
checkpoint remain preserved and closed. A separately frozen read-only receipt
recovery verified the complete trace and every output hash without numerical
replay. See model-receipt-recovery001.json; do not rerun model-resume001.

Both actual four-update run_training children then passed independent parent
raw/H200/optimizer/RNG/full-validation acceptance:4,096 targets and3 H200
snapshots each. CPU/MPS full-validation CE differences were below3e-7. Paired
trained full/cache/reset inference also passed; descriptive timing does not
establish a quality or serving advantage. Runtime integration is frozen in
research/warmup_v1 with all477 predecessor sources unchanged. No scientific
seed, resource-pilot plan or quality plan is reserved/frozen.

Exact obsolete-numerical-cleanup-002 retired two closed FFN resume archives,
recovering1,051,131,606B, with all35,257 other files unchanged. The original
FFN resume archive manifests are retained; those two archives cannot be restored.
All selected/fresh/failed evidence, including the warmup publication-failure
outputs, remains protected.

Next complete minimal finite warmup-pilot orchestration and a full six-screen/
three-fresh-pair design with tested primitives and real payload integration.
Choose a bounded pilot that actually exercises the100-update warmup; the
four-update fixture alone does not demonstrate that execution. Close generated
worker tensor dependencies before considering exact successful-worker retirement.
capacity001 measures complete retained-output high-water13,555,990,528B versus
observed9,365,782,528B. It identifies3,572,856,138B of older numerical archives
for eligibility audit plus1,118,319,700B of successful generated-worker weights
after dependent source/payload checks close. Conditional recovery yields
14,056,958,366B (500,967,838B margin); this is not dispatch readiness. Execute
exact authorized audits/manifests, remeasure actual space, then fresh seed census,
finite pilot freeze, guarded pilots and final quality freeze. No new escalation
or wait is appropriate while this concrete source/recovery path remains.
