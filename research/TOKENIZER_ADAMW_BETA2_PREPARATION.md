# AdamW second-moment beta2 preparation

Current status: optimizer-component001 PASS four generated CPU FP64/FP32 cases.
All738 source bindings unchanged,737 predecessors preserved. No full-model
candidate, scientific seeds, pilot/quality plan or active worker.

Select AdamW beta2 .95 versus incumbent .999. Keep beta1 .9, epsilon1e-8,
Muon/group membership/width-scaled rates, independent per-update decay .00006,
global clipping1, common peakLR .00125, no warmup, cooldown800, H200start800,
unit serial branches and the adopted final LayerNorm model unchanged.
Read-only accepted incumbent RAW inspection confirms44 AdamW groups with
betas(.9,.999), epsilon1e-8, zero child weight decay and outer decay .00006.
The hypothesis changes second-moment adaptation; it may help or hurt.

Both architecture and bounded HP were reviewed after readout centering closed
without adoption. Final LayerNorm learned bias versus fixed zero bias is deferred.
No prior scores/seeds pooled. Prospective quality scope is exactly three fresh
pairs at commonLR .00125, equal zero LR-search budgets, original strict gates,
no screens/selection/rescue/retry/partial panel: standard recipes, not separately
tuned optima. No scientific seeds or pilot/quality plans have been reserved.

The frozen component exercises actual Muon/AdamW with prescribed variable gradients
on CPU FP64/FP32 at both beta2 values. Independent float64 first/second moments,
bias corrections and updates match within preregistered1e-12/1e-6 bounds;
maximum error4.4120593711838296e-7. Manual shrinkage-before-child ordering and
serialized parameter/optimizer-state resume are exact. Missing-gradient weights
stay unchanged, foreign beta2 policy/serialized groups are rejected, RNG is
unchanged, and prescribed-gradient Muon outputs match across arms. These are
small generated optimizer prerequisites, not actual full-model/MPS/EMA/H200/cache
or scientific quality evidence.

Next: narrow additive explicit full-model beta2 policy/group/checkpoint identity
and CPU/MPS raw-model/optimizer/EMA/RNG/clipping-history repeat/resume, final H200
and full/cache/reset inference outside frozen namespaces. Shared initialization,
model logits/RNG and parameter/group membership must match; only declared AdamW
beta2 differs. Validate foreign saved betas before child state loading; never
bypass existing fixed-default policies. Then actual bounded worker/independent
parent validation/inference, fresh census, complete measured retained-output
high-water including metadata/serialization overlap/8GiBreserve, finite pilots
and final quality freeze. Source/small numerical work proceeds; exact authorized
recovery if needed, no stale storage wait or partial panel.

All previous campaigns/fixtures remain closed. Preserve all selected/fresh bytes,
153 lossless RAW archives and H200 originals. Readout closure retains all six fresh
raw/H200 pairs3,355,571,184B; closure free9,388,134,400B is not new-panel readiness.
Storage014 preserves the accepted imported clipping slot00 in its original failed
publication root via exact raw-completed manifest/archive, NOT numerical retirement.

Evidence: tokenizer-evidence/adamw-beta2-preparation-001/direction-decision.json,
optimizer-component001-plan.json, optimizer-component001-result.json and guard/log.
