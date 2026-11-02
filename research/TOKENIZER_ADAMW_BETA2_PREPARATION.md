# AdamW second-moment beta2 preparation

Current status: trainer-worker001 and trained-inference001 COMPLETE both arms.
All760 runtime/acceptance bindings unchanged,753 predecessors preserved. No active
worker, scientific quality result, seed reservation or pilot/quality freeze.

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

Next: minimal additive finite20-update resource-pilot orchestration and complete
three-fresh-pair fixed-LR design outside760 frozen sources. Re-audit actual worker
payloads/cache, finite contracts and dependency closure; fresh census, full retained-
output high-water/8GiB reserve, exact recovery if needed, pilots and final freeze.
See latest worker completion and evidence below.

All previous campaigns/fixtures remain closed. Preserve all selected/fresh bytes,
153 lossless RAW archives and H200 originals. Readout closure retains all six fresh
raw/H200 pairs3,355,571,184B; closure free9,388,134,400B is not new-panel readiness.
Storage014 preserves the accepted imported clipping slot00 in its original failed
publication root via exact raw-completed manifest/archive, NOT numerical retirement.

Evidence: tokenizer-evidence/adamw-beta2-preparation-001/direction-decision.json,
optimizer-component001-plan.json, optimizer-component001-result.json and guard/log.

## Full-model optimizer and resume integration complete

Model003 and model-resume001 passed both beta2 .999/.95 arms. Initial model state,
logits and construction RNG match exactly, with28,492,416parameters/88EMA tensors.
All Muon/AdamW group membership, order and width-scaled rates match; only declared
AdamW beta2 differs. CPU/MPS forward/gradients and full/cache/reset/causality passed.
Foreign configuration and saved group betas are rejected before state loading.

Original model001 launch failed on an absent plan before numerical work; its config
lookup used a nonexistent portable path. Model002 then failed an absolute CPU/MPS
post-update tolerance in the incumbent beta2 .999 arm. Both failures remain closed.
The separate model003 contract uses an exact full-parameter native-child update
oracle on each backend: error0. Absolute CPU/MPS post-update discrepancy is
0.00020608305931091309 for both arms; beta2 adds only2.9802322387695312e-8,
within the preregistered differential bound. Forward/gradient tolerances and resource
guards are unchanged. This does NOT establish absolute backend-update parity.

Each four-update MPS fresh repeat and serialized resume matched raw model,
optimizer, EMA, RNG and complete clipping histories exactly (maxerror0).
Step2 was serialized in capped memory; final-step4 raw/H200 exports have3snapshots.
CPU/MPS inference batches1/8 passed full/cache/reset with196608B cache per item.
All753 source/config bindings unchanged,738 predecessors preserved. This is generated
correctness evidence, not scientific quality, a pilot, or a seed reservation.

Storage015 recovered capacity without retiring any selected/fresh bytes: transparent
lossless compression of1643 historical text files reclaimed259,952,640B and verified
49,952 protected files; archival of16 closed GQA selected/fresh RAWs reclaimed
916,750,821B and verified51,579 other files. Exact restoration passed. Original
copy-only probe stopped before compression on actual capacity; separate002 passed
after text recovery. All169 RAW archives throughstorage006–015 remain protected,
recoverable, NOT retired; all H200 originals and storage004 remain intact.

Next: minimal additive actual run_training/output/worker and independent parent
full-validation/inference adapters using these tested primitives, including beta2
identity and complete clipping histories. Re-audit actual retained resume payloads,
close their tensor dependencies, then take a NEW exact authorized retirement audit
if needed. Four successful generated resume weights total1,118,319,956B and stay
retained meanwhile. Capacity001 free9,222,590,464B versus complete worker-pair
9,814,671,360B conditionally yields10,340,910,420B after dependency closure/audit;
this is a recovery route, not dispatch readiness or a scientific-panel forecast.
Source work proceeds. Then complete scientific census/output high-water/finite
resource pilots before any three-fresh-pair quality freeze. Retain beta2 .999.

Evidence: tokenizer-evidence/adamw-beta2-preparation-001/model003.json,
model-resume001.json, capacity001.json; storage-recovery-015/raw-completed.json,
compression-completed.json and restoration-check.json.

## AdamW beta2 generated worker integration complete

Both fresh generated four-update beta999/beta95 arms passed independent parent
acceptance:4096 training targets,3 H200 snapshots and88 EMA tensors each. Actual
optimizer states contain44 AdamW groups at(.9,.999)/(.9,.95),epsilon1e-8 and zero
child decay, with44 unchanged Muon groups and outer independent decay0.00006.
Full validation recomputed exactly in the parent; CPU/MPS CE differs by less than
2.9e-7. Complete clipping histories agree across raw/summary/metrics. Independent
missing/sequence/threshold/coefficient/nonfinite history, foreign beta2 config/group
and wrong-target-coverage probes were rejected. Trained-inference001 full/cache/reset
passed batches1/8,196608B cache per item. All760 runtime/acceptance bindings unchanged,
753 predecessors preserved. No scientific quality result or seed reservation.

Worker-integration001 first re-audited both real retained resume raw/H200/optimizer,
EMA/RNG/clipping histories/full-validation/cache and closed tensor dependencies.
A separate exact resume-retirement001 audit then reclaimed1,118,319,956B from four
obsolete successful generated weights;42,544 survivors verified unchanged. Original
receipts remain immutable; absent tensors resolve through the retirement manifest,
never restoration or old-fixture reruns. Actual free10,517,405,696B exceeded complete
worker-pair9,814,671,360B including8GiB reserve, enabling both fresh worker arms.
New generated worker raw/H200 weights1,118,325,460B stay retained until pilot
integration re-audits and closes dependencies, followed by a NEW exact cleanup audit.
All169 protected historical RAW archives, H200 originals, selected/fresh bytes,
storage004 and original failures remain intact.

Next: minimal additive finite20-update resource-pilot orchestration and complete
THREE-fresh-pair fixed-LR design outside760 frozen sources. Verify actual worker
payload/cache and finite contracts, close dependencies, fresh scientific census,
measure complete retained-output high-water with serialization overlap/metadata/
8GiB reserve, take exact authorized recovery if needed, then finite pilots and final
quality freeze. CommonLR0.00125/equal zero LR-search/original strict gates/no rescue,
selection or partial panel; beta1 .9/epsilon1e-8/Muon/model/schedule/H200 fixed.
Current free space is not new-panel readiness; source work proceeds and additional
lossless recovery is inventoried if the complete forecast needs it.

Retain incumbent beta2 .999. Final LayerNorm learned bias versus fixed zero is
deferred. Prior model001 missing-plan and model002 absolute CPU/MPS post-update
failures remain closed; model003 exact native updates/differential check is no claim
of absolute backend-update parity. Generated observations do not select quality.

Evidence: tokenizer-evidence/adamw-beta2-preparation-001/trainer-worker001.json,
trained-inference001.json, worker-integration001.json, worker-capacity001.json,
and resume-retirement001/completed.json.
