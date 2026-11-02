# AdamW second-moment beta2 preparation

Current status: quality001 CLOSED_NOT_ADOPTED. Six accepted endpoints and three
paired inference reviews passed; beta2 .95 worsened every seed, mean delta
+0.11006817407836185. All original strict quality gates failed. Terminal STOP
closes training, inference and final commands; retain beta2 .999. See
[TOKENIZER_FINAL_LN_BIAS_PREPARATION.md](TOKENIZER_FINAL_LN_BIAS_PREPARATION.md) for the next prerequisite.

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
tuned optima. Historical prospective design; the final quality panel is now complete and closed.

The frozen component exercises actual Muon/AdamW with prescribed variable gradients
on CPU FP64/FP32 at both beta2 values. Independent float64 first/second moments,
bias corrections and updates match within preregistered1e-12/1e-6 bounds;
maximum error4.4120593711838296e-7. Manual shrinkage-before-child ordering and
serialized parameter/optimizer-state resume are exact. Missing-gradient weights
stay unchanged, foreign beta2 policy/serialized groups are rejected, RNG is
unchanged, and prescribed-gradient Muon outputs match across arms. These are
small generated optimizer prerequisites, not actual full-model/MPS/EMA/H200/cache
or scientific quality evidence.

Historical post-worker next action (completed below): minimal additive finite20-update resource-pilot orchestration and complete
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

Historical pre-worker next action (completed below): minimal additive actual run_training/output/worker and independent parent
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
At that boundary, generated worker raw/H200 weights1,118,325,460B stayed retained until pilot
integration re-audits and closes dependencies, followed by a NEW exact cleanup audit.
All169 protected historical RAW archives, H200 originals, selected/fresh bytes,
storage004 and original failures remain intact.

Historical post-worker next action (completed below): minimal additive finite20-update resource-pilot orchestration and complete
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

## November2 — AdamW beta2 finite resource pilots complete

Both20-update arms independently passed actual raw/H200/optimizer/EMA/RNG and
complete clipping-history acceptance:20,480targets,8H200snapshots,88EMA tensors.
Actual44AdamW groups differ only in declared beta2(.999/.95);44Muon groups,
beta1 .9,epsilon1e-8 and outerdecay .00006 remain fixed. Paired full/cache/reset
inference batches1/8 passed at196608B/cache item. Independent final001 resource
review passed; all766 campaign sources unchanged,760 runtime predecessors preserved.
No scientific quality result. Pilot losses were not used for selection.

The prospective design fixes exactly three fresh pairs/six1200-update trajectories,
commonLR .00125,equal zero search and original strict gates. Panel199740 pilot/eval
used;fresh011/012/013/bootstrap004 remain unused. Next prepare additive
`research/adamw_beta2_quality_v1`, re-audit retained pilot payloads/cache and close
dependencies before a new exact pilot retirement audit and final quality freeze.
Retained pilot weights:1,118,337,488B. Observed free:11,507,736,576B;
conditional post-retirement:12,626,074,064B
versus complete12,381,585,408B including8GiB reserve. Remeasure before dispatch.

Worker retirement reclaimed1,118,325,460B and verified42,735
survivors. Storage016 losslessly archived13 closed accepted readout/wide RAWs,
reclaimed2,236,324,199B and verified52,040
other files unchanged. Exact restoration passed. All182 archived RAWs remain
protected/recoverable,NOT retired;all H200 originals and storage004 remain intact.
Original worker evidence stays immutable; absent generated tensors resolve through
`adamw-beta2-preparation-001/worker-retirement001/manifest.json`,never restore/rerun.
Archived RAWs resolve through `storage-recovery-016/raw-completed.json` and
`research_archives/protected-raw-storage-016/*.tar.zst`;never purge.

Evidence: `adamw-beta2-preparation-001/pilot-integration001.json`,
`pilot001-plan.json`, `pilot001-review.json`, `pilot001-final001-review.json`,
`campaign-capacity001.json`, `scientific-seed-census-001/result.json`;
`storage-recovery-016/raw-completed.json` and `restoration-check.json`.
Retain beta2 .999 and the adopted recipe; final LayerNorm bias ablation deferred.

## November2 — AdamW beta2 quality001 frozen and first fresh arm dispatched

The additive `research/adamw_beta2_quality_v1` binds774 active sources and preserves
all766 campaign predecessors.24 finite contract/publication tests passed, plus
actual retained pilot raw/H200/optimizer/EMA/RNG/clipping-history/full-validation/
cache re-audits. Pilot-to-quality forgery rejected. Declared44AdamW and44Muon
group identities are checked at payload acceptance and each accepted prefix.

Exact pilot retirement reclaimed1,118,337,488B with28,769
retained research files verified unchanged. Original receipts remain immutable;
absent pilot tensors resolve via `quality001-pilot-retirement/completed.json`.
No permanently retired tensors may be restored or closed prerequisites rerun.

Freeze SHA:`b71b2e692637f3599b8c860f63abb081fcfd2d1278ac0e1db2a5a01c63b78a49`.
Observed free12,619,821,056B exceeded complete12,381,585,408B including8GiB
reserve,serialization overlap and metadata. Every dispatch charges retained fresh
outputs and rechecks the remaining full-panel capacity. All182 historical RAW
archives remain lossless/protected,all H200 originals/storage004 remain intact.

Exactly six1200-update trajectories:beta999/beta95,beta95/beta999,beta999/beta95,
commonLR .00125,equal zero LR search,original strict gates,no rescue/selection/
retry/partial panel. Only AdamW beta2 changes;beta1 .9,epsilon1e-8,Muon,grouping,
width rates,outerdecay .00006,model,schedule and H200 remain fixed.
Slot00 dispatched;initial parent81253/worker81264,exec98811. Current live
progress/processes/anchors supersede these launch identifiers. Fresh199740011 is
in use;fresh012/013 andbootstrap004 unused at launch. No quality conclusion yet.
Next poll the actual worker,independently accept each endpoint,then dispatch only
the next frozen slot. After all six:three paired inference reviews and independent
statistics through the frozen `quality001-fresh-inference001.py` and
`quality001-final-review001.py`. Retain beta2 .999 pending all original gates.

Evidence:`adamw-beta2-preparation-001/quality-integration001-review.json`,
`quality001-plan.json`,`quality001-freeze.json`,`quality001-preflight.json`,
`quality001-capacity.json`,`quality001-seed-use.json`,
`quality001-pilot-retirement/completed.json`.

## November2 — AdamW beta2 quality closed without adoption; next final-bias prerequisite

All six1200-update endpoints independently accepted, with1,228,800 targets,
41H200 snapshots,88EMA tensors,44AdamW/44Muon groups and complete clipping
histories each. Three paired full/cache/reset inference reviews passed. Independent
mean beta2 .95-minus-.999 CE delta is +0.11006817407836185; seed deltas are
+0.11153406596318402,+0.10669633482953769,+0.11197412144236384. Every seed
worsens and all original strict gates fail. Retain beta2 .999; no adoption.
This is three paired seeds on reused validation text, fixed commonLR .00125 and
equal zero LR-search budgets; no holdout result or separately tuned optimum.

Terminal STOP closes every beta2 quality command. All774 bound sources unchanged.
Preserve all six fresh raw/H200 pairs3,355,577,712B,182 historical lossless RAW
archives,all H200 originals/storage004 and original failure receipts. Closure free
9,203,707,904B is an observation,not new-panel readiness. No deletion performed.

Both architecture and bounded HP reviewed: select final LayerNorm learned bias
versus fixed zero; defer AdamW epsilon1e-6 versus1e-8 at beta1 .9/beta2 .999.
Keep centering,learned scale,epsilon1e-5 and eight pre-branch RMSNorms unchanged.
The prospective candidate removes256 parameters; this is not parameter-matched.
Standalone component001 passed24 CPU FP64/FP32 and MPS FP32 cases at batches1/3,
widths7/256:explicit equations/gradients,exact native-control parity,initial function
and RNG equality,bias-gradient sum,finite differences and causal independence.
All777 source bindings unchanged,774 predecessors preserved. No candidate full-model,
optimizer,EMA/cache or scientific quality claim; no scientific seeds or panel reserved.
Next explicit full-model policy/optimizer/checkpoint/cache and repeat/resume integration
outside frozen namespaces,then bounded workers,pilots and complete capacity review.
Never bypass existing learned-bias/final-LayerNorm policies.

Evidence: `adamw-beta2-preparation-001/quality001-final-review.json`,
`quality001-closure.json`, `quality001-fresh-inference-pair{0,1,2}-review.json`;
`final-ln-bias-preparation-001/direction-decision.json`, `component001-plan.json`,
`component001-result.json`. See `TOKENIZER_FINAL_LN_BIAS_PREPARATION.md`.
