# Final LayerNorm bias preparation

Current status: trainer-worker002 and trained-inference002 passed both generated
arms. All798 runtime/acceptance bindings unchanged,791 predecessors preserved.
No active worker,scientific quality result,new scientific seeds or pilot/quality
freeze. Next finite20-update resource-pilot orchestration and complete three-pair
fixed-LR design; see latest completion/recovery below. Earlier stage notes are
historical and original failed commands remain closed.

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

Historical post-component action (completed below): implement the additive full-model final-bias policy, optimizer-group,
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

## November2 — Final LayerNorm bias full-model and serialized resume passed

Both learned/fixed-zero policies passed model001 and model-resume001. Shared
initialization/logits/RNG and all unrelated groups match exactly; only final bias
is removed. Parameter counts28,492,416/28,492,160;EMA inventories88/87,Muon44
groups both andAdamW44/43. Explicit actual readout equations,CPU/MPS gradients,
full/cache/reset/causality and foreign-bias checkpoint/cache rejection passed.
Both four-update MPS fresh repeat and serialized resume raw-model/optimizer/EMA/
RNG/clipping histories match exactly(maxerror0). Step2 serialized in capped memory;
final-step4 RAW/H200 exports have3 snapshots,CPU/MPS batch1/8 inference passed.
All791 source/config bindings unchanged,777 predecessors preserved. This is
generated integration evidence;no scientific quality result,seeds or pilot/quality
reservation. Next actual trainer/output/worker and independent parent full-validation
acceptance/inference adapters outside frozen sources,with87/88 tensor acceptance.

Storage017 losslessly archived six closed beta2 fresh RAWs, reclaiming651,815,376B
and verifying52,616 survivors unchanged. Exact restoration passed. The original
APFS copy-only probe kept bytes unchanged but gave no useful compression; the
separate streaming path uses a stricter kernel-enforced512MiB output cap and
complete staging forecast,with unchanged8GiB reserve,time/RSS limits and one
compressor thread. Output-cap self-test passed. Original RAWs resolve through
`storage-recovery-017/raw-completed.json` and `protected-raw-storage-017/*.tar.zst`;
bytes recoverable,NOT retired. All188 historical RAWs and H200 originals/storage004
remain protected. No source,receipt or closed scientific experiment was rerun.

Capacity002 observed9,943,003,136B versus9,781,116,928B complete resume-pair
requirement, enabling both arms. Four successful generated resume weights now
total1,118,313,377B and remain retained until downstream tensor dependencies close
and a NEW exact audit. Capacity003 observed8,625,647,616B versus9,814,671,360B
complete worker-pair requirement;conditional retirement leaves70,710,367B deficit.
An additional118,112,256B of allocated historical text candidates was inventoried
for exact lossless original-path compression if needed;this is not an allowlist or
dispatch readiness. Source integration proceeds;remeasure actual complete capacity
after authorized recovery. No partial panels or relaxed guards.

Retain learned final bias and the adopted beta2 .999 recipe. AdamW epsilon1e-6
versus1e-8 deferred. Prospective scientific design remains exactly three fresh
pairs at commonLR .00125,equal zero LR search,original strict gates,no rescue.
Evidence: `final-ln-bias-preparation-001/model001-plan.json`, `model001.json`,
`model-resume001-plan.json`, `model-resume001.json`, `capacity001/002/003.json`;
`storage-recovery-017/raw-completed.json`, `restoration-check.json`.

## November 2 — Final LayerNorm bias trainer and independent acceptance passed

Both four-update generated arms passed actual bounded run_training, checkpoint
publication and independent parent full validation: 4,096 targets each, three
H200 snapshots, 88/87 EMA tensors and 44/43 AdamW groups (44 Muon groups both).
Complete clipping histories agree across raw/summary/metrics. Parent recomputation
is exact; CPU/MPS CE differences are below 2.9e-7. History, final-bias policy,
optimizer membership and coverage forgeries were rejected in integration001.
Trained-inference002 passes full/cache/reset at batches 1/8 with 196,608 cache
bytes per item. All 798 runtime/acceptance bindings and 791 predecessors remain
unchanged. This is generated integration evidence, not scientific quality.

Original trainer-worker001 accepted learned slot00, then its guard stopped before
slot01 subprocess creation: actual free 9,162,006,528B versus remaining complete
9,210,691,584B, a 48,685,056B deficit. Original STOP/source/config/log/process
receipts remain immutable; slot01 PID is null and no updates ran. A separately
frozen trainer-worker002 independently re-audited/imported accepted slot00 with
zero repeat updates and zero checkpoint copies, then ran only the unstarted
fixed-zero slot01. Original001 commands remain closed. Both accepted generated
RAW/H200 pairs total 1,118,318,945B, retained in their respective original roots;
root001 failure does not make its accepted learned weights obsolete failed data.

After real resume payload/cache audit and dependency closure, exact retirement
removed four obsolete successful generated resume weights, 1,118,313,377B, with
43,535 survivors verified unchanged. Original acceptance/failure evidence stays
immutable; absent tensors resolve via resume-retirement001, never restore/rerun.
Capacity004 measured 9,917,829,120B above complete pair requirement9,814,671,360B;
the later inter-slot decrease was caught by the unchanged guard.

Storage018 compression003 preserves original paths and readable hashes for501
historical text files, reclaiming86,827,008 allocated bytes with52,410 survivors
unchanged. Audit001 stopped before compression because a mutable results file
had grown since the inventory;002 stopped on an import-path error before audit.
Their original sources/logs remain closed. Separate003 excluded the mutable file
and used the correct frozen helper. Capacity005 then measured9,412,763,648B above
remaining9,210,691,584B. No reserve, output, time or memory limits were relaxed.
All188 historical RAW archives and all H200 originals/storage004 remain intact.

Next: minimal additive finite20-update resource pilots and complete three-fresh-pair
fixed-LR design outside frozen sources. Audit actual retained worker payloads/cache
and finite contracts, close dependencies before any NEW exact weight retirement,
fresh scientific census and measured complete retained-output high-water including
serialization/metadata/8GiB reserve. Recover capacity proactively if needed; current
free space is not scientific-panel readiness. No new scientific seeds or pilot/
quality freeze. Keep learned final bias; epsilon1e-6 versus1e-8 remains deferred.
CommonLR.00125, equal zero LR-search and original strict gates remain unchanged.

Evidence: `final-ln-bias-preparation-001/worker-integration001.json`,
`resume-dependencies-closed001.json`, `resume-retirement001/completed.json`,
`capacity004/005.json`, `trainer-worker001-closure.json`,
`trainer-worker002-source-plan.json`, `trainer-worker002.json`,
`trained-inference002.json`; `storage-recovery-018/compression003-completed.json`.
