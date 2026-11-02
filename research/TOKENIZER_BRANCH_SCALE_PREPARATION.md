# Tokenizer fixed residual branch-scaling preparation

Current status: quality001 CLOSED_NOT_ADOPTED. All six endpoints and three paired
inference reviews passed; fixed-half minus unit mean CE delta+0.024273102061827024.
Every fresh seed worsened and all three original adoption gates failed. Terminal
STOP closes all branch-scale quality training/inference/final commands. Retain unit
gains and the incumbent recipe. All six fresh raw/H200 pairs remain protected.
Next direction: TOKENIZER_GRADIENT_CLIP_PREPARATION.md; no active worker.

Compare unit gain with fixed0.5 scaling of both complete post-projection residual
branches at unchanged depth: u=x+g*A(N1(x)); y=u+g*F(N2(u)), g=1 or0.5. Keep separate
RMSNorms and serial composition. Gains are constants, add no learned parameters
or RNG draws, and must remain explicit rather than folded into H200 projections.
The unit arm must exactly reproduce the incumbent. Retain D256 MLA score0.5,
SwiGLU672/SiLU/blockRMS/no product norm/no local mixing/no warmup/independent decay
0.00006/cooldown800/H200. Parameter/optimizer grouping/cache storage are expected
unchanged but require actual full-model verification.

At the boundary both architecture and bounded HP were reviewed. Fixed branch
scaling was the deferred distinct architecture question. Historical character
static0.1 and learned0.1 gains worsened every fresh seed; that adverse prior remains
explicit. This question uses a different gain magnitude, tokenizer/backbone and
horizon, without pooling old scores/seeds, reopening campaigns or assuming benefit.
Global gradient clipping0.5 versus incumbent1.0 is deferred. Its actual clipping
frequency and Muon/AdamW ordering would require measured instrumentation before a
future finite experiment. The closed cooldown grid is not extended or rescued.

Frozen component001 binds three new source files and623 unchanged predecessors.
All12 unit/fixed_half cases passed at batches1/3 on CPU FP64/FP32 and MPS FP32:
explicit independent causal equations, every parameter/input gradient, selected
CPU finite differences, unchanged prefix under suffix perturbation, exactly zero
future-input gradient, identical shared weights and construction RNG. Unit outputs
exactly match the direct serial reference; the candidate intentionally differs.
Maximum forward/gradient errors were5.662441253662109e-7/1.430511474609375e-6.
Distinct normalizers and supported gain kinds are enforced. This generated
one-head attention/SwiGLU composition is not an actual MLA/full-model/cache/
optimizer/checkpoint or scientific quality claim. No corpus or scientific seed
was used. Original sources/plan/log/result remain immutable; do not rerun.

Next: narrow additive explicit full-model policy/optimizer-group/checkpoint identity,
CPU/MPS raw-model/EMA/RNG resume, final-horizon H200 and full/cache/reset integration
outside frozen namespaces; do not bypass serial/unit-only guards. Then an actual
bounded worker and independent parent full validation/inference, fresh seed census,
complete measured retained-output high-water including serialization overlap,
metadata and8GiB reserve, finite resource pilots and final quality freeze.
Prospective comparison: exactly three fresh pairs at common peakLR0.00125, equal
zero LR-search budgets, original strict gates, no screens/rescue/partial panel;
standard-recipe claim only. No new scientific seed or training plan is reserved.

Current free space is an observation, not next-panel capacity. Small source and
numerical work proceeds; authorized exact retirement/lossless archival requires
closed dependencies, allowlisted paths/sizes/hashes, shared locks/process checks,
durable journals and survivor verification. Preserve every selected/fresh model,
original failure sources/logs/configs/receipts and protected storage004/006/007
archives. Sixty historical RAW files remain losslessly recoverable, NOT retired;
all H200 files stay at original paths. Permanently retired numerical payloads
cannot be restored and no closed fixture may rerun. One worker/fourthreads, all
resource guards, closed holdout, no paid compute or subagents.

## Full-model and serialized resume evidence

Full-model model001 and serialized model-resume001 PASS both unit/fixed_half
arms. Shared weights, construction RNG and Muon/AdamW groups are identical;
unit logits exactly reproduce the incumbent. Actual complete-branch equations,
CPU/MPS gradients, suffix causality, foreign-gain checkpoint/cache rejection and
mutated-block rejection passed. Both arms retain28,492,416 parameters,88 EMA tensors
and196,608B cache per item. Fixed gains add no tensor state or RNG draws.

Both generated four-update repeats and serialized step2 resumes matched raw model,
optimizer, EMA and RNG exactly(maxerror0). The intermediate checkpoint existed only
in capped memory. Final-step4 raw/H200 exports contain three snapshots; CPU and MPS
full/cache/reset checks passed at batches1/8. All639 source/config bindings remain
unchanged, including626 predecessors. No scientific seed, pilot/quality freeze or
quality result exists. No active worker. These generated checks do not replace
actual run_training/output and independent parent acceptance.

Storage-recovery-008 losslessly archived six accepted closed cooldown RAWs,
reclaiming653,238,031B with48,501 survivors verified unchanged. Exact restoration
passed. Resolve old paths through storage-recovery-008/raw-completed.json and
protected research_archives/protected-raw-storage-008/*.tar.zst; RAW bytes remain
recoverable, NOT retired, and all H200 files remain at original paths. Together
006/007/008 protect66 archived RAWs. The first copy-only probe was denied sandbox
process visibility before compression; original source/plan/log/guard remain
closed in archive-probe001-failure.json. Separate probe002 passed with process
visibility. No numerical fixture was rerun; only verified duplicate loose RAW copies were removed.

Next: minimal additive actual run_training/output/worker adapters and independent
parent full-validation/inference, including explicit branch-policy data routing.
Re-audit the actual retained successful resume payloads and close their tensor
dependencies. Only then audit exact successful numerical retirement if needed and
remeasure complete worker-pair capacity. The four retained resume weights total
1,118,314,644B; capacity002 records current free space and the conditional recovery
route. Source work proceeds. Afterwards: census, full measured retained-output
high-water plus serialization/metadata/8GiB reserve, finite resource pilots and a
three-fresh-pair fixed-LR quality freeze. No LR screens/rescue or partial panel.

## Actual worker and independent acceptance

Actual trainer-worker001 completed both fresh generated four-update arms under
strict bounded publication and process/resource guards. Each endpoint independently
accepted4updates/4,096training targets/3H200 snapshots/88EMA tensors. Parent full
validation exactly matched each child's report, and independent CPU/MPS CE differed
by less than3.1e-7. Trained-inference001 passed paired full/cache/reset checks at
batches1/8 with196,608B cache per item for both arms. All645 runtime/acceptance
bindings remain unchanged, including639 predecessors. No scientific quality result,
seed reservation, resource pilot/quality freeze or active worker.

Worker-integration001 first re-audited both actual retained resume raw/H200 payloads,
optimizer/EMA/RNG contracts, validation target coverage and foreign-gain rejection,
plus paired inference. That closed the successful resume tensors' dependencies.
Resume-retirement001 then retired only four obsolete generated payloads totaling
1,118,314,644B with39,528 survivors unchanged. Original source/log/config/receipt
bytes survive; absent tensors resolve through the completed manifest and cannot
be restored or rerun. Exact policy-at-audit snapshots match the retirement and
storage008 manifest hashes. Actual free10,996,101,120B exceeded the complete worker
pair requirement9,814,671,360B, including8GiB reserve and all metadata allowance.
Fresh worker raw/H200 payloads total1,118,319,636B and remain retained until tested
pilot integration closes their dependencies and a new exact audit allows retirement.

Next: minimal additive finite20-update resource-pilot orchestration and complete
THREE-fresh-pair fixed-LR design outside all645 tested sources. Reuse tested serial,
acceptance, inference and retention primitives; verify actual worker payloads/cache
and finite contracts. Complete fresh scientific census and full measured retained-
output high-water including serialization overlap/metadata/8GiB reserve. Execute
exact authorized recovery if needed, then finite resource pilots and final quality
freeze. Unit versus fixed0.5 complete serial branches at common LR0.00125, equal
zero LR-search budgets, original strict gates, no screens/selection/rescue/partial
panel. Source work proceeds; current free space is not scientific-panel readiness.


## Finite resource pilots and complete quality design

Pilot-integration001 PASS actual generated worker raw/H200, optimizer/EMA/RNG,
independent full-validation and cache audits against the additive finite campaign.
The exact pilot-only retention test passed, including rejection of fresh-confirmation
deletion, incomplete stages and selection. All651 campaign sources bind645 unchanged
runtime/acceptance predecessors. Worker tensor dependencies then closed; exact
worker-retirement001 reclaimed1,118,319,636B from four obsolete generated payloads
with39,723 survivors verified. Original receipts remain immutable; absent generated
paths resolve through worker-retirement001/completed.json. Never restore or rerun.

Lossless storage-recovery-009 archived nine accepted fresh RAW confirmations from
closed wide-hybrid005, whose hashes were bound by its independent final review.
It reclaimed2,033,806,544B and verified48,929 other files unchanged. Exact temporary
restoration passed. Resolve original RAW paths using storage-recovery-009/raw-completed.json
and protected research_archives/protected-raw-storage-009/*.tar.zst plus materialize.py.
These models are NOT retired: all RAW bytes remain recoverable, and H200 files stay
at their original paths. Together storage006/007/008/009 protect75 archived RAWs;
the storage004 fixture archive also remains protected. Never purge these archives.

Campaign-capacity001 measured12,897,472,512B free against the complete12,381,585,408B
requirement: six retained raw/H200 pairs,256MiB metadata and8GiB reserve, with bounded
streaming publication and same-inode serialization overlap accounted for. This was
checked again at freeze and each dispatch. Seed census001 passed48,515 recorded text
files; panel199410 reserves pilot001/eval003, fresh011/012/013 and bootstrap004.
Pilot/eval are now used; fresh/bootstrap remain unused. No screening seed exists.

Both20-update pilots independently accepted20,480 training targets,8H200 snapshots
and88EMA tensors. Paired full/cache/reset checks passed at batches1/8 with196,608B
cache per item for each arm. Independent final001 resource review PASS; source,
artifact, process and resource guards all checked. No pilot loss is used for
selection, adoption or a quality claim. The common LR is0.00125; scientific arms
retain cooldown800, floor0.1, no warmup, decay0.00006 and H200start800.

Next executable action: minimal additive finite1200-update quality orchestration in
NEW research/branch_scale_quality_v1 outside all651 frozen sources. Exactly THREE
fresh pairs, order unit/half,half/unit,unit/half at common LR0.00125, equal zero
LR-search budgets; no screens/selection/rescue/retry or partial panel. Reuse tested
acceptance/inference/retention primitives; verify actual retained pilots/cache and
all six-trajectory contracts. Close dependencies, independently audit exact successful
pilot retirement of1,118,329,296B, remeasure complete capacity and unused fresh/bootstrap
seeds, then final quality freeze. Preserve every fresh raw/H200 pair and apply all
original strict gates under standard-recipe scope. Final resource review recorded
11,745,796,096B free; adding retained pilots conditionally gives12,864,125,392B.
This conditional recovery is not deletion or quality-dispatch readiness. Source work
proceeds; no stale capacity wait. No active worker or scientific quality result.


## Frozen three-pair scientific quality campaign

Additive research/branch_scale_quality_v1 passed21 finite-contract tests and real
retained pilot raw/H200/optimizer/EMA/RNG/full-validation/cache audits. Pilot-to-quality
forgery and changed source identity were rejected. Paired configurations differ only
in fixed branch gain and output paths; both retain the identical cooldown800 schedule,
decay0.00006 and common LR0.00125. All659 sources are frozen,651 predecessors unchanged.
No closed fixture was rerun and no pilot loss selected a setting.

Independent quality001-pilot-retirement re-audited the actual pilots/cache and then
retired exactly four obsolete successful weights totaling1,118,329,296B, verifying
28,203 other retained files. Original source/config/log/acceptance/inference receipts
are immutable; absent pilot tensors resolve through quality001-pilot-retirement/
completed.json. Never restore these permanently retired generated payloads.

Final freeze measured12,776,476,672B free against complete12,381,585,408B including
8GiB reserve, all six retained raw/H200 pairs, serialization and metadata. Actual
remaining full-panel capacity is rechecked before every dispatch, charging existing
outputs. The original seed census plus current recorded training configs confirmed
fresh199410011/012/013 unused before launch; bootstrap199410004 remains unused.
Pilot199410001 and eval199410003 were already used. First fresh seed011 is now in use.

Quality root: research_runs/tokenizer-branch-scale-quality-v1-001. Freeze SHA:
8947d12c48e087d85d180ee7614900d120cf1f742849e278f6012808c61ddb45.
First slot00 launched via research.branch_scale_quality_v1.quality_execution with
that exact argument; parent38433/worker38444 and initial config/metric verified,
exec session17628. These are launch observations: actual current process/guard/log/
anchor state and progress record supersede them. Never redispatch an existing slot.

Continue exactly slots00–05 in frozen order unit/half,half/unit,unit/half, each1200
updates at LR0.00125, no screens/selection/rescue/retry/partial panel. Preserve every
fresh raw/H200 pair. After all six independent acceptances, execute all three pairs
through prepared quality001-fresh-inference001.py, then quality001-final-review001.py,
with the exact freeze argument. Both require all six endpoints. Independent statistics
retain the original mean<-0.02, every-seed improvement and all simultaneous conditional
text-bound gates. Standard-recipe scope, no old score/seed pooling or holdout scoring.
The adopted recipe remains unchanged until the separate final gate/adoption review.
All protected historical selected/fresh RAW archives and original-path H200s remain
preserved, including storage004 and75 RAWs in storage006/007/008/009.


## Final quality result and closure

All six fresh endpoints independently accepted1200updates/1,228,800targets/
41H200 snapshots/88EMA tensors. All three paired inference reviews passed CPU/MPS
full/cache/reset and retained196,608B cache/item. Independent final statistics and
provenance review passed. Fixed half did not satisfy any adoption gate:

| Fresh seed | Unit H200 CE | Fixed-half H200 CE | Half minus unit |
|---|---:|---:|---:|
|199410011|4.712135067433241|4.740863758615851|+0.0287286911826099|
|199410012|4.723844898324018|4.748150799796942|+0.024305901472923885|
|199410013|4.704536328905343|4.72432104243529|+0.019784713529946707|

Independent mean delta+0.024273102061827024 nat/token. Simultaneous conditional
text-bootstrap upper bounds were+0.02764039132848912,+0.027897883887647648,
+0.026513473821097925 for2048/4096/8192-token blocks. Mean improvement, every-seed
improvement and simultaneous-bound gates all failed. These are three trained seeds
on adaptively reused validation text, not population-seed or held-out inference.
No pilot-loss selection, LR screening, rescue, retry or interim stop occurred.

quality001-closure.json binds the independent final review, all anchors and three
inference reviews. Terminal STOP closes every branch-scale quality command; no
extension or reopening. All659 sources remain unchanged. All six fresh raw/H200
pairs totaling3,354,994,032B are preserved, along with historical selected/fresh
models and protected archives. Closure free9,355,235,328B is an observation, not
next-panel readiness. Retain unit gains/D256 MLA score0.5/SwiGLU672/SiLU/blockRMS/
serial/no product norm/no local mix/no warmup/decay0.00006/cooldown800/H200.

Both architecture and bounded HP were reviewed. Select the previously deferred
global clipping0.5-versus1.0 prerequisite; defer matched-epsilon final-readout
LayerNorm versus final RMSNorm, leaving pre-branch norms unchanged. Earlier block
LayerNorm evidence is adverse and no final-norm benefit is assumed. New clipping
component001 passed four generated CPU FP64/FP32 real Muon/AdamW arithmetic/order/
resume cases, no-gradient preservation, nonfinite rejection, threshold identity and
unchanged RNG. Maximum gradient-oracle error2.2522872067387567e-8. This is not actual
full-model clipping frequency, MPS/EMA/H200/cache or scientific quality evidence.
See TOKENIZER_GRADIENT_CLIP_PREPARATION.md for next executable integration.
