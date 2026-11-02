# Tokenizer cooldown-onset preparation

Current status: quality001 FROZEN; slot00 independently accepted and paired
slot01 cooldown600 is running under the exact
freeze `1a13edadf74644b11c9aca9b9861d1875c156217533916592fabff04bee194d0`. The current
progress record and accepted anchors supersede this launch snapshot. No scientific
quality result exists yet. Six1,200-update trajectories form three fresh pairs at
common peakLR0.00125, with zero LR-search budgets and original strict gates.
Pilot/evaluation seeds are used; fresh199300011 is now in use,012/013 and
bootstrap199300004 remain unused at launch. All earlier campaigns stay closed.

Test linear cooldown beginning at zero-based update600 versus incumbent800 over
1,200 updates. Both use peak LR0.00125 and final multiplier0.1, no warmup and
independent per-update decay0.00006. For start s, multiplier m(t)=1 for t<=s,
otherwise m(t)=1-0.9*(t-s)/(1199-s). Keep the adopted serial D256 MLA score0.5,
SwiGLU672/SiLU, separate block RMSNorms, no product norm/local mixing and H200.
H200 remains start800/every10/half-life200, giving41 snapshots at the quality
horizon. Earlier cooldown also changes total LR exposure: multiplier sums are
1020 versus930, or1.275 versus1.1625 at base LR0.00125 (8.82% less). This is not
an equal-integral schedule-shape comparison.

At the boundary both architecture and bounded HP were reviewed. Fixed serial
residual branch scaling is a distinct deferred architecture hypothesis requiring
explicit gains, initialization/checkpoint/cache contracts. Historical character
schedule evidence at3,000 updates does not answer this subword1,200-update onset
question; do not pool historical scores/seeds or assume improvement.

The frozen schedule-component001 plan binds601 sources, including600 unchanged
predecessors. Both complete schedules match an independent rational oracle;
width-scaled group rates are exact. Generated CPU float64 SGD parameter/momentum
resume matches exactly after zero-based updates0/598/599/600/799/800/1198.
Absolute-step reapplication does not compound rates; malformed schedules are
rejected. Source/plan/log/result/guard are immutable under
[tokenizer-evidence/cooldown-preparation-001](tokenizer-evidence/cooldown-preparation-001).
This is toy-SGD evidence only, not actual Muon/AdamW/model/EMA/H200/cache evidence.
No research corpus or scientific seed was used and no panel was launched.

Next implement minimal additive finite20-update resource-pilot orchestration and
the complete three-fresh-pair fixed-LR design outside609 frozen runtime/acceptance
bindings. Reuse existing serial/acceptance/inference/retention primitives, verify
actual successful worker payloads/cache and finite contracts, then close their
tensor dependencies. Prospective pilot onsets13/10 are compressed20-update
fixtures; actual1,200-update onset800/600 group rates were checked independently.
Complete fresh scientific census and measured complete retained-output high-water
including serialization overlap, metadata and8GiB reserve before finite pilot
freeze, and require successful pilots before quality freeze. Quality remains
exactly three fresh pairs at common LR0.00125, equal zero LR-search budgets,
original strict gates, no screens/selection/rescue/retry; standard-recipe scope.
Proactively audit authorized lossless recovery if the complete scientific panel
needs more space; preserve every selected/fresh byte and existing protected archive.
Do not weaken guards or start a partial panel to fit storage. Current free space
is not readiness; source work proceeds while any necessary exact recovery is
prepared. Preserve selected/fresh models, all original evidence and protected
storage-recovery-004/006 archives; archived RAW bytes are not retired.

Full-model model001 passed both cooldown arms with identical initialization,
logits, RNG and Muon/AdamW grouping:28,492,416 parameters and88EMA tensors.
Both full1,200-update rate sequences match independently computed rates on real
CPU/MPS optimizer groups; no scientific training updates were made. CPU/MPS
all-gradient, full/cache/reset/causality checks passed;196,608B cache per item.
Cross-schedule checkpoint identity and tampered independent decay were rejected.

Serialized model-resume001 passed both fresh four-update generated arms. This
short fixture uses cooldown onsets2/1; it does not execute600/800-update training.
Fresh repeat and step2 resumed raw model/optimizer/EMA/RNG match exactly (maxerror0).
Step2 serialization stays in capped memory. Final-step4 raw/H200 exports contain
three H200 snapshots each; CPU/MPS batch1/8 full/cache/reset inference passed.
All605 source/config bindings, including601 predecessors, are unchanged. Actual
run_training/output publication and independent parent full validation remain
required. No scientific seed reservation, pilot/quality freeze or active worker.

Capacity001 records actual free space and the complete next worker-pair requirement,
including8GiB reserve and metadata. At that point the four successful resume raw/H200 weights were retained pending
real payload adapter audits and an exact cleanup audit. The completed retirement
and worker launch below supersede that conditional recovery route; neither
observation establishes scientific-panel readiness. Small source work proceeds.

Actual trainer/parent/inference milestone: both four-update generated arms passed
run_training with bounded raw/H200 publication and independent parent full validation
(4,096 training targets,3H200 snapshots,88EMA tensors each). CPU/MPS validation CE
differences were below3.1e-7. Trained-inference001 passed full/cache/reset checks
at batches1/8,196,608B cache per item. Timings are descriptive correctness-fixture
measurements, not scientific quality or an adoption route. All609 source/config
bindings remained unchanged. At that prerequisite milestone scientific seeds were not yet reserved.

Adapter integration001 failed before numerical audit/deletion when plan construction
referenced trainer-worker001-source-plan.json before publishing it. Original
source/log/failure remain closed. Separately bound integration002 published plans
in dependency order and passed real resume raw/H200/optimizer/EMA/RNG/full-validation/
cache audits. After tensor dependencies closed, resume-retirement001 independently
audited and retired exactly four obsolete successful weights(1,118,313,556B),
verifying38,697 survivors unchanged. Original resume paths now resolve through
that manifest; do not restore or rerun. Actual post-cleanup free10,033,377,280B
exceeded the complete9,814,671,360B worker-pair requirement and enabled both workers.
Current successful worker raw/H200 weights(1,118,318,420B) stay retained until
pilot adapter dependencies close and a new exact audit. Current free space is not
a scientific-panel guarantee. Original selected/fresh models and storage-recovery-
004/006 protected archives remain untouched; small source work proceeds.

## Finite resource pilots and complete-panel recovery

The additive cooldown_campaign_v1 package binds615 sources, including609 unchanged
runtime/acceptance predecessors. Pilot-integration001 passed actual retained
worker raw/H200/optimizer/EMA/RNG/full-validation/cache audits, both20-update
schedule configurations and the complete six-trajectory/no-screen contract.
Retention-boundary001 passed exact pilot-only deletion and protected-fresh checks.
After dependencies closed, worker-retirement001 retired1,118,318,420B of obsolete
generated weights and verified38,877 survivors. Original receipts remain immutable;
absent generated weights resolve through that manifest, never restoration or retry.

Lossless storage-recovery-007 archived28 accepted historical selected/fresh RAW
checkpoints using exact allowlists, streamed member verification, shared lock,
durable journal and48,037 unchanged survivors. It reclaimed3,157,718,460B; exact
restoration passed. These RAW bytes remain protected and recoverable through
storage-recovery-007/raw-completed.json and protected-raw-storage-007 archives,
NOT retired. Every H200 remains at its original path; storage004/006 remain intact.

Capacity002 measured13,130,375,168B free against the complete12,381,585,408B
requirement: six560MiB weight allowances,256MiB metadata and8GiB reserve. Actual
worker sizes and bounded same-inode temporary publication were included. Fresh
census001 checked47,660 recorded files/4,230,727,100 bytes with no seed collisions.
Pilot freeze measured13,118,279,680B free and bound both20-update arms at common
LR0.00125, cooldown starts13/10 and H200start13/every1, yielding8snapshots.
The actual1,200-update Muon/AdamW800/600 rates had already passed independently.

Both pilots completed20,480 training targets with88EMA tensors and8H200snapshots.
Independent parent full validation and paired full/cache/reset inference passed
at batches1/8,196,608B cache per item. Final001 independently verified artifacts,
rate identities, guards and resource projections. The pilot losses do not select
any setting or support a quality conclusion. Final review observed11,972,616,192B
free plus1,118,328,144B retained pilot weights, conditionally13,090,944,336B after
future exact retirement; pilots stay retained until quality-source dependencies
close and an independent audit authorizes those exact paths.

Quality-source preparation completed in additive research/cooldown_quality_v1.
All623 active source bindings preserve615 unchanged predecessors. All21 contract
tests passed, including exact800/600 schedules, H200start800/41snapshots, six-slot
paired order, commonLR/no selection/no rescue, output guards and statistical signs.
Actual retained pilot raw/H200/optimizer/EMA/RNG/full-validation/cache audits passed;
pilot-to-quality forgery and mutated source bindings were rejected.

The separate locked pilot-retirement audit revalidated both accepted payloads and
paired inference, then retired exactly1,118,328,144B using the tested durable journal
and verified28,023 other run/archive files unchanged. Original receipts remain
immutable; absent successful pilot weights resolve via quality001-pilot-retirement.
No selected/fresh or protected lossless archive was retired. The final freeze
remeasured13,107,888,128B free versus12,381,585,408B complete-panel requirement,
including serialization/metadata and8GiBreserve. All reserved fresh seeds were
still unused before dispatch; the synthetic statistics tests used separate seeds.

Next: poll actual worker/process/log/guard/anchor state; never duplicate a running
or unaccepted attempt. Continue slots00-05 in frozen order800/600,600/800,800/600.
Preserve every fresh raw/H200 pair and recheck remaining full-panel capacity each
dispatch. After all six endpoints are accepted, run all three paired inference
reviews through quality001-fresh-inference001.py and independent statistics through
quality001-final-review001.py, both requiring the exact freeze hash. No screens,
selection, rescue, retry, interim adoption or partial panel.
