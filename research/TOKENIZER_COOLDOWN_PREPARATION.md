# Tokenizer cooldown-onset preparation

Current status: trainer-worker001 and trained-inference001 COMPLETE for both
four-update generated arms. No active worker, scientific quality result, seed
reservation or pilot/quality freeze exists. All earlier prerequisites stay closed.
Parallel-residual quality001 is CLOSED_NOT_ADOPTED; serial remains adopted.

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
bindings remained unchanged. Fresh scientific seeds have not been reserved.

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
