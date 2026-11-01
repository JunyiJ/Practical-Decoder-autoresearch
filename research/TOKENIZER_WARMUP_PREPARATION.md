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


## Warmup resource pilots completed

Finite pilot001 completed both120-update arms with actual100-update warmup in
the challenger. Each independently accepted endpoint has122,880 targets,
88 EMA tensors and21 H200 snapshots. Paired full/cache/reset inference passed
at batches1/8. Independent final002 resource/artifact review passed. The first
read-only final review omitted two strict environment variables and stopped
before auditing; its source/plan/failure are preserved closed. Final002 used
the required environment without repeating training or inference measurements.
No pilot loss selects a learning rate or supports an adoption claim.

The additive warmup_campaign_v1 source is frozen with487 source dependencies.
Exact retention-boundary001 and actual generated-worker payload/parent
pilot-integration001 passed before dependent worker weights were retired.
Fresh census reserved panel198850: pilot001/evaluation003 used as planned;
screen002, fresh011/012/013 and bootstrap004 remain unused. Only the resource
pilot plan is frozen; no scientific quality plan or worker is active.

Exact obsolete-numerical-cleanup-003 reclaimed4,691,175,838B from eight obsolete
numerical archives and four successful generated-worker weights, verifying
35,434 survivors unchanged. All original receipts and every selected/fresh/failed
artifact remain protected, including all model-resume001 publication-failure
outputs. Retired identities resolve through the later manifest and cannot be
restored. Capacity002 then measured14,070,431,744B versus13,555,990,528B required.
After pilots, final002 measured12,916,207,616B; the1,118,332,368B pilot weights
remain present. Their conditional exact retirement would yield14,034,539,984B.
This is conditional capacity, not permission to dispatch an unfrozen panel.

Next implement minimal additive warmup_quality_v1 orchestration outside all
frozen namespaces: six equal-LR1200-update screens, per-arm minimum full-validation
H200 CE/lower-LR tie selection, then all three fresh pairs, paired inference and
independent statistics. Validate real pilot payload/cache integration, close
pilot tensor dependencies, audit exact successful-pilot retirement, remeasure
complete capacity and verify unused scientific seeds before final quality freeze.
Keep warmup100 versus none, LR0.000625/0.00125/0.0025, cooldown/H200start800,
H200interval10/41snapshots, all frozen gates, confirmations and resource guards.


## Warmup finite quality comparison frozen

Warmup quality001 is frozen after75 integration checks and real accepted pilot
raw/H200/optimizer/RNG/full-validation/cache audits passed. Both arms have88 EMA
tensors; pilot-to-quality impersonation and mutated source bindings were rejected.
All487 predecessor sources are unchanged;497 Python sources are bound by the
additive warmup_quality_v1 contract. The only arm difference is the declared
100-update warmup schedule, with equal bounded peak-LR tuning.

Exact successful-pilot retirement reclaimed1,118,332,368B after tensor dependencies
closed. All27,049 other research files verified unchanged. Original pilot receipts
remain immutable; absent weights resolve through quality001-pilot-retirement's
manifest/journal and portable runtime evidence, and cannot be restored or rerun.
The final freeze measured14,039,228,416B free against13,555,990,528B required,
including all retained output/serialization/metadata allowances and8GiB reserve.

Freeze SHA:9f890ee1e534b966713139c9f4740488573133977b5416e43315e94df9775f94.
The first screen has been dispatched through research.warmup_quality_v1.quality_execution.
Keep exactly six1200-update screens at LR0.000625/0.00125/0.0025, persist each
arm's minimum full-validation H200 CE/lower-LR tie selection, independently
re-audit and retire only four losing screen pairs, then all six fresh trajectories
in frozen paired order, three paired inference reviews and independent statistics.
No grid extension, pilot-loss selection, partial panel, retry or adoption from
incomplete results. H200 start800/interval10 yields41 snapshots per endpoint.
Screen/fresh seeds were unused at freeze; screen198850002 is now in use.
Fresh198850011/012/013 and bootstrap198850004 remain unused at launch.

Read current process/log/acceptance-anchor state before any dispatch. Poll the
owned active worker, never duplicate it. Recheck actual remaining full-panel
capacity for each slot. Preserve selected models, every fresh pair, failures,
immutable sources/receipts and the closed test holdout. No new scientific quality
result or adopted-recipe change is claimed at launch.


Warmup quality001 screening and retention boundary completed. All six1200-update
endpoints passed independent parent acceptance and were re-audited at selection.
Both arms selected peak LR0.00125 (selected slots02/03) by the frozen minimum
full-validation H200 CE/lower-LR tie rule. Exactly four losing raw/H200 pairs
were retired:2,236,662,944B reclaimed,72 retained files unchanged. Original
acceptance receipts remain immutable; absent paths resolve via
quality001-screen-retention manifests/journal. Selected pairs remain protected.
Fresh confirmation has started in the frozen paired order using seeds198850011,
198850012,198850013. Preserve all six fresh trajectories, complete all three
paired inference reviews and independent final statistics before any decision.
No adoption or quality conclusion follows from screening alone.
