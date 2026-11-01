# MLA attention-score multiplier transfer

Status: quality001 is frozen and its first screening endpoint has started.
Resource pilots, real acceptance/inference and exact retirement prerequisites
passed; scientific seeds and the complete finite quality plan are frozen. Retain the reviewed D256 MLA/SwiGLU672/SiLU/RMSNorm/H200
recipe unless a new finite comparison passes its prospective rule.

## Why this question now

Review at the completed FFN boundary considered architecture and bounded tuning:

- Internal Q/K normalization: historical MHA QK-RMS fresh deltas were -0.005274
  at D64 and +0.007981 at D256, with the local screening gate already failed.
  Added FFN hidden normalization also missed its gate. Neither supplies a strong
  reason to add new normalization parameters immediately after the block-norm
  comparison. These remain closed character studies, not tokenizer results.
- Position/local mixing: historical NoPE and ALiBi both lost against RoPE at
  both widths; a short-convolution addition remains distinct but introduces a
  new cache/history contract. Defer it, without claiming a subword negative.
- Sparse MoE remains an architectural candidate, with the wider-dense control
  required and its prior full storage projection larger than this two-arm study.
  Closed activation, block-normalization, EMA and FFN comparisons are not reopened.
- Bounded HP: select **attention multiplier 0.5 versus 1.0**. The 0.5 choice was
  adopted with character MHA; historical mean gains were 0.006210/0.009851 at
  D64/D256, with a different selected LR for half-strength scores. The current
  tokenizer MLA uses compressed projections and concatenated content/rotary
  coordinates. Its score multiplier has not been independently tuned in the
  reviewed tokenizer inventory. This is a transfer question, not a rerun of the
  character campaign, a bug fix, or a claim that weaker/stronger scores are better.

Source inspection confirms that current full and cached MLA **do consume 0.5**.
The attention roadmap's earlier missing-multiplier observation describes the
historical pre-integration state and is already resolved in the adopted runtime.
No entropy-collapse or clipping diagnosis has been established. Generic warmup,
clipping and decay sweeps are deferred in favor of this exact inherited choice.

## Exact contrast and prospective finite design

For each head and causal token pair, let content width and rotary width both be32:

`score[t,s] = alpha * (q_content[t]·k_content[s] + RoPE(q_rot[t])·RoPE(k_rot[s])) / sqrt(64)`.

Apply the causal mask after content scaling, then softmax, value weighting and
output projection. Control alpha=0.5; candidate alpha=1.0. Scale both terms
jointly; keep rotation frequencies, low-rank geometry, learned biases and all
other inherited operations unchanged. No trainable temperature or new tensor.
The intended full model has28,492,416 parameters in both arms, identical parameter
shapes, initialization draws and Muon/AdamW groups. Full integration must verify
these properties rather than infer them solely from the equation.

Keep four layers, D256/head32, query/KV ranks64, shared rotary key width32,
SwiGLU672, block RMSNorm, final affine LayerNorm, grouped-embedding v2, untied
GPT2 tables, dropout0, context128, batch8, paired512 sampling,1200updates,
cooldown800 and H200 from800 every10updates. Same data/splits and closed test.

Proposed two20-update resource pilots, then equal three-LR screens per arm
(0.000625/0.00125/0.0025), then three fresh paired seeds at each arm's independently
selected minimum H200 full-validation CE (lower-LR tie break). Six screens and
six fresh trajectories; no boundary extension or alternative winner. Pair order
will alternate, with exact seeds/order frozen after census and prerequisites.
Retain two selected screen pairs and all six fresh raw/H200 trajectories.
Any successful-pilot/losing-screen retirement needs exact audited boundaries.

Adopt only if candidate-minus-control mean H200 CE is below-0.02, every fresh
paired delta is below0, and simultaneous conditional-text upper bounds are below0
for2048/4096/8192-token blocks,2000 common paired PCG64 draws and linear quantile
1-0.05/3. No cost-only route. Full/cache correctness and three fresh paired
inference reviews remain required. Reused validation text gives adaptive local
evidence, not test-holdout or training-seed-population confidence. Independently
selected rates make this a recipe comparison, not an isolated causal score effect.

## Completed narrow numerical prerequisite

`tokenizer-evidence/mla-score-preparation-001/component001.py/json` extracts an
actual incumbent attention module, makes private copies and overrides only their
scalar for this component test. It does not bypass or change full-model validators.
An independent complex-RoPE oracle contracts content and rotary scores separately
and explicitly normalizes exponentials. Both multipliers passed FP64 CPU forward
and all input/parameter gradients; maximum errors7.78e-16 and8.89e-15 respectively.
CPU/MPS FP32 forward errors were at most4.48e-7 and gradient errors4.77e-6.
Uneven3/1/5/7-token caches, nonzero continuation offsets, reset and exact causality
passed on both devices. Cache allocation for one layer and batch2 is98,304B.

The frozen full-model contract correctly rejects alpha1.0. New explicit model,
optimizer/checkpoint/cache identities must be implemented outside the424 frozen
sources, with bitwise incumbent initialization/output checks, serialization/resume
and raw/H200 worker/parent acceptance before scientific training. No corpus,
checkpoint weights or test split were read by this synthetic arithmetic run.
No quality claim follows from numerical correctness.

## Capacity and next executable work

The completed FFN pilots are now eligible for exact policy cleanup because all
of their dependent quality and inference work closed. Their original protection
was scoped to that campaign; selected screens, every fresh pair and failed
artifacts remain protected. A separate manifest and journal record retirement;
historical closure and source snapshots remain immutable.

The next capacity review must bind actual serialized raw/H200 sizes, metadata
and serialization overlap, plus the unchanged8GiB reserve. Do not treat the old
FFN projection or this component check as a new launch authorization. Complete
the small full-model integration while auditing eligible obsolete artifacts;
no partial quality panel or reduced confirmation is permitted to hide a deficit.

## Full-model and serialized resume integration completed

The earlier component-only status is superseded by the isolated
`research/mla_score_v1` runtime and model002/checkpoint-resume001 evidence. Only
block-normalization policy resolution and constructed architecture validation
were extended; unchanged implementation modules resolve under the new namespace.
The named score policy permits exactly float0.5/float1.0 on the incumbent backbone.
Both arms retain28,492,416 parameters, bitwise-identical initial weights and
identical Muon/AdamW groups. Control logits match the incumbent bit for bit.
Cross-score checkpoint/config/cache identities, mutated module scalars and
unsupported settings are rejected. CPU/MPS logits, gradients, uneven/interleaved
cache, reset and causality passed; no quality claim follows.

The first model001 harness stopped before training because it used the new
optimizer validator on the legacy model. Its source/snapshot/failed guard remain
intact. Separately recorded model002 uses each namespace's own optimizer factory;
no historical failed fixture was rerun or validator weakened.

Generated-text strict-MPS tests completed three updates per uninterrupted and
repeated trajectory, plus interrupted replay, for each arm. Model, optimizer,
EMA and RNG comparisons were bitwise exact; raw/H200 export/load and cached
inference passed at batches1/8. Actual serialized sizes are445,082,407B per raw
and114,074,435B per H200. These numerical fixtures are now exact members of the
verified `mla-score-synthetic-001` archive; their original hashes remain valid.
Original-path consumers must use verified archive resolution or restore exact
bytes with adequate space. Scientific worker/parent acceptance remains pending.

The new measured prospective projection is13,690,208,256B: eight retained
trajectories at576MiB each (448MiB raw +112MiB H200 +16MiB per-trajectory metadata),
256MiB round metadata and8GiB reserve. It assumes audited successful-pilot
retirement before screening and losing-screen retirement after saved selection.
The worker must enforce these caps and tested publication/retention boundaries
before this becomes a frozen launch contract. No scientific scope or global
guard is reduced. Later prerequisite output consumption must be charged again.

Completed EMA-pilot retirement reclaimed787,232,443B. Lossless archival of twelve
successful FFN/score numerical checkpoints reclaimed another2,199,083,259B net;
all selected/fresh/failed evidence remained unchanged. Observed free13,986,324,480B
exceeds the prospective projection by296,116,224B at that observation only.
Next: real bounded worker/parent validation and inference, then updated capacity,
fresh seed census, finite pilot freeze/pilots and final quality freeze.

## Real worker, independent parent and trained inference prerequisites passed

The score half/unit arms each completed two actual generated-text run_training
updates through the bounded serial child process and output sink. Independent
parent acceptance verified optimizer/RNG/resume semantics, both raw/H200 identities,
all88 EMA tensors and two snapshots,2,048 processed training targets, and all3,014
validation targets. CPU/MPS validation CE differed by at most2.775e-7. Both raw
checkpoints measured445,084,199B and both H200 exports114,074,819B. Existing streaming
publication and new individual448/112MiB limits passed; no duplicate payload or
temporary checkpoint remained. These are generated-data engineering checks only.

The resulting trained pair passed full/cache/reset checks at batches1/8 before
warmups and eight alternating timing repeats. Cache bytes were196,608 per item.
Timing is descriptive for this two-update fixture; it establishes no score-arm
quality or deployment-performance advantage. The research corpus and test stayed
closed. All frozen and worker-bound sources were rechecked unchanged.

Six successful worker checkpoints (both new score pairs and the prior successful
ungated FFN worker003 pair) are now losslessly archived in
`research_archives/mla-score-worker-001/successful-workers.tar.zst`. Full member
verification preceded loose-copy removal, and26,524 other files were rehashed.
Original1,677,471,074B became849,909,661B, recovering827,561,413B net. The failed FFN
worker002 control and every selected/fresh/failure artifact remain unchanged.
Use worker-archive001's exact manifest to resolve or restore absent loose paths.

Capacity003 corrects duplicate metadata accounting: the enforced256MiB complete
campaign budget already includes8.25MiB per trajectory, below its16MiB allowance,
including temporary metadata publication. Thus the complete proposed high-water
is eight560MiB weight allowances +256MiB metadata +8GiB reserve =13,555,990,528B.
The active serializer is included in its weight allowance and publishes by
same-inode link/unlink. No guard or confirmation requirement was reduced. Observed
free13,654,347,776B gives98,357,248B headroom at that observation only. The earlier
13,690,208,256B estimate double-counted128MiB of trajectory metadata.

Next execute the narrow prospective pilot/losing-screen retention-boundary checks,
then fresh seed census, finite pilot freeze, guarded resource pilots and final
quality freeze. Recheck actual free space and charge all new output before dispatch.
No scientific pilot/quality plan is frozen and no scientific worker is running.
Evidence: mla-score-preparation-001/trainer-worker001, trained-inference001,
worker-archive001, capacity003 and worker003-review.json; implementation in
research/mla_score_v1. No new ledger framework was introduced.

## Finite resource pilots completed; quality freeze is next

The narrow pilot and losing-screen retention test passed using the bounded,
locked executor and tiny disposable byte files. Incomplete or failed stages,
wrong saved LR selection, aliasing, fresh-confirmation paths, missing evidence,
and retry attempts were rejected. Protected sentinels and all remaining evidence
were unchanged. This tests eligibility/file handling; actual retirement still
requires independent complete payload/inference/selection audits.

A read-only census reserved pilot198630001, screen198630002, fresh198630011/12/13,
evaluation198630003 and bootstrap198630004 after scanning42,354 recorded files
(3,789,456,073B) with no collisions. No seed was chosen using a quality result.
The first frozen pilot wrapper (pilot001) stopped before config publication,
model construction or updates because its path variable shadowed the imported
data-contract module. Its source, plan, log and STOP remain preserved and closed.
The separate pilot002 wrapper uses an explicit data_contract alias; static
regression and real execution confirm the correction. Its new frozen root carries
forward the explicitly unconsumed reserved seed panel. No historical fixture was
reopened and no guard or validator was weakened.

Both pilot002 arms completed20updates/20,480 training targets and eight H200
snapshots at LR0.0025. Independent parent acceptance and full-validation coverage
passed, followed by complete paired cache/reset/inference barriers and timing.
Worker process times were40.57s/47.34s; parent acceptance4.81s/4.39s, below600s
per trajectory. Peak worker RSS was1,490,141,184B and MPS driver2,510,094,336B,
below4GiB/6GiB. Pilot losses are excluded from scientific LR selection and quality
conclusions. The test holdout was never scored.

All pilot-bound source bytes remain unchanged. research/mla_score_v1 is now a
frozen pilot namespace; final quality code must be additive outside it. The planned
quality scope remains six1200-step screens (three rates per arm), then three fresh
paired seeds, with identical prospective adoption gates and no extension.
Screen and fresh seeds are reserved and still unused.

Actual free space at final review: 12513693696B. The two accepted pilot raw/H200
pairs occupy1118328272B and remain protected until dependent preparation closes.
After exact successful-pilot retirement, prospective free would be13632021968B
against the complete13,555,990,528B requirement, margin76031440B.
This is conditional recovery credit, not completed deletion or launch authority.
Use the tested exact boundary after independent audits, measure actual space,
and freeze the full quality plan before training. Do not treat retained eligible
pilots as a permanent storage blocker or shrink the confirmation panel.

Evidence: mla-score-preparation-001/retention-boundary001, scientific-seed-census-001,
pilot001-failure-review, pilot002-plan/review/final-review and preserved failed/
completed metadata copies. No scientific quality endpoint has been launched.

## Full finite score quality comparison frozen and started

Additive code in research/mla_score_quality_v1 reuses the frozen score runtime,
bounded writer, full-validation/cache arithmetic and tested retention executor.
All74 finite-rule/config/selection/retention checks passed. Independent additive
acceptance reloaded both real pilot raw/H200 pairs and checked optimizer/EMA,
full-validation and cached inference before closing their dependencies. Neither
frozen pilot source nor the earlier failed attempt was modified or rerun.

The exact successful-pilot retirement removed four audited files totaling
1,118,328,272B under global/campaign locks. Independent payload/inference audits,
path/size/SHA256 allowlist, fsynced intent/deletion journal and post-verification
confirmed26,448 other research files unchanged. The failed pilot001 evidence,
selected models, all fresh pairs and archives remain protected. These four pilot
weights cannot be restored; original receipts and copied deletion evidence remain.

The final quality freeze measured14,089,822,208B free against13,555,990,528B:
eight560MiB weight allowances +256MiB complete metadata +8GiB reserve. Each active
trajectory has separate448MiB raw/112MiB H200 caps and a576MiB process-output cap.
Declared active campaign metadata groups total128,188,416B; bounded external
anchors/reviews/retirement shards add at most32MiB, together below the256MiB
allowance. Temporary publication is charged; checkpoint publication adds names,
not another payload. No speculative pilot cleanup or partial panel is needed.

quality001 freezes all twelve1200-update endpoints: six equal-budget screens,
three rates per multiplier (0.000625/0.00125/0.0025), followed by three reserved
fresh paired seeds. Save each arm's minimum full-validation H200 CE, lower-LR tie
break, only after all six screens independently pass. Then audit and retire only
four losing screen pairs; retain both selected screen pairs and all six fresh
raw/H200 trajectories. No extension, replacement seeds, retry or cost-only route.
All three fresh pairs require paired inference and independent final statistics.
The gate remains mean(unit-half)<-0.02, every fresh delta<0, and all simultaneous
conditional-text upper bounds<0 at2048/4096/8192 tokens (2000 common PCG64 draws).

First screen slot00 (multiplier0.5, LR0.000625) has started under the frozen hash.
No quality result or adoption claim is made here. Both score namespaces are now
frozen; continue the exact serial prefix and actual process/anchor state, keeping
the test holdout closed. Evidence: quality-integration001, quality001-pilot-retirement,
quality001-plan/freeze/preflight/capacity/start-review in mla-score-preparation-001.

### First accepted quality endpoint

quality001 slot00 (multiplier0.5, LR0.000625, screening seed198630002) completed
all1200updates/1,228,800 targets and41H200 snapshots. Independent parent checkpoint,
optimizer/EMA, full-validation and cache checks passed. H200 validation CE is
4.896530022nat/token across34278 targets. This is one
screening endpoint, not a candidate comparison, selected rate or adoption result.
Remaining capacity fits: observed13514715136B versus required
12996825432B after charging the accepted retained pair.
The next frozen slot01 (multiplier1.0 at the same LR) has started serially. Keep
all six screens through selection and every fresh pair through final review.
Evidence: quality001-slot00-result.json and its preserved metadata, plus the
external quality001-accepted-00 anchor. No test scoring or interim selection.


### First matched MLA score screening comparison

quality001 slot01 accepted at 2026-11-01T06:31:57.562317+00:00: multiplier1.0 at
LR0.000625 and screening seed198630002 completed1200updates/1,228,800targets,
with41H200 snapshots and88 EMA parameters. Independent parent checkpoint,
optimizer/EMA, full-validation and cache audits passed. H200 CE across34278
validation targets is 4.863321954179, versus 4.896530022241 for
multiplier0.5 at the same rate: unit-minus-half -0.033208068063nat/token.
This is the first screening comparison only; it does not select a rate or
establish adoption. Finish all six equal-budget screens, then all three fresh
pairs at the independently selected rates under the original frozen gate.

Worker peak RSS1,640,792,064B and MPS driver2,510,094,336B remained within
4GiB/6GiB limits; worker and parent guards finished without violations.
Observed free 12,939,902,976B exceeds the remaining complete-panel
requirement 12,437,660,336B after charging both retained
raw/H200 pairs. Slot02 (multiplier1.0, LR0.00125) has started serially, with
capacity rechecked by the frozen runner. Keep all six screen pairs until the
saved-selection retention boundary; the holdout remains closed. Evidence:
mla-score-preparation-001/quality001-slot01-result.json, preserved slot01
metadata and quality001-accepted-01.json.


### Third score screening endpoint accepted

quality001 slot02 accepted at 2026-11-01T06:37:05.177388+00:00: multiplier1.0,
LR0.00125, screening seed198630002. All1200updates/1,228,800targets and
41H200 snapshots completed; independent parent checkpoint/optimizer/EMA,
full-validation and cache checks passed. H200 CE across34278 validation targets
is 4.692032518055, compared with 4.863321954179 for the same
multiplier at LR0.000625. This single-seed rate comparison makes no rate selection
or adoption decision. The matched multiplier0.5/LR0.00125 endpoint is next,
followed by both frozen LR0.0025 screens. No grid change or interim stopping.

Worker peak RSS1,642,840,064B and MPS driver2,510,094,336B remained below
4GiB/6GiB; worker and parent guards finished without violations. After retaining
all three raw/H200 pairs, observed free 12,364,083,200B exceeds the
remaining full-panel requirement 11,878,495,240B. Slot03
has started serially with the same frozen source and all guards. Preserve all
screens until the complete six-screen selection/retirement boundary and every
fresh pair through final review; no test access. Evidence: quality001-slot02-result.json,
quality001-slot02-evidence/ and quality001-accepted-02.json in
mla-score-preparation-001/.


### Second matched score comparison accepted

quality001 slot03 accepted at 2026-11-01T06:42:22.271513+00:00: multiplier0.5,
LR0.00125 and screening seed198630002 completed1200updates/1,228,800targets,
with41H200 snapshots. Independent checkpoint/optimizer/EMA, full-validation
and cache checks passed. H200 CE across34278 targets is 4.703659447036,
versus 4.692032518055 for multiplier1.0 at the same rate. Unit-minus-half
is -0.011626928981nat/token. Both completed matched-rate comparisons favor1.0
on this screening seed, but the gap is smaller at0.00125 than at0.000625.
This is no rate selection or adoption result; finish the two LR0.0025 screens
and all three reserved fresh pairs under the original gate.

Worker peak RSS1,643,544,576B and MPS driver2,510,094,336B remained below
4GiB/6GiB; both guards finished without violations. Observed free
11,865,415,680B exceeds remaining complete-panel requirement
11,319,330,144B with four raw/H200 pairs retained. Slot04
(multiplier0.5, LR0.0025) has started serially. All screen pairs remain protected
until the complete six-screen saved-selection boundary; no test access, grid
extension or changed source. Evidence: quality001-slot03-result.json,
quality001-slot03-evidence/ and quality001-accepted-03.json in
mla-score-preparation-001/.


### Fifth score screening endpoint accepted

quality001 slot04 accepted at 2026-11-01T06:47:42.882435+00:00: multiplier0.5,
LR0.0025 and screening seed198630002 completed1200updates/1,228,800targets
and41H200 snapshots. Independent checkpoint/optimizer/EMA, full-validation
and cache audits passed. H200 CE across34278 targets is 4.793428589630,
worse by 0.089769142594nat/token than this arm at LR0.00125
(4.703659447036). The upper tested rate therefore did not improve the
incumbent on this screening seed. This does not select a rate or establish
adoption: complete the final unit/LR0.0025 endpoint, then apply the frozen
per-arm minimum-CE/lower-LR tie rule to all six accepted screens.

Worker peak RSS1,611,890,688B and MPS driver2,510,094,336B stayed within
4GiB/6GiB limits; both guards finished without violations. Observed free
11,260,493,824B exceeds the remaining complete-panel requirement
10,760,165,048B with five raw/H200 pairs retained.
Slot05 (multiplier1.0, LR0.0025) has started serially. After it passes,
independently re-audit all six and retire only the four losing screen pairs
through the frozen boundary, then run all three fresh confirmation pairs.
No test scoring, grid extension or checkpoint removal occurred in this iteration.
Evidence: quality001-slot04-result.json, quality001-slot04-evidence/ and
quality001-accepted-04.json in mla-score-preparation-001/.


## MLA score six-screen selection and retention complete

All six frozen1200-update screens passed independent checkpoint/optimizer/EMA,
full-validation and cache acceptance. Each completed1,228,800training targets
and41H200 snapshots; validation covers34278 aligned targets. Final slot05
(multiplier1.0/LR0.0025) reached CE4.832040597159. No grid extension, changed
source, pilot-loss selection or replacement seed occurred.

| Learning rate | Multiplier0.5 H200 CE | Multiplier1.0 H200 CE | Unit minus half |
| --- | ---: | ---: | ---: |
| 0.000625 | 4.896530022 | 4.863321954 | -0.033208068 |
| 0.00125 | 4.703659447 | 4.692032518 | -0.011626929 |
| 0.0025 | 4.793428590 | 4.832040597 | +0.038612008 |

The persisted per-arm minimum-full-validation-CE/lower-LR tie rule selects
**LR0.00125 for both arms**. On this screening seed, multiplier1.0 improves
CE at the two lower rates but worsens it at0.0025. At the selected common rate
the difference is -0.011626928981nat/token. This is screening evidence only;
the three reserved fresh pairs and original strict gate decide adoption.
The result makes an unqualified claim that multiplier1.0 always improves this
backbone inappropriate.

The frozen boundary independently reloaded and re-audited all six raw/H200
pairs before persisting selection and retiring slots00/01/04/05 only. Eight
files totaling2,236,660,384B were removed under the shared worker/campaign locks,
exact path/size/hash manifest, durable intent/deletion journal and bounded
guards. All72 retained campaign evidence files were reverified unchanged.
Selected slots02/03 remain; no fresh, failure, archive or unrelated path was
removed. Retired weights cannot be restored; original accepted receipts and
all nonweight evidence remain immutable, with the later retention manifest
explaining their absent checkpoint paths. A read-only recorder independently
verified the boundary and preserved its selection, inventory, manifest, journal,
receipt and guards in quality001-screen-retention/.

Post-retirement free12,979,437,568B exceeds the full remaining
requirement12,437,660,336B, including the unchanged8GiB
reserve. Fresh slot06 (multiplier0.5/LR0.00125/seed198630011) has started serially;
slot07 is its matched1.0 partner. Complete all six fresh trajectories, all three
paired inference reviews and independent final statistics. Keep every fresh
raw/H200 pair and the test holdout closed. Evidence: quality001-slot05-result.json
and preserved metadata, quality001-accepted-05.json, quality001-boundary-result.json,
quality001-boundary-record-guard.json and quality001-screen-retention/ in
mla-score-preparation-001/.


### First fresh MLA score endpoint accepted

quality001 slot06 accepted at 2026-11-01T06:59:36.996505+00:00: multiplier0.5,
selected LR0.00125, first reserved fresh seed198630011. All1200updates,
1,228,800training targets and41H200 snapshots completed. Independent parent
checkpoint/optimizer/EMA, full-validation and cache audits passed; H200 CE is
4.712354865011 across34278 validation targets. This is one fresh endpoint,
not a paired difference or adoption result. Slot07 multiplier1.0 is now running
at the same seed and rate. No selection change, interim stop or grid extension.

Worker peak RSS1,653,833,728B and MPS driver2,510,094,336B remained below
4GiB/6GiB; worker and parent guards finished without violations. Retained
selected screen pairs and the fresh raw/H200 pair were verified. Observed free
12,407,627,776B exceeds remaining complete-panel requirement
11,878,495,240B. All fresh confirmations remain protected.

The existing read-only three-pair inference driver was adapted outside frozen
namespaces, changing only score-module imports and the exact quality freeze SHA.
Syntax validation passed; execution is deferred until all twelve endpoints are
accepted, when its exact requests/anchors and runner hash will be frozen. No
inference result or new numerical claim is made for this unexecuted driver.
Evidence: quality001-slot06-result.json, quality001-slot06-evidence/ and
quality001-accepted-06.json; driver quality001-fresh-inference.py in
mla-score-preparation-001/. Keep the test holdout closed.


### First fresh MLA score pair completed

quality001 slot07 accepted at 2026-11-01T07:05:12.665572+00:00, completing the first
fresh pair at seed198630011 and selected LR0.00125 for both arms. Multiplier1.0
H200 CE is 4.699208080398, versus 4.712354865011 for0.5:
unit-minus-half -0.013146784613nat/token across34278 aligned validation targets.
Both endpoints completed1200updates/1,228,800training targets and41H200 snapshots
and passed independent checkpoint/optimizer/EMA, full-validation and cache audits.
This one-seed improvement does not establish adoption or justify interim stopping.
Complete the other two frozen pairs and all three inference reviews before the
independent strict-gate calculation; retain both raw/H200 pairs.

Worker peak RSS1,648,738,304B and MPS driver2,518,482,944B remained within
4GiB/6GiB limits; both guards finished without violations. Observed free
11,848,216,576B exceeds remaining full-panel requirement
11,319,330,144B. Slot08 (multiplier1.0, seed198630012,
LR0.00125) has started in the frozen unit-first order for the second pair.

The independent final-statistics driver was adapted from the completed FFN
review outside frozen namespaces, with score identities and neutral handling
of either gate outcome. It rechecks all12 accepted endpoints, all three
inference reviews, aligned validation blocks, explicit independent bootstrap
quantiles and the frozen gate. A passing gate requires a separate adoption
review; the script cannot silently adopt. Syntax validation passed; it has
not been executed and provides no new statistical conclusion yet. Evidence:
quality001-slot07-result.json, quality001-slot07-evidence/, quality001-accepted-07.json
and quality001-final-review.py in mla-score-preparation-001/. Holdout stays closed.


### Second-seed score candidate accepted

Slot08 accepted at 2026-11-01T07:10:56.377832+00:00: multiplier1.0, selected LR0.00125,
fresh seed198630012; H200 CE4.705125006811 across34278 validation targets.
All1200updates/1,228,800training targets and41H200 snapshots completed; independent
checkpoint/optimizer/EMA, full-validation and cache checks passed. Its matched
control slot09 is now running. There is no second paired difference or adoption
conclusion yet; retain every fresh pair and complete the frozen panel.

Peak worker RSS1,558,102,016B and MPS driver2,518,482,944B stayed within4GiB/6GiB;
both guards finished without violations. Free11,279,716,352B exceeds
remaining full-panel requirement10,760,165,048B. No cleanup,
grid/source change or holdout access. Evidence: mla-score-preparation-001/
quality001-slot08-result.json, quality001-slot08-evidence/ and quality001-accepted-08.json.


### Second fresh MLA score pair completed

Slot09 accepted at 2026-11-01T07:16:43.884398+00:00, completing seed198630012 at
selected LR0.00125 for both arms. Multiplier0.5 H200 CE4.722457821258
versus multiplier1.0 CE4.705125006811 gives unit-minus-half
-0.017332814447nat/token across34278 aligned validation targets. Both fresh pairs
so far favor1.0, but there is no adoption or interim stopping decision. Complete
the final pair, all three paired inference reviews and independent frozen-gate
statistics. Each endpoint passed1200updates/1,228,800targets/41H200 snapshots
and independent checkpoint/optimizer/EMA, validation and cache acceptance.

Worker peak RSS1,583,513,600B and driver2,518,482,944B were below4GiB/6GiB;
both guards finished without violations. Free10,710,462,464B exceeds
remaining full-panel requirement10,200,999,952B. Slot10
(multiplier0.5/LR0.00125/seed198630013) has started serially. All fresh raw/H200
pairs remain protected; no cleanup, source/grid change or test access. Evidence:
mla-score-preparation-001/quality001-slot09-result.json, quality001-slot09-evidence/
and quality001-accepted-09.json.


### Final-seed MLA score control accepted

Slot10 accepted at 2026-11-01T07:22:42.657530+00:00: multiplier0.5, selected LR0.00125,
fresh seed198630013; H200 CE4.707110792553 across34278 validation targets.
All1200updates/1,228,800training targets and41H200 snapshots completed, with
independent checkpoint/optimizer/EMA, full-validation and cache audits passed.
The final matched candidate slot11 is now running. There is no third-pair
difference or adoption decision yet. Complete it, then all three paired
inference reviews and independent statistics under the frozen gate.

Worker peak RSS1,551,777,792B and MPS driver2,518,482,944B remained below
4GiB/6GiB; both guards finished without violations. Free10,151,968,768B
exceeds remaining complete-panel requirement9,641,834,856B.
All selected and fresh raw/H200 pairs remain protected; no cleanup, grid/source
change or test access. Evidence: mla-score-preparation-001/quality001-slot10-result.json,
quality001-slot10-evidence/ and quality001-accepted-10.json.


## MLA attention multiplier comparison closed: retain0.5

All twelve1200-update endpoints, three fresh paired inference reviews and the
independent final statistics review passed. Slot11 multiplier1.0 completed the
final fresh pair with H200 CE4.692675884339. Both arms selected LR0.00125 from
the original equal three-rate screens; no grid extension, pilot-loss selection,
seed replacement or holdout evaluation occurred. All449 frozen source entries
were reverified unchanged at closure.

| Fresh seed | Multiplier0.5 CE | Multiplier1.0 CE | Unit minus half |
| --- | ---: | ---: | ---: |
| 198630011 | 4.712354865 | 4.699208080 | -0.013146785 |
| 198630012 | 4.722457821 | 4.705125007 | -0.017332814 |
| 198630013 | 4.707110793 | 4.692675884 | -0.014434908 |

Mean unit-minus-half is **-0.014971502424nat/token**.
Every fresh seed improved and all simultaneous conditional-text upper bounds
are negative. However, the frozen requirement was mean delta strictly below
-0.02; this comparison fails that minimum-improvement gate. Therefore **retain
multiplier0.5 with D256 MLA/SwiGLU672/SiLU/block RMSNorm/H200**. The data support
a small consistent validation improvement from1.0 in these three runs, not
absence of an effect; the prospective practical threshold prevents adoption.
No cost-only route or retrospective threshold change is allowed.

| Text block size | Descriptive95% interval | Simultaneous gate upper bound |
| --- | --- | ---: |
| 2048 | [-0.017749179, -0.011981252] | -0.011724943 |
| 4096 | [-0.016979215, -0.012739207] | -0.012548857 |
| 8192 | [-0.016721147, -0.012374552] | -0.012035767 |

The separate implementation reproduced paired loss arithmetic and2000 common
PCG64 bootstrap draws at2048/4096/8192-token groupings, with linear quantiles
and simultaneous upper quantile1-0.05/3. These are intervals conditional on
reused validation text and these three trained seeds, not population-seed,
independent holdout or universal architecture guarantees.

All three full/cache/reset barriers passed at batches1/8 before timings. Two
warmups and eight alternating repeats per cell were completed under unchanged
guards. Timings are descriptive medians; both H200 models were resident and
model loading/cache allocation were excluded. Mixed timing differences and
identical cache sizes establish no consistent deployment-cost advantage.

| Pair | Batch | Prefill0.5 /1.0 ms | Decode0.5 /1.0 ms | Cache bytes per arm |
| --- | ---: | --- | --- | ---: |
| 1 | 1 | 8.320 / 8.715 | 289.938 / 283.768 | 196,608 |
| 1 | 8 | 14.114 / 13.748 | 298.384 / 298.548 | 1,572,864 |
| 2 | 1 | 8.847 / 9.019 | 291.754 / 292.610 | 196,608 |
| 2 | 8 | 13.667 / 13.826 | 314.810 / 300.039 | 1,572,864 |
| 3 | 1 | 9.067 / 8.519 | 286.192 / 294.305 | 196,608 |
| 3 | 8 | 14.188 / 13.940 | 300.852 / 293.040 | 1,572,864 |

Inference peak RSS was824,164,352B and driver1,155,842,048B. All three guard
reviews finished without violations. Closure preserved both selected screen
pairs and all six fresh raw/H200 trajectories:4,473,320,768 retained weight bytes.
The prior exact four-loser retirement remains final; resolve absent screen
weights through its manifest. Closure observed9,603,719,168B free;
this is not a capacity guarantee for another campaign. No research worker
remains active. All earlier campaigns and failed attempts remain closed.

Evidence in mla-score-preparation-001/: quality001-slot11-result.json/evidence/,
quality001-accepted-11.json, quality001-fresh-inference-plan.json, all three pair
result/review/guard files, quality001-final-review.json/guard and
quality001-closure.json/guard.

At this boundary, both architecture and bounded HP hypotheses were reviewed.
Next prepare per-token affine RMS normalization of the SwiGLU product before
down projection versus none on the retained backbone. This is distinct from
closed block-normalization and FFN-factorization studies. Character results
are a weak prior only.100-update warmup with bounded peak-LR tuning was reviewed
and deferred given weak mixed character evidence and no current stability
mechanism. See TOKENIZER_FFN_NORM_PREPARATION.md and its direction-decision.
No new scientific seeds or training panel are frozen.

The new component001 harness failed before numerical work by parsing Python
source as JSON during hash verification; its source/plan/failure remain closed
and immutable. Separately recorded component002 corrected only this harness
check and passed12 generated CPU FP64/FP32 cases, all forward/gradient comparisons
and per-token causality, with identical shared matrices and672 additional gain
parameters per block. No full-model/MPS/optimizer/resume/quality claim follows.
Next perform narrow explicit model/checkpoint/cache integration outside frozen
namespaces; measure actual retained-output high-water and audit eligible capacity
recovery before any new finite pilot/quality freeze. Small numerical work is
not blocked by a future training-capacity constraint.
