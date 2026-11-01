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
