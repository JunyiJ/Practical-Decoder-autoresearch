# Tokenizer SwiGLU-product normalization preparation

Only prerequisite work is selected. No scientific seeds, pilot or quality panel
are frozen and no new scientific training is authorized by this document alone.

The adopted D256 MLA recipe retains multiplier0.5/SwiGLU672/SiLU/block RMSNorm/
H200 after the closed score campaign. Compare its unchanged SwiGLU product with
affine RMSNorm applied per token after `silu(gate(x))*up(x)` and before the down
projection. Proposed epsilon1e-5, gain initialized to1, no bias; four672-wide
gain vectors add2,688 parameters. This changes hidden-product amplitude and
learned channel scaling, not block centering or gating structure. Do not infer
an optimization mechanism from a CE difference. New gain optimizer treatment
must be explicit; existing parameter groups and shared initialization stay paired.

The character study's width256 RMS mean gain0.003438 was small, mixed across
seeds and insufficient for adoption. It is a weak prior, never pooled with
subword CE. The new tokenizer/MLA transfer question remains untested. QK norm
had unfavorable width256 character evidence; local convolution requires new
cache-history integration; MoE requires a new complete resource projection.
A bounded100-update warmup/peak-LR question was also reviewed and deferred:
character width256 mean gain0.000805 was mixed, and current curves establish
no instability mechanism. Both architecture and HP remain eligible later.

First check component arithmetic, all gradients and per-token causality on
generated tensors. Then add explicit full-model/optimizer/checkpoint/resume/
H200 and full-versus-cached contracts outside frozen namespaces; existing
tokenizer policies must remain closed to unsupported combinations. Complete
real worker/parent validation/inference before fresh seed census, a finite
pilot freeze, guarded pilots and final finite quality freeze. Use equal bounded
LR tuning; do not add retrospective arms or relax fresh confirmation requirements.

Measure actual serialized raw/export overlap, complete metadata/transients and
the8GiB reserve before any new panel freeze. The score campaign closure observed
about9.60GB free; that is an observation, not a ready new-panel budget. Inventory
exact authorized obsolete artifacts or lossless archival options as necessary;
preserve selected models, every fresh pair, failures and protected archives.
Small source/numerical work proceeds independently of training-capacity needs.

Evidence: tokenizer-evidence/ffn-norm-preparation-001/direction-decision.json.

## First component result

component001 stopped before tensor creation: the harness used a JSON reader
to verify Python source hashes. Its original source/plan and failure receipt
are preserved; no numerical or scientific seed was consumed. component002
is a separate frozen correction using exact source-byte hash equality.

All12 generated CPU cases passed: FP64/FP32, both none/RMS arms and input
scales0.01/1/10. Forward and all input/matrix/gain gradient maximum absolute
differences were0 in this implementation; altered suffix tokens left prefix
outputs unchanged. Shared gate/up/down matrices were identical and the gain
adds672 parameters per block,2,688 across four blocks. The guard completed
in0.359s, peak RSS301,154,304B, with the8GiB reserve preserved.

This is component evidence only. The full tokenizer policy still needs an
explicit additive extension; do not bypass frozen policy rejection. Next test
actual shared model initialization/optimizer groups, serialized resume/RNG/EMA,
H200 and cache/causality on CPU/MPS before any scientific launch.

## Explicit full-model integration passed

The additive `research/ffn_norm_v1` runtime introduces an exact none/RMS product
normalization policy without editing the449 frozen MLA score source files.
Generated full-model model001 passed CPU/MPS arithmetic, all parameter gradients,
one CPU optimizer step per arm, real in-memory save/load payload validation,
interleaved request caches, reset and causality. Legacy policy rejection remains.
The control's initialization and logits are bitwise identical to the adopted model;
all shared candidate weights are identical and four new gain vectors start at1.

Parameter counts are28,492,416/28,495,104. Shared optimizer groups are unchanged;
the2,688 gains use AdamW at the test's base LR0.0025 with per-step independent
decay0.00006. This test LR is not a scientific LR selection. Maximum CPU/MPS
logit discrepancy was1.431e-6, all-gradient discrepancy5.253e-7, and cached
logit discrepancy1.282e-6. Prefix causality and reset differences were0.
Cross-arm checkpoint/config/cache identities and altered constructed RMS epsilon
were rejected. Cache storage remains393,216B for batch2/context128.

The guard finished in3.896s with peak RSS1,332,674,560B and MPS driver
1,111,277,568B. No research corpus, holdout or scientific seed was used. This
is implementation evidence, not quality evidence. Serialized raw/optimizer/EMA/
RNG replay and H200 export remain separate prerequisites.

## Capacity recovered for serialized prerequisites

The old six-file resume recipe needs10,972,299,264B free including its complete
raw/export caps,256MiB metadata allowance and8GiB reserve. Before dispatch,
exact successful numerical-fixture archival recovered2,266,198,096B net and
verified27,046 retained files unchanged. Fourteen original weights remain
recoverable from four protected archives; see the retention policy and archive001
manifests. Selected models, every fresh pair and failure artifacts stayed loose.
This is a prerequisite-specific capacity check; a complete future scientific
panel high-water projection remains required.

## Serialized MPS resume and H200 passed

checkpoint-resume001 completed both arms on generated text at batch8/context128.
Three uninterrupted updates, a separate three-update repeat and two updates after
restoring the step1 checkpoint produced exactly equal model/optimizer/EMA/RNG
state (maximum discrepancies0). The test made16 generated optimizer updates
across both arms. Missing optimizer state, reordered optimizer groups, missing
MPS RNG and inference-only H200 resume were rejected before model mutation.

H200 export preserved parent weights/hashes and CPU/MPS RNG and replaced each
learned tensor with its exact EMA value. Exported models passed CPU/MPS full and
cached inference at batches1/8 (maximum discrepancy1.520e-6). No holdout or
research corpus was read. The16.204s guard observed peak RSS2,355,593,216B and
MPS driver2,551,070,720B, below unchanged limits. Six new raw/H200 fixtures are
retained under research_runs/tokenizer-ffn-norm-checkpoint-prerequisite-001;
raw sizes445,082,727/445,132,607B, exports114,074,691/114,087,095B by none/RMS.

Next implement the narrow actual run_training/output/worker path and independent
parent full-validation/inference acceptance using the tested runtime. The current
roughly9.66GB free is not enough to assume two new worker pairs fit above reserve;
finish source work, measure exact complete output high-water, and use audited
lossless archival of completed eligible prerequisites once their dependencies
close. Existing archives are protected and all failed attempts remain closed.
No scientific census, pilot plan or quality plan has been frozen.

## Real worker, independent parent validation and inference passed

trainer-worker001 completed two serial actual run_training workers on generated
text. Each performed2 updates/2,048 targets with2 H200 snapshots; the none/RMS
arms contained88/92 EMA tensors. The independent parent checked raw optimizer/
RNG/EMA state, exact H200 tensors/lineage, output caps and immutable bundle hashes,
and recomputed full validation exactly on MPS. CPU/MPS CE differences were
2.576e-7/2.512e-7. Incorrect target coverage was rejected. Raw/H200 pairs occupy
559,159,594/559,221,814B, and per-trial metadata73,512/73,676B.

All full/cache/reset barriers then passed at batches1/8 before two warmups and
eight alternating timing repeats. Cache remains196,608/1,572,864B per arm.
Batch1 median decode was277.974/282.926ms; batch8 was290.725/290.606ms for
none/RMS. These two-update generated fixtures establish measurement mechanics,
not a scientific quality or deployment-cost advantage. Inference guard finished
in19.289s, peak RSS871,710,720B and MPS driver1,162,608,640B.

The six earlier resume checkpoint files were losslessly archived after dependent
numerical reviews closed; their bytes remain in protected ffn-norm-resume-001
archives. This recovered957,460,848B net and verified27,031 survivors, enabling
both worker runs immediately. Original receipts remain unchanged; use later
resume-archive001 manifests rather than absent loose paths.

Fresh seed census scanned43,153 repository text files/3,829,300,519B and found
no prior occurrence of the reserved finite panel: pilot198740001,
screen198740002, fresh198740011/198740012/198740013, evaluation198740003,
bootstrap198740004. All remain unused. This is reservation only; no pilot or
quality plan is frozen and no scientific training launched.

capacity001 computes the complete prospective panel at13,555,990,528B free:
eight560MiB retained-pair allowances,256MiB complete metadata/transients and
8GiB reserve. Two pilot pairs and six screens fit below the eight-pair peak only
with successful audited pilot/four-loser retirement at the declared boundaries.
Observed9,661,427,712B leaves3,894,562,816B deficit. Successful current-worker
weights total1,118,381,408B; archival alone cannot close that gap even with zero
archive bytes. Remaining unclassified research_state weights total707,285,174B,
an upper bound, not deletion permission. Preserve failures and every protected
scientific checkpoint/archive. Next prepare the minimal finite pilot/retention
source and audit exact remaining eligible recovery; no partial-panel launch or
relaxed guard. If recovery cannot meet the full bound, escalate the remaining
byte deficit once with a concrete storage option while finishing independent work.

## Finite pilot integration ready; exact recovery completed

The additive `research/ffn_norm_campaign_v1` preserves all462 accepted runtime/
worker sources. It prepares two20-step pilots at LR0.0025, with8 H200 snapshots,
and a complete prospective quality design: six1,200-step screens at three equal
LRs per arm, then three fresh paired seeds. Selection remains per-arm minimum
full-validation H200 CE/lower-LR tie. Adoption requires mean RMS-minus-none below
-0.02, all three seeds improving and all frozen text-block upper bounds below0.
Pilot loss never selects; the actual quality freeze follows resource evidence.

retention-boundary001 passed the exact successful-pilot and four-losing-screen
small-artifact boundaries, including failure/fresh/incomplete/alias and selection
rejections. pilot-integration001 independently re-audited both actual generated
worker payloads and exact MPS full validation under the additive source binding;
both20-step configurations, target coverage, CLI entry and exact complete-panel
capacity thresholds passed. Its5.333s guard observed peak RSS1,303,003,136B and
MPS driver1,112,244,224B. No scientific training or seeds were consumed; no pilot
or quality plan is frozen. Do not rerun either closed prerequisite.

After these dependencies closed, four successful worker weights were losslessly
archived with488,444,874B net recovery and27,066 survivors verified. A separate
exact original-review audit archived220 historical successful numerical fixtures,
recovering155,695,215B with27,432 survivors verified. Four unreviewed/failed
fixtures totaling3,686,774B remain loose. The historical apply wrapper fixes the
review-directory set before removal so deleted numerical weights cannot remove
metadata from the final survivor inventory; it ran before any cleanup attempt.
Original reviews/configs/hashes and all protected scientific artifacts remain intact.

capacity002 now observes10,279,190,528B free against13,555,990,528B required,
a remaining3,276,800,000B deficit. This includes eight560MiB pairs,
256MiB complete metadata/transients and8GiB reserve. Both authorized recovery
operations are complete; no further known eligible loose numerical weights remain.
Independent pilot-source/design work is finished. No worker is active. Wake on
actual free space meeting the full bound, new authorized storage evidence, or an
hourly check; remain quiet while unchanged. The already reported storage request
is not reissued. After capacity recovery, verify source/data/unused seed identity,
freeze and execute the two guarded pilots, then prepare the final quality runner.
No partial panel, smaller confirmation set or relaxed guard is permitted.


## Storage recovered and frozen resource pilots completed

The renewed human authorization retired exactly three obsolete numerical
archives, reclaiming5,443,835,089B with34,575 retained files verified unchanged.
Their21 successful historical numerical payloads cannot be restored; original
sources, reviews and hashes remain immutable. Selected/fresh/failed models and
all other archives are unchanged. See obsolete-numerical-cleanup-001/.

Source continuity, data identity and reserved-seed non-use were rechecked before
freezing pilot001. Both20-step workers then passed independent parent acceptance:
20,480targets and8 H200 snapshots each,88/92 EMA tensors for none/RMS.
Worker guard times were35.714/41.472s; maximum sampled RSS1,467,858,944B and
MPS driver2,510,094,336B remained below unchanged limits. Paired trained
full/cache/reset inference also passed. Pilot losses do not select an arm or LR;
no scientific quality result is claimed. Pilot raw/H200 weights total1,118,391,708B
and stay retained until dependent final quality-source acceptance closes.

capacity003 records14,676,287,488B free after both pilots, already above the full
13,555,990,528B quality requirement before any pilot retirement. The old capacity
wait is resolved. No worker is active at this completed boundary. Next prepare
only additive finite quality orchestration, verify both real pilot payload/cache
paths, retire exact successful pilots after dependencies close, then freeze six
1,200-step equal-LR screens plus three fresh pairs. All earlier sources/campaigns
stay frozen/closed; no pilot-loss selection or partial-panel launch. Continuation
returns to five minutes for concrete quality preparation and worker monitoring.


## Finite quality001 frozen and scientific screening started

The additive `research/ffn_norm_quality_v1` passed74 integration tests plus
independent actual pilot raw/H200/optimizer/RNG/full-validation and batch1/8
cache checks. It preserves all467 predecessor sources and binds477 total source
files. None/RMS acceptance requires88/92 EMA tensors. A20-step pilot bundle cannot
impersonate a1,200-step quality endpoint. No pilot loss selects LR or architecture.

After quality-source dependencies closed, both successful pilot pairs were
independently re-audited and exactly four weights totaling1,118,391,708B retired.
All26,752 other research files rehashed unchanged.
Original pilot receipts remain immutable; quality001-pilot-retirement manifests
explain absent pilot paths. Failed, selected and fresh models remain protected.

quality001 freezes six1,200-step equal-LR screens at0.000625/0.00125/0.0025,
then three fresh pairs using per-arm minimum full-validation H200 CE/lower-LR tie
selection. Each endpoint has1,228,800targets and41 H200 snapshots. Exactly four
losing screen pairs retire only after all six acceptance audits and saved selection;
both selected screen pairs and every fresh pair remain retained. Three paired
inference reviews and independent final statistics are required. The RMS-minus-none
mean must be below-0.02, all three paired deltas negative and every simultaneous
text-block upper bound negative; no cost-only adoption or test scoring.

Freeze SHA: `4a10df86ada4f27e89911f6ee231946713e277f9c3eb5f33b683f09e8fbdad26`. Actual free15,810,514,944B exceeds the complete
13,555,990,528B requirement: eight560MiB pairs,256MiB metadata/transients and
8GiB reserve. Streaming publication adds no full-payload duplicate. Slot00 was
dispatched using the exact freeze, one local worker and four threads. Monitor actual
process/anchor state before any later dispatch. No final quality claim is available.


## FFN product-normalization screening complete

All six1,200-step endpoints passed independent checkpoint, full-validation and
cache acceptance. Both arms selected LR0.00125 using the frozen minimum-CE rule.
Screen H200 CE (none/RMS) was4.9243453/4.8871934 at0.000625,
4.7064604/4.7193965 at0.00125, and4.7702461/4.7424677 at0.0025.
These reused-seed screens select rates, not architecture adoption. The selected
RMS recipe trails none by0.0129361 on this screen; all three fresh pairs remain
required regardless of that direction.

The frozen boundary independently re-audited all six endpoints and retired only
eight weights from the four losing screen pairs, reclaiming2,236,787,128B.
All72 retained files, including both selected raw/H200 pairs (slots02/03),
verified unchanged. Original receipts stay immutable; absent loser paths resolve through
ffn-norm-preparation-001/quality001-screen-retention/. Portable exact screen
metadata is saved in quality001-screen-evidence/. The first fresh confirmation
is dispatched with all guards; all fresh raw/H200 pairs will remain protected.


## FFN product-normalization campaign closed: retain no product norm

All twelve independently accepted1,200-step endpoints, all three paired full/cache/
reset inference reviews, and independent final statistics are complete. Both arms
selected peak LR0.00125 from the same three-rate screen. Fresh H200 CE:

| Fresh seed | No product norm | Product RMSNorm | RMS minus none |
|---|---:|---:|---:|
| 198740011 | 4.714070902 | 4.713468354 | -0.000602548 |
| 198740012 | 4.726630458 | 4.734607213 | +0.007976755 |
| 198740013 | 4.732428258 | 4.726370305 | -0.006057953 |

Mean RMS-minus-none delta is+0.000438751: no quality gain. One of three seeds
regressed, and all simultaneous conditional text upper bounds are positive
(0.005718593/0.006452445/0.006978979 for2048/4096/8192-token blocks).
The mean-gain, every-seed and text-bound gates all fail. Retain no FFN-product
normalization, score0.5/SwiGLU672/SiLU/block RMSNorm/H200 on D256 MLA.

Shared-screen effects changed with LR and did not yield a confirmed improvement
after equal tuning. This does not identify an amplitude/stability mechanism.
Intervals condition on three trained seeds and reused validation text; they do
not establish population-seed uncertainty or independent holdout confidence.
All three trained cache/reset reviews passed at batches1/8. Timing is descriptive
(two resident models; allocation/loading excluded), with no cost-only adoption.

Final review and closure are in ffn-norm-preparation-001/quality001-final-review.json
and quality001-closure.json. All477 frozen sources and retained checkpoints were
rechecked. The run root has terminal STOP; do not rerun its training, boundary,
inference or final-review commands. Selected screens02/03 and all six fresh
raw/H200 pairs remain protected. Original receipts and failed attempts are intact.

At this research boundary, causal local mixing and bounded warmup/peak-LR tuning
were reviewed. Select warmup100 versus none for prerequisites only, with equal
bounded peak-LR tuning; local mixing needs explicit placement/cache-history design
and remains deferred. See TOKENIZER_WARMUP_PREPARATION.md. No new scientific seed
or panel is frozen. The small generated schedule/toy-resume prerequisite passed;
full-model Muon/AdamW/raw/H200/cache and real-worker prerequisites remain.
