# Tokenizer gated versus ungated SiLU FFN preparation

Status: architecture hypothesis selected; integration prerequisites remain.
The renewed obsolete-screen cleanup resolves the current capacity projection
(see latest receipt below). No scientific seeds or launch plan are frozen.

## Question and controls

Compare the adopted width256 four-layer RoPE MLA SwiGLU FFN with a bias-free
ungated SiLU FFN at exactly the same matrix-parameter budget. Keep RMSNorm,
serial unit residuals, final affine LayerNorm, untied GPT2 tables, grouped-embedding
v2 numerical policy, context128, batch8,1200updates, cooldown800 and H200 fixed.
The completed EMA study does not justify changing the H200 endpoint.

Control: W_down[SiLU(W_gate x) * W_up x], hidden672, three matrices.
Candidate: W_down[SiLU(W_up x)], hidden1008, two matrices.

Both have516,096 matrix parameters per layer:3*256*672 =2*256*1008.
Four FFNs therefore retain2,064,384 parameters, and total model parameters would
remain28,492,416. The candidate uses a1.5-times wider hidden representation
in exchange for removing the multiplicative branch. This changes factorization,
hidden width, activation statistics and optimizer geometry together. It keeps
SiLU fixed; it is not a repeat of the completed SiLU-versus-GELU gate study or
an isolated test of multiplication. Equal parameter/matrix-multiply counts do
not guarantee equal wall time, activation memory or optimization behavior.

This is a distinct unresolved subword architecture question. Historical
residual attenuation is a weaker next candidate: fixed0.1 and learned0.1 gains
worsened every fresh seed at widths64/256, and no new MLA-specific instability
has been demonstrated. Parallel residual results were also negative. MoE remains
deferred on its18.782GB full-panel requirement, with its wider-dense control
retained. No closed campaign or failed historical numerical fixture is reopened.

## Bounded proposed panel

Two20-update resource pilots; three equal LR choices0.000625/0.00125/0.0025
per arm on a screening seed; select each arm's minimum H200 full-validation CE
with lower-LR tie break; then both selected recipes on three new paired seeds.
Six screens plus six fresh runs, each1200updates. No grid extension or alternative
winner. Keep both selected screen pairs and every fresh raw/H200 pair.

Proposed adoption rule: candidate mean paired H200 CE delta below-0.02, every
fresh seed below0, all simultaneous conditional-text upper bounds below0 for
2048/4096/8192-token blocks with2000 common paired PCG64 draws and linear
quantile1-0.05/3. Freeze exact seeds, ordering and all rules before results.
No cost-only adoption route. Reused validation text is not an independent test
holdout or population-seed confidence. Test scoring stays closed.

## Capacity finding before implementation

CPU construction validated the unchanged baseline's28,492,416 parameters,
65,536 persistent-buffer bytes and optimizer groups:2,752,512 Muon parameters
in44tensors;25,739,904 Adam parameters in44tensors. Candidate FFN components
have the exact proposed shape/count. A full candidate is not yet supported or
validated. Its intended Muon tensor count is40, with unchanged parameter count;
all Adam moments and Muon momentum are charged in the projection.

Raw model + one H200 shadow + H200 export + optimizer payload lower bound is
558,969,520B per trajectory. The conservative allowance is788,953,140B,
including inherited envelope/headroom; final serialization and transient audits
must independently validate it. With prospective audited pilot/losing-screen
cleanup, the high-water stage retains two selected plus six fresh pairs:
6,311,625,120B, plus256MiB metadata and8GiB reserve = **15,169,995,168B**.
Pre-cleanup observed free space was10,181,873,664B, a4,988,121,504B shortfall.
Even six fresh payload pairs alone plus the reserve require11,943,751,712B;
dropping pilots/screens cannot make the full comparison fit in the observed space.
No protected-weight deletion, reduced confirmation or partial-panel start is
credited. The successful LayerNorm pilot pair is being separately audited under
the retention policy; any actual reclaimed space is recorded in its receipt.

## Concrete numerical prerequisite and limitations

Standalone ungated SiLU forward values and input/up/down gradients passed an
independent NumPy analytic-derivative oracle for three cases, including the
actual256-by1008 component. Maximum absolute discrepancy was3.553e-15 in
float64 CPU. No checkpoint, training corpus, test data or scientific trajectory
was used. This is component arithmetic evidence only, not full-model correctness,
MPS behavior, resume equivalence or quality evidence.

The first shape-only attempt failed before the oracle: existing normalization
validation rejects aliased storage pointers, and all meta tensors report pointer0.
Its original source, failed guard and failure decision remain immutable. A distinct
CPU construction attempt used real storage and passed without modifying or bypassing
any validator. The corrected audit took0.502seconds and peaked at447MB RSS.

The current policy correctly rejects the proposed FFN. New integration must live
outside all399 frozen source files and bind the exact module layout, initialization,
shape-dependent Muon scaling, optimizer coverage, checkpoint/resume and H200
identities. Require actual forward/gradient/MPS/cache parity, full-model resource
and serialization audits, then seed census and finite pilot/quality freezes.
Do not build another generic ledger or materialize large fixtures while the full
retained-output projection fails. Safe source inspection and small component
checks can continue while capacity is unresolved.

Evidence: tokenizer-evidence/ffn-structure-preparation-001/review002.json,
review002.py and guard; review.py, review-guard.json and review001-failure.json
preserve the original unsuccessful meta audit. Historical source-document hashes
are bound to the audit's recorded commit, before subsequent documentation updates.


## Authorized cleanup completed; remaining capacity blocker

Exact retention audit removed only the successful LayerNorm pilot002 raw/H200
pair,559,220,996B. The closed normalization campaign and dependent inference
were fully reviewed before deletion. Failed RMS pilot001, all selected/fresh
weights and the explicitly retained EMA pilot remain protected. All234 retained
checkpoint/archive files and25,974 non-checkpoint metadata files were rehashed
unchanged after removal. Original pilot receipts stay immutable; the deleted
weights cannot be reloaded or re-audited and closed controllers must not rerun.

The first audit stopped before mutation because its combined inventory exceeded
the inherited2MiB receipt cap. Original source and failed guard are preserved.
The completed v2 audit used bounded inventory parts while retaining that cap;
its exact two-file allowlist, per-removal intent/journal and completed guard are
under norm-pilot-retention-v2/. No other file was deleted.

Current free space: **10,740,375,552B**. Full proposed panel requires
**15,169,995,168B**, leaving **4,429,619,616B** additional capacity needed.
No next training launch is authorized by this preparation alone. Resolve safe
capacity or an appropriate output volume before full-panel dispatch. Keep all
three fresh confirmation pairs, both tuned controls and the8GiB reserve; do
not change scope solely to fit. Do not repeat unchanged space polling or add
metadata-only implementation work while this prerequisite remains unresolved.


## Renewed old-checkpoint retirement completed

The user explicitly authorized removing old checkpoints. The narrow exception
retired only two obsolete wider-hybrid screening archives (26 completed endpoints),
reclaiming 9,828,356,231 logical bytes. No selected, fresh-confirmation,
failed, or numerical-fixture checkpoint was removed. All 232 surviving checkpoint/
archive files and 25,974 non-weight evidence files were rehashed unchanged.
Original archive/member hashes and results remain; the retired weights can no
longer be restored or re-evaluated.

Free space at completion was 20,517,961,728 bytes, above the proposed
15,169,995,168-byte high-water requirement by 5,347,966,560 bytes.
The previous capacity blocker is resolved against that projection. Revalidate
actual serialization/transient needs before freezing or dispatching a panel.

The earlier restriction on all implementation while capacity was short was too
broad. Capacity constrains disk-consuming dispatch, not bounded source or CPU
prerequisites. Continue narrow full-model FFN integration and numerical/cache/
optimizer/checkpoint prerequisites outside the frozen sources. No new scientific
seeds, plan freeze or training launch has occurred. AGENTS.md now requires active
blocker recovery and distinguishes blocked actions from independent useful work.

Evidence: tokenizer-evidence/obsolete-screen-cleanup-001/manifest.json,
completed.json, journal.jsonl and both resource guards.


## Full-model CPU and strict MPS integration passed

The isolated research/ffn_structure_v1 runtime now accepts exactly the matched
SwiGLU672 and ungated SiLU1008 arms on the grouped-embedding D256 MLA backbone.
It overrides six runtime modules and resolves unchanged dependencies from the
frozen norm runtime under its own namespace. All399 frozen source files remain
byte-identical; the tested new source snapshot is preserved separately.

The control matches legacy initialization and logits bitwise. Candidate total
parameters are28,492,416, with2,064,384 FFN parameters. Complete non-duplicated
optimizer coverage passed: Muon2,752,512 parameters/40tensors and AdamW
25,739,904/44tensors, retaining width-LR scale0.5 and match_rms_adamw. Full-model
synthetic forward/backward and one CPU update passed with finite gradients.
CPU cached/full maximum error was8.345e-7; causality was exact.

Strict MPS tests passed for both arms with repeated token IDs exercising grouped
embedding gradients. Maximum CPU/MPS logits error8.941e-7, gradient error1.500e-7,
and full/interleaved-cache error9.537e-7. Reset and causality were bitwise exact.
Cache remains196608B/item. Existing tolerances and resource guards were retained.
Wrong-arm cache and checkpoint signatures, a changed activation module and use
of the candidate under the old policy were rejected. No corpus/test/checkpoint
file IO or scientific training occurred. Synthetic fixture seeds are not reserved
scientific seeds; the future census must include these sources.

Candidate initialization uses ordinary bias-free Linear draws. Removing a
projection and changing its shape changes RNG consumption, so paired scientific
seeds will not imply identical attention/readout initial values. The comparison
changes factorization, hidden width and optimizer geometry jointly.

Next: real serialized raw/H200 checkpoint and optimizer-resume equivalence,
worker/parent full-validation and inference acceptance; measure actual output
allowance/transients before seed census and finite pilots/quality freezes.
Current capacity remains above the15.170GB projection. No quality claim or
training launch is supported by these numerical prerequisites.

Evidence: model-integration001.py/json/guard/source and
mps-integration001.py/json/guard in tokenizer-evidence/ffn-structure-preparation-001/.


## Serialized resume, actual trainer, parent validation and inference passed

New generated-text fixtures exercised the real GPT2 loader, batch8/context128
paired512 sampling, Muon/AdamW, cooldown and H200 on strict MPS. For both arms,
three uninterrupted updates, a fresh repeat and interrupted file-backed replay
agreed bitwise in model, populated optimizer and EMA state, with exact sampling
and CPU/MPS RNG restoration. Six raw/H200 files were written exclusively under
600MiB raw/128MiB export caps. H200 parameters exactly match the saved average;
the raw parent and RNG were unchanged by export. Wrong optimizer/group order,
missing MPS RNG and inference-only resume were rejected before mutation.
Loaded H200 CPU/MPS and cached inference passed at batches1/8.

Actual run_training then ran two generated-text updates per arm in serial child
processes, with parent/run locks, bounded sinks and resource guards. The first
fixture failed before updates because a two-step cooldown incorrectly started at1.
The distinct second fixture used valid start0 and completed control training and
export, then exposed a namespace bug: reused validation_data called the frozen
normalization architecture validator. Original sources/logs/guards/configs remain
immutable. The corrected FFN validation_data uses the new runtime namespace,
without changing the frozen implementation. Recovery003 accepted the existing
control raw/H200 pair without retraining it and ran only the new candidate child.
No old closed campaign or historical failed numerical fixture was reopened.

Independent parent raw-resume preflight and exact H200 transport passed for both.
Parent MPS scoring reproduced every worker report exactly across3014 generated
validation targets in24windows (including70tail targets). CPU/MPS CE differences
were3.098e-7(control) and2.769e-7(candidate). The read-only view constructs only
train/validation splits. A deliberately incorrect target count was rejected.

The actual trained-pair inference path passed all cache/reset checks and the
predeclared alternating2warmup/8repeat measurement sequence at batches1/8. Cache
is196608B/item for either arm. These two-update generated-text timings validate
measurement mechanics only; they establish neither quality nor a deployment gain.
No research corpus or test holdout was read, and no scientific seeds are reserved.

All399 frozen sources remain unchanged. All10 newly produced fixture checkpoints
are retained (3,126,780,502B), including the control pair from failed validation002.
About17.17GB remains free versus the15.170GB proposed full-panel requirement.
Observed raw/H200 sizes remain within the existing788,953,140B trajectory allowance;
revalidate full-duration metadata and all transient overlaps before scientific
freeze. Do not count speculative fixture deletion as available capacity.

Next implement the narrow finite pilot/config/serial acceptance path using these
proven primitives, perform seed census including all synthetic fixture seeds,
freeze exact pilot configs/source/provenance/resource limits, then run guarded
resource pilots and freeze the finite quality panel. No ledger expansion or
new model hypothesis is required.

Evidence: checkpoint-resume001.*, trainer-worker001/002 failure decisions and
original evidence, trainer-worker003.*, trained-inference001.*, and
transport-worker-inference001-review.json/evidence in
tokenizer-evidence/ffn-structure-preparation-001/.


## Two scientific-corpus resource pilots frozen

The new finite pilot path binds exactly SwiGLU672 then ungatedSiLU1008,20updates
each at LR0.0025, cooldown13 and H200 every update from13. It uses the established
scientific corpus/prior/tokenizer and full-validation target identity. Nineteen
config/output/slot/deadline boundaries passed; no test tokens were encoded during
preparation. The unchanged training loader will materialize test tokens but never
score them. Independent parent acceptance repeats the existing payload arithmetic
through the FFN runtime and recomputes the complete worker validation report.

A census scanned41,516 recorded text files (3,746,599,319B), including all recent
synthetic fixtures, with no collision. Reserved pilot187620001, screen187620002,
fresh187620011/12/13, evaluation187620003 and bootstrap187620004. No quality
training is authorized by this pilot freeze. Failed/partial attempts stop the root
and preserve original artifacts; there is no automatic retry.

The revised projection retains both pilot pairs without cleanup credit. Peak
retention is ten pairs: both pilots, two selected screens, six fresh endpoints.
Each trajectory retains788,953,140B allowance, exceeding simultaneous capped
600MiB raw +128MiB H200 direct writes +16MiB bounded metadata/logs. No full-size
temporary copy is made. With256MiB round metadata and8GiB reserve, the initial
requirement is16,747,901,448B;17,164,787,712B was available. Four losing screen
pairs may only be removed after separately tested, exact audited selection and
retention. That boundary and full-duration output acceptance must precede quality
freeze. No failed, selected, fresh or fixture deletion is credited.

Evidence: scientific-seed-census-001/, pilot001-capacity/boundary-tests/plan/
freeze/preflight/prepare-guard.json, prepare-pilot001.py and pilot001-source/.
Next execute the two guarded pilots, then trained-pair inference and final quality
preparation. These resource pilots are excluded from quality conclusions.


## FFN scientific resource pilots and paired inference completed

Both frozen20-update pilots passed real child training and independent parent
acceptance. Each processed20,480 targets and saved8 H200 snapshots. Raw resume,
optimizer/EMA/export/metric/schedule/tokenizer checks passed; independent parent
scoring reproduced the full34,278-target validation report exactly. These short
pilots are excluded from LR selection and all quality/adoption conclusions.

| Arm | Training seconds | Worker seconds | Parent-inclusive seconds | Worker peak RSS | Worker peak MPS driver |
|---|---:|---:|---:|---:|---:|
| SwiGLU672 | 3.614 | 10.710 | 16.492 | 1.437GB | 2.502GB |
| Ungated SiLU1008 | 3.499 | 10.285 | 16.234 | 1.484GB | 2.502GB |

Both strict resource guards and clean process reaping passed. Linear scaling
puts training alone around210–217seconds per1200-update run; this is only a
rough estimate and does not replace the7,200-second full-trial guard.

Trained H200 full/cache/reset checks passed at batches1/8 before alternating
2warmup/8repeat timing. Median32-token prefill /64-token decode milliseconds:
SwiGLU8.058/279.251 versus ungated8.427/278.982 at batch1;
SwiGLU13.919/294.506 versus ungated13.768/284.885 at batch8. Cache is identical
196,608B/item. This single short-pilot pair is descriptive, with two models
resident and loading/cache construction excluded; no consistent deployment or
quality advantage is established and there is no cost-only adoption route.

All four pilot weight files remain retained, totaling1,118,322,228B. There was
no deletion or test scoring. Actual free space16,045,850,624B exceeds the
remaining conservative full-panel requirement15,629,411,630B by416,438,994B.
This accounts for both retained pilots, full serialization limits and metadata
allowances; future dispatch must recheck capacity. No fixture deletion credit.

Next connect/test only full-duration finite quality acceptance and the exact
losing-screen retention boundary, then freeze the complete3LR-per-arm screen
and3freshpaired-seed panel. Keep all pilot sources immutable; additive quality
work must use a new namespace. Neither accepted pilot is rerun.
Evidence: ffn-structure-preparation-001/pilot001-review.json, both external
pilot001-accepted anchors, pilot001-completed-evidence/ and pilot001-inference-*.


## Finite FFN quality plan frozen

The additive research/ffn_quality_v1 path reuses the established bounded serial
controller, independent full-validation/cache acceptance, paired-statistics and
exact screen-retention logic. It binds FFN configurations and88/84 EMA parameter
inventories without editing any frozen pilot or historical source. Seventy-six
checks passed, including finite orders/ties/adoption arithmetic, all12config
slots, disposable-only cleanup/corruption/re-entry rejection, and real read-only
parent acceptance of both retained scientific pilots. No scientific weights
were deleted.

Integration001 passed all76tests but its external report serialization rejected
pytest tuples. Original source/guard/plan/disposable receipts are preserved.
Distinct002 normalizes report properties to native JSON lists, records its log,
and passed all76checks with the protected pilot roots unchanged. No model or
scientific training retry occurred.

Frozen quality001 contains six1200-update screens at0.000625/0.00125/0.0025,
shared screening seed187620002, alternating arm order. Each arm selects minimum
full-validation H200 CE with lower-LR tie-break. Save selection after all six
accepted endpoints; independently re-audit all six and retire only four losing
raw/H200 pairs through exact manifests/journals before any fresh run. Keep both
selected pairs, both resource pilots and every fresh raw/H200 pair. Then run
three fresh paired seeds187620011/12/13 at the fixed selected rates.

All quality runs retain cooldown800, H200 from800 every10updates, batch8/context128
and1,228,800 targets. Adoption requires mean ungated-minus-gated delta<-0.02,
every fresh delta<0 and all simultaneous conditional-text uppers<0 at the three
frozen block sizes. All three paired inference measurements and a separate final
review remain required; no interim or cost-only adoption.

Full remaining capacity requirement is15,629,411,630B versus16,034,213,888B free.
The bounded checkpoint writer streams to a temporary inode then publishes a hard
link and unlinks the temporary name under the census lock; it makes no second
physical payload copy. Aggregate raw/H200 cap728MiB plus bounded metadata fits
the788,953,140B per-slot cap;256MiB round metadata covers control overlap. The
conservative earlier pilot headroom remains reserved. No speculative deletion
credit or relaxed guard. First quality screen is next.

Evidence: quality-integration001-failure.json, quality-integration002-review.json,
quality-retention-disposable001/002/, quality001-plan/freeze/preflight/capacity, and quality001-protected-pilot-*.


FFN quality001 slot00 accepted (2026-11-01T04:13:05.206410+00:00): SwiGLU672, seed187620002,
LR0.000625,1,200 updates/1,228,800 targets and41H200 snapshots. Full-validation
H200 CE is **4.920534463529** over34,278 targets. Independent parent
raw/optimizer/EMA/export/full-validation/cache audits passed; the child exited
cleanly and all guards finished. Training took225.11seconds; worker265.68seconds
and parent acceptance10.78seconds. Worker peak RSS1.574GB/MPS driver2.510GB.

This first fixed screen supports no selection or comparative quality conclusion.
Next is frozen slot01, ungatedSiLU1008 at the same seed and LR. No checkpoint
deletion or test scoring occurred. About15.50GB is free versus15.07GB remaining
conservative requirement. Both pilot pairs and the accepted screen pair remain
retained. Evidence: quality001-slot00-result.json/evidence/ and
quality001-accepted-00.json in ffn-structure-preparation-001/.


FFN quality001 slot01 accepted (2026-11-01T04:19:11.075544+00:00): ungatedSiLU1008, screening
seed187620002, LR0.000625,1,200 updates/1,228,800 targets and41H200 snapshots.
Full-validation H200 CE is **4.946103441823**, versus SwiGLU672
**4.920534463529** at the same seed/rate: candidate minus control
**+0.025568978294 nat/token**. This first screening pair favors SwiGLU.
It does not settle the independently tuned, fresh-seed comparison or authorize
interim selection, stopping, new rates or a grid extension. Hidden width and
factorization change jointly; this is not an isolated gating-mechanism result.

Independent parent raw/optimizer/EMA/export/full-validation/cache audits passed,
as did clean child exit and all guards. Worker268.54seconds, parent12.04seconds;
worker peak RSS1.579GB and MPS driver2.544GB. Both screen pairs and both pilots
remain retained. No deletion or test scoring occurred. About14.95GB free exceeds
14.51GB remaining conservative requirement. Next frozen slot02 is ungatedSiLU
at LR0.00125, followed by SwiGLU at that rate. Evidence: quality001-slot01-result.json, quality001-slot01-evidence/ and quality001-accepted-01.json in ffn-structure-preparation-001/.


FFN quality001 slot02 accepted (2026-11-01T04:25:21.000319+00:00): ungatedSiLU1008 at
LR0.00125, screening seed187620002,1,200 updates/1,228,800 targets and41H200
snapshots. Full-validation H200 CE **4.722775215414** improves over this
arm's LR0.000625 screen (**4.946103441823**). The paired SwiGLU
endpoint at0.00125 is next; no rate selection or architecture conclusion is
available from this incomplete screen. The remaining frozen rates are unchanged.

Independent parent raw/optimizer/EMA/export/full-validation/cache checks and
clean child exit passed. Training221.93seconds, worker279.39seconds, parent
11.35seconds; peak worker RSS1.558GB/MPS driver2.510GB. All guards passed.
All accepted screen/pilot pairs remain retained; no deletion or test scoring.
Free14.37GB exceeds13.95GB remaining conservative requirement. Evidence:
quality001-slot02-result.json, quality001-slot02-evidence/ and
quality001-accepted-02.json in ffn-structure-preparation-001/.


FFN quality001 slot03 accepted (2026-11-01T04:31:25.581805+00:00): SwiGLU672 at LR0.00125,
screening seed187620002,1,200 updates/1,228,800 targets and41H200 snapshots.
Full-validation H200 CE **4.712840888916** versus ungatedSiLU1008
**4.722775215414** gives candidate-minus-control
**+0.009934326498 nat/token**. Both completed rate pairs favor SwiGLU
on this screening seed. The0.0025 pair and three fresh confirmations remain;
there is no interim rate selection, adoption, early stopping or grid change.

Independent parent raw/optimizer/EMA/export/full-validation/cache audits, clean
child exit and all resource guards passed. Training220.39seconds; worker286.09
seconds and parent10.04seconds. Worker peak RSS1.572GB/MPS driver2.510GB.
All pilot/screen checkpoint pairs remain retained; no deletion or test scoring.
Free13.80GB exceeds13.39GB remaining conservative requirement. Next is frozen
slot04, SwiGLU at LR0.0025. Evidence: quality001-slot03-result.json,
quality001-slot03-evidence/ and quality001-accepted-03.json in
ffn-structure-preparation-001/.


FFN quality001 slot04 accepted (2026-11-01T04:39:27.180350+00:00): SwiGLU672 at LR0.0025,
screening seed187620002,1,200 updates/1,228,800 targets and41H200 snapshots.
Full-validation H200 CE **4.803416296113** over34,278 targets is worse
than this arm's LR0.00125 result **4.712840888916**. All three SwiGLU rates
are now measured; the ungated LR0.0025 endpoint remains before the frozen
six-screen selection boundary. No interim selection or architecture adoption.

Independent parent raw/optimizer/EMA/export/full-validation/cache audits, clean
child exit and all guards passed. Training217.31seconds; worker289.46seconds,
parent10.08seconds. Peak worker RSS1.584GB/MPS driver2.510GB. All pilot and
screen pairs remain retained; no deletion or test scoring occurred. Observed
free13.05GB exceeds the12.83GB remaining conservative requirement; recheck
before dispatch. Next is frozen slot05, ungatedSiLU1008 at LR0.0025. After all
six endpoints pass, save per-arm rate selection and execute the exact audited
losing-screen retention boundary before fresh confirmations. Evidence:
quality001-slot04-result.json, quality001-slot04-evidence/ and
quality001-accepted-04.json in ffn-structure-preparation-001/.


## FFN six-screen selection and retention complete

Slot05 accepted (2026-11-01T04:45:54.354657+00:00): ungatedSiLU1008 at LR0.0025,
screening seed187620002,1,200 updates/1,228,800 targets and41H200 snapshots.
Full-validation H200 CE is **4.749980863365**, versus SwiGLU's4.803416296113
at that rate (candidate minus control **-0.053435432748 nat/token**).
Independent parent raw/optimizer/EMA/export/full-validation/cache checks, clean
child exit and all guards passed. Training213.73seconds; worker291.48seconds,
parent10.09seconds; peak worker RSS1.583GB/MPS driver2.552GB.

All six accepted full-validation screens:

| Peak LR | SwiGLU672 | UngatedSiLU1008 | Candidate minus control |
|---|---:|---:|---:|
| 0.000625 | 4.920534463529 | 4.946103441823 | +0.025568978294 |
| 0.00125 | 4.712840888916 | 4.722775215414 | +0.009934326498 |
| 0.0025 | 4.803416296113 | 4.749980863365 | -0.053435432748 |

The frozen minimum-H200-CE rule selects **LR0.00125 for both arms**, persisted
before any fresh run. The ranking reversal at0.0025 shows why the independently
tuned comparison matters: the selected-rate screen favors SwiGLU by0.009934,
although ungated SiLU wins at the highest fixed rate. This is screening evidence
from one seed, not a fresh confirmation or adoption conclusion. Width and
factorization change jointly, so it does not isolate the gating mechanism.

The boundary independently re-audited all six pairs, saved selection, then removed
exactly eight allowlisted raw/H200 files from losing slots0,1,4,5, reclaiming
**2,236,646,504 logical bytes**. The selected slots2/3, all71 non-deleted files in
the quality evidence inventory and both pilot inventories verified unchanged.
Manifest, fsynced intent/deletion journal, guard and receipt are preserved in
quality001-screen-retention/. Deleted weights cannot be reloaded; original
hashes, complete audits and all non-weight evidence remain. No test scoring.

Post-boundary verification passed: free14,753,619,968B exceeds the remaining
14,511,088,378B conservative requirement. Next is slot06 SwiGLU at LR0.00125,
fresh seed187620011, followed by the paired ungated endpoint. Complete all six
fresh endpoints at frozen rates, all three paired inference reviews and final
independent statistical review. Keep every fresh raw/H200 pair. Evidence:
quality001-slot05-result.json/evidence/, quality001-accepted-05.json,
quality001-selection.json and quality001-boundary-result.json/guards in
ffn-structure-preparation-001/.


FFN quality001 slot06 accepted (2026-11-01T04:53:57.716984+00:00): first fresh SwiGLU672
endpoint, seed187620011, frozen selected LR0.00125. It completed1,200 updates,
1,228,800 targets and41H200 snapshots. Full-validation H200 CE is
**4.700707486073** over34,278 targets. The paired ungated endpoint is pending;
no paired fresh delta, adoption, rate reselection or interim stopping follows.

Independent parent raw/optimizer/EMA/export/full-validation/cache audits, clean
child exit and all guards passed. Training214.88seconds; worker296.82seconds,
parent10.10seconds; peak worker RSS1.623GB/MPS driver2.510GB. This fresh raw/H200
pair is permanently protected under the campaign retention policy. Both selected
screen pairs, both pilots and all existing evidence remain intact. No deletion
or test scoring occurred. Free14,178,803,712B exceeds the remaining13,951,923,794B
projection. Next is slot07 ungatedSiLU1008, same fresh seed and LR. Complete the
three frozen fresh pairs, all paired inference checks and final review. Evidence:
quality001-slot06-result.json/evidence/ and quality001-accepted-06.json in
ffn-structure-preparation-001/.


FFN quality001 slot07 accepted (2026-11-01T05:00:27.705613+00:00), completing the first fresh
pair at seed187620011 and frozen LR0.00125. UngatedSiLU1008 H200 validation CE
is **4.714475188678**, versus SwiGLU672 **4.700707486073**; candidate minus
control is **+0.013767702605 nat/token**, favoring SwiGLU. This first fresh
pair agrees in direction with the selected-rate screen. It is not a completed
three-seed comparison; no interim stopping, reselection or grid extension.

The ungated endpoint completed1,200 updates/1,228,800 targets and41H200 snapshots.
Independent parent raw/optimizer/EMA/export/full-validation/cache checks, clean
child exit and all guards passed. Training212.26seconds; worker298.93seconds,
parent10.02seconds; peak worker RSS1.636GB/MPS driver2.510GB. Both first-fresh
raw/H200 pairs remain protected, as do both selected screens and pilot pairs.
No cleanup or test scoring occurred. Free13,604,524,032B exceeds13,392,765,126B
remaining conservative requirement. Next is slot08 ungatedSiLU1008 at
seed187620012/LR0.00125, then its SwiGLU counterpart in the frozen alternating
order. Finish all three fresh pairs, all paired inference reviews and final
statistical review. Evidence: quality001-slot07-result.json/evidence/ and
quality001-accepted-07.json in ffn-structure-preparation-001/.


FFN quality001 slot08 accepted (2026-11-01T05:07:04.838084+00:00): ungatedSiLU1008 at
second fresh seed187620012 and frozen LR0.00125. Full-validation H200 CE is
**4.722783352829** over34,278 targets. The same-seed SwiGLU endpoint remains
pending; no second paired delta or final architecture conclusion is available.
The first fresh pair still favors SwiGLU by0.013767702605nat/token.

This endpoint completed1,200 updates/1,228,800 targets and41H200 snapshots.
Independent parent raw/optimizer/EMA/export/full-validation/cache audits, clean
child exit and all guards passed. Training211.54seconds; worker301.42seconds,
parent11.11seconds; peak worker RSS1.583GB/MPS driver2.510GB. Every fresh raw/H200
pair remains protected, along with selected screens and both pilots. No cleanup
or test scoring. Free13,024,657,408B exceeds the remaining12,833,606,458B
projection. Next is slot09 SwiGLU672 at the same seed/rate, then the final fresh
pair, all paired inference reviews and final statistical review. Evidence:
quality001-slot08-result.json/evidence/ and quality001-accepted-08.json in
ffn-structure-preparation-001/.
