# Wider hybrid staged tuning draft — not frozen or launch-ready

October27 Los Angeles / October28 UTC. This selects a prospective preparation
direction using closed prior rounds and anchor geometry, before any new quality
screening. It does not reopen hybrid001, wide001, or the failed combined CPU
continuation fixture. The storage and validation gates in
[the preparation record](TOKENIZER_WIDE_HYBRID_PREPARATION.md) remain binding.

## Question and bounded tuning choice

At width512, does a3:1 KDA/gated-MLA recipe preserve or improve quality against
BOTH independently tuned homogeneous controls? Use three LR candidates per
family, then two additional bottleneck-rank candidates at that family's selected
LR. This is broader than hybrid001's two-rate scan, but is still a small staged
search rather than detailed optimization of every gate, schedule or layout.

Choose rank over the alternative shared horizon or decay-bound probes for this
draft because the anchor hybrid has almost the same state size as all-KDA and
all-gated-MLA at context128. Rank offers an explicit capacity/cache tradeoff in
global layers and a decay-parameterization tradeoff in recurrent layers. The
directions of quality and latency effects are unknown. Keep the independently
motivated horizon/regularization and MoE/activation/norm/residual/depth agenda;
this round does not make attention tuning a permanent priority.

The rank divisor b is4 for the anchor; stage2 candidates are8 and2. KDA decay
bottleneck rank is D/b; gated-MLA query and KV ranks are both D/b. The hybrid
changes these bottlenecks in both component types together. This is a combined
rank recipe, not a causal separation of recurrent versus global rank effects.
KDA recurrent state shape is unchanged by this rank. Gated-MLA latent cache size
changes. The other K3-style choices remain fixed: bounded log-decay−5, initial
A=0, convolution4, normalized Q/K, full-rank sigmoid output gates, NoPE, head32,
shared unrotated key32 and four layers. These are local adaptations, distinct
from Kimi Linear's low-rank output gate. No paper-scale reproduction claim.

Retain legacy untied GPT-2 tables, dense SwiGLU21D/8 and inherited normalizations,
score multiplier.5 for gated-MLA, Muon/AdamW width scaling, decay.00006, batch8,
context128 and paired512 sampling. All quality trajectories use1200 updates,
cooldown800→1200 to.1, H200start800/interval10. These are a fixed comparison
budget, not a proven wider/hybrid optimum. Tokens/updates are matched; parameter
counts, FLOPs, elapsed time and initialization draws are not matched.

## Proposed finite order and selection

No new tokenizer request currently uses the proposed pilot3491, screen3501 or
fresh3511/3512/3513 seeds. Repeat the freshness audit before a final freeze.

Three resource pilots would use rank D/2, the largest parameter geometry in each
family, seed3491 and20 updates, in hybrid/gated_mla/kda order. The draft
pilot configuration uses LR.0025 (the highest declared quality-grid rate),
cooldown start13 through20 to.1, and H200 start13/interval1: eight snapshots
at completed steps13–20. Validation cadence is20. This short schedule exercises
the export path; it is not a quality-comparable H200 window or a stability
guarantee for every LR/rank. These are feasibility checks excluded from selection.
The lower-rank configurations still need their own numerical/cache prerequisites;
the largest-rank pilot does not establish every configuration's resource peak.

Stage1: nine anchor-rank LR screens, seed3501:

| LR | Family order |
|---|---|
| .000625 | hybrid, gated_mla, kda |
| .00125 | kda, hybrid, gated_mla |
| .0025 | gated_mla, kda, hybrid |

Independently select the lowest final H200 validation CE for each family;
exact ties choose lower LR. Save selection before stage2.

Stage2: six rank probes, also seed3501, using each family's selected LR:
D/8 in hybrid/gated_mla/kda order, then D/2 in kda/gated_mla/hybrid order.
For each family compare its selected D/4 anchor with its D/8 and D/2 probes;
select lowest final H200 CE, exact ties choose fewer parameters (larger divisor).
Save final choices before confirmations. Do not re-tune LR for each rank or
expand the grid. LR/rank interactions remain a limitation of this staged design.

Confirmation: nine fresh trajectories, preserving the selected family recipes:

| Seed | Family order |
|---|---|
| 3511 | hybrid, gated_mla, kda |
| 3512 | kda, hybrid, gated_mla |
| 3513 | gated_mla, kda, hybrid |

Total27 retained trajectories:3 pilots+9 LR screens+6 rank probes+9 fresh runs.
All raw/H200 artifacts remain. There is no conditional omission or duplicate
control, and no selecting the easier homogeneous comparator after seeing scores.

## Proposed gates, encoded and synthetically tested before freezing

Primary contrasts are hybrid minus all-KDA and hybrid minus all-gated-MLA on
full-validation H200 CE. Both use the same three fresh seeds. A quality route
requires mean delta<−.02, every seed delta<0 and all declared conditional upper
bounds<0 against BOTH controls. A memory route requires mean delta, every seed
delta and every upper bound<=+.01 against BOTH, plus at least10% smaller actual
allocated total recurrent/convolution/latent/shared-key state against BOTH in
EVERY fresh seed at batches1/8. The+.01/−.02 and10% thresholds are prospective
engineering conventions, not numeric tolerances supplied by the human. A
memory route has no speed-win gate; measured latency tradeoffs must be disclosed.
Missing artifacts, incorrect caches or incomplete trials prohibit adoption.

Use paired text-block bootstrap2048/4096/8192,2000 draws, seed20261028; identical
resamples across models, averaging paired deltas across the three trained seeds.
For the six contrast/block-size upper bounds, allocate.05/6 per bound and use
the corresponding one-sided quantile (99.1667%) with an explicitly tested
quantile implementation. This Bonferroni-style allocation is conditional on
trained seeds and adaptively reused validation. It does not provide independent
test evidence, population uncertainty across seeds, or repair adaptive reuse.
Report ordinary descriptive intervals separately; do not swap the gate bounds.

The pure `src/research/tokenizer_wide_hybrid_rules.py` preparation module now
encodes these selection orders and gate arithmetic. Its bootstrap consumes
aligned per-seed/per-family 2048-token block-loss sums and shared target counts;
it resamples grouped blocks with replacement, weighting the short tail by its
target count. It uses NumPy default_rng(20261028), reset for each block size,
2000 draws and the same draw indices for every seed/model/contrast. Linear
quantile interpolation is fixed at sorted position (n−1)q. Block-size keys are
strings so persisted JSON retains its contract. Twenty-nine synthetic tests
pass, including an independent scalar bootstrap reference, exact thresholds,
both-control requirements, corrupt/incomplete panels and JSON round-trip.
The module does not validate saved tokenizer/config/checkpoint identity, dispatch
workers, audit artifacts or adopt recipes. Those remain mandatory integration
steps. The draft is still unfrozen and no real quality records were analyzed.

The separate saved-score adapter now checks independently expected requests,
training/tokenizer metadata, raw/H200 hashes and meta-device tensor geometry,
plus saved score arithmetic. Its36 synthetic checks pass alongside the29 rules
checks. It supplies selection rows/block sums; it never declares tensor values
audited or a recipe adopted. Frozen controller provenance, full tensor-value
audits and the remaining numerical/serving prerequisites are still required.

After quality and saved-artifact audits, plan three serial fresh-seed inference
triples: FP32 MPS/context128, batches1/8,32prefill/64teacher-forced decode tokens,
two warmups and eight rotating repeats. Cache equality precedes timing. Freeze
the exact rotation, input hashes, limits and runner before launch. Report each
model's actual parameters/state and latency, and all-resident memory separately
from deployment peaks. The old RoPE MLA option is not a fresh control here.

## Work remaining before any launch

The separate rank policy now implements D/8,D/4,D/2 with explicit model/cache/
optimizer/checkpoint signatures. Selected geometry/equation/cache/lineage checks
passed; optimizer-resume/EMA and resource-pilot prerequisites remain incomplete.
Constructed meta accounting for every variant and worst-case selections gives
53.2948GiB for the width512 draft including reserve, retained pilot/quality
artifacts and conservative metadata/serialization allowances. Measure
actual pilot resource costs only after the complete prospective panel fits.
The approximate60GiB storage target exceeds this prospective estimate, but actual
filesystem fit and pilot costs still require verification before dispatch.

The failed combined CPU fixture remains closed. New diagnostic and synthetic
buffer tests do not complete missing model continuation checks or authorize a
failed-guard retry. Resolve the validation procedure explicitly while preserving
its failure record and unchanged4GiB RSS/8GiB reserve limits. Quality dispatch
also needs7200s/trial,6GiB MPS-driver, output bounds and global single-worker locks.
Finally integrate/test the serial controllers and audited score adapter, record source/runtime/
data/config hashes, freeze both finite pilot and quality plans, and verify fit.
This draft may change before freeze; it cannot change in response to new quality
outcomes. Test holdout remains closed and no MPS launch follows from this file.

## Prospective inference measurement order

The callback-based `tokenizer_wide_hybrid_timing.py` now fixes the proposed
measurement sequence; it is not a model-loading worker or launch authorization.
For seeds3511/3512/3513 assign offsets0/1/2 into the family cycle
hybrid,gated_mla,kda. At repeat r use offset(seed)+r modulo3. Each seed has eight
measured repeats per batch, giving position counts3/3/2; across all three seeds,
each family occupies each position exactly eight times per batch. Two warmup
repeats use the same rotation, restarting r at0 for measured repeats. Retain
warmup records separately and exclude them from reported timing summaries.

All six family/batch cache-correctness checks must pass before any warmup or
measured timing. The eventual worker must implement the numerical comparison;
a callback's passed flag alone is not independent numerical evidence. Timing
samples require positive finite prefill/decode durations and positive integer
state/current/driver byte observations. Unique backing storage of all cache
buffers measures recurrent, convolution, latent and shared-key capacity, including
unused capacity of views and counting shared storage once. This excludes
allocator overhead; report allocator observations separately. State bytes must
remain identical for a family/batch across warmups and measured repeats.

The proposed input is the first96 training token IDs shared by every model,
identified by SHA256 of96 unsigned little-endian uint32 values. Freeze the
actual IDs/hash after validating tokenizer and split identity before launch.
The helper accepts an independently expected hash; it neither reads the corpus
nor proves that caller-supplied IDs come from training. The eventual worker must
bind checkpoint hashes, selected ranks/LRs/seeds, full configuration/tokenizer,
MPS FP32 device, actual input derivation and cached numerical checks. Guarded
serial worker integration, result auditing and a frozen timing plan remain open.

Saved timing evidence will be audited independently against the frozen seed,
selection and token hash. Require all6 correctness rows,12 warmup rows and48
measured rows per seed in the exact declared order; reject missing/duplicate/
extra records, type aliases in identities, changed state sizes and incorrect
throughput arithmetic. Hash-bind bounded JSON files and reject duplicate JSON
keys. This contract audit does not substitute for model/checkpoint/tokenizer/
device provenance or independent numerical-cache verification.

For each seed/family/batch, compute median prefill and decode latency over the
eight measured repeats, excluding both warmups. For each control/batch/metric,
compute the three seed-paired ratios of hybrid median latency to control median
latency, then report their median alongside all three ratios. Do not pool timing
samples across seeds or select the most favorable seed. Require all three fresh
triples for panel summaries. Evaluate the previously declared state-saving gate
against both controls at every seed/batch, with exact integer byte arithmetic.
Latency summaries have no speed-win gate; neither a timing audit nor its memory
gate can adopt a recipe without the quality and complete-artifact requirements.

The prospective real correctness callback now specifies full forward logits for
all96 input tokens versus a cached32-token prefill followed by64 one-token calls.
Compare each cached chunk to the corresponding full logits with rtol3e-4 and
atol3e-5, rejecting nonfinite FP32 logits. Reset all layer caches and replay the
same complete sequence once more against the same reference. Record the maximum
absolute error across both passes. These are prospective numerical tolerances;
they must pass on the actual selected models, not merely synthetic fixtures.
Checking chunks individually avoids retaining a second concatenated96-token
logit tensor while the full reference is resident.

Each timed call constructs fresh layer caches before synchronization and the
start clock. Prefill timing includes its model call and ending synchronization;
decode timing includes64 sequential teacher-forced one-token calls and a final
synchronization, without per-token synchronization. Input construction, cache
allocation, state-storage counting and allocator queries remain outside timed
intervals. The strict adapter requires MPS availability without CPU fallback,
all three explicit width512 NoPE rank-policy models in eval mode, matching model
and cache configurations, MPS FP32 parameters and floating buffers, and the
independently expected token hash. This adapter does not load/check checkpoints,
start a watchdog, hold worker locks or authorize a launch; the guarded wrapper
must supply those prerequisites and provenance checks.

The prospective selected-export loader takes independent expected requests,
training/tokenizer contracts and artifact hashes for each fresh family. It
rechecks the raw/H200 metadata pair before deserializing each export on CPU,
checks export finiteness and hashes around deserialization, validates rebuilt
tokenizer/splits and first96 training IDs, and loads models strictly. All three
models remain resident. Request/result/corpus/raw/H200 hashes are rechecked after
all loading. The helper does not certify exact-EMA/optimizer audits or freeze
source/runtime, acquire locks or start resource guards; the inference wrapper
must enforce these prerequisites before invoking it. No production loader or
MPS success was established by the mocked preparation tests.

The draft inference contract is now encoded prospectively in
`tokenizer_wide_hybrid_inference_gate.py`:600s per fresh-seed triple,4MiB output,
8GiB reserve,4GiB RSS/6GiB MPS driver, FP32 MPS, declared sequence/tolerances and
seed rotation. Freeze this contract plus actual training-token/tokenizer hashes
inside the quality plan before training. After all quality runs and tensor
reviews, the parent may create an immutable inference request binding that
plan, completed state and exact complete-panel tensor-review hashes. The gate
requires every reviewed request/result/raw/H200 artifact, persisted choices and
underlying audit evidence to remain unchanged, and rechecks source/runtime and
STOP after tokenizer reconstruction. It does not execute the tensor audit or
replace guarded subprocess dispatch; no actual inference plan is frozen yet.

The serial inference integration now applies the4MiB output allowance to the
entire three-seed inference directory, including process logs, requests, guard
journals, result/model receipts and summary. Each child retains the600s timeout;
resource watchdogs cover4GiB RSS,6GiB MPS driver and8GiB free-space reserve.
The parent requires inherited global/study locks, rejects any existing inference
attempt, preserves failure state and stops later seeds after errors. It checks
saved result/model receipts and guard observations before accepting each triple,
then rechecks earlier evidence before proceeding. Final summaries remain pending
final research review and never automatically adopt. This integration has only
mocked/tiny-process test evidence; no current frozen launch plan exists.

## Prospective resource-pilot contract

`tokenizer_wide_hybrid_pilot_config.py` supplies the three draft configurations
and a full-campaign storage check; it has no freeze, dispatcher or retry API.
Before pilot1 the required free space is57,224,879,357bytes, including all27
retained trajectories, metadata and the unchanged8GiB reserve. After an audited
pilot, subtract only that pilot's allocation from required *remaining* free
space. Its files stay on disk, and all24 quality trajectories remain reserved
at the worst possible selected rank. Keep the full metadata allowance throughout.
An ordered family prefix is only an accounting input; the eventual controller
must establish successful audits and unchanged retained artifacts independently.

The prospective per-pilot limits are600s,4GiB RSS,6GiB MPS driver and4MiB log,
with the existing per-trajectory output allowance. The projection is conservative
tensor/serialization accounting, not measured pilot bytes, memory or time.
Required numerical/source/runtime/corpus/tokenizer bindings and full-panel fit
must precede a final freeze. No actual pilot plan is frozen. Resource-pilot
worker integration, raw/H200 tensor/export/cache audits and measured costs remain
unfinished. The short EMA schedule is tested only on a two-scalar CPU fixture;
this neither retries nor completes the failed wider continuation prerequisite.

The pilot metadata adapter now requires the same independently bound artifact
lineage/shape/diagnostic-score checks as quality runs, with the exact20-update
configuration and eight-snapshot EMA contract. It returns a pilot-only receipt,
never a selection row or a passing full-pilot review. Tensor-value finiteness,
exact export-vs-stored-EMA equality, optimizer/resume correctness, cached logits
and guarded actual resource observations remain separate required evidence.
Meta-only test success does not certify any of those numerical prerequisites.

A separate tensor-value audit now checks finite raw/EMA/optimizer/export values,
exact equality of exported parameters to stored H200 and unchanged persistent
buffers. It maps one pair on CPU and compares bounded chunks, after metadata
review and with hashes rechecked afterward. The caller must enforce locks and
resource guards. Its tiny synthetic tests do not measure real checkpoint peak
memory or validate optimizer continuation/EMA recurrence; reports explicitly
leave those checks unapproved. Full numerical and pilot reviews remain required.

The prospective pilot plan schema now binds the full conditional27-trajectory
design, exact pilot requests, quality/state/bootstrap thresholds and full storage
projection before any pilot. Its numerical review must enumerate all nine
width512 family/rank cases with geometry/cache/optimizer/resume/EMA checks and
underlying hash-bound evidence matching source/runtime/data. A read-only gate
rechecks provenance and STOP; it does not create a frozen plan, certify the
review's scientific evidence or enforce live disk/worker limits. Those remain
responsibilities of the final review and guarded serial controller. Synthetic
gate tests do not satisfy the outstanding real numerical prerequisites.

Before pilots, freeze tokenizer identity and the SHA256 of the first96 training
IDs (uint32 little-endian, as in the inference contract). Pilot cache checks
cover raw and H200 separately at batches1/8, full96 versus32+64 cached tokens
and reset replay, rtol3e-4/atol3e-5. Load only one model at a time. Require exact
constructed parameter storage/count and allocated D/2 cache capacity, and retain
allocator observations separately from unmeasured peaks. The prospective helper
rechecks all artifact/data hashes and stops after failures; it does not replace
locks, watchdogs, full-panel fit or a passing numerical prerequisite review.
Its mocked/tiny CPU tests do not establish real MPS correctness.

The prospective serial pilot integration now enforces the frozen gate, inherited
locks, no reentry, full remaining-panel reservation, per-trial and aggregate
output caps,600s timeout and existing4GiB RSS/6GiB driver/8GiB reserve limits.
The parent reviews metadata, tensor/cache receipts and guard observations before
the next pilot, retaining and rechecking evidence. STOP/failures prevent further
dispatch. Even all three completed pilots produce only a pending-review state;
a separate reviewed resource/numerical prerequisite is required for quality.
Only mocked/tiny-process integration has passed. No real plan has been frozen
and no real MPS worker has been verified or launched in this preparation.

## Prospective memory-route feasibility review

Retained context128 meta accounting permits the10% both-control state reduction
only for selected hybrid D/8 versus selected gated-MLA D/2 (any KDA rank):3 of27
possible rank triples. Projected savings are48.9583% versus gated MLA and10.3659%
versus KDA. Same-rank triples fail this memory threshold. This is a workload-
specific mathematical constraint, not an observed memory benefit or a quality
result. Keep lowest-validation-CE selection and both adoption routes unchanged;
no rank may be chosen to manufacture a memory pass. All actual state measurements,
fresh quality gates and audits remain mandatory. See the preparation record and
`tokenizer-evidence/wide-hybrid-preparation-001/readiness-review-001.json` for
coverage limits: seven-token CPU cache evidence and meta optimizer construction
do not establish the missing nine-case rank-policy optimizer/resume/EMA review.

## October 28 retention amendment before freeze

The human authorized proactive non-critical checkpoint cleanup. The 27-trial
scientific design above remains unchanged. Replace the proposed retain-everything
lifetime with audited boundary retention: after pilot review prune three pilot
pairs; after all LR audits and saved per-family LR selection prune six losing
pairs; after rank audits and saved final selection prune six losing pairs from
the three anchors plus six probes. Retain three final selected screens and all
nine fresh confirmation pairs. All non-checkpoint evidence and failures remain.

Follow TOKENIZER_CHECKPOINT_RETENTION.md. The new prospective high-water budget is
30,631,281,412 bytes including 256 MiB round metadata/logs and 8 GiB reserve; see
`tokenizer-evidence/checkpoint-cleanup-002/prospective-retention-projection.json`.
This is not a frozen executable amendment. Existing v1 guards remain intact;
versioned retention-aware runners, immutable deletion receipts, stage audits,
complete future high-water checks and tests are required before freeze/launch.
No quality/memory threshold or trial/rate/seed/order/selection rule changes.

## V2 retention component implementation (draft, October 28)

Separate v2 planning and read-only receipt primitives now have bounded synthetic
coverage (123 passing checks including existing selection/simulation tests).
They enforce candidate eligibility from unchanged scientific decisions and
exact completed receipt provenance, with retained files rehashed. They do not
perform full stage reviews, delete real weights, or authorize dispatch. The
production v1 runners remain unchanged; no automatic receipt exception has been
added to their existence/hash checks. See the preparation document and
`tokenizer-evidence/wide-hybrid-preparation-001/retention-v2-component-review.json`
for exact source hashes, validation and integration prerequisites.

The v2 quality-stage audit binder now additionally checks complete LR/rank stage
reviews and stage-directory inventories (39 new synthetic tests; 112 including
retention regressions). It requires independent nine-case numerical evidence;
metadata/value audit receipts cannot substitute for that evidence. The v2
resource-report producer, pilot/full-panel gates, campaign inventory, guarded
cleanup and complete runner lifecycle remain unimplemented integration work.
See `retention-v2-quality-audit-review.json` in wide-hybrid preparation evidence.
No existing v1 acceptance rule or source of a closed campaign changed.

A separate v2 quality resource recorder now supplies bounded durable event and
summary files; quality-stage binding checks their exact hash/content agreement.
Synthetic coverage is62 checks. It supplements the unchanged existing watchdog,
not a replacement for full gates or measured MPS feasibility. Resource/log caps
use216MiB of the256MiB prospective metadata allowance; remaining metadata and
cumulative enforcement still need full controller integration before freeze.
See `resource-v2-component-review.json` in wide-hybrid preparation evidence.
