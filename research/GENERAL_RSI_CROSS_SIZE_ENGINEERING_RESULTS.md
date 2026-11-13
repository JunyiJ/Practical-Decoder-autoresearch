# Cross-size engineering results — 2026-11-13

The isolated D256 depth4/8/16 implementation passes its bounded engineering
checks for both native and existing byte input. Scientific training is not yet
admitted. The completed context panel and all historical failures remain unchanged.

## New implementation and checks

`general_rsi_cross_size_v1/` gives each model a distinct depth identity. It
reuses the qualified D256 block, composer and optimizer mathematics. The original
four-block initializer supplies the input, first four blocks, final norm and
output head. Each additional four-block group uses its own private initializer
seed. Its temporary input/output tables and final norm are discarded. No block
weights or storage are cloned/shared. The root identity records total depth;
the inherited four-block configuration is an initializer component, not a claim
that the resulting model has four blocks. Common body/output parameters match
within arm pairs and across shared depth prefixes.

CPU fixture001 passes exact old/new depth4 initialization, corrected logits,
loss, gradients, one clipped optimizer update and all optimizer child states for
both arms. All six model inventories match the allocation calculation. Extra
block matrices and storage are distinct. Invalid depths and a native model with
a codebook reject. The guard finishes in10.895seconds, peakRSS1,493,286,912bytes.
This is a parity check required by the new wrapper, not a repeat quality study.

MPS pilot001 completes240 optimizer updates:20 per depth4 arm,40 per new-depth
arm, and ten replayed updates for each of the four new-depth arms. All paired
batch streams match. Each new-depth checkpoint at step30 includes active EMA,
optimizer, sampler, RNG, schedule and history. Loading restores the complete
state exactly; replay through step40 matches the reference state exactly, with
two EMA snapshots. Wrong-depth loads reject before destination mutation.

All six corrected CPU/MPS logit checks pass the fixed atol3e-5/rtol3e-4 rule.
Maximum absolute error is2.384185791e-6. The scorer checks cover258 targets from
training-only text, including a short final chunk. CPU/MPS CE differences are
below the fixed1e-4 bound. No validation model scores or original test contents
were used in engineering qualification.

## Timing and resources

| Depth | Native median seconds/update | Byte median | Byte/native ratio | Native p95 | Byte p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | .330432 | .630841 | 1.909 | .384114 | .678081 |
| 8 | .427230 | .738956 | 1.730 | .472628 | .795355 |
| 16 | .616161 | .875374 | 1.421 | .651811 | .924823 |

These are synchronized complete-update timings from15 observations per cell
(updates6–20), one engineering seed and a short40-step horizon. They support
local feasibility and the overhead-dilution hypothesis. They do not establish
sustained scientific-panel speed, quality superiority or a scaling law.
The full pilot took188.530seconds. PeakRSS is1,898,151,936bytes; peak MPS driver
allocation is4,118,593,536bytes, below the4GiB/6GiB guards.

Six retained raw engineering endpoints total626,847,232bytes. Four transient
step30 checkpoint prefixes were retired only after exact replay and durable
endpoint/review records. Each retirement used the held research lock, exact
path/size/hash checks, an empty exact-path lsof consumer check, and fsynced
intent/deleted journal entries. Failed artifacts and all fresh scientific
weights are untouched. The completion review verifies all six retained hashes
and all four retirement pairs.

For18 fresh scientific confirmations at10,240updates/run, p95 pilot rates imply
120,017.299seconds of training. Conservative scoring extrapolation charges
12,049.177seconds for both endpoints on100,416 development plus32,768 broader
text targets. Provision1,728seconds for432 fixed generations and3,600seconds
for checkpoint/replay work. A30% margin brings the full forecast to
178,612.819seconds (49.6147hours). Revise the prospective tranche from36 to54hours
(194,400seconds) before scientific outcomes. This preserves all18 runs and fresh
pairs. Short-pilot extrapolation remains uncertain; live guards must enforce
the eventual frozen tranche.

The remaining full storage planning requirement is21,192,595,032bytes after
subtracting the626,847,232 engineering bytes now retained from the original
complete forecast. It includes all36 final endpoints, restart/serialization
overlap, metadata, failure allowance,10GiB reserve and2GiB extra headroom.
Free space at review is21,875,994,624bytes, a683,399,592byte margin beyond that
complete requirement. No cleanup is needed at this snapshot. Refresh admission
when the scientific worker and exact output contract are frozen.

## Broader evaluation preparation

Dataset001 rejected itself: all261 unused Python parents fell below2049 eligible
tokens after the initial filter removed paragraphs containing inline references
or indentation. All original source, protocol, selection and failure evidence
remain retained. No model score was computed and no scientific hypothesis lost.

Separate implementation/dataset002 repairs that formatting filter. It retains
ordinary text inside Sphinx roles and inline literals, permits indentation, and
still rejects directives, prompts, tables and remaining markup. The8-parent per
domain requirement,2049-token windows, fixed selection seed, parent exclusions
and exact32-token overlap rule stay unchanged. The resulting dataset has eight
previously unscored PG19 parents and eight Python prose parents,32,768 targets.
All64 previously evaluated parents are excluded. One book candidate was rejected
for selected-excerpt overlap. Coverage passes exactly; no model scores guided
selection. The original test is unopened. Exact-span checks do not exclude all
paraphrases, and mechanically joined prose paragraphs are a narrow domain probe.

Evidence: `tokenizer-evidence/general-rsi-cross-size-fixture-001/`,
`tokenizer-evidence/general-rsi-cross-size-pilot-001/completion-review.json`, and
both `tokenizer-evidence/general-rsi-cross-size-eval-001/` and `-002/` records.
The broader data payload hashes are bound in dataset002's `dataset.json`.

Next freeze fresh scientific seed roles, within-size and interaction criteria,
FLOP accounting and evaluation bindings. Integrate and qualify the new endpoint
loader and serial worker paths without rerunning completed depth qualification.
Then refresh complete resource admission. No training worker is active at review.
