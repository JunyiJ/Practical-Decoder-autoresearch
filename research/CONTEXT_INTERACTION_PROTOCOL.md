# Context, depth and GQA interaction round

Human steering: improve the efficiency and systematic coverage of architecture
interactions, and permit up to +0.01 CE nats/token quality loss for useful inference
savings. The optional choice between the compact interaction design and the older
context-only plan received no reply; proceed with the recommended compact design
within the authorized robustness phase. This document defines the bounded protocol. The readonly run plan binds the final
source, knowledge snapshot and evidence. No training launch until all gates below pass.
All earlier studies, selections, checkpoints and holdout reservations remain closed.

## Questions and finite allocation

Use a complete 2×2×2 factorial: layers2/4 × MHA/GQA × context128/512. All eight
cells use bias-free SwiGLU21/8, head dimension32, RoPE and the Muon/AdamW hybrid.
This answers whether GQA's effect changes with depth and/or training context, with
shared MHA references rather than separate sequential architecture sweeps. Context256
remains a separately bounded follow-up to assess intermediate behavior; the batch
study remains separate. MLA is not added as another factorial dimension: its
reviewed width256 quality/cache tradeoff is preserved for a later targeted comparison.
Nothing here generalizes the two-layer GQA inference timings to four layers.

1. Screen all eight cells at width64 using exactly LRs .00125/.0025/.005 and
   development seeds2021/2022:48 runs. Pick each cell's LR by lowest mean final
   full-validation CE at common evaluation context128; exact ties choose lower LR.
   Freeze those eight choices and the resulting confirmation manifest before any
   confirmation seed or width256 result. Boundary choices remain unresolved;
   no extensions or post-hoc retuning.
2. Evaluate all eight selected recipes on fresh development seeds2031/2032/2033
   at widths64 and256:48 runs. Fresh seeds separate tuning from confirmation,
   but the corpus is still adaptive validation, not an independent holdout.
3. At both widths on the three confirmation seeds, retain fresh two-layer GELU
   MHA LR .002 (six runs), fixed two-layer SwiGLU MHA .0025 and fixed four-layer
   SwiGLU MHA .0025, all context128. If a fixed SwiGLU recipe exactly matches an
   already planned selected factorial recipe, explicitly alias its reference to
   that checkpoint; do not train it twice. Otherwise add its six fixed-reference
   runs. Total102–114 distinct training runs, maximum114. The alias rule is fixed
   before screening and cannot be used to cherry-pick seeds or repeats.

Every trial uses3000 updates and1024 target positions/update (3,072,000 exposures).
Batch8/context128 and batch2/context512 are computational groupings of identical
sampled training targets. Independent per-step decay remains .00006, clipping1,
zero dropout, fixed momentum/betas and base width64. This is equal target exposure
and updates, not equal compute, information, sequence count or gradient variance.
No early pruning: prior small/full-width reversals do not justify an uncalibrated
short-run proxy. Record full learning curves, wall time and resource readings.

## Paired training data and initialization

The new explicit `training.batch_layout=paired_blocks512` samples two training-only
512-token chunks per update from a private seeded generator. For context128,
each chunk is split into four consecutive128-token examples; context512 keeps
the two long examples. Context256 is supported for the later follow-up by splitting
into two. Flattened inputs and targets and the training-generator state must match
exactly across these contexts at every tested update. Targets at short-context
boundaries keep the same labels while their available preceding context changes.

This sampler changes the distribution/correlation of examples relative to the
historical independent-window sampler. All controls in this new round therefore
use the same paired sampler. Historical results are not directly pooled with it.
The default historical sampler remains unchanged. The sampling mode is bound in
checkpoint training contracts; resuming with a changed mode is rejected. Sampled
in-training validation remains a secondary diagnostic with its separate generator.

Under the repaired RoPE source, all active initial weights and global CPU RNG state
must be exactly identical across contexts128/256/512 within a fixed width/depth/
attention/seed. Only deterministic masks and rotary buffers change size. No
cross-architecture initial-weight identity is claimed: projection shapes and RNG
draws differ. Fresh context128 controls are mandatory; no historical checkpoint
is resumed or rewritten.

## Explicit four-layer GQA scaling gate

The new `depth_width_policy=depth4_gqa_unscaled` permits only four-layer bias-free
SwiGLU21/8, GQA half-KV-head32 and widths64/128/256 with base width64 under the
existing Muon/muP implementation. Reject manual hidden/KV/head overrides and other
depths. Preserve the older `depth4_unscaled` contract; it does not silently acquire
GQA support. Query heads are2/4/8; KV heads are1/2/4. QKV projection is2d×d,
output d×d, FFN hidden21d/8. At fixed depth these aspect ratios stay fixed as width
grows. The hidden-matrix LR scales as1/sqrt(width multiplier) for Muon while its
RMS shape factor scales as sqrt(width); this finite algebra is not validation of
optimal-LR transfer or infinite-depth behavior.

Retain existing fan-in initialization, embedding/readout and bias width factors,
normalization, unscaled residual additions, Muon hidden matrices and AdamW vectors.
Changing depth increases the number of residual additions, parameters and compute;
no depth normalization or pure capacity-independent interpretation is introduced.
Check parameter counts, every layer's QKV dimensions, finite gradients/updates,
checkpoint/reload and actual MPS behavior before launch. The width256/context512
four-layer models are the principal memory/runtime feasibility cases.

## Evaluation and interpretation

Final common-context128 evaluation covers every validation target except the first
exactly once with deterministic disjoint windows, unpadded tail and token-weighted
CE, using evaluation batch8 for every training regime. This is the primary LR
selection and interaction score. Also evaluate at the trained context (128 or512)
as a labeled secondary diagnostic; no context extrapolation or test evaluation.
Native-context window boundaries differ by design, so a lower native-context score
is not evidence of a pure architecture effect. Both views preserve all raw
2048-token text-block loss sums/counts and matched seeds. Use conditional paired
text bootstrap at2048/4096/8192 blocks,2000 draws, fixed seed20261024, and show each
training seed separately. Intervals do not include population training-seed
uncertainty or multiplicity correction.

Report paired GQA-minus-MHA contrasts at each depth/context/width, depth effects,
and the differences of those effects across context. At screening, compare fixed
LR .0025 cells to expose interactions without changing LR. Confirmation compares
separately tuned recipes: its interactions include LR choice and must be labeled
as recipe interactions, not pure architectural causal attribution. Compute the
three-way interaction as the depth×attention contrast at context512 minus that
at128; do not infer it from unrelated historical rounds.

For prospective quality eligibility, mark a GQA cell near-neutral only when mean
and every paired seed delta versus matched-depth/context MHA are at most+.01 and
all three conditional text-interval upper bounds are at most+.01 on the declared
primary score. Report native-context results separately, including any regression.
This is a conservative operational rule on three development seeds, not proof of
population non-inferiority. It does not automatically change the quality incumbent.
A subsequent matched inference measurement must establish savings for the actual
new depth/context; current context128 two-layer GQA evidence is insufficient.
The existing inference diagnostic suggests a memory benefit rather than a reliable
speed gain. Preserve separate quality and memory/latency tradeoff records.

## Remaining launch gates and bounds

Focused tests must check exact target pairing and RNG, unchanged legacy sampling,
resume-contract mismatch, context-independent initialization, scaling guards,
finite job/alias counts, screening-only LR selection, confirmation isolation,
full-validation coverage and known interaction contrasts. Run the complete project
test suite and serial short actual MPS training/checkpoint/evaluation pilots.

Proposed hard caps:1200 seconds per trial including final evaluation, eight hours
for the round,4 GiB output, one serial worker, first failure stops, no automatic
retry. A measured feasibility pilot must support these caps; if infeasible, revise
the draft before preregistration rather than extend a live run. Commit/push clean
reviewed source, bind the complete shared knowledge bank and all source/corpus/
runtime contracts, then freeze a uniquely named readonly plan with exact local
jobs and deterministic confirmation construction. No training is authorized by
this Markdown alone; the operational preflight must verify all these gates.

## Prelaunch checks and resource estimate

All 608 project tests passed before isolation, including the new full serial
controller simulation and failure/no-retry checks. Four serial 20-update MPS
checks covered four-layer MHA/GQA at widths64/256 and context512, LR .005,
paired training data, final checkpoint reload, and full validation at128/512.
An additional12-update MPS check exercised the actual bounded trusted worker,
final-result validation and coverage adapter. These are correctness checks,
not quality measurements or completed study trials.

Four-layer MHA/GQA active parameter counts are204544/187904 at width64 and
3152896/2889728 at256, after omitting the unused RoPE positional table.
Observed20-update training times were1.112/.484 seconds at64 and1.423/1.165
seconds at256. Scaling the slower pilot at each width to3000 updates, applying
it to all81 possible width64 and33 possible width256 trials, and adding six
seconds per worker gives a rough5.90-hour proxy. Startup distorts these short
samples; this is not a full-run prediction interval. Checkpoints project to
965 MiB, or1206 MiB with25% margin. Hard limits remain1200 seconds/trial,
eight hours/round and4 GiB output; no automatic extension.

An unrelated GatedMLA edit appeared in the primary checkout during development.
The study is isolated on branch `codex/context-interaction` in the existing
managed worktree. That edit and the human's `kimi` branch remain untouched.
The isolated source passed all 608 tests in28.66 seconds and an additional
12-update bounded MPS worker check in5.76 seconds before preregistration.
The unrelated class is absent from this source snapshot.
