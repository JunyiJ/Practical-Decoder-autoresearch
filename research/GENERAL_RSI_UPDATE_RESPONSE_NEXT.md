# First-update response: prospective diagnostic

## Evidence and competing explanations

The completed RAW/EMA start comparison reduces relative drift by 0.007399052
nats/token when starting from RAW, but every endpoint still trails the unchanged
EMA parent. The preceding group-decay comparison also leaves positive drift
without decay. These controlled local results narrow the remaining question to
update behavior; they do not establish that fresh optimizer state is the cause.
See `GENERAL_RSI_CONTINUATION_START_RESULTS.md` and
`GENERAL_RSI_CONTINUATION_DECAY_RESULTS.md`, with their original first scores.

The closed common-state diagnostic found positive CE-minus-identity changes in
all 24 cases, after an initial CE update. Its geometry records distinguish Muon
and AdamW update magnitudes. The proposed experiment instead isolates the first
no-decay update from pristine optimizer state and varies only its realized
parameter displacement. No old case is replayed or pooled into fresh evidence.

Contrary local evidence is `WARMUP_QUALITY_RESULTS.md`: warmup 100 failed fresh
confirmation despite a favorable screen, and warmup 300 was worse on average at
both tested widths. Those Shakespeare training-from-scratch results at LR 0.005
do not decide this general-corpus continuation question at LR 0.00025. That study
remains closed. A new warmup sweep is not the next action.

## Primary literature and applicability

Liu et al., [arXiv:1908.03265v4](https://arxiv.org/html/1908.03265v4), sections
3–3.2, propose adaptive-rate variance as an early Adam instability mechanism.
Their Adam-2k test accumulates second moments while holding weights and first
moments fixed. The analysis assumes idealized gradient distributions. These
reported results motivate examining fresh-state updates, not a local causal claim.

Ma and Yarats, [arXiv:1910.04209v3](https://arxiv.org/html/1910.04209v3), sections
3.2–3.3, challenge a variance-only account and examine actual update magnitude.
Their local-minimum simulation uses independent zero-mean gradients. Section 4.1
also distinguishes cumulative step suppression through an effective warmup period.
These arguments favor measuring displacement response before proposing a schedule.
Their Adam results do not establish behavior for our hybrid Muon/AdamW optimizer.

Inference: a full first update could overshoot a favorable local direction, or
its direction could already increase loss on disjoint sampled windows. Lowering
step length can distinguish these cases over measured finite scales. It cannot
identify an exact derivative, prove population generalization, or diagnose the
missing original optimizer moments.

## Fixed discriminating experiment

Use the three selected EMA parents, slots 1/3/5, with the qualified no-decay
context. For each parent, freeze eight fresh natural batches and fresh disjoint
training probes before execution. Each case starts from identical parent weights
and pristine optimizer state. Perform exactly one ordinary CE update at LR
0.00025, clip 1. No batch is used for a second optimizer update.

Retain the starting and updated parameter tensors in RAM. Score the line
`theta0 + a * (theta1 - theta0)` for fixed scales `a = 0, 1/16, 1/4, 1`.
Copy exact endpoints at 0 and 1. Freeze one FP32 interpolation rule for the two
interior scales. Restore from original tensors before each scale; never compound
interpolations. These are inference conditions, not extra training arms.

Each parent gets 64 fresh document-local 128-target windows, excluded from all
eight update batches' source positions. Share them across its cases and scales.
Score the unchanged parent once, then all three nonzero scales for each of eight
cases: 75 probe evaluations total, 614,400 scored targets. Also score each update
batch at all four scales: 96 evaluations, 98,304 targets. Total 24 optimizer
updates and 712,704 scored targets. Keep every first score and input inventory.
No validation, test, generation, checkpoint selection or model adoption.

Primary support for finite-step overshoot requires, in each of the three parents,
positive mean probe loss change at scale 1 and negative mean change at scale 1/16
across its eight cases. Failure of this rule is not evidence of a beneficial
smaller step. Report the fixed 1/4 result descriptively, without substituting it
for the primary rule or selecting a winning scale after results.

Report every case/scale, parent means, loss differences and finite secants
`delta CE / a`, plus update norms by optimizer group. If update-batch CE falls
while probe CE rises, this supports a batch-specific direction cost on these
samples. If all measured scales raise probe CE, overshoot is not demonstrated;
unexamined smaller scales remain unresolved. Either outcome determines whether
a schedule proposal has a local mechanism basis. Do not promote reduced harm
into a quality gain or count these correlated conditions as independent seeds.

## Minimal checks, complete cost and next executable action

Implement only the displacement/interpolation harness and variable-target batch
scoring. Verify exact endpoint copies, finite interiors, exact restoration and
correct 1,024-target aggregation. Reuse the qualified model, optimizer, restart,
sampler and FP32 scorer. No repeat of those prerequisites is required.

Provisional cost: 75 probe scores × 1.70 seconds, 96 batch scores × 0.22 seconds,
24 updates × 0.359 seconds and 110 seconds setup/restoration/evidence is about
267 seconds. The batch-score estimate scales measured probe time by target count;
it must be checked by the new scoring fixture before admission. Use a 600-second
ceiling only if measured complete cost fits, RSS 4 GiB and MPS 6 GiB. Forecast
64 MiB retained evidence plus 64 MiB overlap, no model serialization, 10 GiB reserve
and 2 GiB extra headroom. Preserve existing restart payloads and selected models.

Next implement and check those new harness operations. Then freeze unused seed
roles, all 24 batches/probes, source hashes, criterion and measured admission
before scientific execution. Inspect processes and acquire the shared lock
nonblocking. One local worker only. This design is preparation; no worker or
quality campaign is admitted by this document.
