# Fresh native horizon comparison

The evidence and hypothesis are in `GENERAL_RSI_NATIVE_EXPOSURE_BOUNDARY.md`.
This protocol and `general-rsi-native-horizon-001/plan.json` freeze a new
comparison. No prior campaign is reopened. The native recipe is unchanged.

Run three fresh pairs in this serial order: short1, long1, short2, long2,
short3, long3. Each short run gets7,200 synchronized training-update seconds;
each long run gets14,400. All runs start from fresh initialization. Within a
pair, initializer and sampler roles match. Across pairs they are independent.
The plan fixes each role and rejects prior use of the selected seed prefix.
Use LR0.0025 for both horizons, with no new tuning or seed replacement.

The qualified context includes sampling, device transfer, forward/backward,
clipping, optimizer, EMA and batch hashing in its synchronized training clock.
Linear cooldown runs over each horizon's final third to ratio0.1. EMA starts
at the first update beginning in that final third and uses the unchanged
127,184,813,977ns half-life. Keep RAW as a secondary endpoint. Stop after the
first update that reaches the budget. Charge audited lost work after a failure.
Report total worker time separately, including scoring, checkpoint I/O,
generation, failures and recovery. This compares complete horizon recipes,
not exposure isolated from schedule length.

Score all108 original validation documents,100,416 targets, with the qualified
document-wise evaluator. Probe the fixed eight documents at one-third and
two-thirds of each horizon. Probes cannot select checkpoints or stop a poor seed.
Test remains unopened. No transfer holdout is used to tune this comparison.

Adopt the longer recipe only if all three paired final EMA differences
(long minus short) are negative, their mean is at most-0.02 nats/token, the
one-sided99% paired-document bootstrap upper bound is below zero, and neither
source's mean regression exceeds0.05. Use10,000 PCG64 draws of108 document IDs,
seed62111022999. Average paired seed loss sums before resampling and divide by
the sampled target counts. Use NumPy's linear99th percentile. Bounds are
conditional on these documents and seeds. Otherwise retain the short recipe.
Do not pool historical scores or select a best seed.

Generate all144 first completions from the six final EMA models using the
inherited12 prompts, two decoders, seed42 per prompt/decoder and64-token cap.
Retain every output and token journal. Report decoder-specific repetition,
length/EOS/truncation, and blinded relevance/consistency ratings under the fixed
rubric. State the annotator. These diagnostics cannot rescue a failed CE gate.

The worker and reviewer are isolated copies of the qualified equal-time code.
AST checks require identical training/publication and review functions after
removing only the old LR-selection lookup. Retirement, complete restart,
generation and cumulative guards are reused. New path binding, module dispatch,
six-slot range and state labels are reviewed separately. New horizon schedule
and cumulative-bound checks run without model or dataset access.

Resource admission covers the entire six-run panel. Training totals18 hours.
The cumulative panel ceiling is28 hours, including failures. Short slots allow
10,800 seconds each; long slots22,800 seconds each. At most two attempts per
slot are allowed for a demonstrated recoverable execution failure. A numerical
or scientific negative cannot authorize changed rules or a fresh seed. Preserve
all failure evidence and first outputs; a partial endpoint requires an explicit
audited recovery, never blind rerunning or overwriting.

Allocate twelve final weights at their enforced128MiB cap, two512MiB active
restart/serialization slots,256MiB metadata/logs/generations,5GiB failed-write
allowance and10GiB free reserve. Total required free space is19,058,917,376bytes.
The cap-based final-weight allocation replaces the boundary's measured-size
projection to cover changed slot metadata. Existing retained artifacts are
already charged through measured free space. Recheck remaining full-panel
capacity at every worker launch. Preserve selected and fresh weights.

Use one local MPS worker under the nonblocking shared research lock, RSS4GiB
and MPS driver6GiB. Check human stop during training and between endpoint
operations. No remote compute or subagents. Freeze and push authority before
the first scientific update. Review each completed slot before launching its
successor. Numerical qualification is reused, not rerun to create progress.
