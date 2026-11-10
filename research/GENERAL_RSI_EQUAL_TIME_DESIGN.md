# General RSI equal Mac training time

This is a fresh comparison following the validated longer-training result.
The original architecture and exposure campaigns stay closed. The test set stays
unopened. No selected Shakespeare weights or deferred output-lexical work changes.

## Evidence, mechanism and prediction

Original three-pair results in `general-rsi-panel-001/decision.json` favored byte
by 0.052604 nats/token at equal updates. Native needed about 0.3475 seconds per
update and byte about 0.6359 seconds (`general-rsi-next-001/timing.json`). The
fresh longer-byte comparison reduced CE by 0.200184 after twice the updates
(`general-rsi-exposure-001/decision.json`). Those are measured local results.
The hypothesis is that native can use its throughput advantage for more updates
and reduce or reverse byte's advantage at equal time. The contrary prediction
is that byte's sample efficiency still wins. Either outcome informs local
training-time allocation. Three fresh paired bundles discriminate these outcomes.

The schedule-horizon rationale and contrary generation/domain evidence are in
`GENERAL_RSI_NEXT_CAMPAIGNS.md`, citing Hoffmann2203.15556v1 AppendixB,
Holtzman1904.09751v2 sections5.3–6 and Penedo2406.17557v2 sections3.1/4.1–4.2.
Their large models and datasets do not establish our optimal LR, budget, EMA
half-life, generation quality or transfer. This experiment tests the local claim.

## Fixed scientific contract

Use unchanged native and raw-byte D256/four-block/context128 FP32 models, data,
GPT-2 targets, sampler, optimizer grouping and clipping. Use fresh initialization.
Four screens in native/byte order at LR .00125, then native/byte at .0025.
Each screen receives 1,800 synchronized training seconds. Select lowest final
averaged full-validation CE within each arm, with exact ties choosing lower LR.
Then run three fresh native/byte pairs, 7,200 training seconds per arm. Roles,
source bindings, run order and numeric constants are frozen in `plan.json`.
No best checkpoint or best seed selection. No replacing a negative seed.

The clock includes sampling, transfers, forward/backward, clipping, optimizer,
EMA and batch hashing. It also includes the durable start-record write. The end
record write, other logs, initialization, scoring, generation and checkpoint I/O
are outside this clock and reported through end-to-end guard time. MPS synchronizes
at both boundaries. Stop at the first completed update that exhausts the budget,
with at most one update of overshoot. Time limits include all failed attempts.

LR is constant through two-thirds of the budget, then declines linearly toward
ratio 0.1, evaluated at each update's starting cumulative time. Both arms use
the same time EMA half-life, converted from 200 updates at the historical byte
mean: round(200 × 6511.862475605488 / 10240 × 1e9) nanoseconds. Start EMA on the
first update whose start time reaches two-thirds of the budget. Initialize it
from that update's weights, then update every step with decay determined by
elapsed training time between update starts. Averaging work is inside the clock.
This causal boundary avoids defining a duration through its own averaging cost.
RAW is secondary. The chosen half-life is a fixed design choice, not an optimum.

All 108 validation documents and 100,416 targets use the already qualified
scorer. Fixed eight-document probes at one-third and two-thirds are diagnostics.
The primary comparison is final time-EMA CE. For either arm to win: all three
paired deltas favor it, mean improvement is at least0.02 nats/token, one-sided99%
paired-document bootstrap upper bound is below zero, and neither source regresses
by more than0.05. Use10,000 PCG64 resamples of108 documents, average paired seed
loss sums before resampling, denominator from resampled target counts, linear
99th percentile, seed62110700999. Report both directions symmetrically. These
bounds are conditional on these documents and seeds, not population-seed confidence.

The fixed generation protocol provides 12 prompts ×2 decoders ×6 averaged
confirmation models =144 first completions, max64 tokens each. Preserve every
token and partial failure. Reset seed42 per prompt/decoder. No rerolls. Report
repetition, length, EOS and truncation separately from blinded relevance and
consistency ratings, with quotes, annotator identity and unassessable counts.
Keep decoder results separate. Agent ratings are exploratory, not independent
human validation. Generation cannot rescue a failed CE gate.

## Recovery and resource contract

One local MPS worker and the shared nonblocking research lock. A complete
checkpoint is published and read-verified before its exact predecessor is retired
with hash, consumer and durable deletion evidence. Keep all final scientific
weights, first outputs, original logs and failure evidence. Checkpoint every120
training seconds and at the endpoint. Honor stop requests at worker boundaries.

Durable interval records account for completed work discarded after a checkpoint.
Charge it before replay; the same physical compute is never free. An unfinished
interval requires the same-boot supervisor exit timestamp and is conservatively
charged through that timestamp. Record that bound separately. If a journal is
partial/corrupt, the exit timestamp is unavailable, the boot changes, or checkpoint
ownership is ambiguous, stop automatic recovery and perform an explicit evidence
repair. Never guess or subtract compute. No quality claim from implementation
failures. Partial generations require preserved-prefix recovery, never rerolls.

The admission forecast retains all ten final weight pairs (1,794,396,880 measured
projected bytes), two512MiB active restart/serialization slots,256MiB for logs,
metadata and endpoint-size margin, plus ten512MiB failed-write allowances and
10GiB free reserve. At most two attempts per slot are admitted by this resource
contract. Further attempts require a new capacity/failure audit while preserving
scientific rules and original evidence. Failed-write allowance is conservative;
no failing payload is deleted to hide a result.

Training totals50,400 seconds. The numerical admission receipt derives scoring,
generation and checkpoint overhead from measured engineering runs. Each screen
has a3,600-second end-to-end ceiling; each confirmation10,800 seconds. The full
tranche has72,000 seconds (20 hours), including failures and replay. A new
resource diagnosis may be needed if these ceilings fail; do not shorten training,
remove seeds or relax acceptance criteria. No paid compute or subagents.
