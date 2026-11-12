# Pseudo-repetition integration results

The new loss, sampler and native continuation integration passed their bounded
engineering checks. No scientific quality panel has started and no improvement
is claimed. The native-horizon and composer-FP32 comparisons remain closed.

## Training-only span inventory

`tokenizer-evidence/general-rsi-pseudo-spans-001/inventory.json` binds284,735
eligible spans in9,764 of9,807 training documents. There are238,473 FineWeb-Edu
and46,262 Wikipedia spans. Sampling is uniform over spans, with replacement,
so its source mix differs from the ordinary token stream. This is an explicit
recipe difference; source-wise validation guards remain necessary.

These are punctuation-bounded spans, not a grammatical sentence parser. Adjacent
token-aligned .!? boundaries followed by ASCII whitespace delimit spans, with
closing quote/bracket handling. Eligible spans have4–36 tokens and1–16 preceding
tokens from the same document. Initial spans without context, long/short spans
and unterminated tails are excluded. At least three complete copies fit129
positions. Prefix, first copy and EOT padding are masked from the penalty.
There are no partial trailing copies. This differs from DITTO's sentence-level
construction and much longer context. The bounds make aligned full-copy targets
feasible within our inherited context128; they are not tuned on quality scores.

The scalar/gradient check agrees with the analytic detached-anchor reference.
Maximum FP64 gradient error is1.11e-16 and FP32 error6.69e-8. Padding invariance,
saturation protection, sampler replay and global-RNG isolation passed. Nonfinite
inputs, future anchors, mismatched targets, empty masks and wrong inventory state
were rejected. The32 audited batches covered all33 eligible lengths. These checks
use training text and small literal tensors, with no model-quality selection.

## New training integration

`tokenizer-evidence/general-rsi-pseudo-integration-001/` freezes the parent,
sources, runtime, fresh engineering roles and resource contract. Both arms load
the same selected native slot01 time-EMA parent, reset optimizer/EMA, and use
LR0.00025. Candidate updates alternate CE and pseudo loss, beginning with CE.
The ordinary CE generator is independent of the pseudo-span generator.

| Engineering arm | Updates | CE / pseudo | Training seconds | Mean seconds/update | Restart bytes |
|---|---:|---:|---:|---:|---:|
| Native CE |69|69 /0|24.067459|0.348804|445,021,621|
| Native pseudo repetition |65|33 /32|24.066445|0.370253|445,033,525|

The candidate evaluated21,197 valid pseudo targets across32 updates. Synthetic
prediction positions are not additional natural-corpus exposure. Neither arm
reported a saturated target. Their first33 natural CE batches match exactly.

Both complete states match exactly after checkpoint restoration and replay with
the original measured update durations. This includes parameters, optimizer,
time schedule/EMA, ordinary and pseudo samplers, RNG, history and objective phase.
The checkpoint was taken after an averaged odd update, so candidate restoration
crosses directly into a pseudo update. Live clock variability is not claimed to
produce identical trajectories. The fixture used a1-second engineering EMA
half-life; the proposed scientific study keeps the inherited127.184813977 seconds.

The guarded fixture finished in73.113320 seconds, peak RSS2,385,641,472 bytes.
The driver allocation sampled after each replay was2,501,689,344 bytes. The result
field calls this `peak_driver_bytes`, but it is a boundary sample, not a measured
continuous peak. Every update boundary enforced the6GiB driver limit. Checkpoint
serialization and replay overhead are included in fixture wall time, not its
synchronized training clocks. Both restart payloads remain retained.

## Next action

The evidence supports feasibility of the proposed integration only. The mechanism
remains untested for quality. Complete the six-run worker binding and freeze fresh
scientific roles, bootstrap and run order with the unchanged decision rules.
Use prior qualified scoring/generation and durable restart machinery. Do not
repeat completed corpus, clock, scoring, generation or diagnostic prerequisites.

For storage, retain all twelve128MiB final-weight caps, two512MiB restart/write
slots,256MiB metadata,5GiB failed-write allowance and10GiB reserve. This requires
19,058,917,376 incremental free bytes. The new890,055,146-byte qualification
payloads already occupy disk and are charged through measured free space.
Do not charge them twice. The audited cleanup in `tokenizer-evidence/general-rsi-capacity-008/` retired12
obsolete accepted engineering payloads totaling4,090,694,482bytes. All702 protected
files passed hash verification. Actual free space21,407,010,816bytes exceeds the
21,206,401,024-byte requirement including extra headroom. Two pre-deletion audit
harness failures and their separate repairs remain recorded. No payload was
deleted by either failed attempt. Recheck actual capacity at panel admission.
Full worker timing must also cover scoring, generation, checkpoint overhead and
failure allowance within the prospective4-hour ceiling. No partial-panel launch.
