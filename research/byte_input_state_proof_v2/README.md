# Numerical checkpoint prerequisite

This new proof checks `research/byte_input_state_v1.py` against the qualified
`byte_input_model_v2`, preserving native/raw-byte/nonlexical arms, CPU/MPS and
B8T128/V50257. It does not execute any closed historical entry.

`admission.py` checks frozen source/assets/runtime, AST coverage, complete cost
and fresh exclusive roles. `entry.py` owns the nonblocking worker lock and hard
4GiB RSS / 6GiB MPS / 8GiB output / 8GiB reserve / 3600s guards. Entry also
requires 2GiB headroom beyond output and reserve. All attempts are once-only.

The driver independently hashes live parameters and named optimizer state,
verifies capture against those fields, serializes and reads a complete step-one
archive, and verifies exact restore into a fresh model. The uninterrupted and
restored paths both execute the next full forward/backward/update; complete
logits, loss, gradients, parameters, optimizer state and RNG must match bitwise.
The empty optimizer boundary, changed LR, training mode and CPU/MPS random draws
are covered. Each case rejects 33 malformed payloads, five invalid I/O requests
and one populated destination while preserving snapshots. Twelve full numerical
archives remain under `research_runs/byte-input-state-math002`, with hash receipts.
No archive is removed by the proof.

The independent saved reviewer reconciles every recorded tensor digest and shape,
all rejection snapshots and archive byte hashes without importing Torch or the
candidate. This is not a separate deserializer proof or a repeat of model math.
The adapter is not yet a complete training checkpoint: sampler, data identity,
schedule, clipping history, H200 and training-worker qualification remain later.
No real-data quality gain or adoption can be inferred.

Read the current evidence directories `state-admission002`, `state-math002` and
`state-review002` under `research/tokenizer-evidence/byte-composition-boundary-001`
for actual status. A frozen plan or source review alone is not numerical success.
