# CPU resource failure review and diagnostic preparation

The combined wider-hybrid continuation process remains closed after exit90.
No numerical retry, MPS pilot or quality trajectory was launched in this review.
Its first test passed; the two homogeneous-control continuation checks remain
unverified. The lost threshold sample cannot be reconstructed after process exit.

## Confirmed diagnostic defect

The inline watchdog printed its trigger to stdout under pytest's default capture,
then called `os._exit(90)`. A hard exit bypasses pytest's reporting/flush path.
Only the completed first-test dot survived in the redirected log. Post-exit disk
was above reserve and the last observed RSS was close to4GiB, making RSS the
likely trigger, but the surviving evidence does not identify it conclusively.

The new opt-in `src/research/cpu_prerequisite_guard.py` opens a fresh independent
journal before starting its thread. It records limits, PID, elapsed time, peak
RSS and free bytes, and flushes/fsyncs the reason before a resource exit90 or
observer-error exit91. It does not dispatch tests or alter historical training
watchdogs. Existing journals cannot be overwritten by a new guard instance.
Polling still cannot guarantee observation of every transient peak or prevent
an operating-system kill before logging.

Eleven lightweight checks passed, including subprocess exits for deliberately
tiny RSS limits, impossible reserve, timeout and observer errors while stdout
and stderr are redirected. These are guard unit tests, not failed-model retries.
The log is `cpu-guard-tests.log`. No torch/model worker is needed by these tests.

## Fixture memory: observations and hypotheses

Static inspection found three overlapping lifetimes worth addressing in a new
procedure. The fixture keeps the entire serialized partial checkpoint in a
BytesIO closure throughout resumed training. The trainer keeps the deserialized
`ckpt` and its `state` variables through the resumed function. Online EMA load
clones tensors, and `state_dict()` clones them again when saving. Model weights,
optimizer state, gradients and hash/serialization temporaries coexist with
these objects. Some optimizer tensors may share storage, so summing every
Python reference would overcount actual memory.

For all-gated-MLA512, the meta counts imply967,624,896bytes for raw model,
optimizer and one EMA tensor set, excluding serialization/RNG metadata. This
explains why an in-memory fixture can cost substantially more than the model's
253,899,264bytes of parameter storage. It does not prove the failed phase, a
specific leak, or a production/MPS memory requirement. Repeated family tests in
one process may retain allocator memory; that remains a hypothesis.

Before any separately reviewed numerical procedure, specify a single family
per process, phase markers outside pytest capture, unchanged resource thresholds,
and an explicit serialized-buffer lifetime. Closing the BytesIO after a complete
non-mmap deserialization is a candidate fixture-only change to validate on small
synthetic data first. Do not change model math, training budgets, data boundaries
or production resume behavior to make a fixture pass. Do not rerun the failed
combined process or treat the guard implementation as authorization for retry.
Disk-export validation and a complete fitted/frozen panel remain required before
MPS work. Storage and incomplete continuation checks still block launch.
