# Plan-derived checkpoint loading preparation

`research/byte_input_checkpoint_consumer_v1.py` provides a small caller for the
already qualified byte-input inference API. The adopted native recipe cannot
simply be relabeled as a byte model: its expected identity contains typed model
and codebook tuples that ordinary JSON converts to lists. The helper rebuilds
those tuples from the frozen plan's recipe, roles, binding, data and independently
rebuilt codebook. Recorded artifact JSON is used only as a consistency check
after that reconstruction, never as the source of the expected typed identity.

`prepare(plan_record, acceptance_record)` requires trusted path/size/hash receipts
for the original plan and an original accepted slot00–13. It verifies accepted
evidence, finished guard/STOP, source/asset/prerequisite pins and model identity.
Its scope intentionally excludes recovered slots: the already documented
representative convention uses first fresh raw-byte slot10, not a loss-selected
seed. This helper does not select a winner or authorize adoption. A successful
complete-panel independent review remains required before changing the incumbent.

`load(plan_record, acceptance_record, backend)` delegates model construction,
H200 tensor loading, exact identity/parent checks and RNG preservation to the
unchanged `research.byte_input_inference_v1.load` function. It returns the model,
typed expected identity and H200 receipt, which the existing scoring/cache APIs
can consume. The caller owns the shared research lock, source admission and
resource guards; the helper starts no worker and changes no model math. Its
original source/recipe identity does not include the unqualified direct-hash
optimization candidate.

`check.py` prepared slot10 from immutable accepted metadata and original codebook
assets in1.717086seconds under a60-second/1GiB CPU guard with8GiB reserve. Peak
RSS was297,041,920bytes. The expected outer model signature and nested codebook
signature remained tuples, all recorded JSON identity fields matched, and CPU
RNG was unchanged. AST inspection verified the exact delegated loader call and
absence of a training-step or seed-setting call in the helper. No model was
instantiated, no checkpoint tensor payload loaded, no GPU operation requested,
and no quality score or bootstrap draw produced. `result.json` includes the
bound plan/acceptance receipts and representative artifacts for later use.

Consumer SHA256: `fa2bb6b4023c6232a6e4a0927c73adfa34dfd2fca157a769cec2663e293e16ba`.

Runtime loading through the new caller is **not yet qualified**. If the completed
panel passes independent final review, first take the released shared worker
lock and perform one bounded loading integration of slot10 using this helper
and existing accepted inference checks. Reuse all existing model/checkpoint/cache
qualifications; this does not justify another broad test suite or quality sweep.
Bind the consumer and accepted integration evidence into any resulting adopted
recipe. If the scientific gate fails, do not promote this candidate merely
because the loading metadata was prepared successfully.
