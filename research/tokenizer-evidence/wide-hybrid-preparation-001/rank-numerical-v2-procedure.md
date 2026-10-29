# Separate rank-policy numerical prerequisite procedure

This is a fresh CPU validation procedure, not a retry of the closed combined
wider-hybrid fixture. The historical failure and all its artifacts stay closed.
No old continuation test is imported or invoked. This procedure does not freeze
or authorize MPS resource pilots or the quality campaign.

Freeze the script, source/runtime, retained geometry/cache evidence, corpus,
verified local GPT-2 cache and all 27 phase config hashes before execution.
Cases are width512 hybrid, gated_mla, kda, each at divisor8,4,2 in that order.
For each case, run baseline, partial, resumed in separate OS processes, serially.
No process contains two model instances or more than one case. Stop the entire
procedure at the first exception, resource failure, STOP or interrupt. Existing
attempts cannot be retried or resumed by the controller.

Use actual production CPU trainer/model/optimizer/tokenizer/sampler/checkpoint
reader and EMA implementation, seed3490, LR0.00125, four updates, context128,
batch8, four threads, eval_every2/eval_iters1. Use the four-step production
schedule, with H200 snapshot steps1–4 for this correctness-only fixture. These
settings test short continuation; they do not estimate 1,200-step quality or
stability. Only ordinary train/validation paths execute; never evaluate test.

Baseline runs four updates and streams complete model, optimizer, EMA, torch
RNG and loader RNG hashes at the existing save callback, without keeping a
serialized checkpoint. Partial runs one update via the existing positive time
budget, calls the production disk checkpoint writer, and exits. Resumed starts
in a fresh process, reads that checkpoint normally, executes updates2–4 and
requires exact equality with every baseline state hash. No BytesIO, mmap,
production resume lifetime change or source/math patch is used. Only the test
save callback is intercepted to hash full end states and keep one partial file.
The normal trainer still retains its deserialized checkpoint; its memory is
measured under the unchanged guard, not presumed safe from object counts.

Each parameter's first/middle/last EMA values are independently compared after
every update against a double-precision scalar recurrence (H200 beta computed
with exp/log), rtol3e-6/atol1e-7. Full tensor EMA equality between baseline and
resume complements this sampled recurrence check; the sample does not certify
every scalar independently. At the resumed endpoint, load the complete stored
EMA into the existing model, require exact parameter equality and unchanged
persistent buffers, and reject inference-only weights for optimizer resume.
Disk H200 export and real MPS cache behavior remain resource-pilot prerequisites.
Existing nine-case rank geometry/cache evidence is reused only after its saved
source/test/evidence hashes are verified, not relabeled as newly executed tests.

Every phase has an independent durable CPU guard:600seconds,4GiB RSS,8GiB disk
reserve. The parent uses the existing owned-process-group supervisor with a
600second phase timeout and4MiB log cap; it kills/reaps its owned child group on
failure/interrupt. The child checks STOP through the guard observer. Global and
run advisory locks are inherited and verified. A2GiB cap covers each case's
partial checkpoint and all three phase logs/metadata; the complete nine-case
reservation is18GiB plus8GiB reserve, rechecked before every phase. All partial
checkpoints and failures are retained. No cleanup is part of this procedure.

Plan and receipts live in a fresh ignored numerical run directory. Keep full
raw per-phase logs/guards/receipts and partial checkpoints there. After success,
an independent read-only review must verify all27 successful guards/process
records, original file hashes, matching baseline/resume complete-state digests,
EMA checks and unchanged source. Export one compact consolidated trace and one
numerical review for the future campaign, binding the original evidence hashes.
Do not duplicate27 raw phase documents throughout every campaign receipt. The
actual consolidated sizes and inventory propagation must pass the final metadata
high-water audit before any pilot/quality plan freeze. Failed cases cannot yield
a passing numerical review or be silently replaced.
