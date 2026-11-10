# Standalone composer reuse qualification

Frozen before numerical execution. The measured motivation and primary-paper
scope are in `GENERAL_RSI_COMPOSER_EFFICIENCY_BOUNDARY.md`. This is a component
engineering test, not a quality campaign or production change.

Compare the original occurrence-wise composer with sorted whole-token unique
composition and inverse expansion. Its custom backward uses an FP32 membership
matrix multiplication to sum upstream gradients. The existing byte-embedding
reduction stays unchanged. Reuse lasts for one invocation only.

Use one private initializer62111010001 and upstream seed62111010002. Each
upstream entry is a standard-normal FP32 draw divided by512. Cases, in order:
IDs0–1023 once each; ID50256 repeated1024 times; alternating0/50256; the first
historical training batch of equal-time slots4,6,8. Verify historical hashes
against the already audited384-batch inventory. These three fixed batches are
a bounded sample, not a performance characterization of all384 batches.

For each case, execute original then reuse three times each. Require bitwise
repeatability of output and all18 parameter gradients within each implementation.
Require reuse output and every gradient element to satisfy
`abs(reuse-original) <= 2e-6 + 2e-4*abs(original)`. This engineering tolerance
allows FP32 reduction-order rounding; it does not establish optimizer equivalence.
Check inverse-expansion backward against sequential CPU FP64 accumulation with
the same fixed tolerance. Require unchanged parameter hashes and CPU/MPS global
RNG states. Stop at the first failed case, retaining both full output/gradient
payloads. Do not loosen tolerances or proceed to timing after failure.

Only after all correctness gates pass, measure every case. Run two warmup pairs,
then five ABBA blocks (A original, B reuse), yielding10 samples per arm per case.
Synchronize immediately before forward and after backward. Include unique-ID
construction, codebook indexing, all composer checks, inverse mapping, custom
gradient aggregation and full backward. Exclude gradient clearing, CPU snapshots
and guard checks equally. No transformer, optimizer or transfer timing is included.
Require the median of the three historical reuse/original median-time ratios
to be at most0.85, each historical ratio below1, and the all-unique ratio at most1.10.
These practical gates reflect required benefit and a low-duplication control,
not a statistical confidence bound. Failure closes this candidate as specified.

The plan binds both new source files and reused sources. One local numerical
worker holds the shared nonblocking research lock. Limits:900 cumulative seconds,
at most two attempts only for demonstrated harness/resource repair, RSS4GiB,
MPS driver6GiB and10GiB free-space reserve. Each failed pair has7,899,032 raw
tensor bytes (two outputs and two sets of725,235 gradients). Two failed pairs
plus one full serialization overlap and1MiB metadata total24,745,672 bytes,
within the complete64MiB cap used for admission. Successful cases retain hashes
and error metrics, not extra tensor copies. No checkpoint payloads are created.

On success, a separate full-model optimizer/restart and whole-update timing
qualification remains necessary before any new scientific panel. On failure,
retain the exact gate and evidence, stop this candidate, and identify the next
evidence-based question without changing any historical scientific decision.
