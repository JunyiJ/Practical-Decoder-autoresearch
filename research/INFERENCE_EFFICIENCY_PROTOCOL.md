# Saved-checkpoint inference efficiency diagnostic

Status: bounded diagnostic protocol; launch requires the frozen JSON plan. Human
accepted +0.01 CE nats/token as the quality-loss ceiling. Historical architecture
selections and the completed full-validation audit remain closed. No training,
optimizer resume, checkpoint rewrite or held-out test access is needed here.

## Comparators and bounded workload

Use the selected GQA and tuned MHA checkpoints from stage2D (both two layers),
and selected MLA and tuned MHA checkpoints from stage2F (both four layers).
Widths64/128/256 are separate strata. Select the smallest declared seed from each
study (444 and1011) before performance measurements; this samples implementations,
not training-seed quality uncertainty. All checkpoint hashes and source lineage
must match the reviewed original reports. Do not compare GQA2L to MHA4L.

For each pair and width, evaluate batches1/8 and prompt/decode lengths32/32 and
64/64. This is 24 paired workload cells, 48 model/workload measurements, using
12 saved checkpoints. Total positions stay within the trained context128.
Use identical predetermined validation text for each model in a pair, including
teacher-forced continuation tokens, so model outputs cannot change the workload.
No quality claim follows from these timing inputs. A future long-context study
must train fresh models and establish its own workload contract.

Measurement loop: three unscored warmups per model and nine measured
paired repetitions, alternating reference-first and candidate-first. Both models
may be resident to allow alternation; report that fact and do not interpret pooled
allocator readings as isolated model memory. Time synchronized prefill separately
from the entire sequential cached-decode loop, then derive milliseconds per decode
step and aggregate batch tokens/second. Include the normal model forward path and
Python loop; exclude loading, cache allocation/reset, correctness checks and input
copying from timed regions. This is an implementation benchmark, not a serving
system benchmark or an optimized-kernel upper bound. Emit every paired repetition,
median ratio and interquartile range; do not claim timing repetitions are independent
hardware or training replicates.

Count actual allocated cache-buffer bytes at equal capacity128, and logical used
bytes at the matched sequence lengths. Record current MPS allocator memory with
its actual label. This backend helper exposes current allocation, not a device
peak: report peak device memory as unavailable rather than relabeling current
allocation or process RSS. Sampling cannot guarantee a true peak either.

## Correctness, bounds and interpretation

Before timing each model, compare cached prefill plus single-token decode logits
against full-sequence logits on the same tokens. Record tolerances and maximum
errors. GQA must derive cache heads from the same resolver as attention, including
`gqa_half_head32`; MLA caches compressed latents and rotary keys. Neither changes
saved active weights. Legacy RoPE checkpoint loading uses the existing strict
in-memory adapter; saved bytes remain unchanged.

Before freezing: focused and full tests, actual serial MPS correctness pilots,
checkpoint/source/corpus/runtime/full-knowledge hashes, reviewed clean committed
and pushed source. Hard caps:120 seconds per paired workload, one hour
for the round,512 MiB output, one worker, first failure stops, no automatic retry.
A uniquely named readonly plan must define the exact panel, order and raw result
contract; resource limits are not permission to extend the panel. Launch only from the verified readonly JSON plan, not this Markdown alone.

Initially report descriptive efficiency evidence. For the subsequent prospective
adoption protocol, use the human's +.01 quality ceiling and a predeclared meaningful
benefit. Candidate starting thresholds are at least10% lower median decode time
or at least25% lower allocated KV-cache bytes; report any latency regression
alongside memory savings. These thresholds are proposals, not measurements or
already frozen acceptance criteria. Both effect size and repeat stability matter.

Quality must be assessed separately with matched validation targets, paired fresh
training seeds where new training is involved, and conditional text-block intervals.
An upper conditional interval below+.01 alone does not establish population seed
non-inferiority. Report each paired seed and the uncertainty limitation explicitly.
Stage2F full-validation results can inform an adaptive efficiency shortlist;
stage2D has only its original sampled quality panel and must not be mislabeled a
full-validation result. No retrospective winner change follows from this diagnostic.

## Implementation findings before launch

The existing GQA width policy derives KV heads without writing `n_kv_heads` into
config. Generation and the legacy comparison utility previously required that
missing field, while direct cache construction defaulted to all query heads.
The repair shares the attention resolver with the cache and rejects an explicit
cache head count inconsistent with attention. It preserves trained parameter
shapes and does not change historical training results. The standalone comparison
loader now uses the existing historical RoPE weight adapter.

Validation of the repair:570 project tests and nine serial saved-weight MPS
checks passed; maximum cached/full-logit error8.11e-6. Initial unscoped pytest
collected archived test copies and failed; explicit `pytest tests` resolved
collection without touching historical evidence. The timing implementation is `src.research.inference_bench`; its focused tests
cover paired ratios, cache accounting, result tampering and fixed panel selection.
The actual MPS pilot is recorded separately before freezing.

Numerical correctness tolerance is absolute1e-4 and relative2e-5 for FP32
MPS cached/full logits. Raw maximum absolute errors are retained. Timing
repetitions retain both models resident and run with gradients disabled. Decode
throughput counts processed teacher-forced continuation tokens; token sampling,
network/server overhead and end-to-end time to first token are outside scope.
The interval of decode positions is [prompt, prompt+decode), giving exactly
32 or64 single-token forward calls per repetition.

Prelaunch validation:583 project tests passed. The two width256/batch8/64+64
MPS pilots completed in6.16 and7.51 wall seconds respectively (GQA/MHA and
MLA/MHA), with all cached/full checks passing and immutable checkpoint hashes.
Pilot timing samples are excluded from the diagnostic report. The120-second
per-job bound leaves substantial headroom in these feasibility checks; no runtime
guarantee is inferred for another machine or workload.
