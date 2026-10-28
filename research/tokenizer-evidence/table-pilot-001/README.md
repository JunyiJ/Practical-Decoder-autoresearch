# Table pilot001 review

Three width256 GPT-2 GQA table pilots passed under source30a309a and frozen
plan9fc40ed. Each ran20 updates at seed3291/LR0.00125 and exported a separate H200
with three snapshots. Source/runtime/corpus/request provenance was checked;
all six raw/H200 files were rehashed during review. Cache equivalence passed
at batches1/8 for raw and H200. No full-validation selection or test loss.

Untied raw/H200 pairs each occupy560,602,700 bytes; tied occupies303,284,813.
Measured unique FP32 parameter storage is114,352,128 versus62,888,960 bytes,
a45.004% reduction. KV cache allocation is unchanged. This pilot storage finding
is not a quality or speed conclusion. All original projected allowances remain
conservative, with the complete15-trial quality panel fitting above8GiB reserve.

`review.json` binds individual result hashes. The per-arm folders retain exact
requests, launch checks, sampled metrics, summaries and worker logs. Checkpoints
remain under research_runs/tokenizer-table-pilot-001 and are not uploaded to Git.

Initial model/controller CPU suite passed15 checks; an added timeout termination
check brought the controller suite to5. The subsequent complete table study,
benchmark/controller and model suite passed20 checks (table-001/cpu-validation.log).
No failed MPS attempts occurred. No pilot loss selects width, LR or arm.
