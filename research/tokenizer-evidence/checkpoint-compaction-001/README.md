# Lossless checkpoint archival

The user explicitly authorized proactive safe archival/removal while preserving
selected models, fresh confirmation pairs and failure evidence. No eligible
irreversible weight deletion was identified. This operation instead preserves
all logical checkpoint bytes in local Zstandard-compressed PAX tar archives.

`plan.json` pins 35 completed numerical/screen checkpoint files (25,008,600,879
logical bytes), 154 protected loose checkpoint files and 3,740 non-checkpoint
files. The three prospective groups are nine numerical prerequisites, ten
completed campaign002 screen weights, and sixteen completed campaign003 screen
weights. Incomplete campaign results remain descriptive, never selected or
promoted. Failed partials, all selected models and all fresh confirmation pairs
are excluded. No model loading, training, corpus or holdout access occurs.

`archive.py` executes one group at a time and stops once available space reaches
the unchanged full-panel requirement plus 1 GiB of headroom. It uses Zstandard
v1.5.7, level3, a 1 GiB window and one compression worker. A monitor enforces a
600 s subprocess limit, combined parent/child RSS below 4 GiB, 8 GiB free reserve
and 12 GiB archive limit. The overall guarded operation is bounded at 1,800 s.
The full streamed archive is verified against the original member hashes before
any source is unlinked; receipts and an fsynced removal journal are retained.
Selected/fresh/failure weights and all metadata are rehashed after the operation.

Archives live in `research_archives/checkpoint-compaction-001/` and must remain
local and protected. They are not disposable caches and must not be pushed to
GitHub. Original paths in historical receipts remain logical identities. A direct
old-path audit requires restoration, or a new explicit archival-aware audit must
stream the pinned archive and compare member hashes; do not rerun old procedures.

For restoration, first verify the archive SHA256 in its receipt and reserve the
full decompressed size plus the 8 GiB reserve. Decompress using `zstd -d --long=30`
and extract only the receipt-listed regular members to a fresh staging directory.
Reject unexpected names, links, duplicate entries and size/hash mismatches.
After every member passes verification, publish it exclusively to its original
absent path; never overwrite existing evidence. Keep the archive until all restored
bytes are independently verified. This does not authorize resuming a closed run.

Initial copy probes found that `ditto --hfsCompression` and sparse writes did not
reduce allocated blocks on this filesystem. They changed no original. Their
verified scratch duplicates are removed by the archival procedure. A small test
verified the archive roundtrip and rejection of an incorrect expected member hash.

## Completed outcome

All three groups completed.35 checkpoint files (25,008,600,879 bytes) are preserved
in13,454,327,571 archive bytes, reclaiming11,554,273,308 logical bytes.154 excluded
loose checkpoints and3,740 non-checkpoint files verified unchanged. Free space at
completion was33,493,016,576 bytes. `completed.json`, `journal.jsonl` and the
independent `logical-evidence-review.json` bind the outcome and exact restoration
identities. No checkpoint data was discarded; never remove these archives.
