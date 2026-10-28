# All character-model checkpoints removed by human request

On October 27, 2026 (Los Angeles), the human explicitly requested identifying
and deleting all character-based model checkpoints in this workspace. This
supersedes the former weight-retention rule for character models, including
historically selected, stopped-campaign and validation/pilot checkpoints. It
does not reopen any historical study or clear a stop marker.

The inventory examined all 3,582 .pt/.pth/.ckpt/.safetensors files under the
workspace, excluding dependency and Git directories. All were .pt files; a
separate extension check found no .bin/.pkl/.pickle/.tar/.zip model candidates
under research_runs or research_state. Checkpoints were inspected on the meta
device, without training or evaluating the holdout. Character identification
required a 65-row model token embedding matching the corpus character vocabulary
and absent legacy-default or explicit character tokenizer metadata. GPT-2
models have 50,257 rows. Two 16-token models were verified against their saved
intent as synthetic averaging/resume fixtures and retained.

Deleted 3,232 character-model checkpoint files: 25,067,336,573 logical bytes
(23.346 GiB). Retained all 126 GPT-2 model checkpoints, 222 numerical/probe
artifacts and two synthetic fixtures. All 350 retained binary SHA256 hashes
were verified after deletion. All 29,532 non-checkpoint files under research_runs
and research_state (excluding lock files) were verified unchanged at cleanup
completion, before subsequent maintenance updates to campaign progress. This
includes result records, configs, logs, failure records and stop markers.

`manifest.json` records classifications, evidence, paths, sizes and hashes;
`deletion-journal.jsonl` records each successful deletion;
`preserved-metadata.json` records the evidence-file hash inventory. The final
inventory has no unresolved classification. Free storage measured 41.69 GiB
at completion; it can change with other system activity.

Historical numeric findings remain available, but deleted character checkpoints
can no longer be reloaded, resumed, tensor-audited or re-evaluated. Historical
audit statements describe checks performed before this explicit retention
change. Character baselines remain prior numeric evidence only. Do not silently
recreate deleted weights or rerun closed campaigns. Tokenizer holdout remains
closed. The earlier losing-tokenizer-screen cleanup is recorded separately in
checkpoint-cleanup-001 and removed a further 3.444 GiB.
