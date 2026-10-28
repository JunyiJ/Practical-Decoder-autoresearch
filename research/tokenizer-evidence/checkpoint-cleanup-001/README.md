# User-authorized checkpoint cleanup

On October 27, 2026 (Los Angeles), the human requested deletion of non-promising,
non-chosen checkpoints to free storage. This overrides blanket retention only
for the explicitly reviewed files in `manifest.json`.

Deleted 30 raw/H200 checkpoint files from 15 completed, unselected screening
configurations across tokenizer quality, GQA, horizon, MLA and hybrid rounds:
3,697,923,044 logical bytes (3.444 GiB). Selection used the saved frozen per-family
LR or horizon decisions, never a newly chosen score threshold.

Kept all 108 other quality checkpoint files in those five rounds, including
selected screens and every fresh confirmation pair. Their saved SHA256 hashes
were verified after deletion. All 394 non-checkpoint evidence files in those
rounds were verified unchanged, including requests, results, validation block
sums, training summaries, logs, plans and selection decisions. Pilots, failure
evidence and historical campaigns were not cleanup targets. No training ran.

The manifest preserves each deleted path, size, original SHA256, selection
reason, and the retained-file verification. Historical audits remain evidence
of checks performed before cleanup; tensor audits and re-evaluation of deleted
weights can no longer be repeated without retraining. Do not rerun closed
controllers or silently regenerate missing files.

Free space measured 18.36 GiB after cleanup, an increase larger than the 3.444
GiB of files deleted here; other filesystem activity also changed available
space. Do not attribute that entire increase to this cleanup. Both prospective
table panels fit that observation, including the unchanged 8 GiB reserve, but
actual storage must be checked again before dispatch. Width256 is preferred by
the prospective design. Pilot/executable plan validation and source freeze
remain required; no quality or adoption claim follows from freeing storage.
