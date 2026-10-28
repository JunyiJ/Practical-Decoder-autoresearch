# Proactive checkpoint cleanup002

On October 28, 2026 (Los Angeles), the human authorized cleanup of non-critical
checkpoints moving forward rather than silently blocking on storage. This is a
retention change, not permission to change experiment outcomes or reopen runs.

Removed **42 raw/H200 files**, **13,438,900,670 logical bytes (12.516 GiB)**:
28 files from seven successful closed resource-pilot rounds and 14 files from
seven unselected table/width512 attention screening configurations. Losing
screens follow the saved family LR decisions. Successful pilot weights are
superseded by completed audited quality studies; their original reports remain.

All **148 other tokenizer checkpoints** were SHA256-verified before and after
cleanup, including every selected screen and every fresh confirmation raw/H200
pair in all seven quality rounds. All **659 non-checkpoint files** in the tokenizer
run directories were hash-verified unchanged, including failure records, requests,
results, logs, plans, timing and selection evidence. Historical non-tokenizer
files, probe artifacts, corpus and test holdout were not touched. Only allowlisted
`.pt` files were removed, under the shared worker lock with no training process.

`manifest.json` retains original paths, sizes, result and checkpoint hashes,
reasons, decision/review bindings, all retained-file hashes and per-file deletion
journal. `cleanup.py` implements the reviewed one-shot plan/apply path; its plan
SHA256 was `2794f4740b48e4415cb57102b554d5d0af7412a9db0f923ac9d1a3f6e728919e`.
The completed manifest must not be reused to regenerate or rerun closed studies.
Deleted weights cannot be reloaded, independently re-evaluated or tensor-audited;
historical audit receipts describe checks completed before removal.

Observed free bytes changed from 17,645,760,512 to
31,121,584,128 (28.984 GiB). Concurrent filesystem
activity can affect that difference; the exact attributable deletion is the
logical-byte total above.

## Future full-panel retention accounting

`project_retention.py` and `prospective-retention-projection.json` calculate the
complete 27-trajectory width512 hybrid campaign high-water bound after audited
stage-boundary cleanup: **30,631,281,412 bytes (28.5276 GiB)**,
including an unchanged 8 GiB reserve and 256 MiB round metadata/log allowance.
The original per-trajectory allowances (model/optimizer/EMA/export, metadata,
15% allowance) are unchanged. All 27 possible selected rank triples are enumerated.
The peak is three final selected screening pairs plus nine fresh confirmation
pairs, with D/2 bounding each family. Pilot, LR and rank-stage peaks are lower.
The complete future panel fits the post-cleanup observation by
490,302,716 bytes; recheck immediately before
any freeze or dispatch. This is planning feasibility, not measured resource fit.

No training ran, no plan was frozen, and no current v1 guard was changed or
bypassed. The new retention design still needs versioned manifest-aware runner
integration and boundary audit/cleanup tests. Numerical prerequisites and actual
pilot memory/time/export reviews remain outstanding. The failed combined CPU
fixture stays closed. All rates/seeds/order/selection/quality-memory gates and
closed-holdout boundaries remain unchanged. The retain-all 53.29 GiB projection
remains preserved as historical evidence, not silently rewritten.
