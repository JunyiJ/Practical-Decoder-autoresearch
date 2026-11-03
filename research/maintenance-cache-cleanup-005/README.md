# Research archive retirement 005

The human requested a fresh acceptance/dependency audit of large research archives,
overriding old blanket protection labels. This pass deleted exactly two obsolete
payloads and preserved selected models, fresh confirmations, and original evidence.

## Result

- Removed allocation: **105,086,976 bytes (100.2 MiB)** across two regular,
  user-owned files, each with one hardlink.
- Observed free-space increase during deletion: **105,037,824 bytes**.
- Final capacity observation: **8,775,909,376 bytes (8.173 GiB) free**.
- 454 retained evidence/payload files rehashed unchanged after deletion; all
  202 surviving RAW archive identities unchanged. Eight retained RAW archives
  from the affected hybrid study and their H200 checkpoints were checked against
  original archive/acceptance hashes. Unrelated large archives were not needlessly
  rehashed; their receipt/stat identities and original metadata were audited.
- No worker or application was stopped, signalled, restarted, or launched.

`completed.json` and the fsynced `deletion-journal.jsonl` report the actual operation.
`manifest.json` binds exact paths, hashes, sizes, inodes, ownership, link counts,
disposal reasons, retained evidence, dependency proof, and the executing script.
The shared research-worker lock was held during planning and deletion. Process
and exact-file open-handle checks passed, including immediately before unlinking.

## Actual classification, independent of old labels

`archive-audit.json` reclassifies all 197 historical RAW archives from original
acceptance stages/seeds and final selection records. The result is 160 fresh
confirmations, 30 selected screens, six selected final-rank models, and one
obsolete losing-control screen. The old classification strings were not used.

`fresh-epsilon-supplement.json` independently binds the six newly archived epsilon
endpoints to successful original acceptance events and their frozen fresh seed
panel. All six remain, including the control arm. The final retained total is
**202 RAW archives: 166 fresh confirmations and 36 selected models**.

The final size/inode census is `remaining-archive-census.json`; allocation is
deduplicated by device/inode. Two other archives remain: the mixed original
pytest diagnostic archive and a 44KB successful diagnostic-fixture archive.
Neither was deleted solely because it was old or archived. The earlier audit
also found no new failed `.pt.tmp` payload to retire; storage027 failed before
creating an archive and its original failure evidence remains.

## Narrow exception: intermediate winner lost the final control selection

In `tokenizer-hybrid-001`, the gated-MLA screen at LR 0.0025 won its own learning-rate
search, but **KDA won the actual control selection**. The original protocol in
`src/research/tokenizer_hybrid_study.py` selects one control, then runs fresh pairs
only for hybrid and that control. Recomputing selection from all six screen
records gives KDA. The completed final review and all six accepted fresh records
close the hybrid/KDA comparisons for seeds 3011, 3012, and 3013. There was no
pending gated-MLA confirmation or inference. Current research sources and the
adopted epsilon recipe have no exact dependency on the retired payloads.

The deleted files were:

| Exact repository-relative path | Stored bytes | Allocated bytes |
| --- | ---: | ---: |
| `research_archives/protected-raw-storage-010/08-ec50b63af2fc.tar.zst` | 78,524,687 | 78,528,512 |
| `research_runs/tokenizer-hybrid-001/screen-gated_mla-w64-s3001-lr0.0025/checkpoints/h200.pt` | 26,556,995 | 26,558,464 |

The compressed archive hash was checked against its original receipt; its sole
regular member was streamed through decompression and checked against the
original 105,473,449-byte RAW checkpoint hash. No archive extraction or numerical
experiment was needed. H200 matched its original acceptance hash.

This exact exception supersedes storage010's `selected_or_fresh` label and the
later blanket claim that all 197 archives must remain. It does not rewrite those
immutable historical receipts or grant permission to remove any other model.
Old restoration indexes now include one intentionally retired member: do not
attempt to restore that member. Original requests, configs, metrics, logs,
acceptance reviews, final review, selection, sources, and archival receipts remain.

## Capacity and next action

The 8 GiB reserve plus the requested extra 2 GiB totals 10,737,418,240 bytes.
At the final observation, this target remains short **1,961,508,864 bytes
(1.827 GiB)**. System memory reported 56% free; swap used 72.38 MiB in a 1 GiB
allocation. Swap and running applications were untouched. These are point-in-time
measurements; other activity can change filesystem free space.

The next final-LayerNorm-scale panel is not yet measured or frozen. Concurrent
full-model integration can proceed. As a planning reference only, the predecessor
six-trajectory high-water requirement is:

`6 × (469,762,048 RAW + 117,440,512 H200) + 268,435,456 metadata + 8,589,934,592 reserve = 12,381,585,408 bytes`.

The predecessor includes RAW/H200 temporary publication within its aggregate
allowances, metadata serialization overlap, and same-inode publication accounting.
With extra 2 GiB, that reference scenario requires 14,529,069,056 free bytes,
leaving a **5,753,159,680-byte reference deficit**. Existing closed confirmations
are retained evidence, not new-panel outputs to subtract from this requirement.
New integration, pilot, and quality outputs must be measured before a new launch;
the reference scenario is not a new-panel approval or a reason to shrink it.

Concrete escalation: provide at least **1,961,508,864 additional local free bytes**
to meet reserve plus headroom, or identify an external volume for a separately
verified offload of retained archives. The complete future panel may require more,
as the reference scenario shows. Selected/fresh bytes cannot simply be deleted.

The historical RAW classification is exhausted for now. Earlier cleanup003/004
already exhausted the verified unused external cache classes. Wake on a new
dependency closure, new unused cache/download material, a measured panel budget,
or added/offloaded storage; avoid repeating unchanged scans or notifications.

The resume record received a narrow atomic update under the shared lock: corrected
the inaccurate all197 sentence, recorded 202 surviving archives and the latest
capacity/retirement references. All other concurrent research fields were retained;
there was no live capacity blocker to clear. `resume-reconciliation.json` binds
the before/after identities. No schedule was created, restarted, or changed.
