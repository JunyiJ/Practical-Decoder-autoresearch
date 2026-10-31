# Activation inference recovery v1 — preparation only

Activation001 remains permanently failed at its first inference entry. Its twelve
accepted quality trajectories, selection, retention receipts and sixteen retained
checkpoint files remain original evidence. This directory is outside the frozen
`src/**/*.py` plus requirements source set. It does not authorize a launch and
has no inference entry point. No model, checkpoint, corpus or test holdout was
loaded by the metadata prerequisite.

## Lock finding and tested reader

The original `tokenizer_wide_hybrid_process_v2.run_process` already wraps its
preflight, active observation and failure scans in `writer._serialized()`.
The original metadata writer also serializes its capacity checks, atomic JSON
publication and bounded stream appends. These callers do not need a new lock
around their existing locked calls: nesting would contend against themselves.
The uncovered caller is `tokenizer_activation_inference.budget()`, which calls
`snapshot()` while the parent can atomically replace a control document.
Inherited global/campaign lock descriptors establish campaign ownership, but
parent and child share them, so those descriptors do not serialize publication.

`census.synchronized_snapshot(writer)` uses the existing, independently opened
metadata flock. It takes two complete snapshots while holding the lock, rejects
any census difference without retry, and checks a two-second elapsed allowance
including lock wait and scans. It is only callable outside an existing serialized
block. The enclosing watchdog must still bound blocking filesystem operations.
No guard/census work belongs inside a timed inference region.

The reader preserves the original file-count, group-size, regular-file, symlink,
hard-link and declared-path checks. Distinct temporary files retain their full
byte count; only the original exact same-directory temporary/final inode alias
is counted once. The original writer continues to reserve old plus temporary
bytes during publication while readers wait. A failed leftover temporary file
is counted, not ignored. Missing paths still raise; no blanket retry or missing
file suppression is introduced. Inventory revalidation is not content hashing;
immutable provenance needs separate exact hash verification.

The guarded prerequisite passed seventeen tests, including deterministic old
race reproduction, rejecting uncooperative disappearance, two readers waiting
for actual atomic replacement, and the original bounded pipe supervisor with
one disposable child doing thirty control writes and synchronized scans.
Additional tests preserve alias/path/file/group-cap rejection, temporary overlap
accounting, lock and census deadlines, and no retry on inconsistent inventories.
The original campaign inventory and all 422 non-checkpoint file hashes were
unchanged; the full frozen source fingerprint still matches. The check did not
rehash checkpoint payloads; the preceding failure audit verified all sixteen.

Evidence: [review](../tokenizer-evidence/activation-preparation-001/inference-recovery-census-prerequisite-001-review.json),
[log](../tokenizer-evidence/activation-preparation-001/inference-recovery-census-prerequisite-001.log),
[guard](../tokenizer-evidence/activation-preparation-001/inference-recovery-census-prerequisite-001-guard.json).
The evidence-producing wrapper has executed once and must not overwrite its
outputs. A changed prerequisite needs a separately named evidence attempt.

## Required work before any inference

1. Build and test an external binder that reads activation001 without modifying
   it, verifies its freeze, all accepted training/selection/retention records,
   exact source/runtime/input identities and every protected checkpoint hash.
   Explicitly import the preserved failed inference receipt without accepting
   it as timing evidence. Bind the old root separately from the new output root.
2. Build a new-root worker/controller using the tested reader for its unlocked
   budget checks. Reuse the original bounded writers and already-serialized
   process supervisor without monkeypatching their globals or frozen source.
   Original budget/quality-binding entry points tied to the failed root cannot
   be used as a new dispatch path. Audit all reused function dependencies and
   freeze their applicability. Test failures, process exit, no-reentry, STOP,
   resource limits, immutable receipts and serial seed acceptance with doubles
   and disposable metadata, before actual inference.
3. Keep original timing contract: seeds3611/3612/3613, batches1/8, warmups2,
   repeats8, alternating order, FP32 strict MPS/four threads, prefill32/decode64,
   training-token inputs, cache/reset correctness, 600-second jobs, RSS4GiB,
   driver6GiB, reserve8GiB, per-job5MiB/log4MiB, panel16MiB. Import trained H200
   files; no training, weight regeneration, cleanup or new quality scoring.
4. Implement independent final reporting using original six fresh rows,
   selection, paired block statistics and unchanged quality gate. It requires
   all three accepted new timing pairs and an immutable final receipt. The
   original final wrapper cannot run on the failed root. No claim of formal
   closure or new adoption follows from this prerequisite.
5. Freeze the finite dedicated-root plan, exact external source hashes,
   imported evidence hashes and original source applicability before launch;
   commit/push first. Preserve failed attempt001, all selected/fresh pairs and
   closed campaigns. No paid compute, subagents or holdout scoring.
