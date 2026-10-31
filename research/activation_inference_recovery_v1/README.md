# Activation inference recovery v1 — preparation only

Activation001 remains permanently failed at its first inference entry. Its twelve
accepted quality trajectories, selection, retention receipts and sixteen retained
checkpoint files remain original evidence. This directory is outside the frozen
`src/**/*.py` plus requirements source set. The separate recovery001 plan is now frozen after the prerequisites below;
only its pinned controller/final entry points can execute against the new root. No model, checkpoint, corpus or test holdout was
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

1. **Completed prerequisite:** `source_binding.collect_source_import` replays
   the original freeze, fourteen accepted dispatches, selection and both
   retention boundaries; verifies sixteen protected checkpoint hashes and all
   438 original file identities; and anchors the exact preserved failure audit.
   It produces three loader-facing fresh-pair bindings. The new-root controller
   must freeze this import and call `verify_source_import` at its entry/review
   boundaries. Keep the original Session and new output root separate. The
   original remaining-training storage gate is inapplicable to this read-only
   import; no checkpoint allocation is made. The future controller must enforce
   its full new finite inference allocation separately, keeping every original
   resource limit.
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
4. **Quality arithmetic prerequisite completed:** `quality.analyze_imported_quality`
   verifies the complete original import before/after extracting the six fresh
   endpoints, checks each score's checkpoint lineage and uses the unchanged
   original statistical engine. It publishes only `quality_reviewed_inference_pending`.
   Implement the remaining independent final-report assembly. It requires
   all three accepted new timing pairs and an immutable final receipt. The
   original final wrapper cannot run on the failed root. No claim of formal
   closure or new adoption follows from this prerequisite.
5. Freeze the finite dedicated-root plan, exact external source hashes,
   imported evidence hashes and original source applicability before launch;
   commit/push first. Preserve failed attempt001, all selected/fresh pairs and
   closed campaigns. No paid compute, subagents or holdout scoring.


## Completed source import and quality arithmetic

The source-binding prerequisite passed28 contract tests plus actual original-root
integration in4.860s (peakRSS234,258,432bytes). Every protected checkpoint was
rehash-verified:16files,4,473,282,368bytes. All438 original files are imported;
422 are metadata. Failure hashes and the full original source/runtime are bound.
No corpus/model/checkpoint deserialization or numerical scoring took place.
[Review](../tokenizer-evidence/activation-preparation-001/inference-recovery-binding-prerequisite-001-review.json),
[exact import](../tokenizer-evidence/activation-preparation-001/inference-recovery-binding-prerequisite-001-import.json).

The quality adapter passed9 transport/lineage/revalidation tests and actual
analysis in9.333s (peakRSS237,191,168bytes). Source import was fully replayed before
and after the original predeclared arithmetic. Mean GELU-minus-SiLU CE is
+0.001550755925793319. Upper gate bounds for block lengths2048/4096/8192 are
+0.002470789135/+0.002569505209/+0.002788169461, all failing the negative-bound
condition. Descriptive95% intervals are respectively
[+0.000692766861,+0.002409247687],
[+0.000385030443,+0.002518861133], and
[+0.000105328239,+0.002716096356].
These condition on the same trained seeds and reused validation text; they are
not seed-population or independent holdout uncertainty. No new adoption or
formal closure follows; the new inference panel remains unrun.
[Analysis](../tokenizer-evidence/activation-preparation-001/inference-recovery-quality-prerequisite-001-analysis.json),
[review](../tokenizer-evidence/activation-preparation-001/inference-recovery-quality-prerequisite-001-review.json).

Both evidence wrappers have executed once. Their outputs and reviewed source
hashes must remain intact; do not rerun them to overwrite existing artifacts.


## Recovery001 freeze and execution handoff

The complete external worker/controller/final implementation passed the second
integration suite:34execution tests and7final tests, with actual disposable
bounded metadata/guards and explicit GPU/model/process/source-audit doubles.
The original OS supervisor/flock concurrency is separately covered by the real
metadata-child test. Initial suite001 passed37tests; a subsequent review found
that an early failed final attempt could permit a later controller step. Suite002
adds a root-level failure/partial latch and four regressions. All001 evidence
and the exact reviewed source snapshot remain preserved;002 is the applicable
execution/final prerequisite. Frozen original source never changed.

The real-source freeze independently checked all prerequisite hashes/guards,
original imports and current capacity, then created only four new-root files:
plan, exact original-import bytes, campaign lock and metadata lock. Capacity:
12,993,671,168bytes free;8,858,224,705bytes required remaining under the conservative
256MiB metadata envelope plus8GiBreserve; zero checkpoint allocation. Freeze
finished4.848s with222,904,320byte peakRSS. No inference ran during freeze.

Root:`research_runs/tokenizer-activation-inference-recovery-001`.
Plan SHA256:`29b8265276d6bedb933bbb3f522b179ffa1b8403b18ac7f7b62b5b6638ee5dde`.
[Freeze review](../tokenizer-evidence/activation-preparation-001/inference-recovery001-freeze-review.json),
[portable exact plan](../tokenizer-evidence/activation-preparation-001/inference-recovery001-freeze-plan.json),
[execution002](../tokenizer-evidence/activation-preparation-001/inference-recovery-execution-prerequisite-002-review.json),
[final002](../tokenizer-evidence/activation-preparation-001/inference-recovery-final-prerequisite-002-review.json).

Commit/push before dispatch. `controller step` runs exactly the next seed
3611/3612/3613, one per invocation; its fourth invocation writes the summary.
Only then may the separate `final` module publish the recovery terminal receipt.
Both require absolute new root and exact `--plan-sha256`. The old failed campaign
remains failed even after recovery closure. A failure or leftover temporary file
anywhere in the new root blocks retry/advancement. Do not edit any of the seven
frozen runtime modules or the original `src`/requirements after this freeze.
