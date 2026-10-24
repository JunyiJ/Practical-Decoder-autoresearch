# Saved-checkpoint validation coverage audit, October 24, 2026

The human authorized broader validation before context/batch robustness training.
This is an evaluation-only diagnostic of the frozen stage2E and stage2F selected
recipes. It cannot change their historical selection decisions and does not use
the held-out test split. No training, optimization, checkpoint rewriting, LR-grid
extension or rerun is performed.

## Frozen panel and scoring

Exactly81 checkpoint evaluations: E's four selected families × three widths ×
three seeds =36; F's five selected families × three widths × three seeds =45.
Include both fixed and tuned references even where the selected recipe is
identical. Bind each checkpoint hash, both studies' plans/selections/reports/
reviews, corpus hash, evaluator source/commit/runtime and a full knowledge-bank
snapshot in a new uniquely named plan. Check the manifest against the original
selected rows. This is a broader audit of depth and MLA finalists, not a claim to
cover every architecture from earlier studies.

Use original training context128 and evaluation batch8. Score target indices
1 through N−1 of the validation split once each, in consecutive disjoint windows.
Reset context at each window, retain the final short window without padding, and
aggregate loss by target count, not by batch means. This is full coverage with
window resets, not sliding maximum-context perplexity. No random sampling or
new validation split is introduced. The corpus is the same character-tokenized
Shakespeare data, split80/10/10. The test portion is never scored.

Historical checkpoint model/data/device/Torch contracts must match. Use the
strict RoPE compatibility loader introduced in7577129: discard only the dormant
position table in memory after validating its contract, then strictly load all
active tensors. Saved bytes and source-study hashes remain unchanged. Forward
math is unchanged; initialization differences in newly constructed models are
overwritten by saved active weights. Exact historical/new logits are checked on
CPU and on an actual MPS checkpoint. This adapter is not authorized for legacy
optimizer continuation.

## Uncertainty and interpretation

Store per2048-character block loss sums/counts. Pair each candidate with each
reference within its own study/width/seed panel. Report per-seed differences and
three-seed mean differences. Resample text blocks jointly across paired models
and seeds, with2000 bootstrap draws and seed20261024; aggregate adjacent blocks
to report2048/4096/8192-character sensitivity. Use token-weighted ratios so the
partial final block is handled consistently.

These intervals describe text-block resampling conditional on the trained models
and this validation text. They are not training-seed confidence intervals,
independent holdout evidence, or multiplicity-adjusted significance tests. Text
blocks may remain dependent; block-size sensitivity does not prove independence.
Comparisons are descriptive. No architecture winner, LR or historical acceptance
rule is changed in response to this audit. Full-coverage scores can differ from
the historical fixed sampled panel because coverage, overlaps and boundary
placement differ. Do not attribute changes solely to evaluation sample count.

## Finite execution and review

Run one evaluation worker at a time, with120seconds/job,3600seconds active round
time and512MiB new output. No competing training. Stop at the first failure,
preserve failed outputs, and require inspection before any retry. Resume only
finished-job boundaries on the same frozen plan/source. Respect STOP. Verify all
checkpoint hashes before and after the round, and the current checkpoint before
and after each worker; verify source, lineage artifacts and runtime throughout.
No new writes are made inside historical study directories.

Before launch: pass unit checks for exact coverage/tail weighting, paired
bootstrap behavior and malformed result rejection; run the complete repository
suite; run one actual serial MPS saved-checkpoint pilot. The pilot is correctness
and feasibility evidence only. Freeze committed/pushed source and the full bank
before launching the81-job audit. At completion, independently reconstruct
aggregates, verify hashes/results and append findings to the bank, best-config
context, learning guide and HTML report.
