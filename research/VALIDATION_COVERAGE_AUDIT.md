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

### Full-validation audit: depth persists; MLA's largest-width seed sign changes

The frozen `validation-coverage-001` audit completed81 saved-checkpoint evaluations
in125.86seconds active controller time, with zero training runs. All checkpoint
and result hashes, worker contracts,111538 targets per model, source/corpus/full
bank, independently calculated paired means and reconstructed bootstrap/report
were verified on evaluator source `c4ff0a9` before documentation changes. The panel
contains36 selected stage2E and45 selected stage2F checkpoints, preserving repeated
references. The held-out test remains closed; checkpoint bytes are unchanged.

Each validation target except the first character was scored exactly once in
consecutive context128 windows, batch8, with a short unpadded final window and
token-weighted aggregation. This differs from the historical30-batch randomly
sampled fixed panel in coverage, overlap and window boundaries. It is not a new
independent dataset or a sliding maximum-context evaluation. Scores must not be
spliced into the old training study reports.

| Comparison (candidate minus reference) | Full-val delta width64 | Width128 | Width256 |
| --- | ---: | ---: | ---: |
| E: four-layer minus tuned two-layer SwiGLU | −.061794 | −.043728 | −.022303 |
| F: MLA minus fixed four-layer MHA | +.017336 | +.007729 | −.007308 |
| F: MLA minus tuned four-layer MHA | +.013453 | +.006146 | −.007100 |

**Depth:** every seed still improves versus all three E references at every width.
All conditional95% text-block bootstrap intervals are below zero at block sizes
2048/4096/8192. The depth improvement survives this evaluation-coverage change;
its previously measured extra training/parameter cost remains unchanged.

**MLA:** every seed still worsens versus both four-layer MHA references at64;
128 has a worse mean and mixed seeds. At256, all three seeds now improve against
both references. MLA minus tuned MHA seed deltas are−.001353/−.014357/−.005590 for
1011/1022/1033; versus fixed MHA they are−.005353/−.015992/−.000579. Seed1033's sign
therefore changes from the historical sampled panel's positive difference.
The4096-character conditional interval is[−.011499,−.003231] versus tuned MHA
and[−.011526,−.003243] versus fixed MHA; all three block-size intervals remain
negative. This strengthens the largest-width diagnostic under broader coverage,
but neither establishes a causal width interaction nor reverses the frozen
width64 selection. Four-layer MHA remains the historical selected incumbent.

Intervals resample paired contiguous text blocks jointly across the three fixed
trained seeds. They are conditional text uncertainty, not training-seed confidence
intervals, independent holdout confirmation or multiplicity-adjusted tests.
There are55/28/14 blocks at the three sizes, and residual text dependence may
remain. Report the full sensitivity set and per-seed differences; do not select
a favorable block size or infer universal significance. Shared seeds do not make
this adaptively reused text an independent validation source.

Full-validation mean CE, widths64/128/256:
E originalGELU1.672886/1.592635/1.551230; fixed2LSwi1.659265/1.575020/1.547150;
tuned2LSwi1.658437/1.576396/1.544050; four-layer1.596643/1.532668/1.521747.
F originalGELU1.670300/1.585966/1.547213; cheaper2LSwi1.654783/1.574416/1.531199;
fixed4LMHA1.592719/1.528306/1.519030; tuned4LMHA1.596602/1.529889/1.518822;
MLA1.610056/1.536035/1.511722. Fixed/tuned repetitions remain separate.

This audit shows why broader evaluation should precede stronger architecture
claims: a small seed-level contrast can depend on the scoring panel even without
retraining. It cannot isolate whether coverage, boundary placement, or overlap
caused the sign change. Future context comparisons need a common evaluation panel
as well as native-context scores, and fresh controls under repaired RoPE
initialization. Context/batch training is authorized; an interaction-aware redesign
is being discussed and no new training grid has been frozen or launched.

Evidence: [plan](../research_runs/validation-coverage-001/plan.json),
[report](../research_runs/validation-coverage-001/report.json),
[review](../research_runs/validation-coverage-001/review.json).
