# Four-condition extension of the fixed-weight rolling diagnostic

## Prospective human steering

On 2026-11-16 Pacific, while the originally frozen B/C evaluation was active,
the human requested four conditions on identical targets and six saved models.
The B/C scorer, source bindings, first scores, order and original analysis remain
unchanged. This separate extension supplies A from accepted saved scores and
executes only D after the active worker and its analysis finish. No partial B/C
quality effects were inspected to choose this extension or its shuffle seed.

Use the same context256-trained time-EMA models, depths4/16, three bundles each,
and all72876 targets with zero-based within-document position p>=256. Every
condition includes the same terminal EOT targets and excludes the same early
positions. No training, new generations, original-test access or endpoint selection.

| Label | Input before target tokens[p] |
| --- | --- |
| A | tokens[128*floor((p-1)/128):p], existing reset128 policy |
| B | tokens[p-128:p], original active rolling128 evaluation |
| C | tokens[p-256:p], original active rolling256 evaluation |
| D | permutation(tokens[p-256:p-128]) followed by tokens[p-128:p] |

A is extracted by matching document, global target position and token ID from
each saved `time_ema-common128-validation.json` in the completed depth/context
study. Require its hash to match that study's accepted completion audit. Reuse
only the matched subset, never compare its old100416-target mean to72876 targets.
At p divisible by128, A and B have identical inputs; check their saved losses
within1e-5 as an additional scorer control. This is not a quality gate.

## Evidence, prediction and limits

The original completed position analysis found66.74%/67.06% of the native256
versus common128 evaluation-policy gain near resets; late benefits persisted.
See `GENERAL_RSI_DEPTH_CONTEXT_POSITION_RESULTS.md` and its original analysis.
That is local evidence for the reset question, not evidence for D's unseen result.
The user-directed D intervention tests a new ordering question and does not repeat
a closed optimizer or training study.

Khandelwal et al., https://arxiv.org/html/1805.04623v1, section3 and section5.1,
equation2, perturb only selected context spans on fixed weights. Their word-level
LSTM study found distant token identity could matter when ordering effects were
small; they also discuss the train/evaluation distribution mismatch. The exact
version and these sections were retrieved and read. Their result is a plausible
alternative hypothesis, not a prediction established for our subword MLA models.
Dai et al., https://arxiv.org/html/1901.02860v3, section3.1, supplies the already
read stride-one evaluation rationale. No recurrent state is introduced here.

- B-A <0 measures benefit from keeping recent history rather than artificial
  resets, conditional on these fixed models and selected targets.
- C-B <0 is the decisive test of the older128 tokens' additional predictive value.
- C-D <0 indicates sensitivity to ordering/placement in that older prefix.
- D-B <0 measures whether predictive benefit remains after this shuffle. An
  unresolved C-D alone cannot establish that benefit survives unchanged.

Report all four deltas and magnitudes. If C beats B and D does not, describe loss
of supported benefit under permutation, not proof of equality D=B. If D also
beats B, preserved token identities are a plausible source of residual benefit.
Partial retention and partial order sensitivity can coexist. Positive, negative
and unresolved outcomes are all retained. This intervention shuffles **subword
tokens**, preserving their multiset but potentially breaking whole words, syntax,
relative placement and the older/recent boundary. A performance cost does not
isolate semantic structure, and no outcome proves reasoning. One permutation per
target does not estimate variability over all shuffle realizations.

## Fixed shuffle and analysis contract

One permutation per target, shared across every depth and bundle. Form UTF-8 key
`rolling-older-shuffle-v1|621116220001|{split}|{document}|{p}`. Use the first16 bytes
of SHA256 as a big-endian unsigned seed for NumPy PCG64; draw permutation(128).
No rejection or reroll, even for fixed points or repeated token IDs. Keep exact
recent128 token IDs and their order, target ID, length256, and local RoPE policy.
Recompute every window from tokens, with no KV cache or hidden-state carryover.
Record per-document permutation and full input digests. Use the existing qualified
FP32 corrected RMSNorm, no_grad, batch4 and final-position-only projection.

The original B/C three-contrast intervals remain their frozen historical analysis.
The integrated four-condition report uses a separately declared joint family of
12 contrasts: B-A, C-B, C-D, D-B at each depth, plus depth16-minus4 for each.
Use100000 shared paired document bootstrap draws, PCG64 seed621116220002,
Bonferroni family alpha.01, quantiles.01/24 and1-.01/24. Average three bundle
loss sums before bootstrap, with actual matched target counts as denominator.
Report per-bundle and source results. Bounds condition on these reused development
documents, models and fixed shuffles, not population-seed uncertainty. Require
all three bundle signs plus the adjusted interval direction before calling an
effect consistently supported. No equivalence margin or recipe-adoption gate.
Do not overwrite the original three-contrast intervals with extended-family ones.

## Execution and full incremental cost

CPU preparation and saved-A binding can run while B/C scores. No new MPS worker
may start until the previous supervisor exits successfully and the shared lock
is available. Then qualify only D's new permutation semantics on synthetic data
and a fixed training prefix at both depths. Check exact recent-prefix preservation,
old-token multiset, deterministic target-keyed batching and identity-permutation
control. Compare D batched/scalar/full logits at inherited3e-5+3e-4 relative and
CE against float64 logsumexp within1e-5. Verify unchanged model parameters.
Measure128 D targets per depth. Preserve original artifacts if qualification fails.

Provisional D forecast uses the accepted rolling256 timings:3*72876/128 times
the sum of depth4/depth16 costs, plus50% margin and600seconds overhead. Replace
with measured D timing including shuffle construction before admission; require
<=12hours for the full six-model D pass, with600seconds for qualification and
300seconds for final analysis. No early quality stopping or model dropping.

Reserve another256MiB for D scores, analysis, failure and atomic/log allowance,
plus the whole original256MiB B/C allowance conservatively and12GiB disk reserve.
No weight/optimizer/EMA copies are produced. Current free space exceeds this
requirement. Recheck actual host memory, processes and disk when the queued
execution begins. Preserve RSS4GiB/MPS6GiB guards and >=1GiB host reclaimable-page
proxy, normal pressure before load, <=2 during scoring. One serial worker uses the
shared nonblocking lock. Honor stops immediately; no paid compute or subagents.
After D, verify all four aligned inventories and sources, report results, reconcile
state and publish authorized source/evidence. Do not invent another campaign.
