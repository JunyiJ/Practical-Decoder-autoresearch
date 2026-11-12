# Retained-output probability diagnostic — 2026-11-11 Pacific

## Result

The descriptive replay supports investigating repetition reinforcement. It does
not establish that repetition causes increased probability or that a penalty
will improve generation quality. No model was trained or adopted.

All264 frozen sequences completed in68.503 guarded worker seconds, with peak
RSS643,956,736bytes. All2,304 original greedy choices matched exactly through the
original cached inference path. No new generations, checkpoint writes or test
access occurred. The three retained native long-horizon models are unchanged.
The diagnostic's indexing checks passed, and source/input hashes were verified.

Evidence: `tokenizer-evidence/general-rsi-repetition-probability-001/plan.json`,
`attempt-01/rows.jsonl`, `attempt-01/complete.json`, `attempt-01/guard.json` and
`analysis.json`. The plan SHA256 is
`fa739ce96da79ea7274559f3371f559f12c5ce4e0a672751c376e9b6ebb619d7`.
The264 sequences comprise72 original long-horizon continuations and the same64
fixed training windows scored under each of three models. All token probabilities,
argmax IDs and strict ranks are retained. Natural windows predict63 targets each.

## Aligned repeated copies

For each of the23 original terminal loops, take the smallest period reported by
the frozen scorer. Compare aligned tokens in its last three complete copies.
Values below are third-minus-first mean log probability in nats per token.
Each loop receives equal weight within its model. No inferential confidence
interval or new quality gate is attached to this selected set.

| Native slot | Loops | Positive / negative | Mean delta | Median delta | Strictly rising copy means |
|---|---:|---:|---:|---:|---:|
|01|9|7 / 2|+0.163674|+0.036797|6|
|03|7|7 / 0|+0.096590|+0.048844|5|
|05|7|4 / 3|+0.017517|+0.013763|3|

Overall18/23 loop means rise,5 fall, and14 have strictly increasing copy means.
144/212 aligned token positions rise from copy1 to copy3. The descriptive
loop-weighted mean is+0.098774. The49 other retained outputs have no terminal
loop and are excluded only from this aligned-copy statistic; their probabilities
remain in the evidence. Terminal-loop selection does not test the onset of all
repetition. Increased prefix length and absolute position remain confounds.

## Contrary evidence and natural references

The64 training windows contain86 repeated-trigram endpoints. Compare each with
the first occurrence of the identical trigram within that same window, retaining
its different context length and position. Mean log-probability changes are
+1.447656,+1.396091,+1.063845 for slots01,03,05. Positive counts are79/86,77/86,
77/86. These references are not matched to the generated-loop distribution and
must not be interpreted as a comparative effect size. They show that increased
probability upon recurrence also accompanies ordinary training text.

The five decreasing loop means also contradict a universal monotone account.
Greedy choices can remain top-ranked while their probability decreases. Original
weak relevance and consistency ratings are unchanged. Neither increased repeat
probability nor a reduction in that probability is itself a quality measure.

## Interpretation and next action

Local inference: repeated sequence context may help sustain some loops, but the
same behavior can support useful natural repetition. This strengthens the case
for testing a narrowly targeted intervention and weakens the rationale for a
blanket token-repeat penalty. The primary literature and applicability limits
remain in `GENERAL_RSI_REPETITION_BOUNDARY.md`; no new external gain is claimed.

Before training, make one bounded context-order intervention using these exact
retained examples. Freeze its source, masks/permutations and inputs before
scoring. For each terminal loop, keep the third copy, absolute positions, total
prefix length, prefix token multiset and immediately preceding two tokens fixed.
Permute only the earlier two copies except their final two tokens, with one
predeclared deterministic permutation per example and no retries for an
unhelpful permutation. Record unchanged/noninformative cases. Compare intact and
permuted conditions under the same model using the original cached path. This
controls position and length while changing earlier sequence order; it does not
isolate repetition from semantic or syntactic context.

Include the86 natural repeated-trigram endpoints under each model as a necessary
reference. At each endpoint, keep the target and preceding two tokens fixed,
and permute the earlier first-occurrence-to-current-context region. Record exact
ranges and unchanged cases; do not silently shrink the cohort. Report paired
log-probability changes, greedy-choice changes and per-model distributions for
both generated loops and natural repeats. No pooling into a quality decision.
A selective reduction in loop support would strengthen a targeted-loss rationale;
similar reductions in natural repetitions would emphasize collateral prediction
cost and the need for ordinary CE controls. Null results would weaken the
sequence-order reinforcement hypothesis at these contexts.

This next diagnostic remains inference-only, with a600-second cumulative worker
ceiling,64MiB retained-output allowance,10GiB reserve and the existing memory
limits. The current replay provides timing and memory anchors; forecast the
complete pair count before launch. Reuse all qualified model loading, cached
inference and indexing. Test only the new intervention's index/multiset/local-
context invariants. No new decoder, generations, corpus, training panel or
repeated historical qualification is authorized by this report alone. General
research authorization covers the bounded diagnostic after its concrete freeze
and resource admission.
