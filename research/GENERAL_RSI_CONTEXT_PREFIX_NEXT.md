# Natural-text prefix context: next bounded diagnostic

## Local basis and decision

The fresh-optimizer continuation investigations have not produced a model above
the selected EMA parents. The latest independent-batch comparison fails its
frozen criterion in every parent. Close those completed diagnostics and retain
their negative outcomes. No batch-factor, warmup or learning-rate sweep follows.
See `GENERAL_RSI_BATCH_AVERAGING_RESULTS.md` and its original analysis.

The general RSI plan names context length as a question to investigate only
with local evidence. Existing facts justify a bounded measurement, not a training
campaign. The model uses 128-token segments. The train document index has 7,815
FineWeb-Edu documents with median 719 tokens and 1,992 Wikipedia documents with
median 644.5 tokens; all are at least 128 tokens. The closed context-order study
found ordinary repeated-trigram prediction sensitive to earlier order, but its
selected repeat cohort does not measure general-text prefix-length benefit.

Contrary evidence matters: every retained native completion fits within 95 total
tokens, so eviction cannot explain those loops. More context may be unhelpful or
poorly used. Do not repeat the prior permutation intervention, generate new loops,
or infer that a longer model will fix relevance or repetition.

Rank a fresh within-limit context ablation ahead of further continuation tuning:
it tests a separate measured structural limitation using retained models, while
those tuning families currently lack a favorable local direction. It is cheaper
than a new architecture or length panel. No context expansion is yet admitted.

## Primary literature and applicability

Khandelwal et al., [arXiv:1805.04623v1](https://arxiv.org/html/1805.04623v1),
sections 3–4.1, measure context use by dropping older tokens and comparing losses
on eligible common targets. They emphasize perturbation mismatch and variation
by word type. Their PTB/WikiText-2 word-level LSTM results motivate this method;
their reported useful lengths cannot be transferred to our GPT-2-token MLA model.

Dai et al., [arXiv:1901.02860v3](https://arxiv.org/html/1901.02860v3), sections
3.1–3.2, distinguish fixed-context limits from context fragmentation. Section 4.3
warns that effective context comparisons across models need a common baseline.
Their segment-recurrent models and positional scheme differ from ours. We compare
each retained model with itself and do not copy their recurrence or claim their
gains locally. Prefix ablation below the existing limit cannot establish benefit
beyond that limit.

Inference: older natural context may still improve prediction near the upper
end of the present window. If so, measure a longer-context implementation's cost
and semantics before considering training. A null or adverse result weakens that
priority. Neither outcome changes the closed repetition or continuation decisions.

## Prospective comparison and prediction

Use only the three retained selected native EMA parents, slots 1/3/5. Before
scoring, freeze an unused selection seed, 64 distinct training documents per
source, and one randomly located 128-token span per document. Select without
model scores. Use the identical 128 spans under all three models. This is a
source-balanced diagnostic, not the corpus token-weighted evaluation distribution.

In each span, define the final 32 tokens as the target block, beginning at corpus
position t. Compare C = 16, 48 and 96 preceding tokens before that block. For
condition C, input tokens are `[t-C:t+31]`, with lengths 47, 79 and 127, and the
scored logits are positions `[C-1:C+31]`. Targets are always `[t:t+32]`. All input
and target tokens must remain within one source document. Use causal teacher
forcing: later targets see preceding targets, so available history grows across
the block. Report that growth rather than calling each block a constant context.

Primary contrast is CE(C=96) minus CE(C=48). Consistent support for useful added
prefix requires a negative mean in each parent and a negative source mean in
both sources after equal averaging across parents. Report C=48 minus C=16 and
all per-window/per-token scores descriptively. Keep every first result and do not
replace a window or condition based on its score. Standard short-sequence inference
resets the RoPE position origin; this is a truncation experiment, not the former
fixed-position order intervention. Record that scope and validate target alignment.

Total 3 models × 128 spans × 3 conditions = 1,152 sequence evaluations and
36,864 scored target occurrences. There are 4,096 unique corpus targets, shared
across all models and conditions. Total input tokens processed are 97,152.
No training updates, validation, test, generations, checkpoint writes or adoption.
Do not pool these correlated conditions as independent confirmation runs.

If the fixed support criterion passes, the next decision is whether a separate
longer-context pilot has a credible measured resource and modeling contract.
This does not authorize a full length-quality panel. If it fails, do not expand
context merely because documents are long; revisit other evidence-grounded
bottlenecks. No interpretation can claim a solution to the retained short loops.

## Minimal checks and complete cost

Reuse qualified selected-weight loading, corrected FP32 inference and CE arithmetic.
Implement only common-target slicing and source-balanced input selection. Check
synthetic boundary/index cases, identical target IDs across all conditions,
document containment, exact input lengths, count totals and unchanged model hashes.
Cross-check the longest condition's selected logits against a direct full-input
reference on a separate fixed fixture. Preserve any failed attempt; do not repeat
model, optimizer, restart or complete document-scoring prerequisites.

Conservative per-sequence allowance 0.028 seconds comes from the previous maximum
1.79-second 64-window probe. The 1,152 evaluations cost about 32.3 seconds at that
allowance. Add 60 seconds for loading, new checks, input/hash binding and evidence:
provisional complete forecast 92.3 seconds. Measure the new slicing path before
admission; use a 180-second ceiling only if complete measured costs fit. Retain
64 MiB evidence plus 64 MiB overlap, no model serialization, 10 GiB reserve and
2 GiB extra headroom. Limits remain RSS 4 GiB and MPS 6 GiB.

Next implement and check that slicing path, freeze fresh source/role/input bindings
and measured admission, then run one serial worker under the nonblocking shared
research lock. Check actual processes before launch. A full training/confirmation
campaign needs its own future design and complete retained-output forecast.
