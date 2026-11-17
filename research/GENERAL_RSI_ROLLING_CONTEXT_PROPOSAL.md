# Proposed fixed-weight rolling-context diagnostic

Status: recommendation responding to the human question on2026-11-16 Pacific.
No new training or numerical evaluation is launched or admitted by this note.

## Evidence and question

The completed two-depth study confirms the context256 recipe at depths4/16.
Its same-weight evaluation-policy term accounts arithmetically for78.06%/86.53%
of the native-context contrast. About67% of that term occurs in the first32
positions after common128 resets. Later positions still improve. These are
saved-score descriptions, not causal shares or proof of dependence beyond128.
See `GENERAL_RSI_DEPTH_CONTEXT_POSITION_RESULTS.md` and
`tokenizer-evidence/general-rsi-depth-context-position-001/{analysis,verification}.json`.

Question: do tokens129–256 before a target improve prediction when the immediately
preceding128 tokens are already always available? This removes short-history
segment resets as an explanation of that specific paired comparison.

## Proposed controls

Use all six retained context256-trained time-EMA checkpoints, three at each depth.
Keep weights unchanged. On exactly the same within-document targets compare:

| Condition | Supplied history |
| --- | --- |
| A | Original common128 nonoverlapping-reset policy; reuse bound saved scores |
| B | Immediately preceding128 tokens, rolling by one target |
| C | Immediately preceding256 tokens, rolling by one target |
| D | Same as C, but deterministically permute only the oldest128 tokens |

Score only the final next-token prediction per window. Recent128 token IDs and
target ID must be identical in B/C/D. D preserves length, positions and old-token
multiset. Its order perturbation is distribution-shifting, not a pure semantic
causal intervention. Preserve unchanged/noninformative permutations without retries.
B-minus-A measures reset-policy improvement on this target sample. C-minus-B is
the primary incremental older-context contrast. D-minus-C is a secondary order
sensitivity diagnostic. Benefit without shuffle sensitivity can reflect topic or
lexical information; neither result proves reasoning or excludes all dependency
mechanisms. No fresh generation or optimizer updates are needed.

Freeze target selection without model scores, using existing development documents
and targets with at least256 preceding tokens. Sample at most8 eligible targets per
document, balanced where possible across four original128-reset offset bins.
Record exact rules, seed, short-document exclusions, actual bin counts and hashes
before model scoring. Use the same inventory for every model/condition. This is
an eligible-target development diagnostic, not a replacement for full validation.
Keep the original test unopened. Report per-document/source/seed and reset-bin
effects. Predeclare practical thresholds and joint uncertainty across depths before
scores; document bootstrap remains conditional on the three model bundles.

Use uncached full-window recomputation. A cache carrying information older than
128 tokens into B would invalidate the contrast. Verify window indexing, target
alignment, saved-A reconstruction, causal masking and RoPE position conventions,
including common-shift numerical checks. Do not extend context128-trained models
to256 or alter frozen model semantics to fill an unqualified factorial cell.

## Interpretation and next action

If B recovers most reset loss while C adds no supported practical improvement,
prioritize evaluation-history coverage and limit claims about older context.
If C consistently improves on B, evidence supports useful information beyond128
for these fixed weights and sampled documents. If D harms prediction, ordered
older context matters under this perturbation. If it does not, order-insensitive
information remains plausible. Mixed results define source/depth limits rather
than triggering outcome-selected sampling or a larger-context training sweep.

## Literature and complete cost boundary

Dai et al., [1901.02860v3](https://arxiv.org/html/1901.02860v3), section3.1:
one-target sliding evaluation supplies the full window and avoids fragmentation,
but costs more than chunked scoring. Their recurrence architecture is absent here.
Khandelwal et al., [1805.04623v1](https://arxiv.org/html/1805.04623v1), sections3–5:
fixed-weight truncation and order perturbations probe context usage; early targets
with insufficient history are excluded. LSTM word-level findings do not establish
our transformer/subword dependency range. These exact versions/sections were read.

At most108x8=864 targets, six models and three newly scored conditions gives15,552
sequence evaluations, plus bounded qualification and timing. A uses saved scores.
Qualify/timestamp a training-only fixture first; do not inspect scientific scores
in the pilot. Forecast the entire frozen inventory including model loading,
scoring, verification and output publication before launch. Proposed engineering
ceiling120seconds and scientific-worker ceiling3600seconds are planning bounds,
not measured runtime promises. If the full sample does not fit, revise the proposed
inventory prospectively before any scientific score, or decline admission.

One serial MPS worker under the shared nonblocking lock; existing RSS4GiB/MPS6GiB
guards and actual host-memory headroom checks. No checkpoint output. Reserve256MiB
for all per-target metrics, manifests, logs, verification and publication, plus
10GiB disk reserve and2GiB headroom. Measure actual artifact sizes and free space.
Reuse qualified model loading; implement only new sampling/window/permutation
logic and its necessary numerical checks. Preserve all selected weights, failures
and first results. The full timing/memory contract must pass before execution.
