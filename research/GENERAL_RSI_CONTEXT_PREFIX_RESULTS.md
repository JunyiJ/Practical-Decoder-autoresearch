# Natural-text prefix context results

The frozen context-use criterion passes. Increasing the prefix before a common
32-token target block from 48 to 96 tokens lowers mean CE in all three retained
models and both sources. The equal-parent, source-balanced mean improvement is
0.014213562 nats/token. This is evidence of useful context within the existing
model limit, not a new model or evidence of benefit beyond 128 tokens.

## Frozen comparison

The same 128 training spans, 64 distinct documents per source, were used under
three selected EMA parents. Each span supplies identical final 32 target tokens
for prefixes of 16, 48 and 96 tokens. Inputs contain 47, 79 or 127 tokens. Later
targets see earlier targets through causal teacher forcing. The experiment uses
standard short-sequence inference with a reset RoPE origin. No window was selected
by model score, replaced, excluded or rerolled after launch.

Plan SHA256: `2629d477da0ab7b26e763023fd8e662a18b34d0c68a8b03f728f2981ce143299`.
The primary rule requires negative CE96-minus-CE48 in each model and negative
source means after equal model averaging. Every required contrast is negative.

| Parent | CE16 | CE48 | CE96 | CE96 minus CE48 | Favorable windows |
|---|---:|---:|---:|---:|---:|
| 1 | 3.909600082 | 3.821704373 | 3.806985837 | -0.014718536 | 81/128 |
| 3 | 3.984747967 | 3.898161912 | 3.883519899 | -0.014642013 | 70/128 |
| 5 | 3.938137889 | 3.860666642 | 3.847386506 | -0.013280136 | 77/128 |

Source means for CE96-minus-CE48 are −0.011168391 on FineWeb-Edu and
−0.017258732 on Wikipedia. The descriptive CE48-minus-CE16 mean is −0.083984337.
The added 48 tokens help less on average than the earlier 32-token addition.
There are adverse windows: only 81, 70 and 77 of 128 improve by parent. All first
per-token losses, including adverse cases, remain retained.

This source-balanced training sample is not the corpus token-weighted validation
metric. Conditions share targets, and models share windows. The 36,864 scored
target occurrences represent 4,096 unique corpus targets, not independent examples.
Prefix truncation also changes position origin; this is not the closed fixed-
position permutation experiment. No population confidence or exact effective-
context threshold is claimed. Previous generated loops fit within 95 total tokens,
so this result does not make eviction their cause or establish better generation.

## Execution checks and resources

The new indexing fixture passed nine synthetic alignment checks, two invalid-
boundary rejections, identical targets across conditions, and exact longest-
condition agreement with direct full-input logits and token losses. It took
3.423 guarded seconds with no optimizer update. Completed model,
restart and document-scoring prerequisites were not repeated.

All 1,152 scientific sequence evaluations passed, with 36,864 scored target
occurrences and 97,152 input tokens. All 5,200 frozen source bindings and the
three selected checkpoint hashes remained unchanged. In-memory parameter hashes
also remained unchanged. There were no failures. All first scores were retained.
Guard time was 33.438 seconds. Peak RSS was 917,159,936 bytes
and peak MPS allocation 1,114,324,992 bytes. Review measured
2,445,330 evidence bytes before this report. No training, checkpoint
writes or generations occurred. Validation was not scored and test stayed unopened.
The worker exited and the shared lock was reacquired.

## Next action

Prepare the separate context128/context256 implementation and bounded execution
pilot in `GENERAL_RSI_CONTEXT_LENGTH_PILOT_NEXT.md`. The source audit identified
fixed wrapper identities, input limits, runtime architecture contracts and cache
shapes. Those restrictions must remain intact for historical models. Implement
new semantics in a separate namespace and validate only the new context path.
No longer-context quality campaign is admitted, and no model is adopted.

Evidence: `tokenizer-evidence/general-rsi-context-prefix-001/` and
`tokenizer-evidence/general-rsi-context-prefix-fixture-001/` retain the complete
plans, inventories, checks, scores, analysis and reviews.
