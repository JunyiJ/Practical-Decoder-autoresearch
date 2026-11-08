# What the screening code control actually matches

The first-rate screens found real bytes better than arbitrary codes by
0.059273213 nat/token but worse than native embeddings by 0.143202188. The
existing `screen-diagnostic002` analysis also found that both composition models
improve after unseen inputs, with an additional real-byte advantage. Those are
local observations at one rate, not fresh confirmation or causal explanations.

This bounded, training-only analysis asks whether the existing control also
matches byte concentration and literal fragment overlap. It introduces no new
model hypothesis or campaign. The frozen protocol already states that the
control does not match every byte marginal; these measurements quantify that
limitation using the exact codebooks actually used in the accepted screens.

## Definitions fixed before measurement

Reconstruct the raw and nonlexical screening codebooks from the frozen public
assets and the original screening key, and require their complete identities
to match the accepted slot01/02 artifact records. Preserve all token lengths
and unique whole-token codes. Use widths 1–6, matching the six composer filters.
Count only token content, excluding boundary, padding and special markers.

A fragment is shared when it appears in at least two distinct token IDs in the
specified view. Repeating the same token does not by itself establish sharing.
Report full-vocabulary, training-type and training-occurrence views. The last
weights each token by its count in the original training split, not by its count
in the realized optimization stream. The training split has 10,653 distinct token
types. No validation or holdout text is encoded or scored by this analysis.

## Measured local structure

For the training-occurrence view, the fraction of content-fragment occurrences
whose fragment also occurs in another training token type is:

| Fragment width in bytes | Real bytes | Nonlexical codes |
|---|---:|---:|
| 1 | 99.997% | 100% |
| 2 | 98.715% | 56.074% |
| 3 | 97.644% | 0.682% |
| 4 | 89.515% | 0% |
| 5 | 73.690% | 0% |
| 6 | 59.622% | 0% |

The type-based view gives the same qualitative distinction: 97.379% of distinct
token/three-byte-fragment incidences are shared for real bytes versus 0.268% for
codes. The control has no shared content fragments of widths 4–6 among training
token types. Across the complete regular vocabulary, control four-byte sharing
is very small but nonzero; the zero claim above is specific to the training
vocabulary. The exact counts and all three views are in `result.json`.

Byte frequencies differ as well. In the training-occurrence view, the ten most
frequent bytes account for 62.649% of content bytes for real coding versus 13.406%
for the control. Corresponding Shannon byte entropies are 4.7744 and 7.7842 bits.
Both views have exactly the same total content length and same number of windows
at every width; these differences are not caused by changing token lengths.

## Interpretation and contrary evidence

The control matches architecture, parameter count, token lengths and unique
whole-token codes, but the actual training inputs differ greatly in byte
concentration and repeated literal fragments. A raw/control quality difference
can reflect these differences together with useful lexical information. It
cannot separate morphology, substring reuse, frequency concentration or their
optimization consequences. This is a limit on the mechanism claim, not grounds
to label the control defective or replace it after observing outcomes.

Literal fragment overlap is not a measure of learned filter activations,
gradient sharing or semantic similarity. Even unique multi-byte fragments use
shared byte embeddings and filters; the control still permits neural parameter
sharing. Boundary-containing windows are excluded here but are present in the
model, and can create further common patterns. This analysis establishes input
structure, not which structure caused a loss difference. It examines only the
existing screen key, not fresh confirmation keys.

The practical contrary observation remains: both composed arms lost to native
at the first rate. Complete all nine frozen screens and the fresh confirmation
panel. Preserve the protocol's separate practical and lexical-coding contrasts;
do not introduce a new control, new endpoint, subgroup adoption rule or extra
training from this diagnostic. No new external-paper result is claimed here;
the existing campaign's literature basis and applicability limits remain in its
original evidence note and protocol.

## Execution and preserved reader failure

The corrected analysis completed in 2.952 seconds with 166,264,832 bytes peak child
RSS, under its 60-second process timeout. It loaded no model or weights and
assigned no new random roles. Reconstruction used the original deterministic
code permutation and verified the accepted identities, corpus/tokenizer hashes,
all token lengths, all raw token spellings and window/byte count accounting.

Attempt `control-structure001` stopped at the content-length assertion before
producing statistics. Its reader used `bytes(array('H', ...))`, which copies the
packed two-byte representation. The separate corrected attempt converts the
integer values with `bytes(list(...))`. Original source and failure are retained;
`recovery.json` records the exact one-expression correction and source hashes.
All assertions, groups, widths and metrics stayed unchanged. This was an
analysis implementation failure, not a failed scientific criterion.
