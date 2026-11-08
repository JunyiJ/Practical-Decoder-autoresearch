# Padding work in the existing training distribution

This deterministic existing-data analysis uses the frozen corpus, exact GPT2
vocabulary and original training split. The encoded267,688training IDs match
the established SHA256. No model is initialized, no sampler draws are made,
and neither validation nor holdout is encoded or scored. The script completed
in about0.22seconds in the observed invocation. `result.json` records original
input/source hashes, full length counts and the exact arithmetic derivation.

The byte composer includes begin/end boundaries in every code. Training codes
average5.3334positions and range from3to18. Nevertheless, the current forward
path selects codebook rows of width130 and runs all six filter banks across that
width before masking invalid windows.130is needed for the largest vocabulary
entry, not for every token or every batch.

The source sampler independently selects two512-token blocks and reshapes them
to B8T128. Enumerating all267,176admissible block starts gives exact block-maximum
counts. For two independent uniform starts, the batch-maximum CDF is the square
of the block-maximum CDF. This yields an expected batch width14.0673, including
the minimum width6required for all filter banks. No scientific seed trajectory
is replayed or used to choose the result.

| Filter-forward arithmetic | Multiply-adds per1024-token batch |
|---|---:|
| Current fixed width130 |4,408,320,000|
| Expected width trimmed to batch maximum |357,169,101.5|
| Trimmed / current |0.0810216|

For widthL, the count is1024 times the sum of
`(L-w+1) * (15*w) * channels[w]` for w1through6 and channel counts
25,50,75,100,125,150. The expectation weights this count by the exact ordered
pair distribution across71,383,014,976possible start pairs; it does not enumerate
all pairs or approximate them by random draws. The resulting91.90% reduction
applies only to these six linear filter forward operations, excluding backward,
activations, pooling, highway/projection, decoder, validation and device effects.
It is neither an observed runtime reduction nor a model-quality result.

## Mechanism and constraints for a later optimization

The existing computation excludes every window extending beyond the individual
code length from max pooling. Cropping only trailing padding to
`max(6, maximum_code_length_in_this_batch)` should retain every eligible window
and preserve the mathematical model. Both raw-byte and length-matched nonlexical
codes have the same lengths, so this opportunity applies to both arms.

This is a source-derived prediction, not numerical equivalence evidence. Matrix
shape changes can change floating-point reduction behavior, and the complete
parameter/gradient, short-token zero-gradient, padding rejection, save/resume and
cache contracts still apply. Any new path must validate selected rows including
discarded padding before cropping, retain the general maximum130, and handle
arbitrary later inputs. Do not hard-code the observed training maximum18 or
use it to reject legitimate vocabulary entries.

The smaller immediate optimization is direct-byte hashing, whose finite CPU
equivalence and timing are already measured in `../serialization-analysis001`.
Batch-local padding removal is a second concrete target for a separately
identified numerical/integration check after the current worker releases its
allocation. Neither optimization is applied to pinned active sources. These
engineering results do not justify changing the running scientific cases,
tolerances, rate selections, confirmation seeds or training horizon.

No external-paper quality or performance claim is used in this analysis. The
support is the original local training data and executed source; the running
panel remains the test of comparative model quality.
