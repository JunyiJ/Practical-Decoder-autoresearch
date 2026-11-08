# Residual readout frequency attribution: saved-data analysis

Question: does the current residual readout's validation penalty concentrate on
rare target tokens, as the closed output-anchored tying study's penalty did, or
is it spread across common tokens? The current two completed bundles show worse
validation despite better sampled training fit, starting before cooldown.
These aggregate facts do not identify which tokens contribute to the penalty.

Use only the completed current panel after its automatic independent review and
three-bundle saved-log analysis accept. Historical tying results motivate the
question and supply pre-existing bins; they are not pooled into the calculation.
This is descriptive analysis after aggregate outcomes, not a new quality trial,
test of significance, or a rescue of either output transformation.

Implementation: `byte_residual_readout_frequency_v1.py`. Reconstruct original
training and validation token IDs using the recorded tokenizer/version, corpus
hash and split offsets; verify both sequence hashes and validation-target hash.
Do not tokenize the holdout. Count training occurrences and retain the prior
bins: 0, 1–9, 10–99, 100–999, 1000+. Analyze all34278 saved H200 target losses
for all three seeds. Fixed contrasts: nonlinear−byte, linear−byte,
nonlinear−linear. Report target-frequency groups and the same groups for the
immediately preceding input token, preserving the prior diagnostic's two views.

For every group report target count, each seed's mean difference, mean across
seeds and count-weighted contribution to the overall difference. Contributions
must reconstruct the complete contrast within1e-12, with complete disjoint
coverage. Bind input loss/acceptance/result/review hashes and verify the source
files again before publishing. No new model construction, losses, gradients,
weights, random draws, training or holdout scoring. Checkpoint validity relies
on the completed independent review; this analysis does not reload weights.

A concentrated rare-target penalty would prioritize investigating token-specific
behavior; a broad penalty would argue against transferring that interpretation
from tying. Neither identifies a causal decay, gradient or representation effect:
frequency correlates with token identity and context, and the preceding token
is only one part of context. Neither licenses a frequency cutoff, selective
metric, coefficient sweep, model adoption or reopening a closed experiment.
External literature applicability remains in `BYTE_READOUT_REGULARIZATION_REVIEW.md`;
this arithmetic adds no claim about a published intervention's local benefit.

Execution is one new output attempt at
`tokenizer-evidence/byte-residual-readout-frequency-001`. Acquire the existing
nonblocking research lock; do not execute while training or review owns it.
Require the complete panel's controller completion record and final review.
Limits:120seconds CPU wall time,1GiB process RSS,1MiB retained output, zero
weight outputs,8GiB reserve plus2GiB extra headroom and the full output allowance
at entry. Preserve any failure and use a separately identified recovery only
for a demonstrated implementation/environment defect. Preparation is not a
completed numerical analysis or additional exploration credit.
