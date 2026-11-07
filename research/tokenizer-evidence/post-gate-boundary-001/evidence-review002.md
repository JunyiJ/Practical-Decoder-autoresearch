# Depth applicability closed; readout-family feasibility selected

The bounded source/old-record derivation is complete in
`depth-scaling-applicability001.json`. Body blocks have unit residual branches;
hidden matrices use nominalLR/sqrt(width/64), matrix Muon with RMS matching;
vectors/tables use AdamW. There is no explicit depth factor. The old192-to256
comparison also changes vocabulary-table capacity, readout divisor and head
geometry. A scalar branch multiplier is not a proved substitute for a depth-LR
rule under these optimizer equations. The original six fixed-half acceptances
and five-point curves were read without numerical replay: half worsens all three
fresh H200 seeds. Together with the primary papers' limits, this rejects an
immediate depth×scaling panel. No more optimizer instrumentation is admitted.
This does not establish that every possible depth allocation is inferior.

The next structural question changes the output distribution family rather than
another internal attention/FFN block. Native logits are H W^T/4 with width256;
this algebraic rank bound is unchanged by body-only gate or routedFFN variants.
It is not a measured true-language rank bottleneck. Keep that distinction explicit.
Read Yang etal [arXiv1711.03953v1](https://arxiv.org/html/1711.03953v1), §2.1–2.5,
§3 tables1–2/4–5: probability-space mixtures are distinct from feature-space
mixtures. Its RNN word-level results, regularization and optimizer differ from
our nativeBPEMLA setting, and increased expressivity can hurt generalization.
The published gain is not a local gain; no diagnosed output deficiency is claimed.

`readout-mixture-direction001.json` selects **design/resource feasibility only**
for the smallest nontrivialK2 probability mixture against unchanged native and
matchedK2 feature-mixture controls, sharing one untied vocabulary head. This is
distinct from the closed routedFFNMoE,attentiongate,tabletying and8K studies;
those negative findings remain contrary evidence, not pooled scores. Hypothesis:
probability mixing can improve full native loss beyond both controls. Failure
retains the native model; no feature-mixture fallback or cost-only adoption.

B8×128×50257FP32 logits cost205,852,672bytes per component before autograd.
K2 doubles this term; actual peak resources are unqualified. Before component
implementation, finish one exact algebra/shape/parameter/retained-output design,
including context projections/initialization/symmetry, numerical log-prob loss
and model/optimizer/RAW-H200/inference contracts. Resolve finite120s/2MiB metadata
feasibility first;zero model/optimizer/corpus/checkpoint/qualityseed use. Then
qualify or reject the smallest component. No scientific quality panel is admitted,
no new storage blocker is declared, and no training is active. A future panel
requires all correctness/pilot/complete-budget gates and fresh confirmation.
Proposed exploration allocation is carried forward; preparation is not numerical
exploration. Earlier evidence-review.md remains an immutable boundary snapshot.
