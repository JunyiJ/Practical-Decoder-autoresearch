# Can a shared table serve input and output differently?

Evidence review on November8, reconciled after output-anchored-tying-001 completed
without adoption. This answers a literature/applicability question; it does not
select a new candidate or authorize replay of a closed projection study.

## Local observations and restrictions

The complete fresh panel has tied-minus-matched mean CE+0.015965529, worse on
allthree seeds, versus tied-minus-native−0.001488867, with two losing seeds.
The matched control improves native by0.017454395. Sources: immutable records in
`tokenizer-evidence/output-anchored-tying-001/`, final.json and final-review.json.
The saved-loss analysis attributes67.9% of the net shared-minus-matched penalty
to targets seen1–9times in training. This does not establish gradient conflict,
a representational bottleneck, or the cause of the historical GQA tying loss.

`TOKENIZER_READOUT_MIXTURE_RESULTS.md` records a complete contrary result:
MoS improved native by0.033083091nat but improved matched MoC by only0.001065236,
with one losing seed and positive interval upper bounds. The joint gates failed.
Extra transforms can improve a recipe without validating the proposed mechanism.
The old MoC arm is not a newly eligible fallback or fresh observation.

`TOKENIZER_READOUT_PROJECTION_RESULTS.md` and
`tokenizer-evidence/post-readout-boundary-001/evidence-review.md` distinguish the
separate untied-head linear/tanh projection question from table sharing. That
panel closed incomplete at slot07 after a CPU/MPS scoring predicate failed.
The original failing reports were not saved. This is not a negative quality
result for projections, and the missing values must not be reconstructed by
rescoring old checkpoints. Its closed namespace, roles and restrictions stand.

## Primary literature actually read

[Press and Wolf, EACL2017 proceedings](https://aclanthology.org/E17-2025.pdf),
section3, inserts a learned square projection before the shared output embedding
and regularizes that projection in its small LSTM setting. It motivates separate
input/output transformations, not a guaranteed improvement in our model.

[Pappas et al., WMT2018 proceedings](https://aclanthology.org/W18-6308.pdf),
sections3.1–3.2/Eqs14–17,4.2 and4.4/Table3, evaluates flexible sharing in LSTM
translation. Table3 contradicts a blanket claim that adding a linear adapter
helps: bilinear16.23BLEU trails tied16.51. Context-only nonlinear17.52 nearly
matches the full512-dimensional joint model17.54; the larger2048 model reaches
18.11. Thus those numbers do not isolate a benefit from both nonlinear branches
at fixed capacity. Section4.3 also includes an exception favoring the untied
baseline. Training uses dropout, different data/dimensions and sometimes negative
sampling. Its quality and runtime results do not transfer directly to our full
softmax, FP32 MLA/Muon/AdamW experiment.

## Inference and compute implications

A shared table need not imply identical input and output functions. For a table
Q and hidden state h, output Q(P h) permits a learned global relationship while
retaining Q for input lookup. It still restricts the two vocabularies' column
spaces compared with independent tables. A linear P does not increase the usual
linear softmax rank bound; any improvement would need a different explanation.
The literature above supplies contrary evidence to assuming that P helps.

At D256, one dense context transform adds65,536 parameters. At1024contexts it
adds67,108,864 forward multiply-accumulates, about0.51% of the dense vocabulary
projection's13,174,571,008. Transforming all50,257output embeddings with a second
D256map instead adds3,293,642,752 multiply-accumulates per training forward.
That is25% of the dense vocabulary projection alone. These are algebraic counts,
not measured elapsed time, memory peaks or complete training costs. An inference
precomputation for fixed weights does not remove training-time work.

## Decision from this review

Do not launch an adapter as an automatic response to the unfavorable panel.
The completed frequency analysis identifies rare targets as a concrete next
question, without establishing a causal gradient mechanism. Inspect the accepted
paired input/output table geometry by training frequency next, without new
training, scoring or initializer replay. Interpret geometry descriptively and
account for coordinate/scale differences before proposing a new model.

A later proposal would need fresh matched controls, a direct byte-incumbent
comparison, equal bounded tuning, independent confirmation and a full retained
output/runtime forecast. Neither the old incomplete projection panel nor this
literature review supplies a local quality gain or a fresh campaign admission.
The closed nine-run protocol and byte incumbent remain unchanged.
