# Output lexical sharing: qualification question, no campaign admitted

## Local evidence

The adopted byte input composes fixed GPT2 token spellings, while its independent
output table retains12865792 parameters. The input lexical/nonlexical comparison
passed; see the original
`tokenizer-evidence/byte-composition-boundary-001/quality-evidence001/evidence.md`
for mechanism and limits and the selected public inference recipe for adoption.
This demonstrates useful lexical input sharing, not useful output composition.

`OUTPUT_ANCHORED_TYING_RESULTS.md` is contrary evidence to naive sharing: tied
native tables lost to the initialization-matched separate control on every seed.
That trial did not compose output rows from bytes. Neither this distinction nor
input success licenses replaying its closed seeds or rescuing it with adapters.
`BYTE_RESIDUAL_READOUT_RESULTS.md` rejects both context-side residual transforms;
changing the representation of token output rows would test a different object.

`BYTE_SAMPLER_GAP_RESULTS.md` and `BYTE_OUTPUT_MOMENT_RESULTS.md` weaken the
current optimizer rationale and show epsilon already dominates unseen output
rows at one selected endpoint. They do not prove lexical sharing will help.
The957 unseen validation targets establish a generalization question, not a
license to optimize a subgroup metric or to choose a cutoff from its outcomes.

## Primary literature: support and contrary evidence

Labeau and Allauzen, SCLeM2017, final proceedings PDF,
[sections3.2–4.4, Tables4–6](https://aclanthology.org/W17-4101.pdf):
augmenting word outputs with character representations improved their Czech
full-vocabulary experiments. However, character-only outputs performed much
worse; tying and output highway layers also hurt in preliminary experiments.
Small-vocabulary configurations failed to improve, and training was unstable.
They used untied representations, sampled objectives, dropout, checkpoint
backtracking and word units. Their vocabulary and morphology differ from our
English GPT2 pieces; their aggregate gains do not establish gains for rare
targets. These mixed results favor considering lexical augmentation with a
retained independent path over assuming complete replacement is beneficial.

Pappas, Mulcaire and Smith, EMNLP2020, final proceedings PDF,
[section4.1 and AppendixB.3 Table9](https://aclanthology.org/2020.emnlp-main.96.pdf):
their convolutional output baseline, including a per-word correction, lost to
lookup outputs on both Penn and WikiText2 (test perplexities101.6 versus90.8,
and116.6 versus108.3). Their GroC model performed better but adds WordNet
relations and definitions; its headline result cannot be attributed to surface
composition alone. Comparisons used LSTMs, word vocabularies, output dropout
and validation hyperparameter search. This is direct contrary evidence to a
generic convolutional-output improvement claim, and does not authorize adding
external lexical data to our fixed corpus experiment.

## Candidate mechanism and next discriminating action

An output lexical component could share learning across spellings while keeping
an independent output path for distinctions that spelling cannot represent.
The falsifiable quality prediction would be improvement against both the adopted
independent output and an equally sized nonlexical component under fixed global
validation rules. A lexical-only win inside a worse family would not be adoption.
Fully composed replacement, shared input/output weights and context-side tanh
are not interchangeable controls for this question.

This is not yet an admitted candidate: no layer size, coefficient, role, seed,
cutoff, or model update has been selected. The decisive next action is to derive
the actual complete-vocabulary forward/backward cost from the current codebook
and composer shapes, and compare feasible parameterizations supported by these
papers. All output rows participate in the softmax at every training update;
the cheap input lookup workload cannot be used as the cost estimate. Consider
exact recomputation/chunking only if it preserves gradients; stale cached rows
or sampled output objectives would change the scientific question.

A viable proposal needs a bounded correctness/resource pilot, equal comparison
budgets, fresh confirmation, prospective futility rules and the complete retained
output high-water forecast plus reserve/headroom. No quality run is authorized
by this note alone. If the cost or evidence does not support the family, reject
it before building an elaborate implementation; do not use the exploration
allocation to justify arbitrary coefficients or random architectural changes.
