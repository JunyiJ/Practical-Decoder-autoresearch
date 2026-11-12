# Context-order intervention — 2026-11-11 Pacific

## Result and scope

Changing earlier sequence order reduced support for22/23 retained terminal loops.
It also reduced support for most natural repeated-trigram targets. The result
supports a dependence on earlier order, with substantial ordinary-text costs.
It does not identify a repetition-specific causal effect or validate a model
improvement. No new training, decoding, checkpoint writes or test access occurred.

The frozen281 cases contain23 loops and86 natural endpoints under each of three
models. Both conditions were evaluated for every case:562 sequences. All original
condition probabilities reproduced exactly, maximum absolute discrepancy0.0.
All281 index, length, prefix-multiset and local-context invariants passed; three
synthetic edge checks passed. There were no unchanged-input cases, no rerolls,
no exclusions and no failed attempts. Guarded elapsed time54.395s, peak
RSS627,671,040bytes. The three original checkpoints remain hash-identical.

Evidence is in `tokenizer-evidence/general-rsi-context-order-001/`: `plan.json`,
`admission.json`, `attempt-01/rows.jsonl`, `attempt-01/complete.json`,
`attempt-01/guard.json` and `analysis.json`. Plan SHA256:
`c0093e44a3efde3077c7515ee60441965151aba0810752c290c02537d5bf950f`.

## Paired observations

Delta is permuted-minus-intact mean target log probability, in nats per token.
Loop targets are averaged within each loop, then loops equally within a model.
Natural targets are individual repeated-trigram endpoints. These cohorts have
different context lengths, edited spans and selection mechanisms; their effect
sizes are not a matched comparison.

| Slot | Loops with lower probability | Loop mean delta | Loop targets losing top1 | Natural endpoints with lower probability | Natural mean delta |
|---|---:|---:|---:|---:|---:|
|01|8/9|−0.353926|12/87|81/86|−1.576908|
|03|7/7|−0.247202|9/79|80/86|−1.525705|
|05|7/7|−0.192367|4/46|76/86|−1.008047|

Only25/212 original loop targets lose their greedy top1 position after the
intervention. Probability reduction alone does not show that a regenerated
sequence would escape its loop. Natural targets lose top1 in33,22,15 cases and
gain it in1,1,0 cases by slot. This is direct contrary evidence against treating
all context-assisted repetition as undesirable.

## Interpretation and next decision

Lengths, absolute target positions, token counts and immediate two-token context
are controlled. Earlier syntax, semantics and exact repeated sequences change
together. The result establishes sensitivity to that order intervention on these
examples, not a pure repetition mechanism or a population-wide causal effect.
Previous early-onset and probability diagnostics remain descriptive evidence.
The natural references make indiscriminate context corruption or repeat penalties
poor first candidates for this corpus.

Proceed to a new prospective **native pseudo-repetition training comparison**,
with a matched ordinary-CE continuation control and retained natural-text scoring.
The candidate must penalize synthetic redundant copies while preserving ordinary
CE updates. Test whether that separation improves prediction; do not assume it
protects legitimate repeats. This changes the objective, not composer precision,
architecture, horizon exposure study or decoder. The existing native/FP32 panels
remain closed. `GENERAL_RSI_PSEUDO_REPETITION_NEXT.md` specifies the proposed
comparison, limited new engineering and full provisional cost. No scientific
training panel is yet admitted. Do not add another descriptive repetition scan
before the next concrete loss/sampler implementation and cost qualification.
