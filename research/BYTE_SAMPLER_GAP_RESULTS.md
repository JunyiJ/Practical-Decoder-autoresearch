# Sampler gaps do not support a broad long-gap optimizer intervention

The exact existing-data analysis completed successfully. At the current
beta2=.999, only3 of10653 training-observed token types have kappa>1, and none
occurs as a validation target. This does not establish an unbiased optimizer;
it removes support for a broad long-gap explanation of the observed validation
penalty. No new optimizer training panel is admitted on this basis.

Kappa is mean updates between target appearances divided by the second-moment
time constant (approximately1000 updates at .999). These are exact presence
probabilities under the two-block sampler, not realized old-seed histories.
The paper's fixed-point probability-bias formula was deliberately not applied.
Primary-paper scope and predeclared rules are in
[the protocol](BYTE_SAMPLER_GAP_PROTOCOL.md).

| Training count | Types | Validation targets | Median mean gap, updates | Median kappa at .999 | Types with kappa>1 at .999 | Types with kappa>1 at .95 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 39604 | 957 | Undefined | Undefined | Undefined | Undefined |
| 1–9 | 8329 | 2846 | 130.708 | .130708 | 3 | 8329 |
| 10–99 | 1982 | 6892 | 15.645 | .015645 | 0 | 699 |
| 100–999 | 309 | 8788 | 2.454 | .002454 | 0 | 0 |
| 1000+ | 33 | 14795 | 1.008 | .001008 | 0 | 0 |

At .95,9028 training-observed types would exceed kappa=1 and account for4720
validation targets. This is arithmetic under an alternative time constant,
not another beta2 experiment or causal explanation of the historical .95 loss.
The maximum kappa at .999 is7.421806, so it would be incorrect to say that no
training token is in the long-gap regime. None of the positive-frequency types
has zero target-presence probability; the zero-presence group is exactly the
39604 types absent from training. Their957 validation targets cannot be assigned
a finite arrival-gap ratio. They remain a generalization question.

The expected number of training-observed types never targeted over1200 updates
is38.683849 (38.678485 from the1–9 bin and .005363 from10–99). Their expected
validation-target mass is approximately .01661%. These expectations do not
identify which types were actually missed in any completed run. No historical
sampler or score was replayed, and no quality threshold changed.

Verification covered129 deterministic toy sequences with every ordered pair
of starts, including repeated tokens and both boundaries. The actual corpus
has267176 possible starts and the expected total target count reconstructs
1024 exactly. All original corpus/tokenizer/train/validation identity checks
and input/source hash checks passed. No models, gradients, weights, random
draws, new losses or holdout tokenization were used.

Evidence: `tokenizer-evidence/byte-sampler-gap-001/{attempt,guard,result}.json`.
Guarded wall time .318892seconds, peak RSS80756736bytes; output stayed below
1MiB, with11109793792 free bytes at completion. The adopted model is unchanged.

Next executable research action: inspect the schema and retained receipts of
the incumbent's existing RAW optimizer state, then qualify a bounded read-only
analysis of output Adam second moments relative to epsilon by these same
frequency bins. This would measure a different missing premise: whether actual
adaptive denominators are dominated by epsilon, rather than inferring update
behavior from occurrence gaps. It must use already retained accepted states,
verify exact parameter bindings and step/bias correction, and avoid claiming
that endpoint moments establish a causal trajectory or a quality gain. No
coefficient or optimizer replacement is selected. If those states lack the
necessary evidence, reject this diagnostic rather than launching training just
to supply it. Existing closed studies and selected/fresh weights stay intact.
