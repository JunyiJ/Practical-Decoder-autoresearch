# Selected-rate screening: where the loss advantage occurs

All nine screens are accepted and the original selection rule is frozen. This
new saved-data analysis asks whether the selected-rate advantage is concentrated
in the same groups identified at the first, lower rate. It reuses the original
frequency and length bins exactly. It is not another training or inference test.

| Model | Selected LR | Accepted screen | Full validation CE |
|---|---:|---:|---:|
| Native | 0.00125 | 05 | 4.695642038 |
| Real bytes | 0.0025 | 08 | 4.569071185 |
| Nonlexical codes | 0.0025 | 06 | 4.662835554 |

Real bytes improve over native by 0.126570853 nat/token and over the code control
by 0.093764368. The code control improves over native by 0.032806485. These are
selected screening observations from one paired bundle. The rates were selected
using the complete aggregate validation scores; subgroup findings are exploratory
and selection-biased. They are not fresh confirmation or adoption evidence.

## Losses grouped by the immediately preceding input token

Frequency is the token's count in the original training split. Negative deltas
favor the first named model. Each row describes next-target losses after inputs
in the indicated group; it does not describe prediction of those inputs.

| Training frequency | Targets | Bytes minus native | Codes minus native | Bytes minus codes |
|---|---:|---:|---:|---:|
| 0 | 957 | -0.938253 | -0.163382 | -0.774871 |
| 1-9 | 2846 | -0.120221 | +0.257706 | -0.377927 |
| 10-99 | 6893 | -0.118804 | -0.021418 | -0.097386 |
| 100-999 | 8787 | -0.144804 | -0.078101 | -0.066703 |
| 1000+ | 14795 | -0.068079 | -0.058649 | -0.009430 |

Real bytes now improve over native in all five preceding-input frequency bins.
At the first rate, only the unseen-input bin improved. The selected-rate gain is
therefore broader across these input groups than the earlier low-rate result.
The unseen-input group has the largest mean advantage, but its additive
contribution is -0.026194890 of the -0.126570853 total (about 21% of the net gain).
Inputs occurring 10–999 times together contribute -0.061010200, so the aggregate
advantage is not solely an unseen-input effect.

Real bytes also beat the code control in every input-frequency group. The code
control itself loses to native after inputs appearing 1–9 times (+0.257706),
while real bytes win there (-0.120221). The additional byte advantage in that
bin is -0.377927. These are descriptive differences, not an isolated causal
estimate of spelling, substring sharing, or morphology.

## Losses grouped by the prediction target

| Training frequency | Targets | Bytes minus native | Codes minus native | Bytes minus codes |
|---|---:|---:|---:|---:|
| 0 | 957 | +0.382529 | +0.341867 | +0.040661 |
| 1-9 | 2846 | +0.018928 | +0.154910 | -0.135982 |
| 10-99 | 6892 | -0.063326 | +0.082315 | -0.145641 |
| 100-999 | 8788 | -0.173145 | -0.079225 | -0.093920 |
| 1000+ | 14795 | -0.189287 | -0.119207 | -0.070081 |

The rare-target result is unfavorable: real bytes lose to native on unseen
prediction targets (+0.382529) and targets occurring 1–9 times (+0.018928).
Those bins together contribute +0.012251280 to the aggregate delta. Unseen
prediction targets also favor the code control over real bytes (+0.040661).
Much of the favorable aggregate target-group contribution comes from targets
appearing 1,000+ times (-0.081699840). The selected model also loses to native
on targets at least eight bytes long (+0.112124), although it narrowly beats
the code control in that group (-0.006043).

Thus, lower loss after rare inputs must not be described as generally better
prediction of rare tokens. Input and target groups overlap and are correlated
with token identity, context, and difficulty. The output objective is unchanged;
these observations do not isolate a causal pathway. Full byte-length group
results and all additive contributions are retained in result.json.

## Evidence, verification, cost, and next action

The analysis verifies all nine selection review hashes, each arm's original
minimum-CE choice, and accepted loss evidence for selected slots 05/08/06. It
checks paired sampler and target identities, corpus/tokenizer identities, full
loss totals, identical inherited bin counts, and additive partition accounting.
The original diagnostic001/002 sources and results remain unchanged. New losses
and confidence bounds are not generated; the holdout is not encoded or scored.

Execution completed in 0.477 seconds with 84,787,200 bytes peak child RSS and
a 60-second timeout. It made no model call, used no GPU, and drew no randomness.
execution.json binds the source hash; result.json binds the exact input files.
The original successful stdout and empty stderr are retained.

The length-matched code control does not match every byte marginal. Selected
rates also differ across arms. These limits and the single screening bundle
prevent a causal or generalization claim. No subgroup was added to the adoption
criteria, and no rate, seed, or training trajectory was changed.

The next scientific action is already executing: fresh0 real-byte confirmation
slot10, paired with accepted native slot09. Complete all original fresh bundles
and the frozen full-panel decision before considering adoption or a new campaign.
