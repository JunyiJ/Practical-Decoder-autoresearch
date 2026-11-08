# First-rate three-arm comparison

All three original screens at learning rate 0.000625 completed 1,200 updates
and passed their saved-result acceptance. Lower validation CE is better:

| Model | Full validation CE | Difference from native |
|---|---:|---:|
| Native embeddings | 4.896433671 | 0 |
| Real-byte composition | 5.039635859 | +0.143202188 |
| Nonlexical-code composition | 5.098909072 | +0.202475401 |

Real bytes beat the code control by 0.059273213 nat/token at this rate, while
both composition models lose to native embeddings. This is one paired screening
bundle at one rate, not the selected-rate fresh comparison or an adoption result.
The complete frozen panel and its acceptance criteria remain unchanged.

## What the control adds to the earlier diagnostic

This analysis reuses the exact frequency and byte-length groups from
`screen-diagnostic001`. Frequency is the token's count in the original training
split, rather than its count in the sampled optimization stream. Positive deltas
mean the first named model has higher loss. These are losses on the next target,
grouped by the immediately preceding input token:

| Preceding-input training frequency | Targets | Bytes minus native | Codes minus native | Bytes minus codes |
|---|---:|---:|---:|---:|
| 0 | 957 | -0.923899 | -0.568272 | -0.355627 |
| 1–9 | 2,846 | +0.027525 | +0.239922 | -0.212398 |
| 10–99 | 6,893 | +0.459368 | +0.591867 | -0.132499 |
| 100–999 | 8,787 | +0.184457 | +0.203016 | -0.018558 |
| 1,000+ | 14,795 | +0.062675 | +0.063388 | -0.000714 |

The unseen-input improvement also occurs with arbitrary codes. Therefore the
earlier byte-versus-native observation alone cannot attribute that improvement
to spelling. Real bytes provide an additional improvement in this group, and
their advantage over codes decreases across these input-frequency bins. This
is consistent with both generic shared composition and additional information
from lexical coding contributing; it does not establish either mechanism
causally or show that spelling solves the overall quality deficit.

The distinction between input and target grouping matters. On unseen prediction
targets, bytes minus codes is only -0.047666, compared with -0.355627 after
unseen inputs. Those are overlapping, correlated groups, not independent effects.
The model reads more than just the last input token. Across all target-frequency
bins the byte model is better than the code control, but it is worse by +0.019345
on the 8+ byte target-length group. Full group means, counts, additive
contributions and lower-loss fractions are retained in `result.json`.

## Evidence, cost and limits

`analyze.py` reads only the three accepted per-target loss files and original
train/validation token IDs. It verifies acceptance hashes, terminal guards,
paired samplers, target identity, corpus and tokenizer hashes, loss totals and
partition accounting. Every inherited bin, count, native/byte mean and
byte-minus-native delta reproduces the original diagnostic exactly.

The analysis invocation took approximately 0.33 seconds. It executes no model,
collects no new losses, draws no randomness, and neither encodes nor scores the
holdout. The existing control training took 3,795.112 seconds including its
final consumers; the diagnostic adds no training run or GPU allocation.

The control is length-matched and uses the same composer, but does not match
every byte marginal. Frequency, length, token identity and context are correlated.
These exploratory groups have no confirmatory intervals, and were not registered
as adoption endpoints. Their fixed definitions precede the control result but
follow the original aggregate byte/native result. They must not replace the
overall acceptance criteria, choose an extra rate or seed, or reopen closed work.

The next executable scientific action is already running: slot 03, real bytes
at learning rate 0.00125. Complete all nine frozen screens before choosing any
arm's rate, then perform the three fresh paired confirmation bundles. The
native incumbent remains unchanged.
