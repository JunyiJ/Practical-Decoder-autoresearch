# Native horizon comparison completed

The four-hour native recipe passed all frozen validation-loss gates against the
two-hour recipe. Mean time-EMA CE fell from4.9761709785 to4.8347251833nats/token.
The paired differences were-0.1422429413,-0.1465742753,-0.1355201691.
The one-sided99% paired-document bootstrap upper bound was-0.1280419465.
FineWeb-Edu and Wikipedia mean changes were-0.1383577520 and-0.1538142841.
RAW secondary mean CE fell from4.9830909100 to4.8472118445.

Evidence: `tokenizer-evidence/general-rsi-native-horizon-001/decision.json`,
all six accepted slot reviews, and `completion-review.json`. All twelve final
weights remain hash-verified. `adopted-recipe.json` retains the three long-run
primary endpoints without choosing a best seed. Test stayed unopened.

The comparison covers complete7200/14400-second native recipes, including their
own cooldown and averaging horizons. It does not isolate exposure from schedule,
establish an optimum, or establish transfer. Confidence is conditional on the
108validation documents and three paired seeds. Short runs completed20230,
20262,21125updates; long runs40579,42203,42205updates. Total synchronized training
was18.0003hours; total worker time including evaluation, I/O and generation was
18.4655hours. No recovery or scientific failure occurred.

## Fixed generation diagnostics

All144 first completions were rated with horizon/seed mapping hidden until the
annotations were saved. The annotator was the Codex AI assistant in this chat,
not an independent human. Full quotes, reasons, ordinal counts, unassessable
counts and per-seed results are retained in `generation-evaluation/`.
These ratings are exploratory and consistency is not exhaustive factual accuracy.

| Horizon / decoder | Repeated trigram fraction | Terminal loops | Relevance mean | Consistency mean | EOS |
|---|---:|---:|---:|---:|---:|
| Two hours / greedy | .5538 | 15/36 | .8889 | .8485 | 0/36 |
| Four hours / greedy | .6039 | 23/36 | 1.0000 | .8485 | 0/36 |
| Two hours / temperature.8 top-k40 | .0221 | 0/36 | .8611 | .8611 | 2/36 |
| Four hours / temperature.8 top-k40 | .0188 | 0/36 | .8333 | .7778 | 1/36 |

Ordinal means use assessable outputs. Each greedy group had3/36 consistency
ratings unassessable; other fields had none. Equal-weight prompt consistency
means were.8472/.8611 for short/long greedy and.8611/.7778 for sampled outputs.
All greedy outputs reached64tokens. Sampled token-cap counts were34/36 and35/36.
Only3/144 outputs received relevance2 and1/144 consistency2. Loops, broken
grammar, topic drift and failures of simple entity/state tracking remain common.
The first prompt in fixed protocol order, Earth rotation, illustrates the limit:
blinded rowb007 repeats "the sun’s sun"; b002 drifts into unrelated astronomy.
These examples are explanatory, not selected successes or a new scoring gate.
The loss win does not establish improved generation quality.

## Updated hypothesis and next action

More native training improves prediction on this validation corpus at this
budget, while simple generation quality remains weak. No extra horizon panel,
new decoder, or repeated transfer test follows from this result alone.
The next human-directed question is whether composer-only FP32 Muon changes
byte-model training quality at matched updates. Execute
`GENERAL_RSI_COMPOSER_FP32_QUALITY_NEXT.md`: new integration/state/restart/timing
checks, then fresh paired roles and complete measured resource admission before
training. The original FP32 numerical qualification remains failed under its
original rule. FP64 agreement and the old0.01% threshold are diagnostics only
in the new quality study. No batch-reuse arm is authorized.
