# Composer FP32 qualification result

The distinct composer FP32 component failed its frozen direction-accuracy gate.
It is closed without retry or threshold changes. No full-model or quality panel
using this candidate is admitted.

Authority: `GENERAL_RSI_COMPOSER_FP32_PROTOCOL.md` and
`tokenizer-evidence/general-rsi-composer-fp32-fixture-001/plan.json`, frozen in
commit54efe96aa before execution. Original decision, comparisons, guard, log and
failed matrix receipt remain in that evidence directory.

All nine zero-gradient comparisons passed. At step2, the first rank-one
comparison, `composer.highway_weight`, produced direction relative error
0.0002945274107749496 against the fixed0.0001 limit. Parameter maximum absolute
error was4.695743395510821e-8 and momentum error was5.456968139486802e-13. Both
passed their separate tolerances. The combined gate nevertheless failed.

The worker stopped as specified. Dense, reversed, sparse and mixed-gradient
comparisons, independent repeat, serialization/replay and invalid-state checks
were not reached. This result cannot qualify restart or full-model performance.
The guard finished normally after0.371s with peak RSS513,687,552bytes. The
6,617,373-byte failed matrix payload is retained with its SHA-256 receipt.

This is an engineering numerical failure, not a measured quality regression.
The small parameter error matters when interpreting its practical scope. The
earlier saved-momentum diagnostic supported testing FP32 but did not establish
accuracy for every actual Nesterov input. This fixture exposes that limit.
There is no demonstrated harness defect that would authorize another attempt.
The previous reuse v1 failure also remains closed.

Next direction: inspect the already collected native learning curves and design
a fresh horizon comparison for the accepted native recipe. The matched-document
analysis and prospective cost are in `GENERAL_RSI_NATIVE_EXPOSURE_BOUNDARY.md`.
No training worker is active at this boundary. Test remains unopened.
