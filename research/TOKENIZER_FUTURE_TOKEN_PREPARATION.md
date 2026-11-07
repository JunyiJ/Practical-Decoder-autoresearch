# Auxiliary future-token direction: tiny objective accepted

Human steering: keep improving through a different evidence-grounded direction
instead of waiting on CE. Selected two-token auxiliary supervision with a
duplicate-next auxiliary control and fresh genuine native control. Inference
remains next-token prediction; this is not a change to the CE kernel or gates.
See [evidence/design review](tokenizer-evidence/future-token-preparation-001/evidence-review001.md)
for local contrary findings, primary-paper limits, falsifier, controls and costs.

The source-reviewed original component001 command exited0. All54 new literal
CPU64/CPU32/MPS32 objective/analytic-gradient comparisons and81 invalid rejections
passed. Hardcoded target alignment/output ownership, input/populated gradslot
preservation, and caller CPU/MPS RNG preservation passed. CPU64 forward/gradient
maxima8.881784197001252e-16/5.551115123125783e-17; FP32 maxima
4.519583249873449e-7/6.068549146531765e-8. ALL original fixed elementwise predicates
passed; CE3e-5/rtol0 and gradient4e-6/rtol4e-5 were not changed.

Whole guard1.298504208s/RSS417906688B/MPS11354112B passed600s/4GiB/6GiB/reserve8GiB;
both CPU guards finished under120s. Metadata2MiB+reserve8GiB+extra2GiB fits35.13GB.
No models, optimizer steps, corpus/holdout, checkpoints, old oracle replay,
quality seeds/RNG draws, scoring, weight files/copies/deletions or retirement.
Read [original result](tokenizer-evidence/future-token-preparation-001/component001/result.json),
plan/source-review/attempt/allguards/STOP. Original command and two sources are
CLOSED/no replay/retry/source-case-tolerance rescue. Tiny V<=23 is not full-vocab
or actual-model qualification. This is not a quality-gain/adoption result.

Next executable action: [actual-model source integration](tokenizer-evidence/future-token-preparation-001/actual-model-source-handoff001.md)
of native/duplicate-next/future-two with a training-only independent copy of the
last block/norm and shared single W. Implement and review sequential head backward
before any model construction, then new fixture-role census and actual generated
proof after full gates. No old model/fixture replays or new scientific-role values.
Future bindings1226 preserve1224; old scientific runtime1211 remains immutable.

Native incumbent and holdout unchanged; no numerical worker active after the
completed component. Main continuation stays ACTIVE and advances this source work;
the old CE branch remains closed. No full quality budget/freeze/launch admitted.
Preparation is not numerical quality exploration; reserve structural/interaction
allocation for the prospective new objective comparison.
