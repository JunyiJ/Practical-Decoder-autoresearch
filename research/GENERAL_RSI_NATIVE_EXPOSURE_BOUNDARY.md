# Native training-horizon boundary

This is a prospective design, not an admitted panel. The next executable action
is to adapt the existing qualified time-budget worker to a separately bound
native horizon plan, verify that adaptation, then freeze exact seeds, sources,
admission and run order before training. Reuse the qualified training context,
restart, document scorer and generation evaluator. Do not repeat their completed
numerical qualifications without a new requirement.

## Measured local basis

`general_rsi_existing_curve_audit_v1.py` aggregates existing accepted equal-time
slots04–09. Its output, input hashes and per-seed/source statistics are in
`tokenizer-evidence/general-rsi-next-exposure-analysis-001/curves.json`.
It computes no new model scores and opens no test text. All three stages use
the same eight probe documents and8,586 targets. Final RAW document losses are
restricted to that exact inventory before comparison.

| Arm | 2,400 training seconds | 4,800 seconds | 7,200 seconds | Last-third change |
| --- | ---: | ---: | ---: | ---: |
| Native | 5.317995 | 5.037779 | 4.732792 | -0.304987 |
| Byte | 5.575450 | 5.224258 | 4.866381 | -0.357877 |

Values are mean RAW CE across three seeds. Every seed improves in the last
third on both sources. Native last-third changes average-0.272595 on FineWeb-Edu
and-0.370077 on Wikipedia. Native clipping fractions in that interval range
0.0657–0.0797. There is no observed numerical failure in these accepted runs.
The eight fixed documents are a small, nonrepresentative subset. More exposure
and cooldown occur together, so this curve does not identify their separate
effects or establish continued improvement beyond7,200 seconds.

The full equal-time decision in `general-rsi-equal-time-001/decision.json`
accepted native with mean EMA CE4.968966 versus byte5.091068. The earlier byte
exposure decision in `general-rsi-exposure-001/decision.json` accepted a longer
byte horizon, but it supplies no fresh native confirmation. Transfer was
inconclusive under its fixed gates (`GENERAL_RSI_TRANSFER_RESULTS.md`). None of
these historical scores will be pooled into the proposed confirmation.

## Literature, inference and alternatives

Hoffmann et al., [arXiv:2203.15556v1](https://arxiv.org/html/2203.15556v1),
section3.1 and AppendixB, compare terminal training horizons and show that a
decay schedule mismatched to the training endpoint can impair results.
AppendixC also limits scaling consistency to the regime they studied, including
data repetition. Their larger models and corpora do not establish an optimal
budget for our28.5M-parameter model repeatedly sampling a finite10M-token corpus.

Our hypothesis is that additional optimization at the native architecture can
improve held-out CE when its cooldown matches the longer horizon. The competing
explanation is that the current decline mostly reflects cooldown, or that extra
repetition causes overfitting. The experiment tests a complete longer-horizon
recipe, not the isolated causal effect of additional updates. A failed gate
retains the7,200-second incumbent; a passing gate supports a more expensive
native recipe within this corpus. Generation diagnostics separately test whether
CE gains accompany less repetition and better relevance or consistency.

Prioritize this measured exposure question over another composer precision
variation. Both recent composer qualifications are closed. Another model size
would change capacity and throughput together and needs a separate rationale.
This is exploitation of a validated architecture, not numerical exploration or
a replay of the concluded byte exposure study.

## Proposed comparison and full cost

- Three fresh paired native seeds,7,200 versus14,400 synchronized training-update
  seconds. Doubling is a bounded design choice, not a measured optimum. Hold
  architecture, data, LR0.0025, optimizer, clipping and scoring fixed. Within each
  pair use identical initializers and training-sampler streams. No new LR search.
- Each arm gets its own final-third linear cooldown to0.1 and final-third EMA.
  Hold EMA half-life at the qualified127,184,813,977ns for both arms. This changes
  the averaging start with the horizon while keeping its local time scale fixed.
  RAW remains secondary. Compare complete endpoints, not a midpoint of a long run.
- Proposed primary gate: all three long-minus-short EMA CE differences negative,
  mean at most-0.02, one-sided99% paired document-bootstrap upper bound below0,
  and no source regression above0.05. Freeze bootstrap seed and exact aggregation
  in the new plan. Otherwise retain the short recipe. Do not select a best seed.
- Preserve all144 first completions under the existing12-prompt/two-decoder
  protocol. Report repetition, length and termination, plus blinded relevance
  and consistency annotations with the actual annotator identified. No rerolls,
  changed decoder, blended CE score or generation rescue of a failed CE gate.
- Training cost is64,800seconds,18hours, plus evaluation, generation, I/O and
  failure/replay costs. Native measured full validation25.935s and generation
  40.671s imply about555.25s for two validations plus one generation pass per
  endpoint. Restart payload measurement is445,008,055bytes. An executable
  admission must account for periodic probes/checkpoints and initialization,
  and bound cumulative slot and panel time before launch.
- Complete proposed storage: twelve final weight files at113,998,683bytes each
  =1,367,984,196bytes; two512MiB active restart/serialization slots;256MiB logs,
  generation and metadata;5GiB failed-write allowance;10GiB reserve. Required
  free space is18,816,288,836bytes, in addition to all existing retained files.
  Recheck at admission, include any adaptation outputs, and do not launch a
  partial panel. Current approximate24.6GB free suggests it can fit without
  cleanup, but is not admission evidence for a future launch.

Test remains unopened. Existing first outputs, failures, selected models and
frozen studies retain their protections. No new numerical worker is active.
