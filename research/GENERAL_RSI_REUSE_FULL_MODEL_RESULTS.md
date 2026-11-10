# Composer reuse fails the first optimizer-update gate

The reuse v1 full-model candidate is closed as a failed drop-in engineering
qualification. Its component speed result remains valid, but it is not admitted
for training or production integration. No tolerance was changed and no retry
was made. This is a numerical qualification failure, not a negative quality result.

Both paths started with identical parameters and optimizer groups. On the same
first batch, CE matched exactly at10.8321475983 and all105 gradient comparisons
passed. Maximum initial gradient difference was2.794e-8. Actual first-update
gradient norms and clipping coefficients also matched exactly.

Three of105 parameter tensors then failed the frozen first-update tolerance:

| Composer matrix | Maximum absolute parameter difference | Largest tolerance ratio |
| --- | ---: | ---: |
| projection_weight | 9.6321e-5 | 4.6554 |
| transform_weight | 8.2336e-5 | 4.0168 |
| highway_weight | 7.6219e-5 | 2.8527 |

The gate requires a ratio at most1 for every element. All non-composer parameters
were bitwise identical after the update. Fourteen composer parameter tensors
changed at least one bit, with only the three listed matrices exceeding the gate.
Sampler state, loss history, clipping, schedules and optimizer-group options
matched. These facts localize the observed divergence to composition and its
optimizer updates; they do not establish a model-quality effect.

The bound PyTorch2.14.1 implementation in `torch.optim._muon` explicitly converts
the optimizer update to BF16 before normalization and Newton-Schulz iteration.
This is consistent with the earlier local sensitivity evidence. The read-only
audit retained the installed source hash and relevant function excerpt. It did
not rerun the optimizer or isolate which rounding operation caused amplification.

Attempt1 stopped after one update per arm in6.25s, with1,028,358,144-byte peak RSS.
The two complete failure states total364,912,070 bytes. Both remain locally
retained with exact SHA256/size receipts. The complete5GiB output allowance and
10GiB reserve were satisfied. The subsequent CPU-only failure audit took less
than one second and made no training updates.

The planned40-update comparison, complete restart and qualified whole-update
timings were **not reached**. First-update elapsed times are warmup observations
and must not be reported as a valid full-model speed estimate. No validation,
generation or original-test evaluation occurred. Existing accepted recipes,
checkpoints and all first generations remain unchanged.

Evidence is under `tokenizer-evidence/general-rsi-reuse-full-model-001/`:
`plan.json`, `attempt-01/initial-gradient-comparison.json`,
`attempt-01/first-update-parameter-comparison.json`, `attempt-01/decision.json`,
both failure receipts, guards and `failure-localization.json`. The runtime and
protocol were committed as `eede1abd2` before execution.

Updated hypothesis: reuse reduces composition cost, but changed floating-point
reduction order can cause materially different composer Muon updates. It cannot
be treated as a qualified drop-in execution optimization under the declared gate.
The next bounded action is an offline analysis of the retained paired composer
momentum matrices: quantify BF16 quantization disagreement and matrix spectra,
clearly distinguishing saved momentum from the exact Nesterov update input.
Combine that evidence with primary optimizer analysis before deciding whether a
distinct precision-controlled hypothesis merits a new protocol. This does not
authorize retrying reuse v1, relaxing its gate or launching a quality panel.
