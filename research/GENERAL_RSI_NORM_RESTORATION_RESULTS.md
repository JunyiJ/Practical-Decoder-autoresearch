# Retained-weight norm-restoration result, 2026-11-12

Readout-only norm restoration lowers CE in all six retained endpoints, but none
recovers its starting parent. Restoring the other parameter norms, separately or
jointly with the readout, raises CE in every endpoint. Uniform radial restoration
is therefore not a repair for the observed continuation drift.

Evidence: `tokenizer-evidence/general-rsi-norm-restoration-001/plan.json`,
`analysis.json`, `completion-review.json` and all first scores in `attempt-01/`.
The frozen plan hash is
`3f63c03e0c4629b993610ced413712f27f565284c2ff7ef4e7d329168be2dc20`.

## Frozen comparison

Three selected native parents and their six CE-only continuation endpoints use
the same 64 fresh document-local training windows, 8,192 targets per condition.
For each endpoint, four conditions restore neither group, only the readout,
only the other parameters, or both groups. The 27 conditions include unchanged
parent references. Each group uses its parent-to-endpoint L2 norm ratio in FP64,
frozen before scores. There is no scale search or fitted calibration parameter.

RAW readout factors are1.17326–1.17371, other factors1.22815–1.22895.
Time-EMA readout factors are1.14461–1.14490, other factors1.19387–1.19464.
Exact original tensors are restored and verified before every transformation.
The qualified corrected-FP32 scorer retains all first per-window loss sums.

## Results on the shared training windows

Numbers are equal-parent mean CE changes, in nats/token. Negative is favorable.

| Endpoint | Readout − original | Other − original | Both − original | Original − parent | Readout − parent |
|---|---:|---:|---:|---:|---:|
| RAW | −0.070219 | +0.648169 | +0.737652 | +0.187703 | +0.117484 |
| Time EMA | −0.058423 | +0.512498 | +0.563693 | +0.137236 | +0.078813 |

Every parent has these signs. The factorial interaction is positive for all six
endpoints: RAW+0.144995 to+0.173626; EMA+0.101244 to+0.118225. Readout scaling
also raises CE when applied after other-parameter scaling. The effects are not
additive. Per-parent values remain in `analysis.json`; these means do not pool
RAW with EMA or these training windows with prior validation scores.

A positive scalar on a bias-free readout changes logit scale. The limited gain
is consistent with a prediction-probability calibration component. We did not
measure a calibration-error metric, and the result does not establish better
features, greedy generation, generalization, or decay causality. The preflight
logit check had maximum absolute error1.525879e−5 and zero argmax changes on its
single case. This is not an exhaustive generation comparison.

The failed broad restoration is useful contrary evidence. Endpoint rescaling
changes the already-learned function and cannot undo thousands of intervening
gradient updates. It does not identify whether decay during training helped or
hurt. The selected parents remain preferred. No model or recipe is adopted.

## Execution and next action

All27 conditions and all5,138 source bindings passed review. Guarded runtime
was52.413seconds, peak RSS1,005,518,848bytes and sampled peak MPS1,112,227,840bytes.
The admission reserved64MiB evidence,64MiB overlap,10GiB reserve and2GiB extra
headroom. No checkpoint was written. There were no worker failures, training
updates, validation scores or test access. Original weights remain unchanged.

Next prepare the controlled continuation-decay diagnostic described in
`GENERAL_RSI_CONTINUATION_DECAY_NEXT.md`. It tests the actual training-time
mechanism that static rescaling cannot identify. It has not been admitted or
launched. Do not repeat this factorial inference run or select a scale from it.
