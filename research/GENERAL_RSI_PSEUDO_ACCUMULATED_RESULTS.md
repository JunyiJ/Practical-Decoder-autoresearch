# Accumulated-state diagnostic results, 2026-11-12

Both extra optimizer advances and accumulated pseudo gradients worsen the fixed
training-window loss in this diagnostic. After100 matched natural updates,
CE+zero minus CE averages+0.018368nats/token. CE+pseudo minus CE+zero adds
+0.035367. Both contrasts are positive for all three parents. This supports an
accumulated mechanism that the earlier first-step diagnostic did not establish.
It does not establish a model improvement or a universal causal decomposition.

## Frozen execution and verification

Plan `tokenizer-evidence/general-rsi-pseudo-accumulated-001/plan.json`, SHA256
`85073a83ac8e42446e2eca4b2d93f52b75403255ef4a677cb3cfc9263c445557`,
fixed all inputs, fresh roles, arm order, probes and costs before execution.
All three selected native parents ran CE, CE+explicit-zero and CE+pseudo
trajectories. Each received100 identical natural batches within its bundle;
interleaved arms received99 extra updates before the final natural update.
LR0.00025, clipping1.0 and the original optimizer/loss implementations stayed fixed.

All1,494 update records and45 probes passed review. The nine initial complete
state restores matched exactly. The first CE loss, optimizer-state summary and
all64 probe-window losses matched across all three arms within every parent.
Every natural/pseudo input hash and optimizer-step count matched the frozen
inventory. All900 CE updates were clipped. Only1,2,2 pseudo updates were clipped
in parents01/03/05, and no pseudo target saturated. No failures or retries occurred.

Evaluation used8,192 fixed training targets per parent, with positions disjoint
from all current diagnostic updates. The parent models had previously trained
on this corpus. These are mechanism measurements, not a new holdout. Validation
was not scored; test stayed unopened. No generations or new model weights were
saved. All original selected/fresh weights, failures and first outputs remain.

## Endpoint and trajectory evidence

All values are nats/token. Positive contrasts mean higher loss for the first arm.

| Parent | CE | CE+zero | CE+pseudo | Zero−CE | Pseudo−zero |
|---|---:|---:|---:|---:|---:|
|01|4.121749|4.141353|4.185187|+0.019605|+0.043833|
|03|4.249781|4.267868|4.301454|+0.018087|+0.033585|
|05|4.335601|4.353013|4.381697|+0.017412|+0.028683|

| Natural updates | Mean zero−CE | Mean pseudo−zero | Mean pseudo−CE |
|---|---:|---:|---:|
|1|0|0|0|
|10|+0.003215|+0.004274|+0.007489|
|25|+0.005377|+0.013741|+0.019118|
|50|+0.010567|+0.023907|+0.034473|
|100|+0.018368|+0.035367|+0.053735|

Matched natural-batch training losses also diverge. Mean pseudo-minus-CE
pre-update losses over the100 batches are+0.035133,+0.038982,+0.029610.
That reproduces the direction of the original first100-batch gaps using fresh
roles, without pooling old values into a confirmation score. The full probe
windows, moment norms, parameter displacement and10-batch summaries are retained
in `analysis.json` and the original attempt files.

Contrary scope: ordinary CE alone also raises these training-window losses from
the selected parent by0.0256–0.0276nats/token. Reset-state/continuation transients
therefore remain relevant. The zero intervention combines momentum and independent
decay, and pseudo-minus-zero includes their interactions with subsequent gradients.
The endpoint contrasts are algebraically additive here, but they do not assign
general causal percentages to components of the longer quality-panel regression.
There are only three paired prefixes and correlated fixed windows. No confidence
interval, superiority rule, checkpoint selection or adoption applies.

The worker took603.000s, including509.981s of synchronized updates, versus an
821.878s forecast and1,200s ceiling. Peak RSS was1,459,912,704bytes and sampled
MPS driver allocation2,505,883,648bytes, below4GiB/6GiB limits. Final guard free
space was19,928,481,792bytes against10GiB reserve. Metadata/write-overlap admission
was32MiB. No cleanup or model serialization was required.

## Released implementation evidence and next action

A read-only source review during this frozen run found an applicable difference.
At commit `35f5a22cd94a1b35c0a57b1ccb1b74186d2bab1d`, DITTO's
[fine-tuning script](https://github.com/Jxu-Thu/DITTO/blob/35f5a22cd94a1b35c0a57b1ccb1b74186d2bab1d/runs/DITTO.sh)
sets update-freq6, NAG and clipping0.1. Its
[task](https://github.com/Jxu-Thu/DITTO/blob/35f5a22cd94a1b35c0a57b1ccb1b74186d2bab1d/fairseq/custom/language_modeling_pealize_rep.py)
chooses CE or pseudo loss per minibatch. The
[trainer](https://github.com/Jxu-Thu/DITTO/blob/35f5a22cd94a1b35c0a57b1ccb1b74186d2bab1d/fairseq/trainer.py)
accumulates the grouped backward passes, normalizes and clips, then advances the
optimizer once. Mixed gradients can therefore reach one optimizer advance.
Our frozen pure-gradient alternation was a different declared local recipe,
not a faithful optimizer-level reproduction. Its negative result remains valid.

The newly retrieved [supplement, sectionC](https://proceedings.neurips.cc/paper_files/paper/2022/file/148c0aeea1c5da82f4fa86a09d4190da-Supplemental-Conference.pdf)
also reports weaker alternatives and warns that a margin penalty can over-penalize
necessary repetition. It does not justify a stronger-penalty sweep. Pinned source
hashes, inspected lines and applicability limits are in
`released-implementation-review.json`. No active protocol changed after this review.

Updated hypothesis: combining natural and pseudo gradients before clipping and
one optimizer advance may avoid part of the observed damage from separate steps.
It may still worsen prediction or weaken useful repetition suppression. Different
NAG/Muon math, accumulation size, sample weighting, precision and context prevent
importing the published gain into this local setting.

Next design the smallest joint-gradient integration and diagnostic that can
falsify this hypothesis. Explicitly control loss normalization and clipping so a
global gradient rescaling is not mistaken for a pseudo-gradient benefit. Use fresh
roles and preserve the existing pseudo objective and coefficient without a sweep.
Qualify only the new accumulation/state-accounting path, finite updates, restart
and cost needed for that study. Freeze its controls and complete retained-output
admission before training. No joint-gradient experiment or new quality panel is
yet admitted; no worker is active. Ordinary CE remains the selected continuation.
