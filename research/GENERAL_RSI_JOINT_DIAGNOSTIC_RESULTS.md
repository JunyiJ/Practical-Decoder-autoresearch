# Joint-gradient diagnostic results, 2026-11-12

Joint gradients reduce synthetic repetition probabilities but worsen natural
prediction in all three parents at the frozen100-update endpoint. The joint
prediction-and-suppression criterion fails. Ordinary CE remains the selected
continuation recipe. This is a valid short-horizon mechanism result, not a
validation confirmation or a new model-quality adoption.

Frozen plan: `tokenizer-evidence/general-rsi-joint-diagnostic-001/plan.json`,
SHA256 `2ba5d7076744010a54f119da32b63f795b0d8215af1e4d852726a3bef971ea63`.
Execution review: `tokenizer-evidence/general-rsi-joint-diagnostic-001/completion-review.json`.
All900 updates,45 arm probes and nine pristine initial-state comparisons passed.
All5,151 bound source hashes and three selected parent hashes remain unchanged.

## Fixed endpoint contrasts

Positive CE differences mean worse prediction. Negative repetition differences
mean reduced values on the fixed synthetic probe. All comparisons below use
fresh paired streams at100 natural updates, one optimizer advance per update.

| Parent | Joint − full CE | Joint − half CE | Half − full CE |
|---|---:|---:|---:|
| 01 | +0.016246146 | +0.018295450 | -0.002049304 |
| 03 | +0.022606621 | +0.024307873 | -0.001701252 |
| 05 | +0.018824377 | +0.020804708 | -0.001980331 |

| Equal-parent mean contrast | Natural CE | Repeated probability | Previous-copy probability | Repeat/previous ratio | Pseudo penalty |
|---|---:|---:|---:|---:|---:|
| joint_minus_full | +0.019225715 | -0.040442014 | -0.035196508 | -0.017516579 | -0.031062377 |
| joint_minus_half | +0.021136010 | -0.040689448 | -0.035394331 | -0.017697601 | -0.031266654 |
| half_minus_full | -0.001910296 | +0.000247434 | +0.000197824 | +0.000181022 | +0.000204277 |

The joint arm worsens CE against both controls in all three parents. Its pseudo
penalty, repeated probability and aggregate repeat/previous ratio improve against
both controls in all three parents. Previous-copy probability also decreases.
Thus suppression partly accompanies a broader probability decrease. The ratio
decrease supports a differential effect, but these synthetic probes do not
establish better generated prose or a lower real-generation loop rate.

| Natural updates | Mean joint − full CE | Mean joint − half CE |
|---|---:|---:|
| 1 | -0.000033316 | -0.000008260 |
| 10 | +0.002272766 | +0.002388268 |
| 25 | +0.007125715 | +0.007649186 |
| 50 | +0.012780456 | +0.013861219 |
| 100 | +0.019225715 | +0.021136010 |

The tiny first-step joint-minus-full improvement reverses by update10 in every
parent and remains unfavorable at every later frozen probe. No earlier horizon
is substituted for the endpoint. Full CE clipped all300 updates. Half CE and
joint gradients clipped0/300 each. Joint-minus-half therefore retains prediction
damage without an extra optimizer advance or observed clipping. This does not
isolate every nonlinear optimizer interaction. Halving CE improves the short
probe by0.001910nats/token on average, but this is not a validated recipe change.

Full CE itself drifts upward from the parents by mean0.028656972nats/token.
Half CE and joint arms also drift upward. These selected time-averaged parents
are continued with fresh optimizer state. The diagnostic does not separate
optimizer restart, movement away from an averaged model, learning rate, or
early adaptation effects. These remain limits on a broader mechanistic claim.

## Exploratory saved-window audit

`exploratory-window-audit.json` uses the retained endpoint window losses and
original training tokens only. It is posthoc, separate from the frozen criterion,
and computes no new model scores. In every parent, joint-minus-full and
joint-minus-half CE means are positive in both sampled sources and in windows
with and without an internally repeated exact token trigram. Without a repeated
trigram, joint-minus-full means are+0.012241,+0.008819,+0.014090nats/token.
Thus this sampled prediction cost is not confined to visibly repetitive windows.
Exact trigrams are not semantic repetition labels; window averages cannot locate
token-level causes. Small unequal groups are not source-population estimates.

## Execution, limits and next action

The worker finished in457.132seconds, including349.003s of synchronized
updates, under the1,200-second ceiling. Peak RSS was1,629,175,808bytes
and sampled MPS allocation2,501,689,344bytes. No failures,
retries, saturated targets or new checkpoint payloads occurred. All original
logs/probes are retained. The worker ended and the shared lock was reacquired.
No validation, test or new generation scoring ran. Selected/fresh weights,
closed decisions and first outputs remain unchanged.

This result withholds support for expanding the joint recipe to a quality panel.
Do not replay the diagnostic or launch a coefficient sweep. The next executable
work is an existing-data audit of CE-only continuation drift: align the retained
100-update parent-relative probe curves with the original1,200-second CE
continuation histories and RAW/time-EMA endpoint validation. Keep sampled training
CE, online batch loss and full validation as distinct estimands. Determine what
the available evidence can and cannot say about early drift and later recovery
before choosing another training intervention. No new training panel is admitted.
