# CE continuation drift audit, 2026-11-12

The ordinary-CE continuation does not recover below its selected parent on the
retained validation measurements. All three full-validation RAW and time-EMA
endpoints regress. This is an audit of existing results, not new scoring or a
change to the closed CE-versus-pseudo decision. That decision establishes which
continuation is better; it does not establish improvement over the parent.

Evidence: `tokenizer-evidence/general-rsi-ce-drift-001/analysis.json` and
`completion-review.json`. The audit binds original logs, document losses and nine
inference payloads. All inputs remained hash-identical. No test access occurred.

## Full validation, 100,416 targets

| Parent slot | Starting EMA CE | Continuation RAW CE | RAW − parent | Continuation EMA CE | EMA − parent |
|---|---:|---:|---:|---:|---:|
| 01 | 4.836920008 | 4.887541400 | +0.050621392 | 4.862580989 | +0.025660981 |
| 03 | 4.823867243 | 4.876692999 | +0.052825756 | 4.851445698 | +0.027578455 |
| 05 | 4.843388299 | 4.892374112 | +0.048985813 | 4.869009925 | +0.025621627 |

Equal-parent mean RAW regression is+0.050810987nats/token;
time-EMA regression is+0.026287021. Both sources regress for every
parent and both endpoints. Averaging helps relative to final RAW but does not
restore the starting parent performance. All comparisons use identical document
IDs, sources, target counts and the qualified scoring policy.

## Fixed validation subset, 8,586 targets

Parent and endpoint subset values are reconstructed from those exact eight
documents in retained full scores. Intermediate probes use the same documents.
Positive differences below are relative to that parent subset, not full CE.

| Parent | At400s | At800s | Final RAW | Final EMA |
|---|---:|---:|---:|---:|
| 01 | +0.035311704 | +0.032372834 | +0.068646087 | +0.035235061 |
| 03 | +0.038562964 | +0.041469327 | +0.086127359 | +0.047739051 |
| 05 | +0.039893886 | +0.040429789 | +0.077772622 | +0.047764121 |

All recorded probes remain above the parent. This does not reconstruct an
unobserved continuous curve or rule out transient improvements between probes.
The newer100-update training-window drifts are retained separately by campaign
and parent in the JSON. They use different targets and fresh streams; no scores
are pooled with validation. Online minibatch means are also separate. Across
the three400-second intervals, first-to-last online means rise in every parent,
but different sampled batches prevent treating that alone as paired loss drift.

## Weight geometry and candidate explanations

| Parent | RAW Muon norm / parent | RAW AdamW norm / parent | Decay-only algebraic factor |
|---|---:|---:|---:|
| 01 | 0.834743 | 0.832750 | 0.810920 |
| 03 | 0.834576 | 0.833445 | 0.810287 |
| 05 | 0.834848 | 0.833106 | 0.809996 |

Parameter-group norms fall about16.5–16.7%; cosine similarities remain above
0.9937. The original wrapper multiplies every parameter by1−0.00006 on every
update, independently of LR. The continuation keeps this coefficient while
cooling its0.00025 peak LR to about0.000025. The product(1−0.00006)^steps is
about0.810, but it is a no-gradient counterfactual, not the observed norm.
Gradient updates and averaging also affect norms. All10,511 CE updates clip.
These facts motivate a functional norm-sensitivity test; they do not identify
decay, clipping, fresh optimizer state or averaging departure as the cause.

Contrary evidence: the prior fixed-LR Shakespeare zero-decay comparison lost
in all three seeds, mean+0.009165nats/token. Its original closure and acceptance
were read and are bound in `literature-and-boundary.json`. The earlier
weak/strong-decay study also retained its incumbent. Those studies stay closed.
This audit supplies a changed continuation setting, not permission to relabel
or repeat those scientific outcomes.

## Next action and cost

`GENERAL_RSI_NORM_RESTORATION_NEXT.md` specifies the next bounded inference-only
control: mechanically restore parent-relative readout and other-parameter norms
separately and jointly on retained endpoints. It separates a logit-scale effect
from broader functional change before considering another training intervention.
All factors, fresh training windows and27 scoring conditions must be frozen
and admitted first. No learned transform, new training or model adoption is
admitted. The design records primary literature and its applicability limits.

This read-only audit used3.048s guarded CPU time and peak RSS
981,155,840bytes. There were no failures or new model
payloads. The shared lock was reacquired at review and no worker remains active.
