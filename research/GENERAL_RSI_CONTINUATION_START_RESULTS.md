# Continuation starting-weight diagnostic results

RAW starts reduce relative drift in all three paired comparisons. They do not
produce an improved model. Every final endpoint remains worse than the unchanged
EMA parent on the fresh training probes. No recipe or model is adopted.

## Frozen comparison

Three retained native RAW/time-EMA pairs, slots 1/3/5, each received 100 identical
fresh natural updates per arm. Both arms used the qualified no-decay optimizer,
fresh state, LR 0.00025, clipping 1 and the unchanged batch policy. All updates
preceded cooldown and new averaging. Each pair used 64 fresh document-local
training windows, 8,192 targets, excluded from its update source ranges.

Plan SHA256: `ba556f9b2ed434f65fc639a795ca0fb4782241257680f04e303b6216adf0163c`.
The primary contrast is RAW drift minus EMA drift, each relative to its own start.
All three negative signs satisfy the frozen diagnostic support rule.

| Parent | RAW drift | EMA drift | Drift contrast | RAW gap to unchanged EMA |
|---|---:|---:|---:|---:|
| 1 | +0.013566726 | +0.018687593 | -0.005120867 | +0.039187149 |
| 3 | +0.014886280 | +0.020941553 | -0.006055273 | +0.030326048 |
| 5 | +0.008283896 | +0.019304913 | -0.011021016 | +0.041634076 |

Units are nats/token; positive values mean higher loss. Mean RAW drift is
+0.012245634 and mean EMA drift +0.019644686. The paired contrast is −0.007399052.
RAW endpoints remain +0.017404405 above the continued EMA endpoints and
+0.037049091 above the unchanged EMA parents on average. EMA endpoints remain
+0.019644686 above their unchanged parents. These training-window quantities are
not the earlier full-validation scores and are not pooled with previous studies.

## Execution and limits

All 600 updates and 36 evaluations passed review. All 528 loaded parameter
tensors matched their retained payloads exactly. Six nonweight initial-state
checks passed. All 5,170 frozen source bindings were unchanged. All updates were
clipped. No execution failures occurred, and every first score remains retained.

Guard time was 274.222 seconds, including 207.295
training seconds. Peak RSS was 1,515,995,136 bytes and peak MPS allocation
2,497,495,040 bytes. Review measured 1,341,014 evidence bytes
before this report, with 16,934,920,192 free bytes. No checkpoint was written.
The worker exited and the shared lock was reacquired. Validation was not scored;
test remained unopened. Selected and fresh weights remain retained.

Starting weights explain part of the measured relative drift under this policy.
The comparison does not identify optimizer-state effects because both arms start
fresh. RAW models start at higher loss. Smaller drift from that state does not
establish better quality. Neither removing decay nor switching to RAW has repaired
the continuation loss increase. Retain the selected EMA parents.

Next prepare the bounded update-response diagnostic in
`GENERAL_RSI_UPDATE_RESPONSE_NEXT.md`. It separates finite-step loss behavior on
update batches from disjoint training probes before considering another schedule.
No new quality panel is admitted.

Evidence: `tokenizer-evidence/general-rsi-continuation-start-diagnostic-001/`
contains the plan, admission, all first measurements, analysis and completion review.
