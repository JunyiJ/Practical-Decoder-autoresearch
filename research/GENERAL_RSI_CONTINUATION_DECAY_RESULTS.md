# Continuation decay attribution, 2026-11-12

Removing either decay component reduces early CE drift in all three parents.
The predeclared readout-removal contrast is favorable in every pair. Removing
both components helps most among these fixed conditions, but every arm still
finishes worse than its starting parent. Decay contributes to the short-horizon
drift; removing it does not produce a model improvement.

Evidence: `tokenizer-evidence/general-rsi-continuation-decay-diagnostic-001/`.
Frozen plan hash:
`392c8f419d93fe8446118839e60cb9c01e1dc5401c792f87b13a5e1075bc826a`.
All first trajectories and window losses are retained. `analysis.json` and
`completion-review.json` verify the complete fixed comparison.

## Fixed comparison and result

Three selected native time-EMA parents each have four fresh paired trajectories:
decay both groups, other parameters only, readout only, or neither. Coefficients
are0 or0.00006, applied before unchanged canonical optimizer updates. All arms
use identical100 natural CE updates within each parent and shared8,192 training
probe targets excluded from that parent's update-source positions. Initial full
states match apart from the declared policy/configuration. LR stays0.00025,
clipping1; no cooldown or new averaging begins.

At update100, CE differences in nats/token:

| Parent | Remove readout − both decay | Remove other − both decay | Remove both − both decay | No-decay − unchanged parent |
|---|---:|---:|---:|---:|
| 01 | −0.003493711 | −0.006432401 | −0.009810824 | +0.020134234 |
| 03 | −0.003541486 | −0.005778422 | −0.009198939 | +0.020335376 |
| 05 | −0.003491780 | −0.006323971 | −0.009690160 | +0.018493637 |
| Mean | −0.003508992 | −0.006178265 | −0.009566641 | +0.019654416 |

Mean parent-relative drift is+0.029221057 with both decay components,
+0.025712064 without readout decay,+0.023042792 without other-parameter decay,
and+0.019654416 with neither. The mean factorial interaction is+0.000120616;
all three interactions are positive. Removal effects grow across the fixed
1/10/25/50/100 probes. No early checkpoint is selected.

All1,200 updates clip. No-decay readout norms still fall to0.99811–0.99821 of
initial norms, while other norms are1.000077–1.000087. Thus gradient updates also
change norms; endpoint norm shrinkage cannot be attributed entirely to explicit
decay. Both-decay readout norms are0.99224–0.99233 and other norms about0.99410.

These are controlled policy effects for these parents, fresh streams and short
training probes. They do not establish validation, long-horizon or generation
gains. The original fixed-LR fresh-training zero-decay negative remains unchanged.
No recipe is adopted and no quality confirmation is justified solely by being
less damaging than another continuation.

## Execution and next action

All12 initial-state comparisons,1,200 updates,63 evaluations and5,156 frozen
source bindings pass. Guarded runtime527.106seconds includes413.070seconds of
synchronized training. Peak RSS1,738,194,944bytes; MPS2,497,495,040bytes.
No failures, new checkpoint payloads, validation scoring or test access occurred.
The worker has exited. Original selected/fresh weights remain retained.

Next isolate RAW versus averaged starting weights with the qualified no-decay
policy as a mechanism control. `GENERAL_RSI_CONTINUATION_START_NEXT.md` specifies
the prospective comparison and its limits. Original native RAW weights remain
available, but completed optimizer restart states were retired by their original
journals. No exact warm-optimizer control is available. Bind the new start-loader
and fresh paired inputs, then complete measured admission before that diagnostic.
No new quality panel is admitted.
