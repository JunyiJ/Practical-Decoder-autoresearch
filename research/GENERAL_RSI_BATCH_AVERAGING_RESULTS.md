# Independent-batch averaging results

The frozen improvement criterion fails in all three parents. Averaging eight
independent natural batches before one optimizer update does not improve mean
probe loss over the unchanged parent. It is slightly worse than one batch on the
equal-parent mean. No model or larger-batch training recipe is adopted.

## Frozen comparison

Each of three selected EMA parents, slots 1/3/5, supplied eight fresh paired groups.
Control used the first 1,024-target batch. Candidate averaged eight independent
batches, 8,192 targets, before one global clip and one canonical optimizer advance.
Both arms used pristine state, no decay, LR 0.00025 and clip 1. Each parent's
64 document-local probe windows were excluded from all its update source ranges.

Plan SHA256: `f0eeb2c9b099f59894c01976d1ee0246a70ac8ce646e1c915e37ce76dc235ba9`.
Support required negative candidate-minus-control AND candidate-minus-parent mean
CE in every parent. No criterion or batch factor changed after launch.

| Parent | One-batch delta | Eight-batch delta | Eight minus one | Favorable paired cases |
|---|---:|---:|---:|---:|
| 1 | +0.000352955 | +0.000366857 | +0.000013902 | 3/8 |
| 3 | +0.000276568 | +0.000335662 | +0.000059094 | 1/8 |
| 5 | +0.000302436 | +0.000261106 | -0.000041330 | 4/8 |

Deltas are nats/token on the fixed training probes. Equal-parent means are
+0.000310653 for one batch and +0.000321208 for eight, difference +0.000010555.
Eight-batch averaging beats control in 8/24 pairs and the unchanged parent in
2/24 cases. All first outputs, including favorable cases, are retained.
The case count is not independent model-seed replication. These small sampled
loss differences do not establish a population-wide effect or all batch sizes.

## Mechanism and cost limits

Posthoc recorded geometry shows mean pre-clip gradient norm falling from 1.344186
to 0.466854. All 24 controls clip; none of the candidates clip under the same rule.
Mean actual AdamW-group displacement instead increases from 0.370013 to 0.451488;
Muon displacement changes from 0.037027 to 0.038184. These facts oppose a simple
claim that smaller gradient norm necessarily means a smaller realized update.
They do not isolate clipping or optimizer transformation as the cause of probe
harm, and they do not measure a gradient noise scale.

Candidate consumed eight times as many targets per update. Mean update times
were 0.395429 seconds for control and 1.201017 seconds for candidate. The paired
experiment is not an equal-time or equal-token quality comparison. Its negative
criterion does not justify another batch-factor or learning-rate sweep.

## Qualification and execution

The new integration passed exact one-batch parameter, complete child-state,
sampler and CPU/MPS RNG comparisons against the qualified update. All 88 averaged
gradients matched independently computed unscaled-microbatch gradients exactly.
The prospectively fixed tolerance was atol 1e-7 and rtol 1e-5. Qualification took
8.571 guarded seconds, three optimizer updates and 18 forward/backward
passes, with no model output. Original model/restart/scoring prerequisites were
not repeated. No attempt failed.

The diagnostic passed all 48 initial-state checks, 48 optimizer updates, 216
forward/backward passes, 221,184 training targets and 51 evaluations containing
417,792 scored targets. All 5,190 frozen source bindings remained unchanged.
Guard time was 155.366 seconds, including 38.315 training
seconds. Peak RSS was 1,408,958,464 bytes and peak MPS allocation
1,413,267,456 bytes. Review measured 1,171,556 evidence bytes before
this report. No checkpoint was written. The worker exited and the shared lock
was reacquired. Validation was not scored and test remained unopened.

## Research boundary

Close this one/eight-batch diagnostic without a larger-batch quality campaign.
The current fresh-optimizer continuation investigations have supplied partial
mechanistic findings but no model better than the retained EMA parents. Preserve
all closed studies and their failures. Do not turn the smallest regression into
an improvement claim or reopen the old warmup study.

Next assess natural-text context use within the current 128-token model limit,
as specified in `GENERAL_RSI_CONTEXT_PREFIX_NEXT.md`. This addresses a separate
question in the general RSI plan before any longer-context training proposal.
It does not claim that context eviction caused the short retained loops.
No worker or new quality panel is active at this boundary.

Evidence: `tokenizer-evidence/general-rsi-batch-averaging-001/` and
`tokenizer-evidence/general-rsi-batch-averaging-fixture-001/` contain the original
plans, input inventories, checks, measurements and reviews.
