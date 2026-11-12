# First-update response results

The frozen overshoot criterion fails in all three parents. Every tested step
fraction raises mean loss on disjoint training probes. Every update-batch
comparison lowers loss. Smaller steps reduce harm but do not establish a useful
continuation direction on these probes. No model or schedule is adopted.

## Fixed design and result

Each selected EMA parent supplied eight independent fresh first-update cases.
All 24 cases began with unchanged parent weights and pristine no-decay optimizer
state. One ordinary CE update used LR 0.00025 and clip 1. We scored fixed fractions
1/16, 1/4 and 1 of its realized parameter displacement, restoring the original
weights before every interpolation. Each case used its own 1,024-target batch;
each parent shared 64 disjoint document-local probe windows with 8,192 targets.
These windows were excluded from every update source range in that parent.

Plan SHA256: `c784c587e3c9226a91446c54cbb2e26276e34912e1b2e3351ca4ae2d8e69edb3`.
Primary support required positive mean probe loss change at scale 1 and negative
mean change at 1/16 in each parent. Scale 1/4 remained descriptive.

| Parent | Probe delta at 1/16 | At 1/4 | At 1 |
|---|---:|---:|---:|
| 1 | +0.000005482 | +0.000031314 | +0.000276296 |
| 3 | +0.000009740 | +0.000048272 | +0.000341229 |
| 5 | +0.000007330 | +0.000038297 | +0.000296476 |

All values are nats/token relative to the unchanged parent. Equal-parent means
are +0.000007517, +0.000039294 and +0.000304667 at scales 1/16, 1/4 and 1.
Update-batch means are −0.004234311, −0.016912208 and −0.067240703 respectively.
All 72 update-batch deltas are negative. Probe deltas are positive in 19/24,
21/24 and 23/24 cases respectively. Individual favorable probes are preserved;
they do not replace the frozen parent-mean criterion.

This is a valid negative for the specified finite-step overshoot prediction.
It does not prove that all smaller steps or longer schedules fail. The smallest
mean probe differences are small, and the finite training-window sample cannot
establish population direction or generalization. No exact derivative is claimed.
The own-batch/probe contrast supports a batch-specific direction cost on these
samples. Sampling noise, optimizer transformation and probe sampling remain
competing explanations. We did not measure a gradient noise scale.

## New harness checks and execution

The bounded fixture performed no optimizer updates. It passed 440 exact parameter
copy/restoration checks, 176 independently calculated interior interpolation
checks, an unequal-target denominator oracle, and an independent CPU float64
log-sum-exp check for 1,024 scored targets. CE disagreement was
5.65461921731e-08 nats/token, below the frozen 1e-5 engineering tolerance.
The first batch score took 0.281 seconds; fixture guard time
was 3.389 seconds. This qualified only new harness operations.

The diagnostic passed all 24 pristine-state checks, 8,448 exact parameter copies,
171 evaluations and 712,704 scored targets. All 24 updates were clipped. All
5,180 frozen source bindings were unchanged. No execution failure occurred.
Guard time was 178.670 seconds, training 9.383 seconds.
Peak RSS was 1,355,497,472 bytes; peak MPS allocation was 1,404,878,848
bytes. Review measured 1,222,010 evidence bytes before this report.
No model payload was written. The worker exited and the lock was reacquired.
Validation was not scored, test remained unopened, and all first outputs and
selected/fresh weights remain retained.

## Decision and next action

Retain the selected EMA parents. Do not launch a lower-LR or warmup quality sweep
from this result. Prepare the paired independent-batch averaging diagnostic in
`GENERAL_RSI_BATCH_AVERAGING_NEXT.md`. It tests a different direction estimator
at fixed update count, with its extra data and compute charged explicitly.
This is not a batch-reuse arm. No quality confirmation panel is admitted.

Evidence: `tokenizer-evidence/general-rsi-update-response-001/` and
`tokenizer-evidence/general-rsi-update-response-fixture-001/` retain plans,
admission, first scores, checks and completion reviews.
