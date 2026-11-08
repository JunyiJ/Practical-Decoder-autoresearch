# Optimizer-state validation adopted

The active recipe is [recipe.json](recipe.json). It keeps the same selected
16,351,859-parameter model and accepted quality evidence. The new training stack
uses model-v5, optimizer-v6, state-v5, training-v5 and inference-v4. Cache-v3 and
the selected-checkpoint public loader-v3 are unchanged. The recipe keeps the old
model implementation receipt for that loader and adds a separate, explicit
`training_implementation` receipt for the four new training/export sources.
Callers must verify those receipts and hold the shared worker lock/resource guard.

## What changed

Every original optimizer-state structural, ownership, finite, nonnegative and
step-count check remains. Device tensor predicates are combined before reading
their Boolean results on the host. CPU scalar step checks remain immediate.
With multiple independent faults, the first error can change; all faults must
still reject before any update mutation. Optimizer arithmetic, schedules,
gradients, model computation and artifact formats are unchanged.

Exact reversible source maps cover the isolated prototype, the integrated
optimizer/state/training sources and the inference module's two typed imports.
Original source namespaces and historical evidence were not edited.

## Read-only state comparison and harness recovery

The test reused the first-fresh accepted RAW endpoint, without training. Ten
fixed corruptions were applied to both implementations on CPU and MPS: nonfinite
momentum/first moment/second moment/CPU step, negative second moment, wrong step,
wrong shape, missing state field, parameter alias and foreign state key. All
40 checks rejected through `step()` with unchanged parameters, gradients, buffers,
state, storage ownership and counters. Restoration and lazy empty-state acceptance
also passed. Numerical state values were cloned independently; only the model
was shared for this read-only comparison.

Two harness failures remain preserved. Attempt001 completed all CPU cases but
supplied `mps` rather than the actual `mps:0` checkpoint backend. Attempt002 fixed
that transport field, then hit the same spelling mismatch in its direct private
validator argument during warmup. Attempt003 uses the live parameter device in
both places. It ran only the unfinished MPS phase. CPU results were carried
byte-for-byte through all attempts; no measured MPS result was replaced. The
original 180-second budget charged both failed attempts. No scientific gate or
case changed, and no training roles were consumed by these read-only checks.

MPS state-only median fell from 51.963 ms to 13.139 ms. Complete optimizer
validation fell from 99.584 ms to 58.066 ms, with seven of seven paired wins.
CPU complete validation was 42.542 ms versus 42.680 ms: no CPU speedup claim.
All prospective microbenchmark gates passed. These are validator measurements,
not training throughput measurements.

## Exact updates, resume and warmed throughput

A separate prospectively frozen comparison used fresh CPU sampler roles
62101000001/2 and MPS roles 62101100001/2. These roles are now closed/reserved.
CPU parameter/optimizer/averaging/sampler/scheduler/RNG states matched exactly
after updates one and two. A candidate restored at update one produced the same
update-two state, including the first averaging snapshot.

MPS completed 20 paired updates with alternating order. Four pairs were declared
warmups; 16 were measured. All update histories and complete final states matched
exactly, including eight averaging snapshots. The predeclared gate required at
least 5% lower median paired latency, at least 12 of 16 wins, and neither measured
averaging phase regressing. Results:

- Median candidate/reference ratio: **0.822749**, or **17.73% lower latency**.
- Candidate faster in **15 of 16** measured pairs.
- Phase median ratios: **0.814435** before averaging; **0.833001** with averaging.
- Measured totals: reference **17.087397 s**, candidate **14.296959 s**.

This is one short local trajectory, not independent training seeds, a formal
confidence interval, or a full 1,200-step throughput result. Do not multiply
separate historical speedups into an unmeasured combined claim. The prior finite-
parameter optimization's two-step no-gain result remains preserved.

## Saved artifact compatibility and cost

The new training stack restored the accepted 1,200-update RAW checkpoint and
captured exactly the same complete state, including 41 averaging snapshots.
Inference-v4 reproduced the original H200 export exactly in memory and loaded
all saved H200 parameters exactly. This required no new training or weight archive.

The three read-only validation attempts used 42.541 guarded seconds combined;
the CPU/MPS training comparison used 63.052 seconds, and saved-artifact checking
used 6.611 seconds. The largest recorded RSS was 3,418,243,072 bytes, within 4 GiB.
There were 45 numerical updates total, zero quality evaluations, zero holdout
access and zero new weight files. Adoption audited the saved sources, full state
fingerprints, rejection evidence and reconstructed timing gates without rerunning
the experiments. Both harness failures and original scientific records are retained.

## Next executable action

Refresh one warmed training-step profile under this adopted stack, using fresh
sampler roles and the existing small profiling pattern. The old profile predates
both validation optimizations and should not be used to rank their remaining
costs. Select the next implementation change from the new measurements. Keep
infrastructure proportional; do not replay the closed quality or cropping tests.
No worker is currently active.
