# Composer reuse full-model engineering qualification

The component passed its fixed gates with a55.53% median composer time reduction
on three fixed historical batches. The next unknown is full-model gradient and
optimizer behavior, complete restart, and whole-update benefit. This is a new
execution identity, not a change to any frozen runtime or scientific campaign.

`general_rsi_reuse_runtime_v1.py` inherits the original model validation and
forward APIs, replaces only whole-token composition, and exposes a distinct
model identity. The optimizer uses identical grouping, options and inherited
validation, with explicit admission of the new model type. The context retains
the original update/capture/restore implementation and adds distinct execution,
model and complete-checkpoint identities. Cross-loading old/reuse states is
rejected. All inherited sources are bound before execution.

Use one fresh engineering seed bundle62111011001–004, baseLR0.0025 and horizon40.
Both paths must start with identical parameters and optimizer groups. On the
first training batch, compare the scalar CE and every full-model gradient at
`abs(reuse-original) <= 2e-6 + 2e-4*abs(original)`, restoring sampler state before
real updates. Compare every parameter after the first actual Muon/AdamW update
at `abs(reuse-original) <= 2e-5 + 2e-4*abs(original)`. The larger absolute
parameter allowance explicitly permits limited optimizer amplification, up to
0.8% of the0.0025 baseLR in the absolute term. It does not establish identical
training trajectories or quality. Stop at either initial numerical gate failure,
preserve exact failures, and do not tune tolerances.

If those gates pass, run40 updates per arm with paired sampler hashes. Use AB
order on odd steps and BA on even steps. Synchronize around the complete inherited
update, including sampling, forward/backward, clipping, optimizer, gradient
clearing, schedule, EMA and batch-receipt bookkeeping. Exclude the first two
updates per arm as warmup. Require median reuse/original time at most0.90 across
the remaining38 measurements per arm. This tests the40-step fixture, not a
production-time schedule or long-run thermal behavior. Report late parameter
differences descriptively; no claim of trajectory equivalence is permitted.

Serialize reuse complete state at update30 after its first EMA snapshot. Restore
into a fresh context, verify every state entry immediately, then run10 more
updates. Require bitwise equality with uninterrupted update40 for every parameter,
optimizer state, EMA, sampler state, CPU/MPS RNG state, history, configuration and
schedule field. Update40 has two EMA snapshots. Save all state hashes, original
serialization receipt, timings and failures. This extends the qualification to
the new execution path; it does not repeat the already qualified old restart.

Only one numerical worker under the shared nonblocking lock. Limits are900s
cumulative over at most two attempts (second only for demonstrated harness or
resource repair), RSS4GiB, MPS driver6GiB and10GiB reserve. The bounded writer caps
each checkpoint at512MiB. At most four payloads per attempt cover restart and
failure states/gradients. Two attempts plus one512MiB serialization overlap and
16MiB metadata total4,848,615,424 bytes, below the5GiB complete output cap used
for admission. Existing files stay retained. No cleanup is needed if admission
passes. A hard resource failure preserves its original guard and partial writes.

Passing all gates qualifies this small engineering fixture only. A fresh complete
quality comparison, with both implementations and all declared seeds, retained
outputs and a full forecast, is still required for scientific adoption. No
validation scoring, generation or original-test access occurs here.
