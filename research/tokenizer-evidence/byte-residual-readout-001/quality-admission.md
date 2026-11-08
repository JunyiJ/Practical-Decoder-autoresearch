# Admission of one fixed residual-readout comparison

The source-bound pilot passed in55.323seconds:24 updates, six CPU/MPS cases,
exact baseline initial logits and all105 common-parameter gradients, nonzero
learning of both gates and later matrix gradients, complete optimizer/H200
coverage, exact RAW/H200 serialization and CPU/MPS maximum per-target error
1.90735e-6 (limit3e-4). Synthetic analytic forward/input/matrix/gate gradients
matched exactly in the recorded cases. The largest RSS was3,185,377,280bytes.
This is implementation evidence only; no pilot quality score selects a recipe.

The mechanism, contrary local evidence and final UAI2021 ReZero paper limits are
in `research/BYTE_RESIDUAL_READOUT_PROTOCOL.md`. This candidate is qualified for
one new architecture-exploration panel because the observed feature replacement
constraint has a direct, falsifiable intervention and a matched linear control.
The prior failed plain-transform, block-scaling and topology campaigns remain
closed. There is no activation/initialization search or fallback adoption.

Freeze byte, residual-linear and residual-tanh at the existing .0025 recipe and
1200 updates. Use fresh bundles62102600000,62102700000,62102800000, offsets1–5
for native/composer/reserved-nonlexical/train/validation roles and offset6 for
A. Bootstrap seed62102800007. Audit all roles before admission. Rotate arm order
byte/linear/tanh, linear/tanh/byte, tanh/byte/linear. All nine fresh runs count;
none is a screen or selected for favorable loss. Gate starts0, unrestricted sign.
Preserve the protocol's two contrasts, >.02 mean gain, every-seed improvement,
six negative conditional upper bounds, H200 primary and closed holdout. The
linear arm cannot become an unplanned fallback. Successful quality still needs
independent saved-result review and a qualified consumer before adoption.

The completed forecast is `resource-forecast.json`: eighteen checkpoints with
64KiB metadata padding each1,182,239,226bytes; largest overlapping serialization
65,767,529bytes;64MiB quality reports (>3x the two prior quality folders);8MiB
pilot evidence. Total retained-output ceiling1,323,504,227bytes. Adding8GiB
reserve and2GiB extra headroom requires12,060,922,467bytes. The measured free
12,161,163,264bytes exceeds this complete requirement by100,240,797bytes. The
old panel's ceiling and evidence are unchanged; this is a new measured forecast.
Recheck at freeze and each worker start; capacity loss invokes authorized recovery,
not fewer runs or missing checkpoints. No cleanup is currently required.

Observed comparable runtime is roughly105minutes. A deliberately conservative
forecast from1.5x the pilot's warmed maximum update times plus1200seconds per
trajectory for scoring/serialization and600seconds review is19,837.524seconds,
including the55.323second pilot. The hard total is36,000seconds, at most3600
per trajectory,4GiB RSS/6GiB MPS. Charge recovery and final review too. Reserve
300seconds for independent review before launching another whole trajectory.

The quality worker uses the already qualified training-equivalent MPS scorer and
stable FP32 CPU reference; no backend tolerances change. Record initial common
weight hashes, A hashes, terminal train/validation sampler identities and global
RNG identities. Capture scalar gate evolution with existing training losses and
clipping records. Do not interpret a small gate as small Muon matrix updates.
Publication includes exact source pins, original failures if any, full saved
per-target losses, parameters, RNG/schedule/H200 counts and independent bootstrap
reconstruction. Stop on any failed assertion; recover execution defects under a
separate identified attempt, with the scientific contract fixed.
