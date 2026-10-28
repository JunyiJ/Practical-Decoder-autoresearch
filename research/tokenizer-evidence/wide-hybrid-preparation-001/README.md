# Wider hybrid anchor prerequisites: incomplete

The separate wider NoPE policy supports512/1024 meta construction and retains
the old64/256 guard. No training controller or quality plan is frozen. Defaults
in the config helper are preparation values, not selected hyperparameters.

`cpu-geometry.log`:35 checks passed in11.38s, peak reported RSS1,358,561,280bytes.
Command: `pytest -q tests/test_tokenizer_wide_hybrid.py tests/test_tokenizer_hybrid.py -k 'not real_resume'`
before the three continuation tests were added. The process had a600s alarm,
one-second4GiB process peak-RSS/8GiB-free-disk guard. These cover meta geometry
at512/1024, numerical equations/gradients at512, FP32 layerwise cached/full parity
at batches1/8, reset/interleaved requests/causality, optimizer roles/scaling,
checkpoint lineage, invalid mutations, and historical64/256 regressions.

`cpu-resume.log`:the combined1200s CPU continuation process passed its first
test (hybrid), then exited90 at the same resource thresholds. The all-gated-MLA
and all-KDA continuation tests remain unverified. `cpu-resource-failure-001.json`
records the surviving observations and lost trigger diagnostic. No numerical
retry followed. CPU fixtures serialize in memory and do not save model files;
they still cannot replace future disk-export checks in resource pilots.

`project_storage.py` constructs complete models, caches and installed optimizer
state shapes on the meta device, without a forward or optimizer step. It counts
saved model/optimizer/EMA tensors separately from the conservative allowance.
Nonpersistent causal masks are excluded from saved-buffer bytes. The initial
draft and correction record remain. The corrected anchor27-trajectory panel
requires52.96GiB including8GiB reserve at512. Exact tuning variants, actual
checkpoint sizes and MPS resource costs remain unverified. Width1024 is meta-only.

Anchor state at512/context128 is333,824bytes hybrid,327,680 all-gated-MLA and
335,872 all-KDA per batch item in FP32. It provides no meaningful memory advantage
over both controls at that workload. No quality, latency or adoption claim.
All prior campaigns, retained checkpoints and the test holdout remain closed.
