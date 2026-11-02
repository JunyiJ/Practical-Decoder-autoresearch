# Independent weight-decay preparation

Question: does removing independent per-update decay0.00006 improve the adopted
D256 MLA recipe at the standard peak LR0.00125? The architecture stays score0.5,
SwiGLU672/SiLU/blockRMS, no product norm/local mixing/warmup, and H200.

Both architecture (parallel residual branches) and bounded HP were reviewed
in decay-preparation-001/direction-decision.json. Select the previously deferred
regularization question. The proposed comparison is three fresh paired seeds
at one common fixed LR, after complete prerequisites, capacity and resource
pilots. No LR screening, retrospective LR rescue or grid extension is proposed.
This estimates a standard-recipe effect; it does not compare separately tuned
optima. Keep all six fresh trajectories, paired inference and strict quality
gates. No scientific seeds are reserved and no pilot/quality plan is frozen.

## Generated optimizer prerequisite

Frozen optimizer-component001 passed all four CPU FP64/FP32 by decay cases
using the real HybridOptimizer with actual Muon and AdamW child optimizers.
Independent direct child updates after explicit shrinkage match exactly,
serialized resume after three of six updates reproduces all parameters and
optimizer states exactly, gradient-absent parameters stay unchanged, and
cross-arm optimizer-state loading is rejected. All531 predecessor sources
plus the two new bound files were verified unchanged. No scientific corpus,
seed or on-disk checkpoint was used. Original source/plan/log/result/guard
are retained; do not rerun this closed successful fixture.

The isolated zero-gradient shrinkage over1200 updates is0.9305288857862962,
matching(1−0.00006)^1200. This is about6.95% shrinkage in the absence of
optimizer updates, not a claim about actual training weight norms or quality.

Next: narrow additive full-model configuration and optimizer-group identity
checks, CPU/MPS update and serialized model/optimizer/EMA/RNG resume, final-
horizon H200/cache validation, then actual bounded worker/independent parent
validation/inference. Keep original namespaces frozen. Measure complete retained
output high-water including serialization overlap/metadata and8GiB reserve
before freezing a fresh finite panel; current free space is only an observation.
Use exact authorized recovery audits if needed, never partial panels or relaxed
guards. No new model-quality result is claimed.
