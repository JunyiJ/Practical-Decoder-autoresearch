# Wider direct GQA/MLA preparation

68 current CPU checks passed:66 combined geometry/cache/policy/controller checks
plus separate bounded real-trainer continuation tests for GQA and MLA at512.
Each continuation test serializes the partial checkpoint in memory, checks exact
model/optimizer/EMA/RNG/sampler equality and loads H200 inference parameters.
No durable CPU model checkpoints were created. Real disk H200 exports remain a
requirement of the separately frozen MPS pilots. Width1024 has meta-device
geometry/optimizer accounting only; no numerical training claim at1024.

The initial two-update CPU fixture violated the cooldown boundary and failed
before training. Its log and diagnosis are retained; the corrected four-update
fixture initializes EMA at1 and uses cooldown at2. No MPS retry or quality run.

Storage accounting includes two retained pilots and12quality trajectories, plus
8GiB reserve. Width512 fits the current observation;1024 does not. This does not
measure actual runtime or peak memory. No quality claim, test holdout closed.
