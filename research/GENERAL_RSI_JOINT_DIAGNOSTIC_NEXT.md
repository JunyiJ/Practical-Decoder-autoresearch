# Joint-gradient mechanism diagnostic, 2026-11-12

The completed integration permits a new test of the hypothesis in
`GENERAL_RSI_JOINT_GRADIENT_NEXT.md`. Its local observations, contrary findings,
pinned DITTO implementation and supplement remain the evidence basis. The
integration passed all264 gradient comparisons and all three full restarts.
It measured0.343/0.354/0.486seconds per update for full CE/half CE/joint mixture.
Those are engineering measurements, not quality gains.

## Prospective comparison

Use all three selected native parent weights from slots01/03/05. Start fresh
optimizer and sampler states for each arm. Use new paired roles, independent
natural and pseudo streams, and rotate arm order across parents. Compare full
CE, half CE, and half CE plus half pseudo, each for100 fresh natural batches.
The joint arm also draws100 fresh pseudo batches. It accumulates both backward
passes at unchanged weights, clips once and advances the optimizer once.
The other arms also advance once per natural batch. No batch is deliberately
reused across updates; paired arms share their natural stream by design.

Preserve LR0.00025, clip1.0, decay0.00006, model and qualified sampler/loss.
Set the qualified clock budget to1,200seconds and stop at100 updates, before
its800-second cooldown/EMA boundary. Assert ratio1 and zero EMA snapshots each
update. Thus this diagnostic has constant LR, no averaging or checkpoint
selection. No new model payload is retained or adopted. Log every update and
all first probe results. The original closed panels remain unchanged.

For each parent, freeze100 natural and100 synthetic input receipts first.
Select64 document-local128-target natural windows (8,192 targets) with a fresh
training-only seed. Exclude every consumed natural source interval, all synthetic
source spans including preceding context, and other selected probe windows.
Then choose eight batches of eight synthetic spans using another dedicated seed.
Reject a proposed batch if any source interval intersects
a training/probe interval or another span in the batch. (Span plus
prefix has at most52 tokens.) Preserve accepted proposal numbers and input hashes.
The resulting64 spans also have mutually disjoint source intervals. These
exclusions avoid direct input reuse within this diagnostic, not prior pretraining
exposure. They are not an independent holdout or a corpus-representative sample.

Score the parent once and each arm at natural updates1/10/25/50/100. Natural CE
uses the qualified corrected-FP32 scorer. Synthetic scoring uses the same corrected
FP32 forward path and target/previous-copy mask as the qualified pseudo objective.
For each masked target, collect its probability p, the previous-copy probability a,
and −log(1−|p−0.5a|), using the qualified epsilon cap. Aggregate by target count
across all64 spans: mean p, mean a, mean penalty, and sum(p)/sum(a). Record batch
sums and counts, plus saturation. Lower repeated probability alone can reflect
lower probability everywhere; the anchor and ratio expose that alternative.
A tiny analytic oracle checks only this new aggregation before model scoring.
No new generations, validation scoring or test access occur.

## Frozen interpretation

Report joint-minus-full, joint-minus-half, and half-minus-full for every metric
and all five horizons, all three parents, then equal-parent mean contrasts.
Primary diagnostic horizon is100. Do not select a favorable earlier horizon.
Record clipping counts, median gradient norm and parent-relative CE drift.

Directional support requires joint-minus-both-controls to be negative in every
parent for natural CE, pseudo penalty, mean repeated probability and aggregate
repeat/previous probability ratio at100. This stringent descriptive condition is
not a significance test or a model-adoption gate. Prediction improvement without
suppression improvement leaves the proposed joint benefit unresolved. Regression
or mixed signs also withhold support. Even a full pass needs separate prospective
validation confirmation and fixed first-generation evaluation before adoption.
Do not pool historical scores into these fresh contrasts or make causal shares.

## Complete resource forecast

Nine trajectories,900 optimizer updates,900 natural and300 pseudo components.
There are45 arm probes and three parent probes. Forecast300 times each measured
update cost,48 natural scoring passes using the measured22.064seconds per100,416
scoring targets, and384 synthetic forward batches at a conservative0.18seconds
allowance. Add90seconds for nine loads,30seconds state checks/digests,20seconds
bindings/input preparation and30seconds logging/overhead. Whole-worker ceiling
1,200seconds, RSS4GiB and sampled MPS6GiB. Scoring-batch allowance is an estimate,
not a new speed measurement. The remaining margin covers variability and failures.

No checkpoint serialization is required. Retain all metadata/logs/probe results
within64MiB, allow64MiB atomic/output overlap, preserve10GiB reserve and require
another2GiB free headroom at admission. Existing qualification payloads and all
selected/fresh weights remain retained. Inspect actual processes, acquire the
shared lock nonblocking, freeze all source/data/parent/input hashes and the full
admission before launch. Stop on a new human stop; preserve failed attempts.
