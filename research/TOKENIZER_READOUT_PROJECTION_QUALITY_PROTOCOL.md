# Single-projection full-quality source preparation

Status: prospective SOURCE design only. No scientific role, panel freeze, proof
or training launch is admitted here. The new ten-file
research/readout_projection_quality_v1 namespace preserves all1151 accepted
predecessors, including the four/twenty worker that must keep rejecting1200.
The prospective root is tokenizer-readout-projection-quality-v1-001; it must stay
absent until all new source, contract, consumer, role and resource gates pass.
Native incumbent and closed holdout remain unchanged.

## Evidence and question

Original local decisions: TOKENIZER_READOUT_MIXTURE_RESULTS.md and
tokenizer-evidence/readout-quality-preparation-001/readout-quality001-closure001/
review.json, plus the closed existing-analysis001/result.json. Mixture quality
passed runtime gates but failed adoption: MoS-native mean-.033083090568 versus
MoS-MoC-.001065236022, one worse seed and positive MoC interval bounds. Both RAW
contrasts had the same sign pattern. MoC is not a fallback candidate. Averaging,
clipping, rank or conditioning causes were not established. All old scores and
seed roles stay closed and are not pooled into the new comparisons.

Projection component/model/serialized-state and real4/20 prerequisites are
accepted in TOKENIZER_READOUT_PROJECTION_PREPARATION.md and post-readout-boundary-
001/projection-prerequisite-chain001-closure001/review.json. All72 actual updates,
all88or89 exposures, EMA3/8, complete state, FULL34278native CPU/MPS scoring and
cache checks passed. All three twenty pilots were unclipped. Their sampled CE
fell, but native ended lower than both projections: this is contrary short-run
evidence, not a full-horizon ranking. Twenty steps/early cooldown/short EMA do not
establish training stability or justify a source, LR or initializer rescue.

For this boundary, actually retrieved and read
[Yang et al., arXiv1711.03953v1](https://arxiv.org/html/1711.03953v1), sections2.3-2.5
and section3/Table4. Tanh context maps motivate a nonlinear feature control; the
paper distinguishes feature and probability mixing. Its matched MoC improves PTB
but worsens WT2 relative to the baseline, cautioning against a guaranteed gain
from an added hidden layer. Word-level RNNs, regularization and optimization differ
from this local transformer. Published gains are not local gains. All proposed
heads retain logit rank<=256/log-probability rank<=257; no rank-bottleneck claim.

Hypothesis: one tanh projection improves final complete H200 native CE beyond
both native and equally parameterized linear projection under equal finite LR
allowance. Linear WA can be absorbed into another vocabulary matrix as a map,
but is not optimizer equivalence. Tanh changes amplitude and derivatives jointly.
Success would support this exact recipe and permit separate consumer review;
failure keeps native and narrows the local nonlinear-head hypothesis. A linear
win cannot become a fallback. This removes the old K2/router/mixture and asks a
different reduced question. Rank causation, a unique amplitude mechanism and
optimal LR/initialization are not identified by either outcome.

This is LOCAL incumbent improvement under the proposed70/20/10 allocation;
carry broad/interaction allocations forward. Preparation is not numerical
exploration. Architecture/attention roadmaps and original centering/final-scale/
gate/MoE negatives were considered; no random quota or closed-family rescue.

## Exact prospective finite comparison

Arms: genuine native Wz/4, linear W(Az)/4, tanh W(tanh(Az))/4, one shared untied
bias-free vocabulary head and genuine CE in every arm. Native common88 TRUE
construction/order/weights/CPU draws precede the independent private A[256,256],
Uniform[-1/16,1/16], identical between linear and tanh. Private derivation remains
first8SHA256bytes UTF8(readout_projection_model_v1:projection:+decimalbase),
bigendian mod2**63. Native28492416/88/44Muon44AdamW; projections28557952/89/
45Muon44AdamW. A uses Muon eta/2/RMS match; common AdamW(.9,.999),eps1e-6,
independentdecay.00006. These rates are a tested recipe, not an optimality claim.

Nine screens: three arms times rates.000625/.00125/.0025, LR ascending with
native-linear-tanh order, one NEW shared screen role. Only after all nine full
independent acceptances choose minimum COMPLETE H200 MPS native CE within EACH
arm, lower LR on exact tie. No interim choice, pruning or boundary-grid extension.
Then resolve the original nine fresh slots on three NEW paired roles, rotating
native-linear-tanh/linear-tanh-native/tanh-native-linear. Rates are finite tested
choices, not optima. Never pool historical controls or replace a seed/slot.

Each trajectory1200updates/1228800targets, B8/T128/paired512, clip1, dropout0,
cooldown800/minratio.1, H200 start800/interval10/41 snapshots into one average.
Sampled evaluations1/301/601/901/1200 are diagnostic only. Retain all18RAW/H200
pairs, including every screen/control. Independent acceptance binds complete
RAW optimizer/EMA/private samplers/globalCPU-MPSRNG/LR/full clipping/all88or89
exposure1200, exact H200 ancestry and FULL34278targets/34279IDs/111536bytes on
CPU AND MPS. Own target contract61945b178404c035a322de7b7513e6ec4399b06a2275a036931f785018dccd2a
and physical target IDs dad00eaab7c2076736432b83eb9f1cc404c87aa5e4820b5f7abebf384c7fcdaf
remain distinct from the old mixture policy. CPU/MPS aggregate CE atol3e-5;
cached/reset B1/B8 log-probability atol/rtol3e-5,196608cache bytes/item.

All18 acceptances precede all9 fresh H200 serving checks: full128/prefill96/
decode32, B1/B8, two warm and five measured repetitions/cache reset. Timings are
descriptive local measurements, not a quality/cost adoption route. Only tanh can
qualify: mean delta<-.02nat/native-token against EACH native AND linear, all three
paired deltas negative for each, and all six simultaneous conditional-text upper
bounds negative. Blocks2048/4096/8192,2000PCG64 draws reinitialized per width,
same indices across seeds AND contrasts, linear quantile1-.05/6. Separately
implemented grouping/draw/quantile review must agree1e-12. This conditions on
trained seeds/reused validation text, not holdout/population-seed confidence.
No RAW/linear/cost fallback. Separate adoption/checkpoint-consumer review only
after every strict gate passes. Failed or incomplete commands close without retry.

## Required new gates before any scientific freeze

Implement and pin the NEW bounded census/full-contract/pilot-serving proof sources
first. One prospective recorded-text census audits16 distinct roles: six NEW
generated-only fixture roles, six NEW scientific screen/eval/fresh3/bootstrap
roles, and four SHA-derived scientific projection initializer roles. It must
finish120sCPU/4GiBRSS/reserve8GiB/32MiB metadata without Torch/model/corpus loader.
No values are assigned now. Observe the actual plan SHA before the first run.
No prior failed census, role proposal, closed fixture or source is reconstructed.

The new full-contract consumer may reconstruct CPU models/optimizers/EMA from the
three accepted20 pilot configurations/archives, within saved/restored caller RNG.
These are consumers of original checkpoints, not new trajectories or scientific
seed use. Validate actual pilot RAW/H200, then a SYNTHETIC1200-history/exposure/
LR/sampler/RNG/EMA41 schema using accepted pilot tensor values. Count full RAW/H200
archives to a bounded nonretaining stream; enforce RAW512MiB/H200128MiB/pair640MiB.
No real1200 tensors, training, corpus scoring or weight writes. Release each arm's
model/optimizer/EMA/payloads before the next. Whole600sMPS/120sCPU/RSS4GiB/MPS6GiB/
reserve8GiB/64MiB synthetic metadata; working60s/RSS3.5GiB/MPS.5GiB is provisional.

Isolated NEW metadata fixtures must check prefixes0/1/9/18, exact immutable
selection/fresh rotations, attempted/partial/foreign/duplicate/state/source/
resource/ancestry/target/parameter-count/bytes/clipping forgeries, full remaining
budget plus2GiB, both strict statistical paths on NEW nonconstant literal losses
and rejection cases. Only synthetic fixture paths may be mutated. No existing
fixture/source or real root is modified. Generated roles stay fixture-only;
scientific roles stay unused. New proof success does not qualify1200 tensor values.

A distinct three-pilot serving consumer must load all three original own H200
exports with retained RAW ancestry, full128/prefill96/decode32/B1/B8/reset and
two warm/five measured repetitions, NEW literal IDs, unchanged checkpoint hashes
and callerCPU-MPSRNG, zero training/corpus/quality-seed/weight writes. Whole600s/
RSS4GiB/MPS6GiB/reserve8GiB/8MiB metadata. Its actual measurements must replace
provisional serving/time forecasts before runtime admission. It does not replace
the later ALL9fresh benchmark. Observe exact pinned source/plan hashes; each
original command closes on failure with no source/tolerance rescue.

## Complete prospective capacity and time

Retain36 quality files:18pairs*640MiB=12079595520B. Add quality metadata512MiB,
new preparation sidecars128MiB, existing reserve8GiB, external headroom2GiB and
entry2MiB: floor23490199552B, plus2GiB recovery/entry target25637683200B. Sidecars
include census32MiB/full-contract64MiB/pilot-serving8MiB/sourceprep2MiB and22MiB
for remaining design/resource/selection/statistics/receipts. Streaming.tmp occupies
its final slot; publication alias shares allocation, no duplicate copy. Before
each unattempted slot credit only independently accepted retained CURRENT quality
bytes once; revalidate original identities. ZEROcredit for current12prerequisite
3359496280B/prior9proof3016484885B/other studies/hardlinks/future deletion. All
remain protected; closure gives no retirement eligibility. Free~46.88GB covers
the prospective target; no blocker or cleanup is declared. Re-measure at admission.

Original20 pilots: max update.9768425410147756s (including cold first update),
max worker46.66228099993896s plus max parent16.6546714999713s. Conservative
18*(1200*maxupdate+bothfixedcosts)=22239.50403091754s;25% margin=>27799.380038646923s.
This intentionally also retains the fixed20-update work in the overhead. Old
readout ALL9serving52.961390124983154s supplies only a proxy:2x+30=135.9227802499663s.
Old complete-contract CPU proof supplies primary+independent stats proxy66.901974666s;
add300s controller/hash margin=>scientific-panel proxy28302.204793563s. Include
new census120s/full-contract working60s/pilot-serving proxy54.075022332s/sourceprep
120s=>complete remaining-chain proxy28656.279815895s (~7.96h). Current projection
serving/full-contract costs are UNMEASURED; update from actual new guarded proofs
before freeze. No earlier completed output floor is reused or subtracted.

Working RSS/MPS3.5GiB each, one local model/worker, four threads/fallback0/fastmath0/
OMP-OPENBLAS-MKL4. Worker AND parent7200s each, fresh9serving600s, primary AND
independent statistics120s each, CPU validation/cache120s each, RSS4GiB/MPS6GiB/
reserve8GiB/output guards remain hard ceilings. Extrapolations are not promises,
stability evidence, guard ceilings or cost adoption. Full1200schema/serialization,
consumer/source/census/resource reviews and final clean commit/push must precede
any scientific freeze/launch. Existing sources/STOPs never reopen.
