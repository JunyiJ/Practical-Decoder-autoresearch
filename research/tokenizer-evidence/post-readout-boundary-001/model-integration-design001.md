# Single-projection actual-model source preparation

This design follows accepted tiny projection-component001 and source-design001.
It prepares four additive files in research/readout_projection_model_v1. No new
model has been imported, constructed or run in this source-only step. The tiny
functional D<=8/V<=31 guards remain unchanged; actual geometry has its own policy.

Local basis is the original readout quality closure/final decision and closed
existing-analysis001/result.json. MoS-minus-native H200 mean was -.033083090568,
but MoS-minus-matchedMoC was -.001065236022 with one worse seed and positive
conditional-text upper bounds. Both mixtures beat native on every RAW seed too.
These are conditional results under finite, separately selected rates. They do
not identify a rank, averaging, clipping, amplitude or conditioning mechanism;
MoC is not a fallback candidate. Native remains incumbent and all weights stay.

Re-retrieved [Yang et al., arXiv1711.03953v1](https://arxiv.org/html/1711.03953v1)
for this integration boundary: sections2.3-2.5 motivate tanh context maps and
distinguish feature from probability mixtures. Section3/Table4 reports mixed
feature-mixture generalization across PTB and WT2. RNN word models, regularization
and optimization differ from this local transformer. Those reported results
support a matched control, not a predicted local gain. All three proposed heads
retain logit rank<=256; nonlinear features do not establish a high-rank solution.

The new question is whether one nonlinear projection can improve over both native
and equally parameterized linear projection. Linear WA is absorbable into W as a
represented map, but changes optimization. Tanh changes amplitude and derivatives
jointly. A quality failure would reject this exact smaller recipe without a causal
diagnosis; a full success would permit separate integration review. No old K2
replay, initializer/LR/K/horizon/EMA rescue or old score pooling is allowed. This
is local architecture follow-up; carry broad/interaction allocations forward.

## Source contract

Native is the genuine body-allocation incumbent child with its TRUE config. Its
constructor runs first in a saved/restored CPU RNG context, with exact common88
parameter order, values and post-construction CPU draw digest to be tested.
Register a Projection module afterward, containing only one bias-free A[256,256].
Use torch.empty plus private-generator Uniform[-1/16,1/16]; a regular nn.Linear
constructor would consume caller RNG. Linear and tanh use the identical private
initialization stream derived from a NEW audited initializer base and the domain
readout_projection_model_v1:projection:. No role values are assigned here.

Equations remain native Wz/4, linear W(Az)/4 and tanh W(tanh(Az))/4. There is one
untied vocabulary matrix and genuine cross_entropy in every arm. Own tagged
logits, config/fingerprint, weight kind and optimizer signature reject old mixture
and legacy identities. Native28492416/88/44Muon44AdamW; projected28557952/89/
45Muon44AdamW. A uses Muon eta/2/RMS matching; common native group order/rates and
AdamW(.9,.999),epsilon1e-6/independentdecay.00006 remain exact. No optimality claim.
Instance/signature/weight-epoch caches wrap genuine native layers; optimizer steps,
optimizer loads and weight loads invalidate old cache epochs. Weight loading
validates persisted tensors/RoPE before mutation, excluding nonpersistent masks.
This weight consumer is not complete training-state or resume qualification.

## Next finite generated proof, before any scientific worker

First implement and pin a NEW role census and generated proof, then perform source
review and observe exact plan SHA before either command. Census must audit one
unused initializer base AND its SHA-derived private projection role across recorded
text under a 120sCPU/4GiB/reserve8GiB/32MiB output guard. No Torch/model/data loader,
no role reuse, and no reconstruction of old failed censuses. A match/failure closes
that original command. Only successful new census permits actual construction.

Proof is one serial local MPS process: an independent native reference takes two
generated B2/T8 updates; each of native/linear/tanh takes two, eight updates total.
Check true native config/common88 construction/order/draws, identical private A for
both projected arms, exact same-MPS native losses/parameters/optimizer state, all
88/89 gradient and update exposure, A gradients/nonzero changes, and all group
rates/state identities. CPU/MPS forward/CE/all gradients and target CE versus full
normalized-log-probability gather/all gradients must agree. Add one B8/T128 backward
stress per arm after releasing reference snapshots, optimizer and prior graphs.
Stress has no extra optimizer step. Every phase explicitly releases large graphs
before the next; never retain multiple full models on MPS.

Test full/cached/reset B1/B8 length128 using prefill96/decode32 twice, native cache
196608B/item, exact future-token causality, foreign/stale/aliased cache rejection,
and invalid token/target/config/dtype/device/autocast/weight/optimizer identities.
Check rejection transactionality before graph/cache mutation, reconstructible
causal masks, caller CPU/MPS RNG and source hashes unchanged. Literal arithmetic
token fixtures must be NEW; no old fixture/corpus/checkpoint/quality role is used.
No tensor files, corpus scoring, EMA, sampler or resume claim. A complete own
RAW/H200/resume/scorer/worker/pilot chain remains required afterward.

Prospective full-model tolerances: CPU/MPS CE absolute3e-5; elementwise gradients
atol4e-5/rtol4e-4; cached logits/log-probabilities atol3e-5/rtol3e-5; same-MPS native
control and causality exact0. They belong to the NEW actual-model proof, not edits
to closed tiny tests. Pin the implemented cases before first numerical execution;
failure closes its original command, with no retry/source/tolerance rescue.

## Resource scope and limits

Projected FP32 parameters114231808B; one B8/T128/V50257 vocabulary array205852672B;
largest B8 cache1572864B. Two model-sized optimizer moments are a conservative
228463616B bound, plus gradients114231808B. CPU native reference parameter and
optimizer snapshots are bounded by341908992B and must be released before B8 stress.
CPU all-gradient snapshots are also released before that stress. These component
counts exclude runtime workspaces/allocator behavior; they are not a peak guarantee.

Relevant original mixture integration proof completed8 generated updates in17.299s,
peakRSS2899492864B/MPS2489122816B. This is a cost prior only, not a replay/admission.
New single-map proof forecasts60s (roughly twice that17.299s plus three full-batch
backwards and fixed overhead), RSS/MPS3.5GiB each. Enforce600s whole MPS proof,
120s each CPU phase, RSS4GiB/MPS6GiB/reserve8GiB/fourthreads/strictfallback0fastmath0.
Actual guarded results are required. No numerical execution is admitted by this
source review alone: implement, pin and review the new proof and census first.

Complete next-chain output is census32MiB + proof8MiB + source-preparation2MiB,
zero weights/tensorfiles/serialization copies, plus reserve8GiB. Required floor
8633974784B and recovery target10781458432B include the extra2GiB headroom; current
free space exceeds both. No cleanup/blocker/deletion credit is needed. Full scientific
model/EMA/optimizer serialization and all-panel time/storage remain unqualified.
Any later panel needs both controls/equal finite tuning/three fresh paired roles,
practical mean<-.02 each/all paired deltas negative/all6 conditional-text bounds
negative, all retained outputs/serving/independent statistics budget before freeze.
