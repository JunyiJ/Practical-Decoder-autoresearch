# Native readout mixture: generated component closed; own model next

The additive two-file readout_mixture_v1 component passed63literal CPUFP64,
CPUFP32 andMPSFP32 native/MoC/MoS forward/analytic-gradient cases and21invalid
shape/dtype/target/nonfinite/kind cases. Independent scalar math.fsum/logsumexp
and explicit z/W/A/R calculus agree: FP64maxforward1.78e-15/gradient7.40e-15;
FP32maxforward1.21392355e-6/gradient1.26477125e-6 under frozen tolerances.
Normalization, target-only versus full-gather loss/gradients, permutations,
identical contexts and a discriminating MoS/MoC case passed. GlobalCPU/MPS RNG,
inputs and all1087source pins were unchanged. ZEROmodel/optimizer/corpus/
checkpoint/qualityseed calls or tensor files. Both120sCPU/600sMPS guards passed
in2.433s, peakRSS389,922,816B/MPS11,255,808B. This does not qualify a full model.

Closed metadata review found missing explicit mixed-device/autocast adversaries.
A distinct single new harness rejected16new N2/D2/V3 device/autocast cases,
without replaying any original positive/gradient/invalid fixture or editing
component sources. Its guards and source/input/globalRNG invariance passed.
All1088pins preserve1084scientific predecessors. Both numerical commands are
terminal and immutable; never rerun/edit them. Metadata review0.026s loaded no
models and replayed no numerical cases. Total two proof roots295,378Bmetadata.

Original launch001 used the shell's Python3.14 lacking torch and failed before
plan/attempt/guards/fixtures. Preserve that preimport execution failure separately;
it is not a numerical or quality failure. Verified project .venv/bin/python is
Python3.13.12/Torch2.14.1/git5c4886908584029761b579af026dcfb627c84070.
Launch002 used that existing runtime for the first numerical fixture execution;
no packages were installed or source changed. Use this explicit project runtime.

Next implement an additive own readout_mixture_model_v1 model/config/loss,
optimizer grouping and weight/cache consumer. Keep frozen tiny component limits
D<=8/V<=31; do not relax them to pass actual model state. Full-model equations
must be independently integrated and checked under the exact design, with a
genuine native child/backbone/head first, common88construction/RNG/weights exact,
then explicit private context initialization and zero router. Root configuration
must identify readout_model/backbone_template; native child keeps its true exact
configuration. Never masquerade as old gate/MoE/legacy model kinds or bypass
validators. Own nested-backbone parameter grouping must preserve common88order
and add twoMuon context maps plus oneAdamW router,91tensors/28,624,000elements.

Before any model construction audit NEW unused initializer roles and fix the
private initializer domain/seed derivation. No quality seeds assigned now.
Use forked caller RNG state and independent private context generator. Own tagged
distribution/NLL APIs preserve native CE exactly, implement MoC genuine CE and
MoS stable target-only logsumexp; own full scorer/cache/weight contracts follow.
No old generic-logit consumer may reinterpret normalized mixture probabilities.

Admit the smallest full generated model integration proof after source review
and resource gates: exact native construction/outputs/gradients and same-MPS
optimizer-control updates; genuine additional parameter exposure; generated
CPU/MPS loss/gradient agreement, causality, full/cache/resetB1/B8, transactional
foreign-arm weights and complete inventory/grouping. Start bounded shapes, then
fullB8×128 target-only loss/backward stress under4GiBRSS/6GiBMPS/8GiBreserve.
Keep one resident MPS model at a time where possible. Full memory/time/transport
remain unqualified. A failure closes its own new command; no rescue or tolerance
change. Do not reopen old optimizer-response/precision diagnostics.

Future own RAW/H200/resume/private-sampler/EMA/fullclipping/exposure/target
validation and real worker/pilot prerequisites remain required before any
scientific freeze. No actual training, corpus scoring, holdout access, quality
seed assignment, adoption or full panel occurred. Current native incumbent and
all prior evidence/weights remain unchanged; continue source integration next.
