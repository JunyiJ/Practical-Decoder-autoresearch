# Ungated squared-ReLU: tiny component accepted; actual model source next

Original tokenizer-evidence/post-projection-boundary-001/relu-squared-component001
is CLOSED_ACCEPTED/no replay or retry. PlanSHA
f0699f1609bc706eea13c4a270d91d6cd23d224aac6f3adf62a3be237c19069b.
All99 NEW literal comparisons (11families x3arms xCPU64/CPU32/MPS32) passed
independent scalar math.fsum/stable-sigmoid forward and analytic input/up/down/
native-gate gradients, including zero derivative at the ReLU-squared kink.
All89 transactional kind/type/shape/dtype/device/autocast/nonfinite/overflow
rejections passed. Caller inputs/nonzero gradslots/globalCPU-MPSRNG preserved;
1178 source bindings preserve1175 predecessors. No models, optimizer, checkpoints,
corpus, RNG draws, quality roles or tensor serialization. All four guards finished.

FP64 maximum forward2.66453525910e-15/gradient7.10542735760e-15; FP32 maxima
forward6.43493943464e-7/gradient2.85046305493e-6. All original fixed elementwise
predicates passed: CPU64 forwardatol1e-11/gradatol1e-10/rtol1e-10; FP32
forwardatol4e-6/gradatol4e-5/rtol4e-4. Whole1.838612s/RSS376274944B/MPS9125888B
under600s/4GiB6GiB/reserve8GiB/2MiBmetadata. Independent metadata-only review
agrees, with no numerical replay. Tiny D<=8 success does not admit D256 models.

CURRENT NEXT: SOURCE implementation/review ONLY, following
tokenizer-evidence/post-projection-boundary-001/actual-model-source-design001.md.
Own exact model/optimizer/cache identities; genuine native common construction
first, then privately paired ungated FFNs. No actual-model construction/roles,
generated proof, full-state/resume, real corpus/scorer, pilot, quality budget or
launch admitted. Those need separate new source/correctness/resource/role gates.
Native/holdout unchanged. All earlier closed sources/commands stay immutable.

The remainder records the historical pre-execution design; its NEXT wording is
superseded by the completed tiny gate and current source-only action above.

Read tokenizer-evidence/post-projection-boundary-001/evidence-review.md and
 direction001.json. Primer2109.08668v1 section3/Figure6/A.7/A.8 and
GLUVariants2002.05202v1 section2/equations4-6/section3.1/Table1 were actually read.
The candidate is W_down[max(W_up x,0)^2], NOT a squared independent GLU gate.
Controls: native W_down[SiLU(W_gate x)*W_up x], matched-budget ungatedSiLU.
At actualD256: gatedH672/ungatedH1008 give516096matrixelements per layer.
This is source algebra, not a runtime/model budget or an optimality claim.

Local contrary original results inspected: ungatedSiLU-minus-SwiGLU
+.007388630366807889/two seeds worse; GELUgate-minus-SiLUgate
+.001550755925793319/two worse. Bothclosed/no replay,unpooled. Table sharing/
localconvolution/depth/gate/MoE/readout remain closed. Anchor-normalization
functional gate passed, but model integration was rejected for worse same-case
MPSinput-gradient error. No normalizer/kernel change is part of this question.

NEXT: implement smallest additive purefunctional three-arm component plus
INDEPENDENT scalar forward/analyticbackward oracle with NEWliteral inputs.
D<=8/N<=16; atD8 gatedH8 versusungatedH12 gives192matrixelements each. Biasfree/
dropout0/no rescale/normalization/router. SquaredReLU derivative2max(u,0),zeroat0.
Review invalid shape/device/dtype/autocast/nonfinite and signed/zero/positive/
negative/boundaryinputs; caller input/gradslots/globalCPU-MPSRNG preservation.
Pin/review exactnewsource/scope/cases/tolerances/resources/actualplanSHA BEFORE
first numerical command. No oldfixtures/retries/source/tolerance rescue.

Working tiny10s/<512MiBRSS/<32MiBMPS; strict120sCPU/600sMPS/4GiB6GiB/reserve8GiB/
2MiBmetadata/fourthreads/fallback0fastmath0. No tensorfiles/weights/serialization;
floor8592031744B/extra2GiBtarget10739515392Bfits~42.27GBfree. Actualsourceadmission
stillrequired. No model/optimizer/corpus/checkpoint/randomroles/fullpanel budget
or qualitylaunch admitted. Futureactualmodelinitialization/grouping/state/scorer/
cache/workers/pilots/fullquality andmandatoryconsumers need separate complete gates.

LOCALFFNarchitecturefollowup,notbroadquota/preparationnotnumericalexploration.
Native incumbent and holdout unchanged; no active worker.
