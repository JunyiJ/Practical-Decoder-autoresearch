# Ungated squared-ReLU: tiny-component source preparation next

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
