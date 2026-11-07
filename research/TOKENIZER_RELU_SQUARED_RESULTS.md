# Tiny functional FFN gate: accepted, no quality result

Original tokenizer-evidence/post-projection-boundary-001/relu-squared-component001
and its metadata-only independent review are CLOSED_ACCEPTED. All99 new literal
comparisons (11fixtures x3equations xCPU64/CPU32/MPS32) and89 transactional
rejections passed. Scalar math.fsum/stable-sigmoid forward and analytic x/up/down/
native-gate gradients agree under the original fixed elementwise predicates.
ReLU-squared uses2max(u,0),including zero at zero; no independent gate exists.
Every arm uses3D^2 matrix elements, up to192, with D2/4/6/8 and N1..16.

| Backend | Maximum forward error | Maximum gradient error |
| --- | ---: | ---: |
| CPU64 | 2.66453525910e-15 | 7.10542735760e-15 |
| CPU32 | 6.43493943464e-7 | 2.85046305493e-6 |
| MPS32 | 5.79805159925e-7 | 2.85046305493e-6 |

CPU64 fixed forwardatol1e-11/gradatol1e-10/rtol1e-10; FP32 forwardatol4e-6/
gradatol4e-5/rtol4e-4. No predicate was adjusted after observing errors. All inputs,
populated caller gradient slots and global CPU/MPS RNG stayed unchanged;1178
source bindings preserve1175 predecessors. All four guards finished: whole
1.838612333s/RSS376274944B/MPS9125888B under600s/4GiB/6GiB/reserve8GiB/2MiBmetadata.
Independent metadata review .001758s/RSS21069824B, no numerical replay.

No model, optimizer, checkpoint, corpus, RNG draw, scientific role or quality
comparison ran. This is bounded functional correctness, not evidence of a gain,
cost advantage, actual-model stability or the cause of failed projection QUALITY001.
Native incumbent/holdout unchanged. All original commands are closed/no retry,
source/tolerance rescue or replay. No weights/deletions/retirement eligibility.

Next SOURCE-only action: actual-model-source-design001.md and
TOKENIZER_RELU_SQUARED_PREPARATION.md. Actual D256/V50257 model construction,
generated correctness/state/resume/scoring/cache/pilots and any quality campaign
need separate new source, correctness, resource and role gates. Old contrary FFN
results remain unpooled. No full-model/scientific-panel budget is qualified.
