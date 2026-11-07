# New literal final-normalization/readout sensitivity diagnostic

This is a bounded correctness diagnostic, not a scientific replacement panel or
an attempt to recover the missing slot07 scores. The original failed panel,
all1170 bound sources (1167 predecessors plus three new files), old tiny D<=8
proofs, checkpoint payloads and original guards remain unchanged. No actual GPT,
optimizer, trainable module, corpus, heldout data or quality seed is used.

Local evidence: failure-closure001/review.json and original parent-quality-07
guard; accepted slot06 RAW/H200 CPU-MPS CE differences2.8386214446562974e-5/
2.801248809358725e-5; frozen validation.py combined3e-5/target predicate and
missing reports. These do not identify an operator or trained activation range.
Source inspection establishes the actual final path:
research/final_ln_scale_v1/component.py calls F.layer_norm(x,(D,),gamma,beta,1e-5);
research/readout_projection_model_v1/model.py then uses Wz/4, W(Az)/4 or
W(tanh(Az))/4. Existing successful generated/full-model and real4/20 proofs are
contrary evidence to a universal backend failure; they do not cover all possible
learned feature/affine magnitudes. No old numerical proof is replayed.

Primary documentation actually retrieved this boundary:
[PyTorch2.14 LayerNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html),
formula/biased variance/affine/statistics paragraphs;
[F.layer_norm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.layer_norm.html),
signature and link to LayerNorm semantics;
[numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html),
opening and Batched computations or slice computations. Backend reduction order
may differ. General documentation is not a local error guarantee or evidence
that this norm caused the failed checkpoint. No CUDA TF32/half-precision recipe
is imported into this FP32 MPS scope.

Mechanistic question: can a mathematically irrelevant common feature offset
change the finite-precision final norm/readout, and how does fixed affine gain
propagate that discrepancy through these three readout equations? In exact
arithmetic LN(x+c)=LN(x), with biased variance and unchanged epsilon/affine.
Use exactly representable dyadic literals so input quantization cannot explain
a paired difference. Independent scalar math.fsum/sqrt/tanh/stable logsumexp
computes all five stages; all three centered/shifted scalar controls must agree
exactly. No tensor/matrix/autograd operation is used in the oracle.

Nine NEW fixture families, four rows/D256/V31, three readout arms, CPU64/CPU32/
MPS32:81 finite stage comparisons. Regular normalization and regular bypass
control distinguish a well-spread normalized path from direct readout; constant
input controls affine-only behavior. Three centered versus +1024 pairs use the
same narrow dyadic pattern with affine gain1/16/64. A,W,gamma,beta are fixed new
arithmetic formulas, not initializers, learned weights or claimed observed ranges.
The full 256x256 A is the largest tensor65536elements/524288B in CPU64. Old
D<=8/V<=31 guards are not touched or invoked. No random role/census is needed
because no random draws or model construction occur.

Prediction: if rounding sensitivity is exposed, shifted and centered FP32 results
differ despite exactly equal scalar results, potentially amplified by affine gain
and altered by tanh. If no difference appears, this controlled regime does not
exhibit the proposed sensitivity; it does not certify a trained transformer.
Report per-stage absolute deviations from scalar and translation-pair differences
without selecting a numerical acceptance tolerance from the observed errors.
Finite outputs, exact shape/input/RNG preservation, exact scalar pair equality,
source identity and guards are required. Numerical deviations are the measured
outcome, not a pass/fail gate. The old scientific3e-5 acceptance is unchanged.

Complete cost: all81comparisons and scalar oracle in ONE serial process, no
tensor files/serialization/retained weight outputs. Working30s/<512MiBRSS/
64MiBMPS; strict600s wholeMPS/120s eachCPUphase/4GiBRSS/6GiBMPS/fourthreads/
fallback0/fastmath0. Complete retained metadata cap4MiB plus8GiBreserve gives
floor8594128896B; extra2GiBtarget10741612544B, covered by42271039488B observed
at source admission. No storage block, cleanup or old-output capacity credit.
Shared worker lock is nonblocking. Exact original plan SHA observed before the
first command:93e147b5c47cea5843f06ffe610e5da9071cf1682bbb346feba0826242ec1b7c.
Failure closes this original command without retry/source/tolerance rescue.

Either outcome answers a new isolated forward-numerics question. It establishes
neither the cause of slot07 nor a validated numerical repair/quality gain. Any
future actual-model, gradient, optimizer, full-vocabulary, scorer or scientific
integration requires separate evidence and admission. Carry broad/interaction
allocation forward; this is diagnostic work, not numerical architecture exploration.
