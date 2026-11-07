# Source-only candidate after incomplete forward diagnostic

## Evidence and limits

forward-sensitivity001 exited at its thread/MPS environment predicate BEFORE
attempt/guard/tensor construction. Its entry-failure and STOP remain final.
Host preflight confirmed Torch2.14.1/fourthreads/MPSavailable. Distinct002 used
byte-identical numerical source with a new plan/output and explicit host access.
The scalar oracle and all27CPU64+27CPU32 rows completed. The first MPS row stopped
at reporting .double().mean().cpu(): the requested FP64 cast happened on MPS.
MPS has no FP64 support. This is a source/reporting defect, not an MPS accuracy
result. No MPS row/result/finalRNG assertion was published;002 is CLOSED_FAILED,
and neither original command is edited, retried or reconstructed.

The existing54CPU rows remain legitimate partial measurements. Regular CPU32
normalized-feature error versus scalar was2.3475546928608537e-7. The narrow
+1024 fixture increased it to8.15836623855537e-4 at affine gain1 and
5.221354392675437e-2 at gain64. Across saved CPU32 cases, maxlogit error
6.800607095021505e-3/maxper-targetNLLerror1.1139385255489742e-3. Scalar centered/
shifted outputs were exactly equal; literals were checked exactly representable
in FP32. All CPU64 errors were much smaller (maxfeature5.553609128128301e-9,
maxlogit3.6918113011097375e-10/maxNLL2.9095392761746552e-11).
These are controlled functional stress measurements at V31 and synthetic affine
gains, not observed trained activations, MPS comparisons, native validation CE,
a tanh quality result or the cause of the original slot07 failure.

## Primary sources and algebra

Actually retrieved and read Ba et al.,
[Layer Normalization, arXiv1607.06450v1](https://arxiv.org/html/1607.06450v1),
section3/equation3 and section5.1/Table1/invariance discussion. It specifies
per-example mean/variance and adaptive gains/biases. Its RNN learning results
are not a numerical error guarantee for this FP32 MPS transformer. Also read
PyTorch2.14 LayerNorm formula/biased variance and numerical-accuracy opening/
batched-or-slice sections. No reported paper gain is used as a local quality claim.

For row x, define u=x-x_0*1, mu_u=mu_x-x_0 and Var(u)=Var(x). Thus
LN_epsilon(u;gamma,beta)=LN_epsilon(x;gamma,beta) in real arithmetic, including
unchanged epsilon1e-5 and learned affine parameters. This is an algebraic kernel
formulation, not an epsilon/normalizer-type/LR/initialization change.
The derivative through the anchor must remain attached: d_u/d_x=I-1e_0^T.
Since the exact normalization Jacobian annihilates1, J_LN(I-1e_0^T)=J_LN.
No parameter, extra state, RNG or optimizer grouping is introduced.

This is not invariance to arbitrary dataset re-centering in the paper's Table1;
it is subtraction of a scalar common to all coordinates of one feature row,
derived directly from the mean/variance equations. Floating point can change both
the current kernel and anchored formulation. Subtraction may overflow for opposite
extremes; the source explicitly rejects nonfinite centered values with no clamp,
fallback or silent cast. No universal accuracy improvement is claimed.

## New source and next discriminating gate

research/anchored_layer_norm_v1.py is SOURCE ONLY. It uses functional
F.layer_norm(x-x[:, :1],(256,),gamma,beta,1e-5), strict N<=16/D256,
CPU64/CPU32/MPS32, finite same-device/dtype tensors/no autocast. It creates no
nn.Module or actual GPT and is not imported by any frozen model. Source-only
AST/algebra/config review does not establish runtime correctness.

Before any execution, implement a NEW independent scalar forward+analytic
backward oracle, NEW exactly representable literals and complete source/plan/
resource gates. Compare original functionalLN and anchor formulation to the same
oracle, including x/gamma/beta gradients, zero-variance/regular/narrow-offset,
signed-affine/permuted-coordinate/overflow-rejection cases. Use distinct literal
inputs, never old001/002cases or trained checkpoint values. Preserve caller input/
grad slots/globalCPU-MPSRNG. CPU transfer MUST precede any FP64 metadata conversion;
do not repair or replay the closed failed harness. Future actual-model/optimizer/
resume/scorer/cache/worker/quality integration remains unadmitted.

Falsifiable prediction: anchor subtraction reduces error for NEW narrow-spread,
large-common-offset literals while preserving scalar forward/analytic gradients.
A worse result or wrong gradients rejects the formulation for integration. A pass
only supports the tested functional regime; it cannot qualify the old panel or
prove that trained models need it. Native incumbent and holdout stay unchanged.
The current/source/metadata work counts as correctness work, not broad exploration.

Prospective next functional scope only: working30s/<512MiBRSS/<64MiBMPS,
strict120sCPUphase/600swhole/4GiBRSS/6GiBMPS/fourthreads/fallback0/fastmath0/
8GiBreserve/4MiBmetadata/no tensor files or serialization. Floor8594128896B/
extra2GiBtarget10741612544B fits current~42.28GBfree. Exactimplemented source,
all cases and tolerances must be reviewed/pinned before a new execution. No new
scientific role, quality budget or training launch is assigned here.
