# Composer FP32 optimizer component qualification

This is the distinct arithmetic hypothesis supported by
`GENERAL_RSI_MOMENTUM_PRECISION_RESULTS.md`. It does not retry or reinterpret
the closed reuse v1 qualification. Local measured evidence, primary-paper
references, contrary findings and prospective full scientific costs remain in
that report. No full-model or quality panel is admitted here.

Implement only the nine composer Muon matrix shapes. Preserve momentum0.95,
Nesterov, five Newton-Schulz steps, coefficients3.4445/-4.775/2.0315,
normalization floor1e-7, match-rms-AdamW shape scaling and zero weight decay.
All momentum, Nesterov, normalization and polynomial operations use FP32.
Other model/optimizer policies are outside this component. It has a separate
state identity and validates all gradients before state or parameter mutation.

Freeze private parameter seed62111012001 and gradient seed62111012002. Use
all nine shapes with six gradients: zero, random rank-one, dense, sign-reversed
dense, diagonal-sparse and half rank-one plus half dense. Nonzero constituent
gradients have Frobenius norm0.01 before mixing. Parameters start uniformly
in[-0.05,0.05]. The six group learning rates are0.00125,0.00125,0.00125,
0.000625,0.00025,0.000125. No data or saved failed gradients are reused.

An independent CPU FP64 reference computes momentum/Nesterov and an expanded
finite polynomial, not the exact polar factor. At every step and matrix require:

- Parameter error <=2e-6 +2e-4*abs(reference).
- Momentum error <=2e-9 +2e-5*abs(reference).
- Orthogonalized Nesterov-direction Frobenius relative error <=1e-4, with
  absolute norm error for a zero reference.

These are fixed engineering tolerances, not quality criteria. Stop at first
failure and retain the failed matrix's candidate/reference parameter and momentum
tensors. Do not tune around a negative result.

After all54 comparisons pass, require bitwise equality of a full independent
rerun and a midpoint serialization/restore/replay. The snapshot includes all
parameters, momentum, step count, learning rates and identity. Inject NaN into
the last gradient and into the last saved momentum tensor, separately; reject
both before any parameter/state mutation. Verify private inputs and global
CPU/MPS RNG states remain unchanged. Retain first final state, midpoint, FP64
reference, comparison hashes/errors and all failures. These checks qualify only
this component and its explicit snapshot API.

One local numerical worker under the nonblocking shared research lock. Fixed
240s cumulative, RSS4GiB, MPS driver6GiB,64MiB complete retained-output cap and
10GiB reserve. Exact forecast:719,775 matrix elements;5,758,200 bytes per FP32
parameter/momentum pair and11,516,400 for FP64. Two midpoint pairs, two maximal
failure matrix pairs, one final pair, one reference pair, one reference-sized
serialization overlap and4MiB metadata total57,731,704 bytes. This covers at
most two attempts, where a second is allowed only for a demonstrated harness or
resource defect. No numerical retry is allowed. A subsequent full-model fixture
needs independent admission, unchanged-body controls, complete restart and
whole-update timing before any quality experiment.
