# Saved-momentum analysis supports a separate precision hypothesis

The controlled matrix diagnostic passed its predeclared hypothesis gate. FP32
reduced paired finite-polynomial output differences by119–173 times for the three
matrices that failed the previous first-update gate. This is evidence about
arithmetic on saved arrays, not a model-quality result or a rescue of reuse v1.

## Local measurements

`general-rsi-momentum-spectrum-001/` includes all nine composer Muon matrices,
not only the failed three. Of719,775 saved momentum entries,568,011 differ in
FP32 and168 differ after CPU BF16 conversion. The highway, transform and
projection matrices need respectively2,2,1 singular directions for99% of their
squared singular-value energy. Their stable ranks are1.013,1.014,1.007. These
are energy concentration measures, not claims of exact rank1 or2.

The new diagnostic evaluates the same five-step finite polynomial, coefficients
and normalization on each saved momentum array. It compares MPS FP32 with MPS
BF16, using CPU FP64 as a finite-polynomial reference. It does not reconstruct
the exact Nesterov input or perform an optimizer update.

| Matrix | FP64 pair relative difference | MPS FP32 | MPS BF16 | BF16/FP32 ratio |
| --- | ---: | ---: | ---: | ---: |
| highway | 0.00006059 | 0.00006220 | 0.01075496 | 172.91 |
| transform | 0.00005845 | 0.00005978 | 0.01019930 | 170.62 |
| projection | 0.00006819 | 0.00006973 | 0.00832626 | 119.41 |

All54 arm/matrix/precision cases were bitwise repeatable over three evaluations.
All18 FP32-versus-FP64 relative errors were below1e-4; maximum was3.112e-5.
Every first output is retained, along with source bindings and guards. The CPU
spectrum pass took0.41s and the mixed-device precision pass1.56s. No training,
validation, generation or test evaluation occurred.

Contrary detail: BF16 maps the first four filter pairs to identical outputs,
whereas FP32 retains their small nonzero differences. FP32 does not minimize
every pair discrepancy. Also, the FP64 map amplifies relative differences in
the three large matrices from roughly6e-7 in momentum to roughly6e-5 in output.
Thus finite-polynomial sensitivity remains even without BF16 arithmetic.
Individual operation timings include initialization effects and are not a
qualified performance benchmark.

## Primary evidence and applicability

Keller Jordan's [Muon design note](https://kellerjordan.github.io/posts/muon/),
8 December2024, Definition, Eliminating alternatives, Tuning coefficients and
Runtime analysis, motivates finite polynomial orthogonalization and reports
stable BF16 execution. It also treats imperfect orthogonalization as useful
and argues against expensive SVD in training. These findings oppose assuming
that higher precision automatically improves quality. Its hardware and batches
differ from our Mac,1,024-token updates and byte composer.

Li and Tsuchiya, [arXiv:2608.26288v1](https://arxiv.org/html/2608.26288v1),
sections2.2,5.1/Lemma5.2 and6, analyze the stability/approximation tradeoff of
finite spectral maps. Their theorem uses different coefficients, fixed
normalization and a different optimization framework. It is not a bound on
our BF16 rounding or local training. It supports retaining the same finite
polynomial as the reference here, rather than assuming an exact polar/SVD
update would be the correct training target. A separate OpenReview low-bit
paper could not be retrieved beyond its browser challenge and is not relied on.

## Next hypothesis and cost boundary

Inference: a separately identified variant that uses FP32 orthogonalization
for composer Muon matrices may reduce sensitivity while preserving enough of
reuse's measured component saving. It changes optimizer arithmetic and must
be treated as a new recipe. Reuse v1 remains closed under its failed gate.

Next executable test: implement a standalone **actual Nesterov update and
optimizer-state** fixture for composer matrices, with explicit FP32 finite
polynomial arithmetic. Use FP64 finite-polynomial references, deterministic
repeats, fixed gradients and state serialization checks before any full-model
integration. Keep the five iterations, coefficients, momentum, scaling and
non-composer policy fixed. Freeze tolerances before outputs. If it fails, stop
the distinct variant; if it passes, qualify complete full-model restart and
whole-update cost. Neither case changes the old failure or scientific decisions.

Prospective bounds for that component fixture:240s,64MiB complete outputs,
RSS4GiB, MPS6GiB,10GiB reserve, one nonblocking locked worker. A subsequent
full-model fixture needs its own exact inventory and at most900s; the previous
5GiB conservative complete-state allowance is a planning ceiling, not admission.

If engineering passes, a discriminating fresh equal-time quality design would
include native, original byte, byte with composer FP32, and byte with composer
FP32 plus reuse. Two1,800s LR screens and three7,200s confirmations per arm total
100,800 training seconds (28h); a prospective40h tranche must include scoring,
generation, failures, initialization and I/O. Forty endpoint weights project
3,103,224,660 bytes from existing native/byte payload sizes. Two512MiB restart
slots,256MiB metadata,5GiB failed writes and10GiB reserve bring the requirement
to20,551,529,300 free bytes **before additional prerequisite outputs**. Include
those actual retained outputs and serialization overlap at admission. The full
worst-case prerequisite allowances plus that panel exceed current free space,
so this note admits no quality panel. Recompute from measured prerequisites and
use exact authorized recovery if an actual admission deficit remains. Do not
shrink confirmations or drop controls to fit capacity.

This prioritizes a bounded explanation of an observed arithmetic failure over
an arbitrary optimizer or model-size change. A successful engineering fixture
still would not establish a validation, generation or transfer improvement.
