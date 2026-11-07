# CE arithmetic error budget and static implementation evidence

This bounded question is complete. NEW ce-error-budget001.py consumes only saved
CPU32/MPS32 scalar CE and gradient-maximum reports plus selected installed source
headers. It imports no Torch/NumPy and calls no model, tensor, backend, old scalar
oracle, corpus, scoring, seed or RNG operation. Exact source/inputs/assumptions/
resources were pinned before execution: planSHA
0804c883b9a145540a03c59e5967bddcfb019be74aa512fd891adc2587eb9bdb.
Result/STOP complete_existing_scalar_source_analysis, .000865916s/RSS21381120B
under120sCPU4GiB/reserve8GiB/2MiBmetadata. One new saved source extends1223 future
bindings to1224; predecessors were untouched, not re-audited. All prior commands
and failed model/literal fixtures remain closed without replay/retry/rescue.

## Source facts and assumptions

Actually retrieved/read the official files at Torch commit
5c4886908584029761b579af026dcfb627c84070. Also read corresponding installed headers
under .venv/lib/python3.13/site-packages/torch/include/ATen/cpu/vec, recording exact
path/SHA refs for functional_base.h,vec.h,vec128/vec128.h,vec128_float_neon.h and
ATen/Config.h. Packaging guard wrappers change local line offsets; no byte-identity
claim against web-rendered source is made. Unavailable web paths are not evidence:
the initially tried vec256_float_neon.h path was wrong; the actual specialization
is vec128_float_neon.h and was retrieved successfully. Some other header web paths
returned cache misses; installed copies were read directly.

[functional_base.h](https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/cpu/vec/functional_base.h),
map_reduce_all lines228-249, maps each vector then adds it to a running vector,
handles the partial tail and finally combines lanes. This is lane-wise sequential
accumulation, not a balanced tree over the entire vocabulary. The aarch64 generic
float lane reducer uses two shuffle/add stages; its separate std::plus specialization
is distinct from the lambda reducer used by the log-softmax source.

[vec128_float_neon.h](https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/cpu/vec/vec128/vec128_float_neon.h),
class size/exp-log macros/operator+ sections, defines four FP32 lanes and FP32
addition. Ordinary exp/log can select SLEEF or scalar C++ math according to a build
macro. The separately named exp_u20/fexp_u20 helpers are not the ordinary exp call
used by the read log-softmax source. Existing fallback0/fastmath0 flags do not imply
FP64 accumulation, identical library algorithms or a universal numerical guarantee.
Installed Config.h does not resolve that SLEEF build macro by itself.

[Blanchard-Higham-Higham arXiv1909.03469v1](https://arxiv.org/pdf/1909.03469v1),
§3 eq3.3/Theorem3.2 and§4 eq4.6-4.10 were retrieved/read. Their forward-error analyses
include finite-precision summation, shifted inputs and transcendental evaluation;
shifting prevents overflow but is not a dimension-independent absolute-error
certificate. Their arithmetic assumptions are conditional and do not certify our
installed SLEEF binary or opaque MPSGraph implementation.

## Conditional addition budget, not a measured kernel error

Assume the four-lane path is used, FP32 round-to-nearest basic additions obey the
standard relative-error model, and there is no overflow, harmful underflow or
reassociation that changes the path. For50257 positive mapped terms, lane counts
are12565/12564/12564/12564. Initial lane loads do not add; the longest lane has12564
additions and its horizontal combination adds two. Withu=2^-24, k=12566 and
 gamma_k=k*u/(1-k*u)=6283/8382325≈0.0007495533757042348.
This is a worst-case allowance for addition of already mapped terms, not an
observed error and not a full CE bound. Even assuming exact mapped exponentials,
the associated possible log-normalizer allowance -log(1-gamma_k)≈
0.0007498344312886336 exceeds3e-5. It therefore cannot prove that threshold. It
also cannot prove a failure: worst-case bounds can be loose.

If mapped terms have relative error at mosteta, a conditional relative sum budget
isrho=(1+eta)(1+gamma_k)-1. Forrho<1, normalizer error is bounded by
-log(1-rho), before adding actual log evaluation and final subtraction errors.
Shifted-input subtraction, exp/log error, backward exp/subtraction, rounding mode,
actual dispatch and build details remain separate terms. No values are invented
for them. The scalar math log/expm1 outputs are approximate interpretations,
not directed-rounding formal certificates.

The saved result also records generic serial gamma50256≈.00300449 and a purely
hypothetical balanced-tree gamma16≈9.53675226e-7. The latter was not executed,
implemented, proposed as a replacement or authorized to rescue a gate. It merely
shows why arithmetic order matters to a theoretical bound.

## What saved scalar errors can and cannot explain

For the published correct_peak14 row, letDelta be savedCE minus savedoracleCE.
CPU32Delta=2.7842204301922746e-5. The scalar expm1(Delta)=2.7842591899690125e-5
is the relative difference between the exponentials of these published scalars.
It is NOT a recovered internal sum, hidden logit, probability or gradient entry.
If, hypothetically, all log probabilities differed only by the same normalizer
errorDelta and backward arithmetic were exact, each gradient error would be at
most abs(expm1(-Delta))=2.7841816711349686e-5. The recorded CPU gradient maximum
2.6695281533428705e-5 fits that hypothetical envelope. Compatibility is not
localization of sum versus exp/log or proof of a common-error model.

The same calculation for saved MPSDelta4.781338509840527e-8 gives an envelope
4.781338395534539e-8, below the recorded gradient maximum5.200531882909587e-8.
Thus the normalizer-only/exact-backward model does not cover all recorded errors;
additional arithmetic terms cannot be dropped. This is not evidence of an MPS bug
or a contradiction of the passed fixed predicates. Full gradient coordinates were
not saved, and none were reconstructed. These calculations do not explain the
closed trained-native281-target failure or missing old projection scores.

## New static binary evidence: compiled function, not run tracing

One read-only otool/nm inspection and one bounded disassembly were saved in
ce-kernel-static001/{inspection,float-function-disassembly,completed}.json. They
read the installed386016816B ARM64 libtorch_cpu.dylib, SHA
886a21732227bea91a64dcc355bdea98808fab5014a84891de2fe3a081744079, and changed or
copied no binary. Original inspector tool arguments/argv are preserved; there is
no fabricated filesystem-saved inspector source hash. Neither inspection called
Torch/dlopen or executed a tensor/math kernel. The scalar-analysis saved script
and the unsaved read-only tool orchestration are explicitly different evidence.

The symbol table locates DEFAULT serial_vec_log_softmax_lastdim_range<float> at
0x2338288; disassembly was bounded to [0x2338288,0x2338a60),508lines. It contains
four-lane fadd.4s, horizontal adds at0x2338648/0x2338650, and direct calls to
_Sleef_expf4_u10 at0x4c6bcc4 and_Sleef_logf4_u10 at0x4c6b8ec. These targets agree
with saved symbol addresses. Thus the actual installed float function contains
SLEEF calls and FP32 lane arithmetic; source-macro ambiguity for this function is
resolved. Unresolved stub annotations are not used as evidence. Presence of this
function does not retroactively profile which function was selected by a stopped
run, expose intermediate values or isolate its measured error.

[Shibata-Petrogalli arXiv2001.09258v1](https://arxiv.org/pdf/2001.09258v1),
§1/§5.5-5.6/§6.1-6.3 were actually retrieved/read. The authors describe1-ULP library
variants and MPFR-based empirical accuracy testing rather than formal verification.
A high-accuracy elementary function still leaves the surrounding FP32 sum and
backward arithmetic. The paper's library/version/platform experiments do not
validate every argument in our installed binary. No SLEEF test was run locally.

## Decision and next bounded action

CLOSED_COMPLETE existing-data/source analysis; all previous numerical gates retain
their original outcomes and exact predicates. No source, dependency, backend,
precision, initializer, LR, EMA, horizon, normalization or tolerance is changed;
no old tensors/oracles/models/scoring are replayed. This does not qualify a new
model/quality panel or waive the failed native full-horizon prerequisite.

The next concrete independent work is one evidence-grounded applicability decision:
map the frozen full-horizon worker/acceptance/serving requirements to the accepted
short prerequisites and the two distinct numerical failures. Identify which claims
remain unsupported and whether a new discriminating question qualifies without
replaying a closed fixture or weakening a predicate. Read the actual relevant
sources/receipts and current roadmap; do not rerun source/weight audits or add
ledger/optimizer instrumentation. If no new experiment qualifies, record a precise
new-evidence wake condition and carry exploration allocation forward instead of
creating another arbitrary numerical probe. This decision is not a new launch gate.

All1224sources/old selected/fresh/screen/prerequisite/failed/partial artifacts remain
protected; no new weights/copies/deletions. ~35.11GBfree;metadata2MiB+reserve8GiB+
extra2GiBtarget10739515392B covered, no storage blocker or cleanup needed. Native
incumbent/holdout unchanged, no numericalworker active. Analysis is not numerical
quality exploration and does not spend the carried broad/interaction allocation.
