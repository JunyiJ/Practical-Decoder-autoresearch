# New full-vocabulary literal CE probe: closed numerical counterexample

The NEW original literal-vocabulary-ce001 command,planSHA
d5f9b82bfe006ffd3d8a6ff8efb9f8b6184fe65490f100b09ad591141c15cd24,
worker65628/session74887 EXITED1. STOPclosed_failed/no_retry. Two new sources
native_vocabulary_ce_v1.py and literal-vocabulary-ce001.py preserve1220 bindings
and add1222. They were pinned/reviewed before first execution. The separate saved
metadata-review source adds1223; all are immutable. No models, optimizer, training,
corpus/holdout, checkpoint files, quality roles, seeding or RNG draws. No original
full-state or earlier primitive/forward fixtures were invoked, edited or retried.

The bounded six-row V50257 matrix contains flat, dyadic-periodic, correct-peak14,
wrong-peak20/target-12, and periodic+/-768 rows. A separate known dyadic perturbation
and single-periodic-row call give8 reported row measurements per backend. All
inputs are exact dyadic literals in CPU64/CPU32/MPS32. The scalar math.fsum shifted
LSE and analytic softmax-minus-onehot oracle does not call Torch. Measurements were
durable BEFORE the fixed predicate rejection. Raw tensors and individual gradient
coordinates were not saved and must not be reconstructed by replay.

All SAME-INPUT CPU/MPS CE errors passed fixedatol3e-5/rtol0, maximum
2.779439091682434e-5 on correct_peak14. The perturbation difference was also below
3e-5. Every backend passed the declared common-shift, batched-versus-slice and
known-perturbation predicates. Thus this finite probe did NOT observe an isolated
CE forward violation of3e-5. It does not prove that bound for all logits/shapes or
the closed trained states. In particular N<=6 is not the model scorer's128-target
window shape, and no trained logit ranges are known.

The complete gate nevertheless FAILED the prespecified elementwise gradient
predicates. The sole oracle-failing row was CPU32 correct_peak14,target32003:
CE0.041360486298799515 versus oracle0.04133264409449759, forwarderror
2.7842204301922746e-5 (passes3e-5), maximumgradienterror2.6695281533428705e-5
with originalatol4e-6/rtol4e-5. CPU/MPS gradient predicate also FALSE, maxdifference
2.664327621459961e-5. Do not substitute a max-only gradient test: the executed
comparison was elementwise atol+rtol*abs(reference); full coordinates are unavailable.
No tolerance, oracle, input margin, backend or source is changed to rescue this test.

Across all8 reported rows, CPU64 forward/gradient maxima2.814415367424772e-13 /
2.7000623958883807e-13; MPS32 maxima9.400133027526181e-7 /
7.47033193082558e-8; CPU32 maxima2.7842204301922746e-5 /
2.6695281533428705e-5. AllCPU64/MPS32 oracle predicates passed. This only compares
the declared cases, not general backend superiority. It also means CPU FP32 is not
an exact numerical ground truth for these literals. The native incumbent and its
training/scorer arithmetic remain unchanged.

All73 transactional rejections passed (CPU64=24,CPU32=25,MPS32=24), including
kind/shape/layout/dtype/device/autocast/targets/nonfinite and finiteFP32 CEoverflow.
Actual passed inputs, populated caller gradient slots and globalCPU/MPS RNG were
preserved. All1222 pinned source bindings unchanged. Whole .735266417s/
RSS478543872B/MPS44843008B under600s/4GiB6GiB/reserve8GiB reportsfixture_exception
only because the final numerical predicate assert failed. ThreeCPUguards finished:
oracle.038588s/CPU64.285356s/CPU32.028562s. No storage blocker or weights/deletions.

Independent original-metadata review in literal-vocabulary-ce001-review001 passed
.054293333s/RSS21790720B,120sCPU4GiB/reserve8GiB/2MiBmetadata. It checked1223source
bindings,23 original metadata hashes, reported gradient flags, exact failed-predicate
set, finite forward errors, CPU/MPS CE arithmetic,73rejections and allguards. It
DID NOT recompute full gradient predicates from missing tensors or rerun the oracle,
backend, model, census or scoring. Never repeat this closed audit for unchanged state.

## Pinned primary-source trace, with limits

Retrieved/read official PyTorch source at runtime-reported commit
5c4886908584029761b579af026dcfb627c84070. GitHub HTML retrieval returned cache misses;
raw.githubusercontent.com retrieval succeeded. These are source facts, not profiling
of the installed binary's selected vector specialization or Apple's proprietary graph
compiler. No build/installation/backend was changed.

[LossNLL.cpp, cross_entropy_loss_symint, lines587-619](https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/LossNLL.cpp)
composes integer-target, zero-smoothing CE from log_softmax at input scalar dtype and
NLL. Its unreduced two-dimensional NLL selects the negative target log probability.
Our wrapper used that target kind and default zero smoothing.

[SoftMaxKernel.cpp, lines30-54/151-177/931-964](https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/SoftMaxKernel.cpp)
connects the last-dimension CPU path to a vectorized implementation and its backward
path uses the stored log-softmax output. Dispatch depends on scalar type.

[LogSoftmaxKernelImpl.h, lines15-91](https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/LogSoftmaxKernelImpl.h)
stores maxima and sum intermediates as scalar_t and uses map_reduce_all for shifted
exponentials, then log and ordered input-minus-max-minus-logsum. This source path
does not automatically promote FP32 input summation to FP64. The order explicitly
avoids adding the large maximum to a small logsum before subtracting it.

[MPS Activation.mm, lines26-105](https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/operations/Activation.mm)
builds a graph of max reduction, subtract, exp, sum reduction, log and subtract;
backward also uses exponentiated stored output. Source does not expose Apple's
internal reduction order/accumulation precision. SoftMax.mm was inspected but its
plain-softmax implementation is not used as evidence for log-softmax internals.

Inference: a precision/reduction explanation is consistent with these source paths
and the literal CPU32 deviation, but we have not isolated sum, exp, log or the actual
compiled CPU vector path. We cannot attribute the earlier trained-model CE failure
to this gradient counterexample, nor claim upstream layers are at fault because
these sampled forward cases pass. [The prior numerical-analysis boundary](post-full-state-failure-evidence001.md)
records arXiv1909.03469v1§2-4 and its applicability limits; no new paper gain or MPS
root-cause claim is made. No new model or quality campaign is qualified.

## Next bounded independent work

Complete SOURCE/LITERATURE analysis of the exact pinned CPU map_reduce_all and
ARM vector exp/log/reduction definitions, including source-level precision and
summation order and what is NOT guaranteed. Derive an error budget for log-normalizer
and CE/gradient sensitivity using saved scalar measurements only; no recreated
logits/gradient coordinates/old oracle calls, no tensor/model/backend execution.
Distinguish source-level possibility from actual binary dispatch and actual error
localization. Use the existing arithmetic-paper assumptions, do not infer a universal
3e-5 guarantee or retune a failed gate. No additional numerical fixture, model,
source replacement or scientific panel is admitted. This is a concrete existing-data
question; it does not block unrelated source/literature work or consume exploration
quality allocation.

All1223sources and all prior RAW/H200/selected/fresh/screens/prerequisite/failed/
partial evidence remain protected; no tensor output or deletion here. ~35.12GBfree,
metadata-only remaining work2MiB+reserve8GiB+extra2GiBtarget10739515392B fits; old
fullquality25637683200B forecast is HISTORICAL/unfrozen. Native incumbent/holdout
unchanged, no numericalworker active. Carry broad/interaction allocation forward.
