# Supplied-logit proximity objective SOURCE prepared; static gate closed complete

Original top-objective-source001 pid74857 exited0. CPU guard finished
.002676833071745932s/RSS22855680B under120s/4GiB/reserve8GiB/2MiB metadata.
Original plan SHAea6c36c4dd5a590a6092a9d7af249bf1c98317e78607ac7e82a4798f2a6cbaad,
reviewer42e133dc7531514e26d4ae2df7110bdc60ae7968cc8ea393886a8397a32c25d2.
All sources/plans/review/guard/STOP are closed/no replay/retry/source rescue.
This review is AST and literal-input metadata only: no Torch, objective, accepted
target helper, scalar oracle, backend, RNG, model, corpus, scoring or tensor calls.

New research/top_proximity_objective_v1.py3657B/SHA
d425365fad2836f267f641632779213406c8802403d0700c7db7db216efe5de5 has one public
sparse_proximity_ce(logits,targets). Targets are immutable flattened B*T rows of
unique ordinary IDs and positive finite Python-float masses, fsum-normalized
within1e-14 before tensor staging. All target rows are preflighted before staging
indices/masses or constructing the differentiable loss. Inputs must be plain
strided CPU64/CPU32/MPS32 finite logits; autocast is forbidden. Source bounds
B<=8/T<=128/V<=50257 do not qualify those full dimensions for execution.

The module allocates explicit int64 same-device coordinates and casts masses
ONCE to logits.dtype/device. It does not renormalize represented weights. It
calls ordinary F.log_softmax(dim=-1) without a dtype override, gathers only sparse
support and divides the weighted sum by all B*T input positions. It does not
backward, assign/clear .grad, choose/mix primary objectives, construct models,
mask missing targets or import scorer/worker/state/config adapters. Caller target
provenance remains separate; supplied logits do not establish model causality.

Read the installed functional.py log_softmax wrapper2293–2324 and local typing
declarations without importing Torch. With dtype=None it delegates to the input
tensor's log_softmax. Its exact file hash is in the new source plan; no binary,
old weight, old failed source/fixture or installed CE-kernel audit was repeated.
No native backend, precision, normalizer, dependency or incumbent arithmetic was
changed. This source is a separate auxiliary consumer, not physical validation.

For represented weights w with real sum m, the mathematical gradient is
(m*softmax(z)-w)/(B*T). In particular, independently rounded FP32 target masses
need not sum exactly1. The future oracle must use those declared represented
weights, without candidate intermediates. Positive Python weights can underflow
when cast; the implementation declares ordinary dtype conversion, no hidden
mass correction. Current prospective literals have support<=4 and no extreme
underflow claim. Raw auxiliary CE includes target entropy and is not quality.

top-objective-literal-source-plan001.json contains10 NEW explicit fixture designs
and41 negative-family IDs/51 unique design IDs. Static validation checked their
shapes, finite logits, ID support, positive Python masses and row sums. Flat,
signed, sharp-correct/sharp-wrong, distinct mass patterns, batch/time, single-type
and W1 genuine-CE controls, maximal B2T8V31 and noncontiguous/rounding cases are
prepared. No loss or gradient values were computed. The41 families are NOT41
executed rejections. Concrete backend-specific payloads, harness, oracle and
actual backend-output counts remain unimplemented/unfrozen.

Prospective unchanged gate anchors: CPU64 lossatol1e-11/gradatol1e-10/rtol1e-10;
FP32 loss+gradatol4e-6/rtol4e-5; same-input CPU/MPS scalar lossatol3e-5/rtol0.
The full future functional resource envelope remains forecast only:30s/1GiB RSS/
256MiB MPS, hard120sCPU/600sMPS/4GiB RSS/6GiB MPS/reserve8GiB/2MiB metadata.
The implemented harness/negative tensors/RNG and input copies must fit a new
complete pinned plan before any tensor/oracle execution. No seed roles/census.

Source gate admits only future functional-proof SOURCE work, not a runtime,
model, full-vocabulary distribution, real-data worker, scientific role/budget/
selection/fresh confirmation or adoption. Target-math proof remains separately
closed/accepted; all original future-model/index/CE failures remain closed and
unqualified, never recaptured or promoted by these static checks. No weights,
copies/deletions/retirement. Future1253bindings preserve1251 plusmodule/reviewer;
closedmodelproof1236/scientificruntime1211 unchanged. Next exact source task is
top-objective-proof-source-handoff001.md. Reading/preparation is not numerical
quality exploration; native and holdout remain unchanged.
