# Block normalization prerequisites v1

This directory provides a prospective normalizer specification, CPU reference
checks and a separately namespaced unmodified runtime baseline. A named candidate model policy now has construction/optimizer-coverage tests.
Raw/H200 signature serialization now passes bounded synthetic tests. Synthetic CPU populated-optimizer and interrupted replay checks now pass.
Independent gradient/cache numerical, MPS and tokenizer-bound trainer/controller
integration remain unvalidated; no quality training is ready. Frozen original/recovery source and evidence are unchanged.

`spec.py` binds kind, width256, FP32, epsilon2^-23, affine/bias and pre-attention/
pre-FFN placement. The control uses exact native RMSNorm with `eps=None`; the
candidate uses native LayerNorm with explicit2^-23 and learned bias. The factory
initializes constants without RNG draws. Runtime validation rejects wrong module
type, shape, epsilon, affine/bias, frozen gains, extra buffers, mixed/shared/aliased
normalizers, non-FP32 parameters/input and autocast. Eight distinct normalizers
are required. Signature encoding rejects mismatches even at matching widths.
This is signature metadata testing, not a torch checkpoint load/resume test.

The test oracle computes separate FP64 equations and analytic input/gain/bias
derivatives; it does not differentiate through native normalization. Cases cover
random/shifted, zero/constant, near-zero and shifted near-constant inputs. Actual
execution is native FP32 on CPU. Six control cases require bit-exact output,
input gradients and gain gradients against the incumbent native RMS path.

Tolerances were fixed before the first run and did not change afterward. Ordinary
output/input-gradient/gain-gradient atol are3e-5/3e-3/3e-4, rtol2e-4; shifted
near-constant LayerNorm uses8e-4/0.2/3e-3, rtol2e-4 because FP32 mean rounding is
amplified by the reciprocal scale. Bias gradients use rtol2e-5/atol3e-5. These are
bounded agreement checks, not bit-exact analytic equivalence or training-stability
claims. Observed maxima are recorded per case in the review. Near-constant
LayerNorm output/input-gradient/gain errors were0.0002103/0.01536/0.0009631;
ordinary random/shifted output errors stayed below6.26e-7.

Attempt001 passed all66 tests but FAILED its enclosing audit: the final metadata
hash loop passed a string to file_digest, which requires Path. Its log, failed
guard, report and full source snapshot are preserved. Attempt002 changed only
that wrapper conversion, reran under a new evidence prefix, passed66 tests and
verified original source, all13 frozen recovery modules, closed-root inventories
and noncheckpoint hashes unchanged. Spec/test code and tolerances were identical.
Guard002 finished0.388s, peakRSS285,900,800bytes, four CPU threads,8GiB reserve.
No corpus, checkpoint, training, MPS or cleanup operation was performed.

Do not rerun/overwrite either evidence attempt. Next map full-model/optimizer/
checkpoint/cache integration and source applicability without editing frozen
runtime artifacts or bypassing RMS-only validators. The prospective panel's
2.186GB storage deficit remains a training gate; do not poll unchanged capacity
or narrow the design merely to fit. See TOKENIZER_BLOCK_NORM_PREPARATION.md.


## Runtime integration baseline

See [INTEGRATION_CONTRACT.md](INTEGRATION_CONTRACT.md) for the import map, source
applicability, prospective edit allowlist and required full-model tests. The43
module copy under `runtime` is byte-identical to its original source. It passed
five guarded tests: import isolation, exact initial state/RNG/logits/loss and
old-new cached state/output, optimizer group equivalence and RMS-only rejection.
No candidate policy or optimizer update was exercised. Attempt001 stopped before
tests on sandbox PermissionError; attempt002 passed unchanged sources using
authorized process access. All snapshots/reports remain immutable. Do not rerun
those evidence attempts or invoke the copied Hydra CLI.

The baseline commit's staged whitespace check reported only six inherited lines: the exact baseline
and two immutable snapshots each retain original `attention/mha.py:106` and
`attention/mla.py:66` whitespace. These bytes are intentionally unchanged to
preserve origin hashes. No new authored whitespace error was found.


## Named policy construction

The new `runtime/models/block_norm.py` and three explicit model-file changes
passed40 tests (attempt001), then41 (attempt002 adding direct named-RMS versus
original equivalence). The runtime code was identical across attempts. Both
source snapshots are immutable. The current runtime has diverged from the
unmodified baseline only as recorded in `model-policy-delta-001.json`; historical
baseline evidence applies to its own saved sources, not silently to later edits.

New RMS control initial weights/RNG/full logits/loss and old-new cached execution
are bit-exact. LayerNorm adds2,048 parameters, preserving shared initial weights
and RNG, with all eight new biases uniquely grouped in AdamW at base LR. Config,
constructed module, epsilon/dtype/autocast and same-shape metadata rejection
checks passed. These are not gradient, serialized resume or H200 export tests.

Next bind actual normalization signatures in new raw/H200 checkpoint producers
and strict loaders, test serialized transport/EMA/resume, then complete numerical
cache/gradient and MPS checks. No old source or recovery runtime may change.
The original standalone spec remains unchanged. Training capacity remains gated.

The two model-policy source snapshots likewise preserve the two inherited
attention whitespace lines each; source hashes take priority over reformatting
immutable captures. Authored model changes have no whitespace-check errors.


## Serialized signature transport

The checkpoint writer/loader and H200 exporter now bind the dedicated constructed
normalization signature and validate exact typed metadata, saved configuration
and tensor inventory/shape/dtype/finite values.26 CPU checks passed on fresh
synthetic raw/H200 files for both arms. H200 gains/biases match a separate FP64
two-snapshot oracle; export leaves parent bytes and RNG unchanged. All four
fixture files (684,354,160bytes) are retained under an explicit size/SHA manifest.
No trained checkpoints or corpus were accessed. Source delta and exact tested
snapshot are recorded under checkpoint-transport-prerequisite-001.

This supersedes the prior pending-transport note. Empty optimizer group transport
and final EMA accumulator restoration passed; populated optimizer-state and
interrupted trajectory/loader RNG replay remain pending. Tighten new-policy raw
resume requirements and test those before declaring resume ready. No candidate
cache/gradient/MPS or quality claim. No evidence attempt may be overwritten.


## Populated resume replay

30 CPU checks now establish bit-exact batch1/context2 primitive replay for both
arms over three synthetic updates with an interruption after the first. Model,
Muon/AdamW slots, schedule, EMA, losses, samples and CPU/loader RNG agree exactly.
The new copied `utils/block_norm_resume.py` checks complete raw state before
weight loading; malformed groups, rates, slots, counters and RNG/EMA are rejected.
All source deltas and test snapshots are retained. Two raw fixture files and ID
files total890,211,626bytes; no cleanup or original corpus access occurred.

The accompanying three-test read-only H200 regression remains exact on prior
synthetic fixtures and preserves old-policy resume behavior. These results
supersede the earlier populated-resume pending note, but do not prove production
trainer/tokenizer integration, independent gradients, candidate cache parity or
MPS stability. Next complete numerical/cache checks without new binary fixtures
where feasible, then resource and controller readiness. Training remains gated.
