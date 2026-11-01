# Tokenizer SwiGLU-product normalization preparation

Only prerequisite work is selected. No scientific seeds, pilot or quality panel
are frozen and no new scientific training is authorized by this document alone.

The adopted D256 MLA recipe retains multiplier0.5/SwiGLU672/SiLU/block RMSNorm/
H200 after the closed score campaign. Compare its unchanged SwiGLU product with
affine RMSNorm applied per token after `silu(gate(x))*up(x)` and before the down
projection. Proposed epsilon1e-5, gain initialized to1, no bias; four672-wide
gain vectors add2,688 parameters. This changes hidden-product amplitude and
learned channel scaling, not block centering or gating structure. Do not infer
an optimization mechanism from a CE difference. New gain optimizer treatment
must be explicit; existing parameter groups and shared initialization stay paired.

The character study's width256 RMS mean gain0.003438 was small, mixed across
seeds and insufficient for adoption. It is a weak prior, never pooled with
subword CE. The new tokenizer/MLA transfer question remains untested. QK norm
had unfavorable width256 character evidence; local convolution requires new
cache-history integration; MoE requires a new complete resource projection.
A bounded100-update warmup/peak-LR question was also reviewed and deferred:
character width256 mean gain0.000805 was mixed, and current curves establish
no instability mechanism. Both architecture and HP remain eligible later.

First check component arithmetic, all gradients and per-token causality on
generated tensors. Then add explicit full-model/optimizer/checkpoint/resume/
H200 and full-versus-cached contracts outside frozen namespaces; existing
tokenizer policies must remain closed to unsupported combinations. Complete
real worker/parent validation/inference before fresh seed census, a finite
pilot freeze, guarded pilots and final finite quality freeze. Use equal bounded
LR tuning; do not add retrospective arms or relax fresh confirmation requirements.

Measure actual serialized raw/export overlap, complete metadata/transients and
the8GiB reserve before any new panel freeze. The score campaign closure observed
about9.60GB free; that is an observation, not a ready new-panel budget. Inventory
exact authorized obsolete artifacts or lossless archival options as necessary;
preserve selected models, every fresh pair, failures and protected archives.
Small source/numerical work proceeds independently of training-capacity needs.

Evidence: tokenizer-evidence/ffn-norm-preparation-001/direction-decision.json.

## First component result

component001 stopped before tensor creation: the harness used a JSON reader
to verify Python source hashes. Its original source/plan and failure receipt
are preserved; no numerical or scientific seed was consumed. component002
is a separate frozen correction using exact source-byte hash equality.

All12 generated CPU cases passed: FP64/FP32, both none/RMS arms and input
scales0.01/1/10. Forward and all input/matrix/gain gradient maximum absolute
differences were0 in this implementation; altered suffix tokens left prefix
outputs unchanged. Shared gate/up/down matrices were identical and the gain
adds672 parameters per block,2,688 across four blocks. The guard completed
in0.359s, peak RSS301,154,304B, with the8GiB reserve preserved.

This is component evidence only. The full tokenizer policy still needs an
explicit additive extension; do not bypass frozen policy rejection. Next test
actual shared model initialization/optimizer groups, serialized resume/RNG/EMA,
H200 and cache/causality on CPU/MPS before any scientific launch.

## Explicit full-model integration passed

The additive `research/ffn_norm_v1` runtime introduces an exact none/RMS product
normalization policy without editing the449 frozen MLA score source files.
Generated full-model model001 passed CPU/MPS arithmetic, all parameter gradients,
one CPU optimizer step per arm, real in-memory save/load payload validation,
interleaved request caches, reset and causality. Legacy policy rejection remains.
The control's initialization and logits are bitwise identical to the adopted model;
all shared candidate weights are identical and four new gain vectors start at1.

Parameter counts are28,492,416/28,495,104. Shared optimizer groups are unchanged;
the2,688 gains use AdamW at the test's base LR0.0025 with per-step independent
decay0.00006. This test LR is not a scientific LR selection. Maximum CPU/MPS
logit discrepancy was1.431e-6, all-gradient discrepancy5.253e-7, and cached
logit discrepancy1.282e-6. Prefix causality and reset differences were0.
Cross-arm checkpoint/config/cache identities and altered constructed RMS epsilon
were rejected. Cache storage remains393,216B for batch2/context128.

The guard finished in3.896s with peak RSS1,332,674,560B and MPS driver
1,111,277,568B. No research corpus, holdout or scientific seed was used. This
is implementation evidence, not quality evidence. Serialized raw/optimizer/EMA/
RNG replay and H200 export remain separate prerequisites.

## Capacity recovered for serialized prerequisites

The old six-file resume recipe needs10,972,299,264B free including its complete
raw/export caps,256MiB metadata allowance and8GiB reserve. Before dispatch,
exact successful numerical-fixture archival recovered2,266,198,096B net and
verified27,046 retained files unchanged. Fourteen original weights remain
recoverable from four protected archives; see the retention policy and archive001
manifests. Selected models, every fresh pair and failure artifacts stayed loose.
This is a prerequisite-specific capacity check; a complete future scientific
panel high-water projection remains required.

## Serialized MPS resume and H200 passed

checkpoint-resume001 completed both arms on generated text at batch8/context128.
Three uninterrupted updates, a separate three-update repeat and two updates after
restoring the step1 checkpoint produced exactly equal model/optimizer/EMA/RNG
state (maximum discrepancies0). The test made16 generated optimizer updates
across both arms. Missing optimizer state, reordered optimizer groups, missing
MPS RNG and inference-only H200 resume were rejected before model mutation.

H200 export preserved parent weights/hashes and CPU/MPS RNG and replaced each
learned tensor with its exact EMA value. Exported models passed CPU/MPS full and
cached inference at batches1/8 (maximum discrepancy1.520e-6). No holdout or
research corpus was read. The16.204s guard observed peak RSS2,355,593,216B and
MPS driver2,551,070,720B, below unchanged limits. Six new raw/H200 fixtures are
retained under research_runs/tokenizer-ffn-norm-checkpoint-prerequisite-001;
raw sizes445,082,727/445,132,607B, exports114,074,691/114,087,095B by none/RMS.

Next implement the narrow actual run_training/output/worker path and independent
parent full-validation/inference acceptance using the tested runtime. The current
roughly9.66GB free is not enough to assume two new worker pairs fit above reserve;
finish source work, measure exact complete output high-water, and use audited
lossless archival of completed eligible prerequisites once their dependencies
close. Existing archives are protected and all failed attempts remain closed.
No scientific census, pilot plan or quality plan has been frozen.
