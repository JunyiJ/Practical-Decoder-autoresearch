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
