# Tokenizer 3:1 KDA / gated MLA preparation

October27,2026. Implementation preparation, not a frozen quality protocol.
MLA001 is complete: a scoped RoPE MLA option passed three fresh quality pairs
and measured81.25% allocated KV savings versus fresh MHA, with slower decode.
The GQA recipe remains separate. Next implement the human-prioritized3:1
hybrid before choosing a finite training panel. No new quality launch yet.

## Version and local adaptation

The primary [Kimi Linear report, section4](https://arxiv.org/html/2510.26692v1#S4)
uses three KDA layers per NoPE MLA layer, low-rank output gates, and head128.
The [Kimi K3 report, section2.1](https://arxiv.org/html/2607.24653v2#S2.SS1)
uses bounded log-decay with minimum-5, full-rank output gates and NoPE gated
MLA, plus an additional global layer at the end of its much deeper backbone.

The initial local proposal is four layers KDA/KDA/KDA/gated-MLA. Keep dense
SwiGLU21D/8, RMSNorm blocks, final affine LayerNorm, serial unit residuals,
untied GPT-2 tables, head32, context128 and the reviewed optimizer. This is
a small K3-style attention adaptation, not a reproduction of K3's MoE,
attention residuals, per-head Muon, scale or extra fifth global layer. Four
layers already end in global attention; adding a fifth would confound depth.

KDA should retain the checked sequential recurrence, convolution width4,
L2-normalized Q/K, FP32 recurrent state, alpha bottleneckD/4, learnable per-head
A initialized0, bounded log-decay-5*sigmoid(exp(A)*z), head RMSNorm and a
full-rank sigmoid output gate. Keep the existing projection/bias initialization
explicit; it is not claimed to reproduce the paper's decay-bias initialization.
Sequential implementation cost is a backend limitation, not a prediction of
optimized KDA kernels. Any later chunk implementation needs equation/gradient
agreement before performance claims.

Global gated MLA needs an explicit0.5 score multiplier in full/cached paths,
query/KV ranksD/4 and NoPE with no learned-position fallback. Initially retain
the existing shared32-dimensional unrotated key/query component and normalize
scores by sqrt(32+32), documenting that this local NoPE formulation still
stores latent plus shared key. A latent-only formulation/absorption is a
separate implementation and geometry change; do not silently claim its cache
savings for this baseline.

## Implementation and prerequisite review

Before this change, GPT constructed every block from one global attention
family and cache callers repeated the global family. The new exact contract
now binds a validated layer layout; each block and cache resolves its family
from that shared contract, with matching per-layer state construction.
The top-level checkpoint signature must include the entire layout, positions,
geometry and gate semantics. Reject swapped or incompatible cache layouts,
noncontiguous recurrent offsets and mutated configs. Keep historical contracts
and old checkpoint semantics unchanged.

`GatedMLA` now consumes the declared score multiplier, preserving1.0 for legacy
configs. A separate `tokenizer_k3_hybrid_nope_v1` contract binds3:1, all-KDA
and all-gated-MLA layouts without loosening old tokenizer or KDA contracts.
The top-level config retains the whole layout and each block's index/family is
validated. NoPE is explicit for all layers, with no positional table fallback.
Optimizer validation accepts only the new exact backbone; actual2D block
projections use the hidden-matrix rule while convolution kernels and vectors
retain AdamW base LR. Canonical parameter names and tensor layout are preserved
for strict optimizer/resume and H200 state continuity.

Required checks: independent recurrence/gated-attention equations and gradients;
causality; full/chunked/tokenwise cache parity; reset/interleaved requests;
per-layer shapes and actual byte counts; batch/offset errors; absence of RoPE
and learned position tables; exact tokenizer/optimizer/checkpoint/resume/EMA
contracts; old-family regression checks. CPU synthetic checks precede serial
MPS feasibility pilots. Freeze a pilot plan, source and resource projection
before running any pilot; pilot CE cannot select a quality winner.

Completed CPU validation:163 checks passed in33.16seconds across hybrid and
existing tokenizer MHA/GQA/MLA, historical KDA/gated-MLA and cache suites.
This includes strict FP64 full/chunked/tokenwise cache parity and independently
computed attention equations/gradients, plus exact real training resume and
averaged-checkpoint/tokenizer loading. The initial18-pass/2-failure log is
preserved: FP32 projection rounding at width256 exceeded the initial2e-6
absolute tolerance by2.3–3.0e-6; stricter FP64 checks distinguished this from a
state/equation error. FP32 tolerance is now1e-5 absolute/2e-5 relative, frozen
before the MPS pilot. Nine focused contract checks passed after a type-error
cleanup, and three pilot guard/configuration checks passed. Counts overlap.
New implementation/test files pass lint; the broader touched-file lint log
also records existing style findings, without unrelated formatting churn.
Completed temporary pytest fixtures were cleared after preserving pass/failure
logs and a file manifest; no research checkpoints were removed.

The [pilot protocol](TOKENIZER_HYBRID_PILOT_PROTOCOL.md) fixes one20-update
width64 run and MPS cache checks. Width256 CPU contracts are validated, but no
width256 training or quality transfer is claimed.

## Resource gate and experiment choice

After MLA001, free disk is about9.9GiB, with8GiB reserved. Another ten width256
raw/H200 pairs needs roughly6.05GiB plus reserve and does not fit. Preserve all
research checkpoints. Measure candidate storage before freezing the next
quality plan; possible smaller-width feasibility work does not imply width256
transfer. No new quality launch may ignore the remaining-trial projection.

Candidate controls are full NoPE gated MLA and all-KDA using the same newly
validated tokenizer backbone; the inherited RoPE MLA result alone does not
control gating/position changes. Choose a finite panel and equally bounded
family-specific LR tuning after correctness/runtime/storage checks. Keep fresh
confirmation, fixed endpoint metrics and prospective quality/cost thresholds;
do not reuse the MLA001 confirmation seeds for adaptive tuning.

This priority follows the explicit human hybrid request and closes a missing
topology comparison. MoE is the next untested major family, followed by targeted
activation/normalization/residual/readout hypotheses as evidence warrants.
Standalone schedule, LR and regularization studies remain first-class options;
attention parameter refinement must not indefinitely displace those questions.
Historical all-KDA character results remain prior evidence only. Test closed,
one local worker, no paid compute, no subagents, all previous stop markers intact.
