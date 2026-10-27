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

## Read-only implementation findings and next edits

GPT currently constructs every block from one global attention config, and
cache callers repeat the same global KVCache config. Add a validated immutable
per-layer layout, derive each block's local config and build matching caches.
The top-level checkpoint signature must include the entire layout, positions,
geometry and gate semantics. Reject swapped or incompatible cache layouts,
noncontiguous recurrent offsets and mutated configs. Keep historical contracts
and old checkpoint semantics unchanged.

`GatedMLA` already has a full-rank channel gate and an explicit `enable_rope`
switch, but does not consume the score multiplier. The current tokenizer
architecture validator allows only MHA/GQA/RoPE MLA; introduce a new exact
hybrid contract rather than loosening the old ones. KDA's historical geometry
validator assumes four identical KDA layers. New per-layer configs must bind
alpha/head dimensions without bypassing the old study guards. Optimizer
parameter roles must be resolved per actual block type, including convolution,
norm, decay A/bias and gated-MLA projections. Preserve canonical parameter names
for strict optimizer/resume and H200 state continuity.

Required checks: independent recurrence/gated-attention equations and gradients;
causality; full/chunked/tokenwise cache parity; reset/interleaved requests;
per-layer shapes and actual byte counts; batch/offset errors; absence of RoPE
and learned position tables; exact tokenizer/optimizer/checkpoint/resume/EMA
contracts; old-family regression checks. CPU synthetic checks precede serial
MPS feasibility pilots. Freeze a pilot plan, source and resource projection
before running any pilot; pilot CE cannot select a quality winner.

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
