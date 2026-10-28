# Wider hybrid: next finite preparation

October27,2026 Los Angeles. The direct width512 MLA/GQA round is complete: MLA
passes the scoped quality and memory gates, with slower decode. The next proposed
question is whether a wider3:1 KDA/gated-MLA hybrid benefits from broader bounded
tuning versus homogeneous controls. This responds to the human's width and HP
steering. It is a new experiment, not a reopening of closed hybrid001.

## Decision and alternatives

Prioritize preparing width512 hybrid/all-KDA/all-gated-MLA contracts and resources.
The earlier width64 hybrid received only two LR trials per family and cannot be
called an optimum. New RoPE MLA evidence does not establish NoPE gated-MLA or
hybrid quality. Keep the newly adopted RoPE MLA recipe separately scoped.

MoE adds capacity and storage, activation/norm/residual comparisons are useful
controlled alternatives, and standalone schedule/regularization tuning remains
valid. They are deferred for this preparation to address the explicit broader
hybrid question; attention refinements must not indefinitely displace them. No
new quality launch is authorized by this preparation note alone: executable
contracts/tests, full projection, pilot freeze and quality freeze are required.

## Proposed bounded stages, not a frozen plan

Use width512 four layers, GPT-2 legacy untied tables, existing dense backbone and
same closed-holdout boundaries. Declare NoPE for all three new families, K3-style
bounded-decay/full-rank output gates and complete layerwise recurrent/convolution
and latent/shared-key cache semantics. This is distinct from Kimi Linear's
low-rank output gates. No claim to reproduce a large Kimi model or long context.

Proposal: three retained resource pilots, then at most24 quality trajectories:
9 independent LR screens (three per family),6 bounded second-stage probes
(two per family),9 fresh confirmations (three seeds for all three families).
Second-stage hypotheses may address KDA decay/gate initialization and MLA latent
rank; exact candidates, fair budgets, screening seeds and deterministic selection
must be declared before any quality data. Do not adaptively enlarge a grid.
A shared horizon/schedule study is an alternative second-stage design; choose
once from prior evidence before freezing, not after viewing new screening scores.
Fresh confirmation compares the selected hybrid with BOTH selected homogeneous
controls. Freeze quality neutrality/strict and actual memory benefit criteria,
including multiplicity/conditional-interval limitations and honest latency costs.
No claim that any finite grid finds the global optimum.

## Resource gate and next useful work

Current free disk was17,551,732,736bytes (~16.35GiB). A deliberately optimistic
lower bound for27 retained trajectories is40,617,872,799bytes (~37.83GiB), including
8GiB reserve. It counts ONLY the two vocabulary tables with raw weights, Adam
moments, saved EMA and H200 export, then2MiB/trajectory plus15% allowance and4MiB
round metadata. Attention/FFN/norm weights and optimizer state are excluded, so
the actual complete projection will be larger. The panel cannot fit locally.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/storage-lower-bound.json`.
This lower bound is accounting, not a policy-validated construction or launch plan.

Do not lower reserve, silently shrink width, delete more checkpoints, reuse
closed controllers or launch a partial panel. Request an external checkpoint
location or additional space; roughly60GiB free is a provisional planning target,
not a measured sufficient budget. Full model/pilot accounting must determine it.
Meanwhile useful bounded work is explicit wider hybrid geometry/optimizer/lineage
support, CPU equation/gradient/causal-cache/resume/EMA tests and exact meta-device
accounting. Preserve all failures. Only after resources fit should a finite pilot
source/plan be frozen; actual pilot costs then gate every remaining quality run.
Single worker, strict MPS, sampled4GiB RSS/6GiB driver and8GiB reserve remain.

## Anchor implementation and accounting

A separate `tokenizer_wide_k3_hybrid_nope_v1` policy now admits only widths512/1024
with head32 and the three declared layer layouts. The old policy still admits
only64/256. The new anchor keeps D/4 decay and query/KV bottlenecks, convolution4,
bounded log-decay−5, full-rank output gates, NoPE and the inherited dense backbone.
No optional rank/decay/gate overrides are admitted yet. The config helper is not
a dispatcher or frozen experiment; its default seed/rate are preparation values.

Meta construction gives width512 parameter counts65,037,792 hybrid,
63,474,816 all-gated-MLA and65,558,784 all-KDA. At FP32/context128/batch1 their
allocated cache/state sizes are333,824/327,680/335,872bytes respectively. The hybrid
is only0.61% smaller than all-KDA and1.875% larger than all-gated-MLA at this anchor.
Therefore a meaningful memory-saving route against both controls cannot be
assumed. Quality and actual latency remain unknown; no trained hybrid comparison
or long-context claim follows from these shapes.

The complete anchor model/optimizer/EMA accounting and conservative27-trajectory
projection are in `tokenizer-evidence/wide-hybrid-preparation-001/anchor-storage-projection.json`.
This substitutes the anchor geometry for every stage2 probe; changed tuning
geometries must be accounted for again before freeze. It is not a sufficient
resource guarantee, measured checkpoint size or measured MPS training peak.
Width1024 has meta accounting only. Numerical prerequisites run at512 on CPU.

The corrected conservative anchor panel requires56,866,579,251bytes (52.96GiB)
at512 and124,702,393,587bytes (116.14GiB) at1024, including reserve. Exact saved
model/optimizer/EMA tensor counts are separated from conservative allowances.
An initial accounting draft included nonpersistent causal masks as saved buffers;
the draft and correction are retained. No launch decision used the draft.

Thirty-five geometry/equation/cache/optimizer/lineage and historical-policy checks
passed. A separate combined CPU continuation process passed the hybrid's exact
model/optimizer/RNG/EMA resume check, then exited90 at its resource guard. The
gated-MLA and all-KDA continuation checks are unverified. The guard diagnostic
was lost inside pytest capture on hard exit, so the exact trigger sample is not
available. Last observed RSS was4,018,368KiB, and post-exit disk remained above
reserve; an RSS crossing is likely, not a proven sampled peak. This is fixture
process memory, not an MPS training measurement. Preserve
`cpu-resource-failure-001.json`; no numerical retry or MPS launch followed.
Review fixture lifetime and a separately recorded validation procedure before
further numerical work. The wider anchor is not launch-ready.
