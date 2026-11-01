# Causal FFN-input mixing preparation

Current question: does identity-initialized causal three-tap per-channel mixing
of the normalized FFN input improve the adopted tokenizer D256 MLA recipe under
equal bounded LR tuning? This is an untested architecture hypothesis. Existing
short convolutions in KDA do not establish a benefit here; no Kimi reproduction
or long-context claim is intended.

After the warmup campaign closed without adoption, both architecture and bounded
HP were reviewed. Independent decay0.00006 versus0 remains a valid separate
regularization hypothesis, deferred while addressing the previously queued local
mixing family. No closed campaign or failed fixture is reopened.

For each block, after the attention residual, let u be the existing FFN RMSNorm
output. Feed the existing SwiGLU with
v[t,c]=a0[c]u[t,c]+a1[c]u[t-1,c]+a2[c]u[t-2,c], zero-padding the left edge.
There is no bias or additional nonlinearity. Initialize three learned channel
vectors a0=1,a1=a2=0 without RNG draws, so the initial function and shared model
weights can match the incumbent exactly. Four D256 layers add3072 parameters;
report the small capacity difference explicitly rather than claim parameter
matching. The prospective optimizer assignment is inherited AdamW at base LR
with the same independent decay for these 1D vectors; actual group/state policy
still requires full-model verification.

Streaming inference retains exactly two normalized FFN-input tokens per layer.
Preserve history across uneven chunks; reset to zeros for a new sequence. Four
layers add8192 FP32 bytes per batch item to the current196608-byte MLA cache;
full-model buffer enumeration and timing must verify this projection. A standalone
component is implemented in research/local_mix_v1/component.py. It is not yet
accepted by a full-model/checkpoint policy or wired into an existing runtime.

Frozen component001 passed six generated cases: CPU FP64/FP32 and MPS FP32,
batches1/3, width7, length11, chunks4/1/6. An independent grouped Conv1d oracle
checked forward values and input/all-three-vector gradients. Maximum FP64 errors
were1.11e-16 forward/1.67e-16 gradients; maximum FP32 errors2.98e-8/1.79e-7.
Cached output was exact in all cases; identity initialization, future-token
causality, reset and wrong-history rejection passed. All497 predecessor sources
remained unchanged. No corpus, optimizer training, scientific seed, checkpoint
resume or quality evidence was used. Preserve this successful source/plan/result;
do not rerun it merely for progress.

Next add narrow explicit full-model policy, shared-weight identity, vector optimizer
grouping, raw optimizer/RNG/EMA resume, H200 export and full/cache/reset integration
outside frozen sources. Then real bounded worker and independent parent full
validation/inference, fresh seed census and complete measured output/transient
capacity, finite guarded pilots and final quality freeze. Prospective quality
comparison remains six equal-LR screens and three fresh pairs with strict gates;
no training panel or scientific seed is frozen/reserved yet. Keep all existing
selected/fresh/failed weights and immutable evidence. Small source/numerical work
proceeds while future full-panel capacity is audited; do not reuse stale blockers.

Evidence: tokenizer-evidence/local-mix-preparation-001/direction-decision.json,
component001-plan.json, component001-result.json and component001-guard.json.
