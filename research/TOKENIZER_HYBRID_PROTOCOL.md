# Frozen width64 NoPE hybrid comparison

October27,2026. Compare four-layer KDA/KDA/KDA/gated-MLA with homogeneous
all-gated-MLA and all-KDA controls under the exact implementation in
[TOKENIZER_HYBRID_PREPARATION.md](TOKENIZER_HYBRID_PREPARATION.md). This is a
small dense K3-style adaptation, not a Kimi reproduction. Bounded LR tuning
and structural comparison are both part of the experiment.

All families: width64, head32, dense SwiGLU21D/8, unit serial residuals,
RMSNorm blocks/final affine LayerNorm, untied GPT-2 tables, NoPE and no learned
position fallback. KDA: convolution4, Q/K normalization, FP32 recurrent state,
alpha rank16, bounded log-decay -5, A initialized0, full-rank output gate.
Gated MLA: query/KV rank16, shared32-dimensional unrotated key/query component,
score multiplier0.5/sqrt64, full-rank output gate. Cache includes each recurrent
matrix and convolution history, or latent plus shared key; no latent-only claim.
Inherited initialization, Muon/AdamW parameter groups and independent decay
0.00006 are fixed. Architecture changes consume RNG differently; same seed
is not a guarantee of identical shared weights. Report total parameters,
training time and measured serving cost; equal token budgets are not equal compute.

## Finite selection and confirmation

- Six screens: seed3001, LR0.00125/0.0025 for each family. Family order hybrid,
  gated_mla, kda; reverse LR order for gated_mla. Choose each family's lower
  full-validation H200 CE (lower LR on exact tie). Choose the lower-CE winner
  among the two homogeneous controls (lexical family order on exact tie).
- Freeze the three selected LRs and selected control in selection.json before
  confirmation. Six fresh trials: hybrid and selected control at seeds3011,
  3012,3013, alternating pair order. Twelve trajectories maximum, no grid
  expansion, replacement seeds or retries. The unselected control has screening
  evidence only; make no confirmed superiority claim against it.
- Every trial:1200 updates, batch8/context128, cooldown and H200 start800,
  averaging every10 updates (41 snapshots). This recipe transfers the shorter
  GQA horizon as a fixed comparison budget, not an established hybrid optimum.
- Same raw80/10/10 Shakespeare boundaries and GPT-2 BPE identity. Primary endpoint
  full-validation H200 CE over34278 targets, disjoint context128 with tail;
  raw endpoint CE/BPB, sampled validation and fixed samples are diagnostic.
  Never select an earlier checkpoint. Keep test loss closed. Character CE is
  not comparable. Adaptive validation reuse limits generalization claims.

Two prospective routes for a scoped width64 hybrid option, both requiring all
artifact audits and measured final cost review:
1. Strict quality: mean paired CE delta < -0.02, all three seed deltas <0,
   upper conditional95% text-bootstrap bounds <0 for blocks2048/4096/8192,
   2000 draws/seed20261024 (existing paired_blocks implementation).
2. Memory route, only if all-gated-MLA is the selected control: mean, each seed
   and all three interval upper bounds <= +0.01 nat, plus at least40% smaller
   actual allocated cache/state at BOTH batch1 and8 in EVERY fresh pair.
   The +0.01 margin and40% benefit threshold are prospective engineering
   conventions, not human-specified numeric tolerances. No speed-win requirement;
   report the actual latency tradeoff. All-KDA has smaller state than the hybrid,
   so it cannot support this memory route.

Intervals condition on trained seeds and reused text, not seed-population
uncertainty. No automatic adoption by the quality controller. No replacement of
width256 GQA/RoPE MLA or MHA without a fresh direct comparison. Quality success
against the selected control is not proof that it is the best homogeneous family
in the population. A failed panel stays failed; no selecting another control.

## Cost, prerequisites and resources

Before quality launch, retain the completed20-step hybrid MPS pilot audit and
run exactly two synthetic MPS optimizer updates for each homogeneous control
(seeds2992/2993, LR0.00125, batch8/context128, first fixed training tokens).
Check finite loss/gradients/optimizer state and full/cached outputs at batch1/8.
No quality scoring or checkpoint export in this control feasibility check;
retain its raw records. A600-second process cap and existing memory guard apply.
Freeze its helper/source/protocol before running. No failure retry.

After all12 quality records are artifact-audited, run three serial source-frozen
cached-inference comparisons on their H200 checkpoints,600s per process, no
concurrent training/timing. FP32 MPS, context128 allocation, first96 training
tokens,32-token prefill/64 teacher-forced decode, batch1/8, two warmups and eight
alternating paired repetitions per seed. Validate cache parity before timing.
Count all recurrent/convolution/latent/shared-key buffers. Report parameter bytes,
allocated cache, prefill/decode medians and throughput separately. Two models
resident; allocator observations are not single-model deployment peaks or cloud
cost. No long-context/optimized-kernel claim from this sequential implementation.

One worker, four CPU threads, no CPU fallback. Freeze source/runtime/recipe,
protocol/pilot/resource/preflight/corpus digests.7200s quality process cap,
RSS4GiB/MPS driver6GiB sampled watchdog,8GiB free disk reserve. Per-trial
projection uses measured hybrid raw/H200 pair132348300 bytes plus24 bytes for
any extra parameters of the largest control, plus2MiB metadata, then15% safety
headroom. All12 remaining projections must fit before launch and after every
trial; cap total output at12 projections plus4MiB. Poll free disk/output while
running. Preserve every research checkpoint. Space is tight; a guard failure
stops the finite round and requires review, never deletion or automatic restart.
STOP/signals prevent new trials and terminate an active worker. Global/run locks
exclude other workers. Retain failures and interrupted directories.

CPU equation/gradient/cache/resume checks and the hybrid MPS pilot precede this
round. Controller tests cover finite selection, strict/neutral rules, resource
reserve, source freeze, serial order and non-restart. This width64 choice is
resource-driven; width256 quality is untested. Broad MoE, activations, norms,
residual/depth/readout and standalone HP hypotheses remain first-class next-round
options; evaluate them after this finite panel rather than extending attention
search indefinitely.
