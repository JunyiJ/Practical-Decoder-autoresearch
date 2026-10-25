# Prospective SwiGLU hidden-normalization quality protocol

Preparation only. No run is registered or launched. The output-gate round must
complete and pass its full independent audit before this experiment resolves
its common attention family and fixed reference learning rate. Partial screens,
larger-width observations and historical repeated recipes cannot select that
backbone. All closed studies and human stop 006 remain closed.

## Question and rationale

Test normalization of the SwiGLU product immediately before its down projection.
NormFormer adds normalization after the FFN activation along with attention
changes. Its larger GELU-based models motivate this location; they do not
establish benefit for this small Muon/SwiGLU model. This experiment tests one
adaptation, not full NormFormer or a replication. It does not assume that this
project has the paper's measured gradient mismatch.

Sources: [NormFormer, section 2.1](https://arxiv.org/pdf/2110.09456) and
[authors' implementation guide](https://github.com/facebookresearch/fairseq/blob/main/examples/normformer/README.md).
The earlier block LayerNorm-versus-RMSNorm round changed the pre-branch norms;
the completed QK round changed queries and keys. Their failures do not test the
FFN-product location. They do caution against assuming normalization gains
transfer from width 64 to 256. Keep all existing block/final norms unchanged.

## Exact intervention and scaling contract

For the existing normalized block input X, let
Z = SiLU(X Wgate^T) * (X Wup^T), with hidden width H=21D/8.
The arms are `none`, `rms`, and `layer`:

- none: pass Z directly to the unchanged down projection.
- rms: gamma * Z / sqrt(mean(Z^2) + 1e-5).
- layer: gamma * (Z-mean(Z)) / sqrt(mean((Z-mean(Z))^2) + 1e-5) + beta.

All statistics are over H for each token independently, with population rather
than sample variance. There is no normalization across batch or sequence.
Gamma starts at ones, beta at zeros; constructors must consume no RNG. Keep
common weights and their initialization order unchanged. Apply no additional
residual, attention or activation scaling. Down-projection matrices stay
bias-free; the LayerNorm arm alone adds its explicitly declared norm bias.

RMS and LayerNorm add 4H and 8H parameters: respectively 672/2,688 and
1,344/5,376 at widths 64/256. The latter changes centering and introduces beta,
so their difference is not an isolated centering effect. Both also change
activation magnitudes and gradients; no amplitude-only or causal mechanism
claim follows from a quality difference. Extra normalization has compute cost.

Use `ffn_hidden_norm_policy=swiglu_product_norm_21_8`. Permit only four-layer
dense SwiGLU 21/8, standard MHA head32, RoPE, RMSNorm blocks/final LayerNorm,
unscaled residuals, dropout zero, context128, muP base64, widths64/128/256.
One common output-gate family is resolved from its final reviewed adoption.
Reject MoE, GQA/MLA, QK normalization, hidden-width/norm-epsilon/affine/placement/
initialization overrides and other unreviewed shape or residual policies.
Default/missing `none` preserves historical keys, RNG and numerical behavior.

New norm gains and biases are vectors in AdamW at the base LR, with the existing
independent per-update decay. Existing hidden matrices retain Muon and the
1/sqrt(D/64) LR factor. This is an explicit algebraic policy, not a guarantee of
optimal LR transfer under the new normalization geometry. The full config must
bind resume; all norm parameters must be included in H200 shadows and inference
exports. Cached decoding remains query-local; verify it rather than assume it.

## Finite panel proposed before any new quality measurement

Reserve screen seeds2221/2222 and confirmation seeds2231/2232/2233, subject to a
fresh all-plan collision check at registration. Width64 screens: three arms x
peak LR[.00125,.0025,.005] x two seeds =18 full-budget trajectories. Select each
arm's LR by mean final H200 full-validation CE, ties lower LR. Choose the lowest
mean selected arm, ties none/rms/layer. Require mean gain>.005 and both seeds
negative versus fixed incumbent, and tuned none for a nondefault arm. If the
best arm fails, freeze retention, with no alternative fallback.

Confirm all three selected arms at widths64/256 on all three fresh seeds:
18 trajectories. Add six fresh fixed incumbent references, prospectively
aliasing selected none only when its full recipe and LR match exactly. Total
36 or42 physical trajectories,72 or84 correlated raw/H200 model outputs.
Only the eligible frozen winner may replace the incumbent: mean gain>.005,
every fresh seed negative and all three conditional text interval upper bounds
negative at both widths against fixed incumbent and tuned none for nondefault.
Report all pairwise contrasts and fixed-LR screening contrasts. No pruning,
grid extension, larger-width reselection or historical pooling.

Common training:3,000 updates, cooldown starting at2,000 to.1 of peak,
H200 snapshots every10 steps from2,000 through3,000,context128/B8,
paired_blocks512,3,072,000targets,Muon/AdamW,independentdecay.00006,clip1.
Primary evaluation:111,538validationtargets once each,context128/B8.
Raw endpoint diagnostic only. Bootstrap2,048/4,096/8,192textblocks,
2,000draws,seed20261024. Adaptive reused validation and conditional text
uncertainty do not establish held-out or seed-population significance.

## Required gates, resources and audit

Do not launch or run competing model tests while output-gate training is live.
Develop only in the separate copy. Before registration require independent
normalization and complete SwiGLU output/input/parameter gradient checks,
zero/constant/small-input behavior with explicit epsilon, exact default/RNG,
causality and cached decoding, parameter counts, finite updates, all optimizer
roles/states/width factors, strict save/load and changed-norm resume rejection.
Actual trainer tests must verify raw/H200 noninterference, private sampler RNG,
untouched test RNG and exact CPU resume before/at the averaging boundary.
Run the full project suite after integration, then six serial MPS shortened
training/reload/full-validation/gradient/cache pilots, excluded from research.

Proposed hard caps:42trajectories,2,400s/job,21,600s/round,4,096MiB artifacts and
sampled process RSS,6,144MiB current MPS driver allocation. First failure stops,
no automatic retry,one local worker,no paid compute. Independently reviewed
pilot-based time/storage forecasts with50%margin must fit the limits; do not
raise caps in response to failure. Current allocation is not peak memory.

Resolve only the complete output-gate review; verify source/runtime/corpus,
fresh seeds, all predecessor evidence and the full append-only bank. Commit and
push clean reviewed source, freeze a unique readonly plan and knowledge
snapshot, then launch serially. At completion independently audit every raw/
derived checkpoint/request/result/hash/config/norm/QKV/FFN/optimizer/RNG/
schedule/token/snapshot/coverage contract, frozen choices and aliases, all
aggregates and bootstraps before any promotion. Preserve all failures, append
findings, maintain best config and guide/HTML, then continue the quality goal.
