# Tokenizer sparse versus dense FFN preparation

Status: design and implementation feasibility only, October31,2026. No new
model, training pilot, screen or confirmation has run. This is not a frozen
launch plan. Completed wide-hybrid005 is closed and must not be rerun.

## Direction and baseline

Prioritize the architecture roadmap's next untested tokenizer family: routed
expert FFNs. Question: does top2-of4 half-width SwiGLU expert capacity improve
GPT2-tokenized validation quality versus both active-FFN-matched dense and
stored-FFN-matched wider dense controls? Historical character MoE improved at64
but did not transfer at256 and was not adopted; wider dense explained much of
the narrow gain. These are motivation for the controls, not tokenizer evidence.

Use the already adopted width256 quarter-latent RoPE MLA recipe as the proposed
common attention backbone: four layers, head32, score multiplier0.5, context128,
batch8,1200 updates, cooldown800, H200 start800, untied GPT2 tables. All arms
train fresh; incumbent weights are preserved. Width256 is an initial resource
scope, not a result-selected claim of width transfer. Width512 remains deferred
until full retained-capacity accounting fits. Do not substitute the005 gated
NoPE MLA control for the adopted RoPE incumbent without a separate prospective
comparison. Source baseline: tokenizer-evidence/mla-001/adopted-width256-mla-recipe.json.

## Proposed controlled arms and arithmetic

| Arm | Hidden width at D256 | Active FFN matrices per layer | Stored FFN matrices per layer |
| --- | ---: | ---: | ---: |
| dense SwiGLU |672|516,096|516,096|
| wider dense SwiGLU |1344|1,032,192|1,032,192|
| four experts, top2, SwiGLU |336 per expert|516,096 +1,024 router|1,032,192 +1,024 router|

Each SwiGLU expert has three bias-free D-by-H matrices. Ordinary hidden width
is21D/8; expert width21D/16; wider dense21D/4. Four layers multiply these counts
by4. Shared vocabulary/attention/norm parameters must be counted separately.
Equal active FFN weights do not imply equal FLOPs, wall time or optimizer exposure.

Propose softmax router, top2 selection and renormalized selected probabilities,
weighted expert sum, no shared experts/capacity cap/token dropping/noise/z-loss;
auxiliary load coefficient0.01 separately disclosed and excluded from validation
CE. Keep router scaling explicit and test it. Experts with no gradient currently
skip update and decay; document exposure and cumulative shrinkage differences.
Collect cumulative per-layer assignment counts, entropy/load/unused experts and
auxiliary loss, rather than claiming whole-run balance from one sampled batch.

## Read-only implementation findings

- `src/moe/scaling.py` permits only historical MHA, widths64/128/256 and its exact
  field set. It cannot be combined with MLA merely by toggling mlp_type.
- `src/models/tokenizer_architecture.py` fixes dense SwiGLU and validates dense
  gate/normalization attributes. Introduce a separate narrow tokenizer-FFN
  contract; preserve old validators and checkpoint signatures unchanged.
- `src/models/blocks.py` routes historical half-expert policy through that MHA
  validator. Wider dense policy is also bounded. New explicit dispatch is needed.
- `src/train/optimizers.py` validates historical MoE and identifies router weights
  by its exact width-policy name; new contract must preserve router Adam scaling,
  hidden-matrix grouping, complete unique coverage and dormant-expert behavior.
- `src/moe/moe_block.py` supplies a reference routing implementation, not a claim
  of optimized MPS throughput. Existing `tests/test_moe_study.py` numerical,
  gradient, count and resume tests cover historical contracts only.

## Finite candidate design to finalize before launch

Propose three20-update resource/correctness pilots, one per arm, followed by
three equal LR choices0.000625/0.00125/0.0025 per arm on one screening seed
(9 full runs), choose each arm's minimum H200 CE with lower-LR tie break, then
all three fixed recipes on three fresh seeds (9 full runs). Rotate family order
across fresh seeds. Freeze exact seed identities, requests, limits and metadata
allowances before pilots; do not extend a boundary-winning LR grid.

Only MoE is the adoption candidate; wider dense is a stored-capacity diagnostic
control, not a posthoc fallback winner. Proposed strict rule: mean H200 gain
>0.02 versus each tuned dense control, improvement on every fresh seed, and
negative simultaneous one-sided conditional-text upper bounds for both contrasts
at block2048/4096/8192,2000 common paired draws, frozen RNG/quantile conventions.
Report all seeds, raw endpoints, wall time, memory and inference costs. No
quality-neutral memory adoption route is proposed for extra-capacity MoE.
Freeze this rule and its tests before new quality results; reused validation
text does not provide unconditional seed-population or holdout confidence.

## Required work and storage gate

1. Implement a narrow new policy and model/load/optimizer validation while
   retaining old errors and signatures. Test routing values/gradients against
   an independent dense oracle; counts, zero-route handling, CE/aux separation,
   causality, cached decoding, checkpoint/resume and H200 export/load identity.
2. Add cumulative routing diagnostics and bounded artifact manifests. Reuse
   reviewed checkpoint transport concepts with a new source applicability audit;
   never rerun the closed failed numerical fixture or closed campaigns.
3. Compute exact stage high-water capacity before any resource pilot: all old
   protected artifacts, three selected screening pairs, nine fresh raw/H200 pairs,
   optimizer/EMA/export overlap, metadata/logs and unchanged8GiB reserve. At closure
   free space was about15.52GB. No fit is asserted from parameter arithmetic.
   Do not delete005 fresh pairs or selected models to make this study fit.
4. If the proposal does not fit, document the exact gap and lawful retention
   options or prospectively redesign the scope before freezing. No partial-panel
   start, reduced confirmation after results, paid compute or weakened guard.
5. Freeze the complete finite protocol, source/runtime/input/prerequisite pins,
   immutable stage selections and guarded serial controller before training.

## Other directions considered

Activation/normalization/residual comparisons remain next after this feasibility
assessment; a same-shape gate-nonlinearity comparison is the fallback if MoE's
storage/implementation cost cannot be justified prospectively. More hybrid
ratio/rank tuning is deferred after005 failed quality and memory gates; its
closed grid stays closed. Bounded LR tuning is included equally for all FFN arms.
A separate EMA/horizon study is deferred:005 raw-versus-H200 differences motivate
future hypotheses but do not justify changing the finished H200 decision metric.
No new empirical quality claim follows from this preparation document.
