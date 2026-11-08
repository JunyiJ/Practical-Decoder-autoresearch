# Byte-context transform: a new fixed-recipe question

Status: admit a bounded implementation/resource pilot only. A scientific panel
requires the measured pilot and a separate immutable full-panel freeze.

## Evidence and hypothesis

Local evidence: the adopted byte recipe is
`tokenizer-evidence/byte-composition-boundary-001/bulk-hash-adoption001/recipe.json`.
Its real-byte input model improved both native and nonlexical controls in fresh
confirmation. It retains an independent linear output table. The original
`readout-quality-preparation-001/readout-quality001-primary-statistics.json` and
`readout-quality001-selection.json` show that the old native-input MoS recipe
improved native by 0.033083091 nat/token, but only 0.001065236 versus matched MoC,
with one losing seed and positive bounds. MoS failed. The descriptive MoC gain
cannot become an adopted fallback. Its two nonlinear maps, router and selected
learning rate confound an explanation based on nonlinearity alone.

The completed sharing experiment and `shared-table-geometry-001/analysis.md`
do not support imposing a shared table or a rare-specific distortion fix. This
proposal keeps the separate output table. The old native-input single-projection
panel closed incomplete on a numerical scoring failure, not a quality rejection
(`TOKENIZER_READOUT_PROJECTION_RESULTS.md`). Its failed fixture, seeds, namespaces
and restrictions remain closed. This is a new interaction question on the adopted
raw-byte composer, with fresh roles and no reuse of historical scores or weights.
It is not a reconstruction or rescue of that failed native-input attempt.

External evidence actually read: Pappas et al., WMT 2018 proceedings,
https://aclanthology.org/W18-6308.pdf, sections 4.2–4.4 and Table 3: context-only
nonlinearity improves their tied translation model, while a linear map does not;
the full same-width joint model adds little beyond the context-only map. These
are LSTM translation results with different data, dropout, Adam and widths.
Yang et al., arXiv:1711.03953v1 (10 November 2017),
https://arxiv.org/html/1711.03953v1, sections 2.1–2.5 and Table 4: feature mixing
improves PTB but worsens WT2 relative to the standard AWD-LSTM settings. Their
high-rank probability mixture result does not imply a gain from one context map.
No published or historical improvement is counted as a new local result.

Inference to test: a learned bounded nonlinear reparameterization of the final
context may improve generalization of the byte model at its existing training
budget. It changes amplitude and derivatives together. The falsifiable prediction
is that tanh beats both the unmodified byte baseline and a same-sized linear map
under the same fixed recipe. Linear isolates added matrix capacity and part of
the optimization change; it does not isolate saturation from amplitude. Failure
against either control rejects this fixed-recipe candidate, not every nonlinear
output family. All arms retain logit rank at most 256; no softmax-rank claim.

## Arms and fixed proposed scientific comparison

For unchanged final affine LayerNorm feature z and untied head W:
- byte: W z / 4 (16,351,859 parameters, 105 tensors).
- linear: W A z / 4 (16,417,395 parameters, 106 tensors).
- tanh: W tanh(A z) / 4 (same inventory as linear).

A is bias-free 256x256, initialized U(-1/16,1/16) with a new explicit private
role after genuine native and composer initialization. Linear/tanh share A.
Every arm preserves the complete genuine native initializer before displacement
of its input table and uses the same real-byte codebook and composer.

Proposed panel: nine 1,200-update trajectories, three fresh paired bundles by
three arms, rotated order, fixed base LR .0025 for all arms. This tests a change
to the adopted byte recipe; no tuning or screening allowance in any arm, no
claim of optimized families. The .0025 choice comes from the existing byte recipe,
not outcomes from this pilot. Keep B8 T128, paired blocks, original corpus/splits,
FP32, no dropout, clip1, independent decay .00006, AdamW(.9,.999), eps1e-6,
Muon match_rms_adamw at half base LR for inherited matrices and A, cooldown
800–1200 to .1, H200 snapshots 800..1200 every10 (41 total).

Primary endpoint: saved/reloaded H200, full 34,278 validation targets, reset128,
FP32 per-target CE and FP64 aggregation. Candidate tanh must beat BOTH byte and
linear: mean delta < -.02, all three paired deltas negative, and all six upper
bounds negative (2,000 joint PCG64 draws, widths2048/4096/8192, quantile1-.05/6).
RAW and online curves are descriptive only. No linear fallback, endpoint switch,
seed replacement, grid extension, horizon change or old-score pooling. Holdout
remains closed. A passing panel still requires independent result review and
artifact/consumer validation before adoption. No cached-serving claim follows.

## Smallest useful implementation and pilot

Use an isolated wrapper around the adopted v6 model and unchanged composer;
retain its forward validation. Add only A and ordinary PyTorch linear/tanh math.
Keep canonical parameter order, the original clipping/schedule/EMA operations,
and independent optimizer groups. The existing model, consumer and frozen
studies are not edited. The new experiment uses its own identities and exports
weights only; it promises neither optimizer resume nor public cache support.

One bounded pilot on CPU and MPS, four updates per arm plus a paired official
baseline reference (32 total updates). Verify exact baseline initial logits,
all gradients, each updated parameter, optimizer state and EMA on each backend;
verify common initialization and matching linear/tanh A; finite all-parameter
exposure; independent analytic projection gradients; and exact weight-envelope
roundtrip including codebook identity. Initial batch CPU/MPS mean CE must agree
within 3e-5 and per-target CE within 3e-4. Preserve numerical diagnostics before
asserting the predicates. No validation-quality selection or holdout scoring.

Pilot roles are distinct from all closed roles, audited before use. Guard:
1,800 seconds, RSS4GiB, MPS6GiB, reserve8GiB, 8MiB persistent output cap, shared
nonblocking worker lock. Forecast pilot retained output <8MiB; in-memory weight
serialization ~66MiB. All original failures remain if recovery is necessary.

## Complete cost forecast and decision boundary

Nine trajectories imply 10,800 updates and 11,059,200 training targets. Each
projected model adds 65,536 parameters (0.401%), 262,144 FP32 weight bytes and
67,108,864 forward multiply-accumulates per B8 T128 batch. This is 0.51% of the
vocabulary projection MACs, not a measured runtime prediction. Byte composition
cost remains. Both final RAW weights and H200 weights are retained: tensor bytes
for all18 files total 1,180,480,728. Allow 64MiB for serialization/container and
metadata plus 66MiB one-file overlap, then round full output cap up to 1.5GiB.
Codebooks are reconstructed from bound public assets, not duplicated per file.
Reserve8GiB plus extra2GiB must remain after that complete requirement. Current
~13.7GB free exceeds the ~12.35GB requirement; measure again at freeze.

A conservative provisional runtime ceiling is 9x3,600s plus 1,800s pilot and
1,800s final scoring/review = 10 hours. Before freezing, use measured pilot update,
serialization, and numerical-check costs to forecast all runs and evaluations;
do not dispatch if the complete forecast exceeds the admitted cap. All costs are
local compute, no paid service. This is targeted architectural refinement of the
byte incumbent, not a relabeled broad-exploration quota contribution.
