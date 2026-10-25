# Query/key RMS normalization quality protocol

This is the finite prospective protocol for a separately registered round.
The historical horizon study is completed and reviewed. Its frozen candidate
failed width-256 confirmation, so this round uses the retained 3,000-update
recipe, cooldown starting at 2,000, and H200 returned weights. No completed
study is resumed. The run plan binds the exact source, predecessor review,
prerequisites and full shared knowledge snapshot after every launch gate passes.
The active-campaign pointer and live processes establish execution status.

## Question and predecessor

Does per-head RMS normalization of queries and keys improve the adopted
four-layer MHA model under equal, bounded learning-rate tuning?

Resolve the training horizon from the final independently verified horizon
review. If its frozen candidate passes both-width adoption, use that candidate's
total steps and cooldown start. Otherwise retain the 3,000-update / start-2,000
recipe. Never choose the horizon from partial results or the best alternative
after confirmation. Freeze the exact resolved recipe and predecessor hashes
before screening. This experiment retains the reviewed H200 returned weights.

## Exact intervention

Use separate learned RMS gains for queries and keys, each of shape `[32]`,
shared across heads, positions and batch examples. For each head vector x,
compute `x / sqrt(mean(x*x) + 1e-6) * gain`. Gains initialize to one without
consuming random numbers. Apply normalization **before RoPE**. Preserve the
existing dot-product scale `1/sqrt(32)` and leave values unchanged. Cache the
normalized, rotated keys once; incremental attention must not normalize them
again. This adds 256 learned scalar parameters to the four-layer model.

This RMS variant is not an exact reproduction of Henry et al. (2020), which
uses L2 normalization and a learned scalar attention scale. External translation
and vision findings motivate a hypothesis; they do not establish a decoder
quality gain or a local attention-instability diagnosis.

All other architecture choices remain fixed: four layers, MHA head dimension
32, dense bias-free SwiGLU hidden width 21d/8, RMSNorm blocks, final LayerNorm,
RoPE, unscaled residuals, zero dropout, context 128. Explicit policies are
`depth4_qkrms_unscaled` and `mha_head32_qkrms`. Reject unreviewed shape,
normalization, attention-scale, residual, latent or expert overrides.

Standard MHA retains its original policies and missing/default `qk_norm=none`
behavior. Its state keys, common initialization draws, RNG state and forward
arithmetic must remain unchanged. No global math-version migration is intended:
the explicit config, source digest and strict state keys distinguish the new
architecture. A changed normalization contract must reject optimizer resume.

## Optimizer and scaling contract

Use Muon/AdamW hybrid with muP base width 64, clipping 1 and independent
per-update decay 0.00006. Existing hidden matrices retain width factor
`1/sqrt(d/64)` and Muon's RMS-matched update. Embeddings, readout and vectors
retain AdamW at the base learning rate. Q/K gains are one-dimensional vectors
and use that same AdamW/base-LR/independent-decay rule. Head dimension and gain
shape stay fixed as the number of heads increases with width.

These are explicit implementation rules, not an empirical proof of optimal
learning-rate transfer or a new asymptotic muP theorem. Q/K normalization changes
logit magnitude and gradient geometry. Tune each family independently on the
same finite LR panel; do not assume the inherited LR remains optimal.

## Fixed finite panel

Screen at width 64: standard MHA and QK RMS MHA, each at peak LR
`[0.00125, 0.0025, 0.005]`, on fresh seeds 2181 and 2182. This is 12 trajectories.
Verify these seeds were unused in all earlier registered plans before launch.
Alternate family ordering across seeds. No early pruning or LR-grid extension.

Select each family's LR by mean final H200 full-validation CE; ties choose the
lower LR. Before any confirmation, freeze both choices and the complete
confirmation/reference manifest. QK RMS is locally eligible only if its mean
gain exceeds 0.005 nats/token and both paired seeds improve against **both**
the equally tuned standard family and the fixed incumbent standard LR 0.005.
The incumbent screening cells already exist in the LR grid; do not duplicate
them or treat them as independent repeats.

Confirm both selected families at widths 64 and 256 on fresh seeds
2191, 2192 and 2193: 12 trajectories. Add six fresh fixed-incumbent references
at LR 0.005 unless the selected standard family uses exactly that recipe;
then prospectively alias those six references to its selected runs. Never
alias across seeds, widths, schedules, normalization or returned-weight policy.

Total: **24 or 30 physical trajectories**, producing **48 or 60 correlated
raw/H200 endpoints**. Confirm both selected families even after local failure.
Freeze QK RMS as the winner when it meets its local rule. Otherwise, freeze the
tuned standard family if it improves the fixed incumbent by >0.005 mean CE
and improves both screening seeds. Otherwise freeze retention of the incumbent.
This hierarchy preserves a possible learning-rate improvement without making
any choice from confirmation data.

Only that frozen winner can replace the incumbent. QK RMS must beat both
tuned-standard and fixed-incumbent references at each width; a frozen standard
winner must beat the fixed incumbent at each width. Each required contrast
needs mean gain >0.005, every fresh-seed delta negative, and all three
conditional text-bootstrap upper bounds negative. An unselected alternative
cannot be promoted if the frozen winner fails. No larger-width reselection,
retrospective reference choice or automatic grid extension.

## Training and evaluation

All arms share the final adopted total steps and cooldown start, batch 8,
context 128, paired_blocks512 sampling, 1,024 targets per update, and the same
cooldown minimum ratio 0.1. Fix H200 averaging every ten completed updates from
the adopted cooldown boundary, without feedback to the online optimizer or RNG.
Use raw parents for optimizer resume; averages are inference-only artifacts.

Primary: final H200 full validation at context128/batch8, exactly 111,538 targets
once each. Preserve raw final validation as a separate diagnostic. Training-log
sampled CE cannot drive selection. Pair seeds and text blocks; report block
sizes 2,048/4,096/8,192 with 2,000 bootstrap draws and seed 20261024. Include
QK RMS minus tuned standard, QK RMS minus fixed incumbent, and tuned standard
minus fixed incumbent. Aliased contrasts are explicitly identical observations,
not independent repeat evidence. Fixed-LR screening contrasts stay separate
from tuned confirmation contrasts.

Training tokens/updates are matched within this round. Measured runtime and
parameter count may differ; this is not equal compute. Adaptive validation and
conditional text uncertainty are not independent holdout or seed-population
significance. The held-out test remains closed.

## Gates and limits before launch

1. Independently check normalization, input/parameter gradients, pre-RoPE
   placement, causality, unchanged default behavior, and full/cached attention.
2. Verify model initialization/RNG parity for common weights, exact added
   parameter counts, optimizer parameter coverage/roles/scales, finite updates,
   scheduler and H200 inclusion, strict save/load, and changed-contract resume
   rejection. Actual training checks run serially after the predecessor exits.
3. Run the complete project suite on the integrated clean source. Preserve any
   failures and fixes. Then run four serial MPS train/reload/full-validation
   pilots (two families × widths64/256), plus cached/full correctness. Pilot
   scores and timings are excluded from research rankings.
4. Review maximum checkpoint/RSS/driver-memory estimates and measured pilot
   costs for the resolved horizon. Proposed hard limits: 30 trials, 2,400 seconds
   per trial, 21,600 seconds per round, 4,096 MiB artifacts/RSS and 6,144 MiB MPS
   driver allocation. These are launch gates, not a runtime prediction.
5. Preserve the full shared knowledge bank, lineage, source/corpus/runtime and
   prerequisite hashes. Review, test, commit and push clean isolated source;
   freeze a uniquely named plan before a single serial launch. First failure
   stops the round; no automatic retry. No competing training.

At completion independently verify every raw/derived checkpoint and request/
result hash, configuration, optimizer state, RNG state, schedule, step/token/
snapshot counts, head/QKV/FFN/gain shapes, full coverage, frozen choices/aliases,
all report aggregates and conditional bootstrap intervals. Append results,
update the best configuration only under the frozen rule, and publish the
learning guide and readable HTML. Never rerun completed training to interpret it.
