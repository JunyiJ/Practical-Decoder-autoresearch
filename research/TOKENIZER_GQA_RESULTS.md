# Tokenizer GQA comparison

The [finite protocol](TOKENIZER_GQA_PROTOCOL.md) compares MHA and half-KV-head
GQA on the adopted GPT-2 tokenizer backbone. The previous LR round adopted
0.0025; this round gives each family equal tuning at0.0025/0.00125 before fresh
confirmation. The round is complete: adopt width256 GQA as a scoped serving recipe; retain MHA at width64.

## Completed round and inference review

All20 trials and six fresh-checkpoint inference comparisons completed. All saved
artifacts passed review; source/runtime/protocol/corpus stayed frozen. The three
width256 seed deltas are -0.061297, -0.076319 and -0.027171 nats/token. Mean
H200 CE is5.500381 for MHA versus5.445452 for GQA. The4096-token conditional
95% interval is[-0.069829,-0.039910]; all declared quality checks pass. Width64
fails as detailed below. Test loss remains closed.

Final MPS FP32 timings used three checkpoint pairs, batches1/8,32 prompt tokens
and64 teacher-forced decode steps, two warmups and eight paired repeats each.
Full/cached checks passed for every model and batch (maximum absolute error
0.000023842). Actual cache allocation is halved at each width and batch.

| Width | Batch | MHA / GQA KV bytes | MHA / GQA median decode seconds |
|---:|---:|---:|---:|
|64|1|262144 / 131072|0.169812 / 0.174436|
|64|8|2097152 / 1048576|0.171028 / 0.175389|
|256|1|1048576 / 524288|0.174082 / 0.176071|
|256|8|8388608 / 4194304|0.175977 / 0.179972|

Adopt half-KV GQA at width256 and LR0.00125 as a scoped serving recipe relative
to equally tuned MHA at LR0.00125. Its50% KV saving passes the40% route despite
observed1.14%/2.27% slower median decode at batches1/8. At context128 the absolute
cache savings are0.5MiB/4MiB, not a halving of total application memory. Timing
uses two resident models and has outliers; no guaranteed speed, peak-memory,
long-context, cloud-cost or production claim follows. Width64 is not adopted.
The general MHA LR0.0025 incumbent is not replaced by an untested fresh LR-only
comparison. Training totaled6600.20seconds; all checkpoints are preserved.

[Final review](tokenizer-evidence/gqa-001/final-review.json),
[width256 serving recipe](tokenizer-evidence/gqa-001/adopted-width256-serving-recipe.json),
and [raw inference records](tokenizer-evidence/gqa-001/final-inference/).

Next finite hypothesis: shorten the width256 GQA training schedule, with
prospective duration choices and fresh confirmation. Repeated late validation
degradation at multiple LRs motivates this HP round before further attention
comparisons. MLA and3:1 KDA/MLA remain the next structural priorities; MoE,
activation, normalization and residual alternatives remain in the broader agenda.
This does not change or reopen the completed GQA experiment.

## Prerequisites completed

Source `dbc4d9f` passed140 targeted CPU tests in287.11seconds. The20 initial new
checks are included in that total. Checks include exact MHA initialization,
forward, gradients and optimizer-step parity with the adopted implementation;
GQA against an expanded-KV reference and its input gradients; full/cached causal
behavior, uneven/nonzero offsets, separate/reset state, parameter/cache sizes,
strict contract rejection, real training/resume and EMA/tokenizer loading.
This is targeted regression, not the entire historical repository suite.

Four serial40-step MPS pilots (MHA/GQA, widths64/256, seed2691) completed and
passed saved-artifact review, including finite tensors, optimizer steps,
21 EMA snapshots, parent hashes, full validation coverage, replayed training
sampler state and untouched test RNG. Their scores are feasibility data only.
The separate new architecture contract fixes the inherited model semantics and
implements the0.5 score multiplier in GQA; historical contracts stay intact.
MLA and hybrid support are still future work.

MPS full/cached checks passed at batches1/8 and widths64/256. Paired pilot timing
completed with the frozen32-token prefill/64-step teacher-forced decode workload,
two warmups and eight repetitions per arm and batch. Actual KV allocation is
50% lower for GQA. Timings use two resident models and are feasibility evidence,
not a trained quality/cost adoption result or a total-memory reduction claim.
Fresh selected checkpoints must pass the final cost review after quality training.

Projected20-trial raw/H200 output is5,223,069,680bytes (about4.86GiB), or7.30GiB
with50% headroom. The measured free space passed that projection plus the8GiB
reserve; the round cap is8GiB. Successful temporary pytest fixtures were cleared
after preserving logs; research checkpoints and failure evidence remain retained.

An initial lint check found an unused benchmark import; it was removed after
the unchanged-source CPU regression, and final lint passed. No training or
numerical prerequisite failure occurred. [Pilot and timing review](tokenizer-evidence/gqa-pilots-001/review.json).

## Interpretation and continuation

The20 quality trajectories use one serial MPS worker, eight screen cells and
twelve fresh confirmation cells. Frozen per-family LR choices precede confirmation.
GQA may qualify independently by width only after the+0.01-nat quality margin,
cached correctness and measured cache-storage route pass. Report latency tradeoffs
separately. The known late-validation degradation remains unresolved: this round
tests architecture under the inherited3000-update budget. A horizon/schedule
comparison remains active in the backlog alongside MLA, the Kimi hybrid and other
architecture families. The test holdout stays closed.

## Completed screening and frozen confirmation choices

All eight screening trajectories passed saved-artifact review. A separate
recomputation verified the controller's selection: both MHA and GQA use
LR0.00125 for the twelve fresh confirmation trajectories. Source, runtime,
protocol and corpus match the frozen plan. This is same-agent review.

| Family | Mean H200 CE at LR0.0025 | Mean H200 CE at LR0.00125 |
|---|---:|---:|
| MHA | 5.218836 | 4.946603 |
| GQA | 5.212367 | 4.952413 |

Both choices lie at the lower grid boundary, so the best LR remains unresolved.
Do not expand this frozen round or replace the general LR0.0025 incumbent from
these screens. Fresh confirmation compares both architectures at the same LR.
The fixed-LR GQA-minus-MHA mean changes from -0.006469 at LR0.0025 to +0.005810
at LR0.00125; paired seed differences have mixed signs at each LR. One lower-LR
screen difference is +0.013444, above the prospective +0.01 margin. Screening
does not establish neutrality or failure of the fresh-confirmation gate.
This sensitivity supports retaining fair tuning in architecture comparisons.
The known horizon/schedule concern also remains unresolved. Final cached
inference measurements are still required before adoption.
[Screening audit](tokenizer-evidence/gqa-001/screen-review.json).

## Completed width64 confirmation: quality gate fails

All six fresh width64 trajectories passed artifact review. At LR0.00125,
mean H200 CE is MHA4.947097 versus GQA4.964981, a +0.017884-nat difference.
The paired seed differences are +0.023516, +0.040203 and -0.010066. The mean
and two seeds exceed the frozen +0.01 margin; all conditional text-bootstrap
upper bounds also exceed it. Width64 GQA is not adopted under this rule.

| Text-block tokens | Conditional 95% interval for GQA minus MHA |
|---:|---|
| 2048 | [+0.008256, +0.027745] |
| 4096 | [+0.007822, +0.028360] |
| 8192 | [+0.006359, +0.031489] |

Intervals were separately recomputed from audited saved loss sums, with no
rescoring. They condition on these three models and the reused validation text;
they are not training-seed uncertainty or independent holdout evidence. One seed
favors GQA, so do not present this as uniform degradation or evidence against
all GQA designs. The width64 setting is two query heads sharing one KV head;
width256 uses eight query heads and four KV heads and has its own decision.

The frozen width256 trials continue. Final inference measurements remain
pending and will document tradeoffs, but cache savings cannot reverse this
failed quality gate. Retain the general baseline and the existing horizon,
MLA and Kimi hybrid hypotheses. Test holdout remains closed.
[Width64 confirmation audit](tokenizer-evidence/gqa-001/width64-confirmation-review.json).

## Width256 trajectory warning persists at LR0.00125

The first completed width256 MHA confirmation (seed2711) passed artifact review:
full-validation raw CE5.601754 and H200 CE5.520442. Its sampled validation loss
was lowest among logged evaluations at step1200 (4.638358), then rose to5.394212
at step2999 while the logged training minibatch CE fell from3.235198 to0.968407.
These sampled diagnostics are not the full-validation endpoint metric, and the
retrospective minimum is not an adopted checkpoint or proven optimal horizon.
The trajectory is consistent with the earlier late-validation degradation even
at the lower LR, supporting a prospective horizon/schedule experiment. It does
not isolate a causal mechanism. Complete the frozen architecture comparison
before changing the training recipe. The paired GQA run subsequently passed
artifact review with H200 CE5.459145, a -0.061297 difference versus MHA.
This first width256 pair does not establish improvement or neutrality; finish
the other two pairs and the frozen quality/cost review.

## Reviewed quality trajectories

Only completed, artifact-reviewed trajectories appear here. Quality eligibility
alone does not adopt GQA: cached correctness, measured inference cost and final
review are also required. The test holdout remains closed.

| Stage | Family | Width | Seed | LR | Raw CE | H200 CE | Training seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| screen | mha | 64 | 2701 | 0.0025 | 5.283691 | 5.229085 | 254.43 |
| screen | mha | 64 | 2701 | 0.00125 | 4.964187 | 4.950018 | 254.70 |
| screen | gqa | 64 | 2701 | 0.0025 | 5.265618 | 5.212528 | 253.65 |
| screen | gqa | 64 | 2701 | 0.00125 | 4.956483 | 4.948193 | 254.50 |
| screen | gqa | 64 | 2702 | 0.00125 | 4.969298 | 4.956633 | 254.48 |
| screen | gqa | 64 | 2702 | 0.0025 | 5.270827 | 5.212207 | 254.04 |
| screen | mha | 64 | 2702 | 0.00125 | 4.952350 | 4.943188 | 254.78 |
| screen | mha | 64 | 2702 | 0.0025 | 5.265823 | 5.208587 | 254.70 |
| confirm | mha | 64 | 2711 | 0.00125 | 4.963383 | 4.950808 | 253.68 |
| confirm | gqa | 64 | 2711 | 0.00125 | 4.982797 | 4.974324 | 254.33 |
| confirm | gqa | 64 | 2712 | 0.00125 | 4.994932 | 4.977332 | 253.64 |
| confirm | mha | 64 | 2712 | 0.00125 | 4.955060 | 4.937129 | 253.90 |
| confirm | mha | 64 | 2713 | 0.00125 | 4.960276 | 4.953354 | 255.46 |
| confirm | gqa | 64 | 2713 | 0.00125 | 4.949657 | 4.943288 | 254.46 |
| confirm | mha | 256 | 2711 | 0.00125 | 5.601754 | 5.520442 | 509.48 |
| confirm | gqa | 256 | 2711 | 0.00125 | 5.542281 | 5.459145 | 505.40 |
| confirm | gqa | 256 | 2712 | 0.00125 | 5.528917 | 5.434492 | 505.62 |
| confirm | mha | 256 | 2712 | 0.00125 | 5.605393 | 5.510811 | 509.29 |
| confirm | mha | 256 | 2713 | 0.00125 | 5.557112 | 5.469891 | 506.29 |
| confirm | gqa | 256 | 2713 | 0.00125 | 5.522909 | 5.442720 | 503.35 |

Paired GQA minus MHA H200 CE (fixed-LR screens; selected-recipe confirmation):

- screen, width64, seed2701, GQA LR0.0025 / MHA LR0.0025: -0.016558.
- screen, width64, seed2701, GQA LR0.00125 / MHA LR0.00125: -0.001825.
- screen, width64, seed2702, GQA LR0.00125 / MHA LR0.00125: +0.013444.
- screen, width64, seed2702, GQA LR0.0025 / MHA LR0.0025: +0.003620.
- confirm, width64, seed2711, GQA LR0.00125 / MHA LR0.00125: +0.023516.
- confirm, width64, seed2712, GQA LR0.00125 / MHA LR0.00125: +0.040203.
- confirm, width64, seed2713, GQA LR0.00125 / MHA LR0.00125: -0.010066.
- confirm, width256, seed2711, GQA LR0.00125 / MHA LR0.00125: -0.061297.
- confirm, width256, seed2712, GQA LR0.00125 / MHA LR0.00125: -0.076319.
- confirm, width256, seed2713, GQA LR0.00125 / MHA LR0.00125: -0.027171.

Frozen family LR selection: {'gqa': 0.00125, 'mha': 0.00125}
