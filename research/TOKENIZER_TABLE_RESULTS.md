# Width256 table-sharing results

Completed October 28, 2026: 15 quality trajectories, 30 retained raw/H200
checkpoints, and three serial fresh-seed inference comparisons. Neither table
alternative is adopted. Retain the existing legacy-untied width256 GQA recipe;
the separately retained MLA option is unchanged.

Source89309b3, plan4de77f0, digest
`6abcfdfe160acaa34d214bb456720cd8ec16591a3647640201eb2628d16ba353`.
See the [frozen protocol](TOKENIZER_TABLE_PROTOCOL.md) and
[complete audit](tokenizer-evidence/table-001/final-review.json).

## Fresh quality and decision

Six seed3301 screens independently selected LR0.00125 for legacy_untied and
LR0.0025 for matched_untied/scaled_tied. Selection was saved before confirmation.
All three arms ran fresh seeds3311/3312/3313 for1200 updates with cooldown and
H200start800/interval10. No rate or seed was reselected.

| Arm | Selected LR | Mean H200 CE | Unique FP32 parameter bytes | Mean training seconds |
|---|---:|---:|---:|---:|
| legacy_untied | 0.00125 | 4.745130 | 114,352,128 | 204.47 |
| matched_untied | 0.0025 | 4.847797 | 114,352,128 | 213.05 |
| scaled_tied | 0.0025 | 4.878305 | 62,888,960 | 183.78 |

| Candidate minus control | Mean CE delta | Seed deltas3311/3312/3313 | Conditional95% interval,4096-token blocks |
|---|---:|---|---|
| matched_untied_vs_legacy_untied | +0.102667 | +0.113643 / +0.094751 / +0.099606 | [+0.087902, +0.116390] |
| scaled_tied_vs_legacy_untied | +0.133175 | +0.139312 / +0.126814 / +0.133398 | [+0.111468, +0.152707] |
| scaled_tied_vs_matched_untied | +0.030508 | +0.025669 / +0.032063 / +0.033792 | [+0.021848, +0.037515] |

Both alternatives are worse than legacy on every fresh seed. Tied is also worse
than matched on every seed. All three contrasts fail both strict quality and the
prospective +0.01-nat neutrality band; all2048/4096/8192-token conditional interval
lower bounds are positive. This margin was an agent-chosen engineering convention.
The tied45.004% parameter-memory reduction exceeds the declared30% cost threshold,
but cannot qualify because neutrality against both controls failed.

The matched control already loses0.102667 nat versus legacy. It changes correlated
initialization and readout scaling/coordinates. Tying adds0.030508 nat versus this
matched control while coupling gradients/optimizer histories and reducing
expressivity. Independently selected learning rates also form part of these
recipe contrasts. The total0.133175-nat loss cannot all be attributed to sharing,
nor does it establish that all forms of tying fail.

## Measured costs

Actual parameter bytes are identical across all three fresh seeds. Tied reduces
parameters from28,588,032 to15,722,240. Allocated KV storage is unchanged:
524,288 bytes at batch1 and4,194,304 at batch8/context128. Tied mean training time
is183.78s versus204.47s legacy (10.12% lower); timing is descriptive.

Inference uses fresh H200 triples, FP32 MPS,32-token prefill and64 teacher-forced
decode tokens, two warmups and eight rotating repeats per arm/batch. Ratios below
are medians across seeds of within-seed median latency ratios; below1 is faster.

| Comparison | Batch | Prefill latency ratio | Decode latency ratio |
|---|---:|---:|---:|
| matched_untied_vs_legacy_untied | 1 | 1.012560 | 0.994631 |
| scaled_tied_vs_legacy_untied | 1 | 1.016067 | 0.994895 |
| scaled_tied_vs_matched_untied | 1 | 1.003463 | 0.997106 |
| matched_untied_vs_legacy_untied | 8 | 1.040391 | 0.991528 |
| scaled_tied_vs_legacy_untied | 8 | 1.105087 | 0.994594 |
| scaled_tied_vs_matched_untied | 8 | 1.047000 | 1.004808 |

Tied decode is about0.51%/0.54% faster than legacy at batches1/8, while prefill is
1.61%/10.51% slower. These small decode differences do not establish a general
speed advantage. All three models were resident during comparison, so observed
working-memory readings are not single-model peak memory or maximum capacity.
No long-context, larger-batch, cloud-cost or width64 claim follows.

## Validation and evidence

All15 records passed frozen source/runtime/data/request/selection provenance,
checkpoint SHA256, exact config/tokenizer/EMA lineage,1200-step/41-snapshot and
saved full-validation arithmetic review. All30 saved checkpoints passed CPU tensor
finiteness; every H200 parameter exactly equals the saved raw EMA state and each
persistent buffer equals its raw counterpart. This verifies exports, not a replay
of optimization or a new full-validation model evaluation. Gates were recomputed
from retained score blocks and matched the controller conclusion exactly.

All three inference comparisons passed six cached/full checks each (largest
absolute logit discrepancy1.24e-5), exact checkpoint lineage,48 timing-row coverage,
parameter/cache accounting and identical fixed training-token inputs. Per-seed
medians and original timing rows are retained in the audit and timing directory.
Three20-step feasibility pilots and20 CPU tests preceded quality; pilots never
selected arms or rates. No training/timing failure or retry occurred.

The initial metadata reviewer counted16,384 persistent RoPE buffer elements as
parameters and asserted. This review-only failure remains in
`tokenizer-evidence/table-001/reviewer-failure-001.json`; corrected counting uses
canonical EMA parameter names and separately checks the eight buffers. No model,
result, checkpoint, selection or training source changed to fix that reviewer.

All remaining checkpoints and evidence are retained. Historical authorized
cleanup manifests stay authoritative. The test holdout is closed. Conditional
text bootstraps condition on three trained seeds and an adaptively reused
validation split; they are not seed-population or multiplicity-adjusted inference.

## Learning and next direction

Large vocabulary tables make sharing materially cheaper, but the tested scaled
tied recipe did not preserve quality after equal bounded tuning. Initialization,
readout scaling and shared optimizer dynamics need explicit controls. A memory
saving alone is insufficient under the predeclared quality/cost rule.

A same-shape activation comparison was proposed, then deferred by explicit human
steering to wider targets and a fresh direct MLA/GQA comparison. See
[the revised preparation](TOKENIZER_WIDE_ATTENTION_PREPARATION.md). Activation,
MoE, normalization, residual/depth and standalone tuning remain first-class
alternatives; the closed hybrid result does not establish an HP optimum.
