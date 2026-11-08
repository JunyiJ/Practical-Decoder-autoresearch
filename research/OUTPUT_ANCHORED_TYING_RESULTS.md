# Output-anchored table sharing: complete, not adopted

Allnine fresh trajectories completed1200updates each:10,800scientific updates,
11,059,200sampled targets and18retained RAW/H200 weight files. The independent
saved-result review passed. The shared model does not satisfy either required
quality comparison. Keep the byte bulk-hash-adoption001 incumbent unchanged.
No new worker is running for this completed panel.

## Primary H200 validation result

| Fresh bundle | Native separate | Matched separate | Shared | Shared−native | Shared−matched |
|---|---:|---:|---:|---:|---:|
| 62101800000 | 4.674577 | 4.648745 | 4.666682 | -0.007895 | +0.017937 |
| 62101900000 | 4.695631 | 4.687429 | 4.696511 | +0.000879 | +0.009082 |
| 62102000000 | 4.680066 | 4.661738 | 4.682615 | +0.002549 | +0.020878 |
| Mean | 4.683425 | 4.665970 | 4.681936 | -0.001489 | +0.015966 |

Lower CE is better. Each result covers all34,278validation targets with the
original context128 reset windows. All41H200 snapshots and allthree paired
training-sampler endpoint hashes were verified. The matched control beats shared
on every seed. Sharing beats native only on the first seed; its mean difference
is negligible relative to the predeclared0.02nat minimum gain.

Both primary contrasts fail allthree gate types: mean gain, every-seed improvement
and negative conditional upper bounds. The scalar independent reviewer reproduced
the controller's deltas/bounds within1e-12 using the saved target losses.

| Contrast | Mean delta | Upper2048 | Upper4096 | Upper8192 |
|---|---:|---:|---:|---:|
| tied_vs_native | -0.001489 | +0.011624 | +0.013726 | +0.013665 |
| tied_vs_matched | +0.015966 | +0.024523 | +0.026073 | +0.021744 |
| matched_vs_native | -0.017454 | -0.008587 | -0.009021 | -0.008124 |

These are one-sided conditional text-bootstrap bounds at quantile1−.05/6,
conditioning on these three seeds and an adaptively reused validation split.
They are not seed-population confidence or untouched-holdout evidence.
The matched control's mean improvement also falls below0.02; it is not an
alternative candidate selected after observing the shared model's failure.

## Cost and diagnostic endpoints

| Arm | Parameters | Mean training-update seconds | Mean final online CE |
|---|---:|---:|---:|
| native | 28,492,416 | 272.552 | 4.671634 |
| matched | 28,492,416 | 272.424 | 4.663665 |
| tied | 15,626,624 | 215.857 | 4.667037 |

Sharing saves45.16% of parameters and used20.80% less
training-update time than native in this local panel. Timing excludes diagnostics
and export and is descriptive, not a universal serving claim or a cost-only
adoption route. Peak process RSS was not materially reduced in these runs;
parameter savings must not be presented as the same percentage of total memory.

The final online weights are a diagnostic, not a replacement selector. Their
shared-minus-matched mean is+0.003372, with two losing
seeds and one winning seed. H200 increases that penalty, but neither this
observation nor sparse curves establishes an averaging or optimization cause.
Do not switch endpoints, extend the horizon, or retune the closed panel to rescue it.

The nine guarded scientific jobs used39.99minutes in total. The CPU/MPS
preflight used24updates in25.8seconds; it verified native outputs/allgradients,
matched/shared initial predictions and summed gradients exactly, unique optimizer/
averaging ownership, and exact saved-weight roundtrip. The final read-only review
used2.414seconds. Total retained weight bytes are1,744,386,504; all18files remain.
Final RAW files contain weights, not resumable optimizer states. Independent EMA
reconstruction from every intermediate snapshot is outside this saved artifact
format; no such proof is claimed.

## Exploratory saved-loss analysis

The analysis uses the prior byte-study frequency bins and allthree seeds, with
no model execution, additional losses, new training or holdout access. Relative
to matched, targets seen1–9times in training have mean CE penalty
+0.130519 on2846validation targets, positive in allthree
seeds. Their weighted contribution+0.010837
accounts for67.9% of the net+0.015966penalty.
Unseen targets improve, partially offsetting the loss. These descriptive groups
are correlated with token identity and context; no subgroup significance or
causal gradient explanation follows.

[Projection applicability review](SHARED_TABLE_PROJECTION_EVIDENCE.md) records
supporting and contrary primary literature and the closed local projection
restrictions. A linear adapter is not automatically supported. The next small
research question is whether accepted matched input/output table geometry differs
most for rare tokens. Inspect these saved weights and training-frequency groups
without retraining, rescoring or reconstructing old failed results before choosing
a new model hypothesis. No follow-up scientific campaign is frozen here.

## Bound evidence

- [Protocol](OUTPUT_ANCHORED_TYING_PROTOCOL.md)
- [Frozen plan](tokenizer-evidence/output-anchored-tying-001/plan.json)
- [Controller result](tokenizer-evidence/output-anchored-tying-001/final.json)
- [Independent review](tokenizer-evidence/output-anchored-tying-001/final-review.json)
- [Frequency analysis](tokenizer-evidence/output-anchored-tying-001/frequency-diagnostic001/result.json)

No comparison with the byte incumbent was made on these fresh seeds. No historical
scores were pooled. All outcomes, including unfavorable seeds, remain intact.
