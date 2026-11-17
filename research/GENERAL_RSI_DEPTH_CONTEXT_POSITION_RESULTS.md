# Context benefit by segment position

The completed two-depth study's evaluation-policy benefit is concentrated near
common128 segment resets. It also persists later in the segment. This is a
posthoc description of saved scores, with no new training, scoring or generation.
The frozen superiority and unresolved-interaction decisions remain unchanged.

## Measured evidence

Each row averages the three fresh paired bundles. Delta is loss under native256
evaluation minus common128 evaluation of the **same context256-trained weights**.
Negative favors supplying the longer prefix. Counts are identical across bundles.

| Position in256 segment | Targets | Depth4 EMA delta | Depth16 EMA delta |
| --- | --- | --- | --- |
| 1–128, same prefix in both evaluations | 53820 | 0 exactly | 0 exactly |
| 129–160, first32 after common128 reset | 12223 | −0.411038 | −0.428967 |
| 161–192, next32 | 11877 | −0.119210 | −0.119533 |
| 193–256, last64 | 22496 | −0.048343 | −0.051395 |
| Combined remaining96 after first32 | 34373 | −0.072830 | −0.074939 |

The first32 contain12.17% of all100416 scored targets and account for66.74%/67.06%
of the respective depth4/depth16 mean evaluation-policy gain. The first32 benefit
per target exceeds the remaining96 benefit in all three bundles at both depths,
for both RAW and EMA. RAW shares are67.05%/67.25%. This supports the declared
reset-concentration prediction but rejects an exclusively first32 explanation.

All same-prefix losses are bitwise identical across native256/common128 in the
saved files, including every RAW/EMA endpoint. Exact target-ID and global-position
matches rule out different target inventories as the explanation for these deltas.
Short document tails are included using their actual counts.

## Decomposition and contrary evidence

| EMA contrast, nats/token | Depth4 | Depth16 |
| --- | --- | --- |
| Total native-context delta256-trained minus128-trained | −0.096031 | −0.089993 |
| Same-common128 difference between trained weights | −0.021068 | −0.012125 |
| Same256-trained-weight evaluation-policy difference | −0.074963 | −0.077868 |

The final two rows sum to the first, with the original100416-target denominator.
The evaluation-policy term represents78.06%/86.53% of the signed total difference
on this chosen decomposition path. These fractions are arithmetic descriptions,
not identified causal shares or population estimates. The context128-trained
weights were not evaluated at256, so a full training-by-evaluation interaction
and an order-independent attribution are unavailable.

Common128 gains are not uniform over position. For targets129–160 in the original
256 segment, the256-trained weights regress by+0.025142/+0.028508 under common128.
All six paired EMA contrasts in this bin are positive. Later-position gains and
shared-prefix gains outweigh this regression in the aggregate. This posthoc bin
has no frozen adoption gate or adjusted confidence interval. It does not overturn
the prespecified aggregate decision or justify selecting a new cutoff after results.

## Interpretation and literature

- Local inference: avoiding short available history immediately after resets is
  consistent with the largest observed evaluation-policy contribution. Later
  benefit remains. These observations do not establish semantic reasoning or the
  usefulness of dependencies beyond128 tokens: the added prefix also supplies
  nearby tokens that common128 discards at its reset.
- Dai et al., [1901.02860v3](https://arxiv.org/html/1901.02860v3), §§3.1–3.2,
  discuss fragmentation and sliding evaluation. Our nonoverlapping evaluation
  policy is different, and our model has no recurrence. Their architecture and
  speed results are not established locally.
- Khandelwal et al., [1805.04623v1](https://arxiv.org/html/1805.04623v1), §§3–4.1,
  use fixed-weight context perturbations and show heterogeneous context needs.
  Their LSTMs, word tokens and exclusion of short-history targets differ here.
  Their results neither imply a depth scaling law nor quantify our dependency range.
- Representation, exposure, optimizer, clipping and schedule uncertainty remain.
  The analysis supports inspecting evaluation-history coverage before another
  context extension. It does not support reopening closed training studies.

## Verification and cost

`tokenizer-evidence/general-rsi-depth-context-position-001/analysis.json` retains
all12 endpoint/bundle descriptions, per-document/source summaries and hashes of
36 original score files. Every input hash matches the accepted completed-panel
manifest. An independent ordered-vector implementation reproduced every bin
summary with zero discrepancy and verified identical IDs, positions and shared
prefix losses. The decomposition reconciles with aggregate CE within1e-10.

Analysis took2.86s with177618944-byte peak RSS. Independent verification took1.13s
with114966528-byte peak RSS. No MPS allocation, model load, corpus/test read or
new endpoint occurred. Shared nonblocking locks were acquired. Before execution,
process inspection found no worker and memory pressure was normal. Free storage
was16718843904bytes during analysis, above12GiB reserve plus64MiB allowance.

## Next executable action

Keep the completed context256 recipe and its original scope. No new context,
recurrence or optimizer training panel is admitted by this analysis. A useful
next source/literature review is whether the observed global-clipping difference
across depths changes the qualified mixed Muon/AdamW update in the way suggested
by gradient-norm counts. Establish that mechanism and inspect closed optimizer
records before proposing any diagnostic. No arbitrary clipping sweep is selected.

Public publication remains blocked by the previously reported automatic approval
review. Preserve local commits and await the already requested explicit public
publication authorization. Do not retry an unchanged push or treat that blocker
as a reason to rerun scientific experiments.
