# Cross-size attribution and next comparison — 2026-11-14 Pacific

The weakening byte advantage is distributed across the development set. It is
not explained by one source, a few influential documents, or H200 averaging.
The retained intermediate probe is too narrow and unstable to identify when the
full-development interaction emerged. This is posthoc diagnosis, with no new
acceptance gate, model scoring, training, or test access.

## Document attribution

The interaction is (byte minus native at depth 16) minus (byte minus native at
depth 4). Positive means the byte advantage weakens. We average paired loss sums
across the same three seeds and divide by the same scored target counts.

| H200 subset | Documents | Targets | Interaction | Documents with positive mean interaction |
| --- | ---: | ---: | ---: | ---: |
| All | 108 | 100,416 | +0.048436 | 100 |
| FineWeb-Edu | 85 | 80,354 | +0.046780 | 79 |
| Wikipedia | 23 | 20,062 | +0.055072 | 21 |

All three seed interactions are positive within each source. Documents with a
positive mean account for 95.41% of scored targets. Only 59 documents have a
positive interaction in every seed, so document-level signs still vary by seed.
The eight negative-mean documents and every per-seed result remain in the evidence.

Removing any one document leaves an interaction between +0.047140 and +0.050400.
Removing the five largest positive contributors leaves +0.045345; removing ten
leaves +0.042366. These removals are descriptive sensitivity checks selected
using the observed effects, not new confidence bounds or scientific gates.
The five largest contributors are document IDs 29, 72, 95, 106 and 46. Together
they contribute 25.16% of the net interaction and contain 20,143 targets. Their
source, count, paired effects and contributions are recorded without reading
additional corpus text. This evidence does not localize a linguistic mechanism.

RAW has the same broad pattern: +0.047120 overall, with positive means in 97 of
108 documents. Its source interactions are +0.045297 and +0.054420. Averaging
changes the magnitude slightly but cannot be the sole origin of weakening.
The independently recomputed overall H200 deltas and interaction reproduce the
frozen decision to numerical precision. All six arms within each seed bundle
used identical batch-hash sequences over all 10,240 updates.

## Timing of the effect

The intermediate probe contains only documents 0 and 1, both FineWeb-Edu, with
4,503 targets. It uses online weights, unlike the primary final H200 checkpoint.

| Update | Phase | Mean depth-16-minus-4 probe interaction |
| ---: | --- | ---: |
| 1,024 | Before cooldown | +0.033684 |
| 2,048 | Before cooldown | +0.025416 |
| 3,072 | Before cooldown | +0.043370 |
| 4,096 | Before cooldown | +0.059280 |
| 5,120 | Before cooldown | −0.008091 |
| 6,144 | Before cooldown | −0.032922 |
| 7,168 | During cooldown | −0.014522 |
| 8,192 | During cooldown | −0.001540 |
| 9,216 | During cooldown | +0.016589 |

Cooldown starts at schedule index 6,826. Early positive effects exist, but their
sign is not persistent before or during cooldown. At the final endpoint, these
same two documents show only +0.007183 RAW and +0.010984 H200 interaction, versus
+0.047120 and +0.048436 on all 108 documents. This probe cannot establish a
full-development onset time or show that cooldown caused the final interaction.
Do not turn its early points into matched-horizon or equal-time claims.

Training-block losses decrease through the final block at every depth and arm.
Byte has higher mean training loss than native in every retained block after
averaging seeds. The last 1,024-update byte-minus-native differences are about
+0.043320, +0.055062 and +0.057155 at depths 4, 8 and 16. Training batches are
paired, but these moving online losses are not endpoint development losses.
There is no convergence proof. Clipping differs substantially between arms;
that alone does not identify a suitable new learning rate or a causal remedy.
All block losses, gradient norms, clipping fractions and LR ranges are retained.

## Explanations that remain open

- Representation: more transformer capacity may reduce the usefulness of shared
  spelling features. The widespread interaction is compatible with this, but
  representation was not separated from optimization.
- Finite exposure: all sizes saw the same 10,485,760 sampled targets from a
  10,001,163-token stream. Deeper models may need different exposure. Falling
  training loss is compatible with this but does not prove that longer byte
  training would restore its relative benefit. Closed exposure studies remain
  closed; this is not permission to replay them.
- Optimizer: fixed LR and mixed Muon/AdamW may transfer differently across depth
  or representation. Gradient and clipping differences make this plausible,
  without selecting a specific replacement or demonstrating a harness defect.
- Schedule: intermediate probe signs vary, while final RAW and H200 agree on the
  broad effect. A schedule-only explanation is not established. A fresh schedule
  intervention would need its own discriminating prediction and matched budget.

## Next comparison ranking

Select **native context 128 versus 256 across D256 depths 4, 8 and 16** for
preparation. The purpose is to test transfer of the accepted four-hour recipe.
This boundary selects a question, not permission to skip numerical qualification
or scientific admission. There is no active training worker.

1. Context transfer has a specific, measured local basis: the completed six-run
   depth-4 panel improved native-context CE by −0.095712 and common-128 CE by
   −0.022221. It offers a direct new size interaction and a useful result even if
   transfer fails. The larger body is already qualified at context 128; only the
   combined depth/context path and its consumers need new engineering checks.
   The new depth-4 pairs are required fresh reference controls for that
   interaction. They do not reopen the old study or pool its scores.
2. A bounded size-specific byte/native LR screen is a plausible alternative.
   It would test fixed-recipe optimization sensitivity, but there is no measured
   optimum shift or specific new LR prediction. A two-LR, full-horizon screen at
   depths 8 and 16 would add eight runs before 18 fresh confirmations: roughly
   46 training hours, with evaluation, failure and engineering allowances on top.
   It also tests selected recipes rather than explaining representation alone.
3. A byte/native exposure or schedule factorial needs fresh controls at each
   size. A short/long-horizon design with 18 runs at the original horizon and 18
   at twice that horizon would require about 93 training hours from the retained
   30.90-hour base cost, before overhead. The narrow probe does not pick such an
   intervention. Do not launch it as a rescue of the unresolved depth-16 result.
4. A new representation variant lacks a qualified local mechanism at this
   boundary. No composer-precision replay, batch-reuse arm or output-lexical
   pilot is selected.

This ranking weighs the strength and specificity of evidence against cost, not
raw CE effect sizes from unequal horizons. Context is more expensive than the
completed equal-token study, but it directly transfers another accepted finding
without adding a tuning-selection confound.

## Primary literature revisited

These external results motivate controls; they do not establish local gains.

- Khandelwal et al., [1805.04623v1](https://arxiv.org/html/1805.04623v1), §§3–4.1,
  use truncation to measure effective context. Their model-size changes preserve
  the context-usage trend, which challenges an assumption that a larger model
  must gain more from longer context. They study LSTMs and word-level PTB/WikiText,
  unlike our tokenized MLA/RoPE models. Keep identical scored targets and separate
  native-context from common-128 effects.
- Dai et al., [1901.02860v3](https://arxiv.org/html/1901.02860v3), §§3.1–3.2,
  identify fixed-segment fragmentation and propose recurrence. Our local
  comparison only changes segment length; it adds no recurrence. Their dependency
  and speed gains cannot be transferred to this implementation.
- Hoffmann et al., [2203.15556v1](https://arxiv.org/html/2203.15556v1), Appendix B,
  show that schedules ending too late impair endpoint loss; Appendix C limits
  their cross-dataset scaling statement to the single-epoch setting. Preserve
  endpoint-aligned cooldown. Do not infer compute-optimal token allocation from
  our repeated finite corpus or compare an intermediate point as a final run.
- Yang et al., [2203.03466v1](https://arxiv.org/html/2203.03466v1), §6.1,
  report empirical depth-transfer limits, including initialization sensitivity
  and pre/post-layernorm differences. Our mixed optimizer is not qualified muP.
  The proposed context comparison tests a fixed recipe, not each size's optimum.
- Muennighoff et al., [2305.16264v2](https://arxiv.org/html/2305.16264v2), §§4–5 and
  Appendix P, find repeated-data returns depend on resources and leave important
  hyperparameter dependence unresolved. Their smallest unique-token budget is
  100M and their whole-dataset epochs differ from our random-block sampling.
  Record actual targets and reuse; their fitted scaling law does not prescribe
  our horizon.

## Prospective scope and complete planning cost

Prediction: context 256 retains a practically useful negative paired CE effect
at depths 8 and 16 under the same 14,400-second training budget. A failed fresh
within-size gate limits transfer at that size. Estimate the largest-minus-reference
interaction directly; do not assume its direction or interpret nonsignificance
as equivalence. Native-context improvements without common-128 improvements would
localize benefit mainly to available inference context. They would not establish
better short-context weights or readable generation.

Plan three fresh paired bundles at each size, 18 runs total, fixed native input
and independent output tables, D256, same corpus/tokenizer, inherited LR 0.0025,
optimizer, clipping and decay. Preserve 1,024 targets/update, the paired flattened
sampling stream, final-third time cooldown and inherited physical-time EMA
half-life. Report different update counts and actual tokens/FLOPs/time. Keep
native-context primary scoring and common-128 secondary scoring, RAW diagnostics,
source results and fixed first outputs. A broader probe reused after inspection
must be called development evidence, not untouched confirmation.

Before training, freeze fresh role values, source/runtime bindings, all three
within-size quality rules and the interaction rule with multiplicity control,
scoring inventories, generation policy and the full measured admission. Retain
at least three pairs per size. The initial planning rules are the existing
0.02-nat practical threshold, all-seed sign and 0.05 source-regression bound;
finalization precedes any scientific score. No outcome-dependent reduction to two
sizes is allowed. A prospectively selected two-size fallback, if truly necessary,
would be replication rather than a scaling law.

The complete planning envelope is 259,200 training seconds (72 hours), plus 30%
for evaluation, generation, serialization, failure and timing variability, and
3,600 engineering seconds: 340,560 seconds (94.6 hours). This is an upper planning
allocation pending measured new-shape cost, not a launched scientific tranche.
Storage requires 23,199,221,760 free bytes: 36 retained RAW/EMA endpoints,
1,310,965,760 bytes for active optimizer/EMA restart and atomic overlap, 1 GiB
metadata, 1 GiB engineering endpoints, 2 GiB failure allowance, 10 GiB reserve
and 2 GiB extra headroom. Existing retained scientific outputs remain charged
through measured free space. Details: `context-transfer-planning-cost-v2.json`. The metadata allowance was increased
from 512 MiB after measuring 250,979,857 bytes for the old six-run context panel.
The original planning estimate is retained.

Exact audited cleanup retired 22 obsolete accepted engineering payloads,
recovering 5,833,396,224 allocated bytes while verifying 242 protected files.
The original closure-reader schema failure is retained separately; it deleted
nothing. Cleanup and the revised complete forecast now leave more than 0.7 GB
beyond all allowances and the 12 GiB reserve/headroom. This resolves the planning
capacity deficit. New-shape memory and timing remain unqualified.

Next executable action: implement an isolated native combined depth/context wrapper. Verify parity against both qualified paths,
independent initialization, new context-256 shapes at depths 8 and 16, complete
time-EMA restart, endpoint load/scoring and timing/memory under the shared lock.
Reuse existing math and tests where the contract is unchanged. No old scientific
panel or failed numerical qualification is relaunched.

Evidence: `tokenizer-evidence/general-rsi-cross-size-attribution-001/analysis.json`
and its verification, plus the unchanged cross-size/context completion audits.
The analysis reads saved scores and step records only. The original test remains
unopened. All historical scientific decisions remain unchanged.
