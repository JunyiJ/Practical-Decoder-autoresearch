# Accepted table geometry: descriptive analysis

The read-only analysis completed in 1.942 seconds with peak RSS 536,363,008 bytes. It inspected all three accepted matched/shared H200 pairs. No model was constructed, no initializer replayed, no training or scoring performed, and CPU RNG state was unchanged. The plan, input hashes and limitations are recorded in plan.json and result.json.

Q is the independent matched input table, W its output table, and S the shared table. Stored tables use common output coordinates; effective inputs are Q/k and S/k, where k = 1/sqrt(192). Values below are validation-target-weighted means within each training-frequency bin, then equally averaged over the three seeds. Higher cosine means closer directions.

| Training occurrences | Vocabulary types | Validation targets | cos(Q,W) | cos(S,Q) | cos(S,W) | Input norm S/Q |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 39604 | 957 | 0.446954 | 0.442928 | 0.984901 | 2.151261 |
| 1-9 | 8329 | 2846 | 0.521985 | 0.652833 | 0.854807 | 1.369017 |
| 10-99 | 1982 | 6892 | 0.565771 | 0.732696 | 0.858196 | 1.379991 |
| 100-999 | 309 | 8788 | 0.588016 | 0.720929 | 0.903985 | 1.629517 |
| 1000+ | 33 | 14795 | 0.472350 | 0.624832 | 0.922725 | 2.114980 |

The shared table is closer to the matched output table than to the matched input table in every bin. This is consistent with output-role influence, but endpoint geometry does not measure gradient competition or establish a cause of the quality loss. Cross-model coordinates can evolve differently even from shared initialization.

The earlier saved-loss diagnostic found that targets seen 1–9 times contribute 67.9% of the net shared-minus-matched penalty. However, those rows do not have the largest input/output mismatch or input norm inflation here. Unseen and very frequent rows show larger discrepancies. This weakens a simple rare-specific endpoint-distortion explanation; it does not rule out contextual or optimization mechanisms.

Vocabulary-type summaries in result.json include many rows absent from validation. They are distinct from the target-weighted population shown here. These exploratory post-result comparisons do not supply subgroup significance, functional equivalence, or a new adoption criterion.

Decision: no automatic adapter or input-scale experiment is justified by this analysis. The sharing panel remains closed without adoption and the byte incumbent remains unchanged. The next evidence question is whether an inexpensive context-side output transform has a justified, genuinely new application to the byte incumbent, considering original mixture evidence, contrary findings and closed projection restrictions. No candidate or numerical panel is admitted by this note.
