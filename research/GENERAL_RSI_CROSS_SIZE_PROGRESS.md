# Cross-size execution progress

The first paired seed bundle across depths4/8/16 passed execution review.
Eight runs have now passed execution review. Each completed10,240updates and
342EMA snapshots. All16 final weights and192first outputs are retained. Every endpoint covers100,416development targets
and32,768broader-probe targets. Semantic ratings remain pending.

| Depth | Native H200 CE | Byte H200 CE | Byte minus native | Byte/native training time | Parameter saving |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 5.196423356 | 5.146958298 | -0.049465058 | 1.840× | 42.61% |
| 8 | 5.120939116 | 5.102819586 | -0.018119530 | 1.648× | 38.85% |
| 16 | 5.064510033 | 5.050969629 | -0.013540404 | 1.441× | 33.01% |

These are descriptive results from one paired seed bundle. Both larger-size
advantages are below the frozen0.02practical magnitude threshold. The descriptive
16-minus-4 interaction is+0.035924654; the secondary8-minus-4 interaction
is+0.031345528. The ratio of depth16 to depth4 runtime penalties is0.782845.
Both families improve validation loss as depth increases in this bundle.

The shrinking byte quality advantage and runtime penalty support the dilution
hypothesis descriptively. Three-bundle confirmation and the frozen interaction
and transfer decisions remain unresolved. This is not a scaling law or a
population-seed estimate. Keep all fixed comparisons and criteria unchanged.

| Slot | Arm | Depth | Training seconds | Worker seconds | RAW CE |
| --- | --- | ---: | ---: | ---: | ---: |
| 00 | native | 4 | 3464.333 | 3615.742 | 5.200483851 |
| 01 | raw_bytes | 4 | 6375.625 | 6719.651 | 5.154076443 |
| 02 | raw_bytes | 8 | 7275.498 | 7663.806 | 5.111235728 |
| 03 | native | 8 | 4414.755 | 4611.317 | 5.126989868 |
| 04 | native | 16 | 6259.661 | 6538.401 | 5.070047314 |
| 05 | raw_bytes | 16 | 9018.402 | 9432.577 | 5.058856420 |
| 06 | native | 8 | 4449.442 | 4663.488 | 5.127501022 |
| 07 | raw_bytes | 8 | 7133.353 | 7467.710 | 5.097987630 |

The second depth8 pair also favors byte. Native H200 CE is5.124055823 and
byte H200 CE is5.093561120, a difference of-0.030494703. Byte training takes
1.603× native time, compared with1.648× in bundle1. Both depth8 differences
favor byte, but their magnitudes straddle0.02. The third pair and second-bundle
depths4/16 remain pending. This supports a recurring depth8 benefit without
resolving the frozen quality, transfer or cross-size interaction criteria.

Evidence: `tokenizer-evidence/general-rsi-cross-size-quality-001/slot-00/review.json`
through `slot-07/review.json`, plus `slot-07-completion-and-next-admission.json`.
Remaining-panel admission passes: 19,338,417,174bytes required and
19,518,595,072bytes available. This includes12GiB reserve/headroom and all
retained outputs, restart, serialization, metadata and failure allowances. The
conservative total forecast is42.749hours against the frozen54-hour tranche.

Slot08, depth16 byte bundle2, is running after full-panel admission. The single
worker holds the shared lock and its sampler prefix matches bundle2. Review
its completed execution, then admit fixed slot09, depth16 native bundle2, after
live full-panel checks. Do not repeat prerequisites or reopen the test.
