# Cross-size execution progress

The first fresh depth4 and depth8 pairs, bundle1, passed execution review.
Each run completed10,240updates and342EMA snapshots. All eight final weights and
96first outputs are retained. Every endpoint covers100,416development targets
and32,768broader-probe targets. Semantic ratings remain pending.

| Depth | Native H200 CE | Byte H200 CE | Byte minus native | Byte/native training time | Parameter saving |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 5.196423356 | 5.146958298 | -0.049465058 | 1.840× | 42.61% |
| 8 | 5.120939116 | 5.102819586 | -0.018119530 | 1.648× | 38.85% |

These are descriptive results from one paired seed bundle. The depth8 difference
of-0.018119530 is below the frozen0.02practical magnitude threshold. The descriptive
8-minus-4 interaction is+0.031345528. The smaller byte quality advantage and runtime
penalty are consistent with dilution as the body grows, but fresh confirmation
and the primary16-minus-4 interaction remain unresolved. Two sizes at one bundle
do not establish a scaling law. Keep all fixed comparisons and criteria unchanged.

| Slot | Arm | Depth | Training seconds | Worker seconds | RAW CE |
| --- | --- | ---: | ---: | ---: | ---: |
| 00 | native | 4 | 3464.333 | 3615.742 | 5.200483851 |
| 01 | raw_bytes | 4 | 6375.625 | 6719.651 | 5.154076443 |
| 02 | raw_bytes | 8 | 7275.498 | 7663.806 | 5.111235728 |
| 03 | native | 8 | 4414.755 | 4611.317 | 5.126989868 |

Evidence: `tokenizer-evidence/general-rsi-cross-size-quality-001/slot-00/review.json`
through `slot-03/review.json`, plus `slot-03-completion-and-next-admission.json`.
Remaining-panel admission passes: 20,233,269,766bytes required and
20,797,624,320bytes available. This includes12GiB reserve/headroom and all
retained outputs, restart, serialization, metadata and failure allowances. The
conservative total forecast is47.177hours against the frozen54-hour tranche.

Next: launch fixed slot04, depth16 native bundle1, after live admission checks.
Do not repeat prerequisites or reopen the test.
