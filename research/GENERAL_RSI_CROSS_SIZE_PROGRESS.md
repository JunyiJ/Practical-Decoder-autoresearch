# Cross-size execution progress

The first paired seed bundle across depths4/8/16 passed execution review.
Nine runs have now passed execution review. Each completed10,240updates and
342EMA snapshots. All18 final weights and216first outputs are retained. Every endpoint covers100,416development targets
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
| 08 | raw_bytes | 16 | 9125.205 | 9603.407 | 5.066292152 |

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

Slot08, depth16 byte bundle2, passed execution review. Its H200 CE is5.061700673
and RAW CE is5.066292152. The second depth16 contrast remains unpaired until
slot09 completes, so this result does not resolve the dilution hypothesis.
All18final endpoints and216first outputs are retained. Fixed slot09, depth16
native bundle2, is running after full-panel admission. The sole worker holds
the shared lock and its sampler prefix matches slot08. Review its completion
and the second depth16 pair before admitting slot10, depth4 byte bundle2.
Do not repeat prerequisites or reopen the test.

Storage recovery at 2026-11-14T05:03:32.738770+00:00: a new free-space decline left a projected full-panel
deficit. Exact external-cache cleanup removed2525files and recovered114,556,928
allocated bytes, with71protected hashes verified. Free space is now18279223296bytes
against19088137381bytes remaining after one250279793byte current-restart credit.
The remaining deficit is808914085bytes. The12GiB reserve/headroom,
serialization overlap, future endpoints, metadata and failure allowance remain
included. The active worker stays above its reserve and remains untouched.
Three obsolete accepted projection-screen parents are audited for retirement,
covering1,353,510,912allocated bytes. The study is closed with no selection,
adoption or fresh confirmation. All H200 weights, failed-slot payloads and
original failure/source/log/config/metric/receipt evidence remain preserved.
Latest human cleanup authorization narrowly supersedes older retain-all wording.
Execute the exact manifest after slot08 releases the shared lock, then remeasure
capacity before slot09. Do not repeat unchanged inventories. Scientific hypotheses
and frozen criteria are unchanged. Evidence: `storage-cleanup-2026-11-13-active-cross-size-001/reconciliation.json`.

Recovery completed at 2026-11-14T06:05:58.164650+00:00. The exact three-parent deletion journal confirms
1,353,510,912allocated bytes removed, zero skips and312protected hashes verified.
This resolves the preceding pending cleanup. Remaining-panel admission passes:
19669118976bytes free versus19141124524bytes required, a527994452byte margin after the
12GiB reserve/headroom and all output/failure allowances. No transient restart
credit is used. Conservative total time is41.568hours against54hours.
Evidence: `storage-cleanup-2026-11-13-active-cross-size-001/projection-parent-completed.json`
and `tokenizer-evidence/general-rsi-cross-size-quality-001/slot-08-completion-and-next-admission.json`.
