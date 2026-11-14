# Cross-size execution progress

The first paired seed bundle across depths4/8/16 passed execution review.
Thirteen runs have now passed execution review. Each completed10,240updates and
342EMA snapshots. All26 final weights and312first outputs are retained. Every endpoint covers100,416development targets
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
| 09 | native | 16 | 6333.691 | 6613.158 | 5.059108271 |
| 10 | raw_bytes | 4 | 6507.495 | 6856.893 | 5.150909192 |
| 11 | native | 4 | 3527.550 | 3678.337 | 5.209806629 |
| 12 | native | 16 | 6316.000 | 6595.571 | 5.057882101 |

The second depth8 pair also favors byte. Native H200 CE is5.124055823 and
byte H200 CE is5.093561120, a difference of-0.030494703. Byte training takes
1.603× native time, compared with1.648× in bundle1. Both depth8 differences
favor byte, but their magnitudes straddle0.02. The third depth8 pair remains
pending. This supports a recurring depth8 benefit without
resolving the frozen quality, transfer or cross-size interaction criteria.

The second depth16 pair reverses the first pair's direction. Native H200 CE is
5.055027309 and byte H200 CE is5.061700673, a byte-minus-native difference of
+0.006673365. RAW also favors native by0.007183881. Byte training takes1.440740×
native time, close to1.440717× in bundle1. The first H200 difference was
-0.013540404. These mixed seed signs preclude either arm from meeting the frozen
every-seed sign requirement at depth16. A consistent larger-size byte quality
benefit is therefore unsupported by these paired results. The full three-bundle
interaction and remaining size decisions are pending. Keep all runs and gates
unchanged. These are execution-reviewed descriptive results, not a completed
panel decision or scaling law.

Slots10–11 complete the second depth4 pair. Native H200 CE is5.207382848
and byte H200 CE is5.146265301, a difference of-0.061117546.
Byte training takes1.844763× native time. The second bundle now spans all three
depths. Its16-minus-4 interaction is+0.067790911, compared with+0.035924654
in bundle1. Its depth16/depth4 runtime-penalty ratio is0.780989. Both completed
bundles descriptively support dilution of byte's quality advantage and runtime
penalty. The mixed signs at depth16 remain. The third bundle and frozen
multiplicity-adjusted quality and interaction decisions remain pending.

Slot12, depth16 native bundle3, passed execution review. H200 CE is5.054405951
and RAW CE is5.057882101. Paired byte slot13 is pending. This unpaired
result does not resolve the third depth16 contrast or the full interaction.

Evidence: `tokenizer-evidence/general-rsi-cross-size-quality-001/slot-00/review.json`
through `slot-12/review.json`, plus `slot-12-completion-and-next-admission.json`.
Remaining-panel admission passes: 18193430592bytes required and18628456448bytes available,
a435025856byte margin. This includes12GiB reserve/headroom and all retained
outputs, restart, serialization, metadata and failure allowances. Conservative
total time is37.700hours against the frozen54-hour tranche.

All26final endpoints and312first outputs are retained. Fixed slot13, depth16
byte bundle3, is running under the shared lock. Its sampler prefix matches
paired slot12 exactly. Live admission passed with18627198976bytes free against
18193430592bytes required, a433768384byte margin. After execution review, admit
fixed slot14 depth4 native bundle3. Finish all remaining runs unchanged before
the frozen analysis and generation ratings.

Slot11 preflight attempt01 failed before training because its receipt basename
still named slot10. Exclusive creation preserved the existing slot10 receipt;
no scientific run started. Attempt02 changed only the receipt basename, passed
the complete admission checks and launched slot11 with unchanged bound sources,
roles and schedules. Both preflight sources and the original failure are retained
in `tokenizer-evidence/general-rsi-cross-size-quality-001/prelaunch-11-failure-01/`.
Live verification is in `slot-11/launch-verification.json` under the same panel.

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
