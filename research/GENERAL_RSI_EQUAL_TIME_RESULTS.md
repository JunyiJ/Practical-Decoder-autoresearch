# Equal Mac training-time comparison

All four screens passed review. The frozen rule selects LR0.0025 for both arms.
`tokenizer-evidence/general-rsi-equal-time-001/selection.json` binds the choice to
the original plan. No confirmed architecture decision has been made.

| Slot | Arm | LR | Updates | Training seconds | Time-EMA CE | RAW CE |
| --- | --- | --- | --- | --- | --- | --- |
| 00 | Native | 0.00125 | 5,338 | 1,800.200685 | 5.6160593260 | 5.6100533674 |
| 01 | Byte | 0.00125 | 2,975 | 1,800.220744 | 5.9043330308 | 5.8894565185 |
| 02 | Native | 0.0025 | 5,339 | 1,800.299251 | 5.4689027207 | 5.4637743363 |
| 03 | Byte | 0.0025 | 2,975 | 1,800.479829 | 5.7425720106 | 5.7293537405 |

All screens passed source/weight hashes, durable time accounting, validation
coverage and resource guards. Both paired sampler prefixes match. Each overshoot
was within one update. All eight final weights are retained; transient restarts
were retired with exact audited receipts. Guarded times were1,866.154,1,917.650,
1,866.499 and1,918.304 seconds.

Native leads by0.273669 nats/token at the selected screen LR. The result supports
the throughput hypothesis for this screening horizon. It does not establish
superiority at the longer confirmation horizon or better generated text.

Fresh native confirmation slot04 passed execution review. It completed21,348
updates in7,200.237572257 training seconds, with a0.237572257-second overshoot
within one update. Time-EMA CE is4.975752952122 and RAW CE is4.984698500579.
Time-EMA source CEs are4.961011621457 for FineWeb-Edu and5.034796162385 for
Wikipedia. Guarded worker time was7,380.481886 seconds. All24 first generations
and both final weights are retained. No failure or recovery charge occurred.

Paired byte confirmation slot05 also passed review. It completed11,919 updates
in7,200.587296762 training seconds, with a0.587296762-second overshoot within one
update. Time-EMA CE is5.078865061978 and RAW CE is5.083875641980. Guarded worker
time was7,449.936635 seconds. Its24 first generations and both weights are retained.
The paired training-sampler prefix matches. Neither arm needed failure recovery.

| Fresh pair | Native time-EMA CE | Byte time-EMA CE | Byte minus native | RAW difference |
| --- | --- | --- | --- | --- |
| 1 | 4.975752952122 | 5.078865061978 | +0.103112109855 | +0.099177141401 |

Native has lower loss on both sources: byte-minus-native differences are
+0.104245076918 for FineWeb-Edu and+0.098574255435 for Wikipedia. This first pair
supports the throughput hypothesis at the confirmation horizon. Two further
fresh pairs are required before applying the frozen acceptance rule. These
results do not establish better generated text. All48 scientific first outputs
remain unannotated until the complete144-output blind evaluation is assembled.

Fresh native confirmation slot06, bundle2, is running under PID5479 and
supervisor session5559. Each arm receives7,200
training seconds with budget-matched cooldown and a shared127.184813977-second
EMA half-life. Preserve all24 first completions per final averaged model and
exclude the48 engineering completions. Keep test unopened and all gates fixed.
