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

The paired byte confirmation slot05 is running under PID99876 and supervisor
session41607. This single native endpoint cannot
resolve the throughput hypothesis. Review its paired byte result, then complete
all three fresh pairs before applying the frozen decision rule. Each arm receives
7,200 training seconds with budget-matched cooldown and a shared127.184813977-second
EMA half-life. Generate24 first completions per final averaged model for144
scientific completions total. The48 engineering completions remain excluded.
Keep the original test unopened and all frozen gates unchanged.
