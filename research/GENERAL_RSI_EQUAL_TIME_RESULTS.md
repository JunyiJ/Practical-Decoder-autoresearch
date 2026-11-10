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

Fresh native confirmation slot04 is running under worker PID94793 and
supervisor session69399. Next: review its endpoint and all24 first generations. Three fresh native/byte pairs each
receive7,200 training seconds per arm, with budget-matched cooldown and a shared
127.184813977-second EMA half-life. Generate24 first completions from each final
averaged model for144 scientific completions total. The48 engineering completions
remain excluded. Keep the original test unopened and all frozen gates unchanged.
