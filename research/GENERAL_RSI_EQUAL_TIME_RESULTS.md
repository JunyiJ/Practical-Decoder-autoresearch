# Equal Mac training-time comparison

Frozen authority: `tokenizer-evidence/general-rsi-equal-time-001/plan.json`.
Four LR screens receive30 minutes per arm/candidate. Three fresh confirmation
pairs then receive two hours per arm, with budget-matched cooldown and a shared
127.184813977-second EMA half-life. Test remains unopened.

| Slot | Arm | LR | Updates | Training seconds | Time-EMA CE | RAW CE |
| --- | --- | --- | --- | --- | --- | --- |
| 00 | Native | 0.00125 | 5,338 | 1,800.200685 | 5.6160593260 | 5.6100533674 |

Slot00 passed source/weight hashes, durable time accounting, validation coverage
and resource guards. Total guarded wall time was1,866.154 seconds. Its0.201-second
overshoot was within one update. Both final weights are retained; transient
restarts were retired with exact audited receipts. This is a screen result only.
There is no LR choice or equal-time architecture conclusion yet.

Slot01, byte LR0.00125, is now running under worker PID90326 and supervisor session63005. Next: finish and review its paired sampler prefix and endpoint.
The48 first engineering generations remain separate from the144 planned first
scientific confirmation completions.
