# Equal Mac training-time comparison

Frozen authority: `tokenizer-evidence/general-rsi-equal-time-001/plan.json`.
Four LR screens receive30 minutes per arm/candidate. Three fresh confirmation
pairs then receive two hours per arm, with budget-matched cooldown and a shared
127.184813977-second EMA half-life. Test remains unopened.

| Slot | Arm | LR | Updates | Training seconds | Time-EMA CE | RAW CE |
| --- | --- | --- | --- | --- | --- | --- |
| 00 | Native | 0.00125 | 5,338 | 1,800.200685 | 5.6160593260 | 5.6100533674 |
| 01 | Byte | 0.00125 | 2,975 | 1,800.220744 | 5.9043330308 | 5.8894565185 |

Both slots passed source/weight hashes, durable time accounting, validation
coverage and resource guards. The paired training sampler prefix matches.
Guarded wall times were1,866.154 and1,917.650 seconds. Both overshoots were within
one update. All four final weights are retained. Transient restarts were retired
with exact audited receipts.

Native leads by0.288274 nats/token at this single screen LR and used the time
for more updates. This supports the throughput hypothesis at this setting only.
The frozen LR0.0025 screens and three fresh confirmations remain required;
no LR selection or confirmed architecture conclusion yet.

Slot02, native LR0.0025, is running under worker PID91858 and supervisor session96547. Next: review its endpoint, then launch slot03 byte LR0.0025.
The48 first engineering generations remain separate from the144 planned first
scientific confirmation completions.
