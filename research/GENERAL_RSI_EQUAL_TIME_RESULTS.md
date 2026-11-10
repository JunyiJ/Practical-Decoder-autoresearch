# Equal Mac training-time comparison

Frozen authority: `tokenizer-evidence/general-rsi-equal-time-001/plan.json`.
Four LR screens receive30 minutes per arm/candidate. Three fresh confirmation
pairs then receive two hours per arm, with budget-matched cooldown and a shared
127.184813977-second EMA half-life. Test remains unopened.

| Slot | Arm | LR | Updates | Training seconds | Time-EMA CE | RAW CE |
| --- | --- | --- | --- | --- | --- | --- |
| 00 | Native | 0.00125 | 5,338 | 1,800.200685 | 5.6160593260 | 5.6100533674 |
| 01 | Byte | 0.00125 | 2,975 | 1,800.220744 | 5.9043330308 | 5.8894565185 |
| 02 | Native | 0.0025 | 5,339 | 1,800.299251 | 5.4689027207 | 5.4637743363 |

The first three slots passed source/weight hashes, durable time accounting, validation
coverage and resource guards. The paired training sampler prefix matches.
Guarded wall times were1,866.154,1,917.650 and1,866.499 seconds. All overshoots were within
one update. All six final weights are retained. Transient restarts were retired
with exact audited receipts.

Native leads by0.288274 nats/token at this single screen LR and used the time
for more updates. This supports the throughput hypothesis at this setting only.
The frozen LR0.0025 screens and three fresh confirmations remain required;
no LR selection or confirmed architecture conclusion yet.

Native LR0.0025 improved its screen loss to5.468903 versus5.616059 at0.00125.
Slot03 byte LR0.0025 is running under worker PID93251 and supervisor session88500.
Next: review it, select each arm’s LR after all four accepted screens, and launch
fresh confirmation.
The48 first engineering generations remain separate from the144 planned first
scientific confirmation completions.
