# Pseudo-repetition continuation progress

The frozen six-run panel is incomplete. No quality decision is available.
The first two pairs challenge the targeted-objective hypothesis: the candidates
regress on validation CE. Complete all three pairs and the fixed generation
evaluation before the frozen decision. Test remains unopened.

| Slot | Parent | Recipe | Updates | Training seconds | Time-EMA CE | RAW CE | First outputs |
|---|---|---|---:|---:|---:|---:|---:|
|00|Native-horizon01|Ordinary CE|3,493|1,200.128546|4.8625809893|4.8875414003|24|
|01|Native-horizon01|CE / pseudo|3,464|1,200.285049|5.0042306204|5.0462512488|24|
|02|Native-horizon03|Ordinary CE|3,506|1,200.290195|4.8514456982|4.8766929991|24|
|03|Native-horizon03|CE / pseudo|3,467|1,200.328832|5.0067426033|5.0480475979|24|
|04|Native-horizon05|Ordinary CE|3,512|1,200.089785|4.8690099254|4.8923741118|24|

Slot00 passed the bound execution review in
`tokenizer-evidence/general-rsi-pseudo-repetition-001/slot-00/review.json`.
It scored all108 validation documents/100,416 targets and retained both final
weights. Guarded worker time was1,308.106422s, with no failure or recovery charge.
The exact audited transient restart retirement completed after final publication.
Generation ratings remain pending the full blind panel. A single-arm endpoint
is not evidence for either recipe's superiority.

Slot01 also passed execution review, including exact agreement of its1,732
ordinary CE batch hashes with the control prefix. Its1,732 pseudo updates used
1,161,507 valid penalty targets, with no saturation. Guarded time was1,308.842068s,
with no failure or recovery charge. Both final weights and all24 first outputs
remain retained. The candidate-minus-control deltas are+0.1416496312time-EMA and
+0.1587098485RAW nats/token. These are descriptive first-pair results, not the
complete three-pair decision. Do not repair or retune a valid quality regression.

Slot02 passed execution review with3,506CE updates/3,590,144 natural targets.
Guarded time was1,307.635223s with no failure or recovery charge. Both final
weights and24 first outputs remain retained. Its paired candidate is pending.
The remaining complete storage requirement is18,374,922,718bytes after accounting
for retained endpoints, versus20,684,509,184 actual free bytes at this boundary.

Slot03 passed execution review, including its1,734 ordinary-CE batch hashes
against the paired control prefix. Its1,733 pseudo updates used1,160,520 valid
penalty targets with no saturation. Guarded time was1,308.216052s with no
failure or recovery charge. Both final weights and all24 first outputs remain
retained. Second-pair candidate-minus-control deltas are+0.1552969051time-EMA
and+0.1713545988RAW nats/token. Both completed pairs regress; this is valid
quality evidence. The full three-pair decision and generation ratings remain pending.

Remaining complete storage requirement is18,146,924,456bytes, compared with
20,443,299,840actual free bytes after four accepted runs and96 first outputs.
Slot04 passed execution review with3,512CE updates/3,596,288 natural targets.
Guarded time was1,306.766926s with no failure or recovery charge. Both final
weights and24 first outputs remain retained. Five runs now retain120 first
outputs and10 final weights. Remaining complete storage requirement is
17,918,926,258bytes, versus20,188,889,088 actual free bytes at this boundary.
Next execute final paired slot05 from parent05.
Keep the frozen objective, roles, resources and acceptance rules unchanged.
