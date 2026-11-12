# Pseudo-repetition continuation progress

The frozen six-run panel is incomplete. No quality decision is available.
The first pair challenges the targeted-objective hypothesis: the candidate
regresses on validation CE. Complete all three pairs and the fixed generation
evaluation before the frozen decision. Test remains unopened.

| Slot | Parent | Recipe | Updates | Training seconds | Time-EMA CE | RAW CE | First outputs |
|---|---|---|---:|---:|---:|---:|---:|
|00|Native-horizon01|Ordinary CE|3,493|1,200.128546|4.8625809893|4.8875414003|24|
|01|Native-horizon01|CE / pseudo|3,464|1,200.285049|5.0042306204|5.0462512488|24|

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

Next execute slot02 ordinary CE from selected parent03, then paired slot03.
Keep the frozen objective, roles, resources and acceptance rules unchanged.
