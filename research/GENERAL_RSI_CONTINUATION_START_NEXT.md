# RAW versus averaged continuation starts: next diagnostic

## Evidence and question

The completed group-decay diagnostic leaves positive100-update CE drift in all
three no-decay trajectories. Removing shrinkage is insufficient to make those
continuations improve their starting averaged parents. See
`GENERAL_RSI_CONTINUATION_DECAY_RESULTS.md` and its original first probes.
This motivates separating the initial weight choice from the remaining shared
fresh-optimizer and gradient-update effects.

The original long native runs retain both RAW and time-EMA inference weights.
Their existing full-validation RAW-minus-EMA differences are+0.011800680,
+0.014122660 and+0.011536643nats/token for slots1/3/5. RAW starts therefore have
a known disadvantage on that validation set; they are not presumed better models.
The original `latest.json` and final `deletion-journal.jsonl` entries record
retirement of the completed optimizer restart states. Do not claim an exact
warm-optimizer control, reconstruct missing moments, or replay the closed native
training to recover them. Both inference endpoints remain protected.

Izmailov et al., [arXiv:1803.05407v3](https://arxiv.org/html/1803.05407v3),
sections3.1–3.2, distinguish averaged weights from individual trajectory points
and describe further training from pretrained weights. These previously read
sections motivate distinguishing starting points. Their image SGD/SWA gains do
not predict that our time-EMA parent will regress or that a RAW start will help.
The local result may instead reflect fresh optimizer state, learning rate or
sampled-gradient noise. Keep these alternatives explicit.

## Prospective comparison

Use each of the three matched retained native RAW/time-EMA endpoint pairs.
Both arms use the qualified no-decay policy to remove the shrinkage effect just
measured. This is a mechanism control, not adoption of no decay. Keep fresh
optimizer initialization, LR0.00025, clipping1, batch policy and1200-second clock
identical. Compare100 fresh paired natural CE updates; all must precede cooldown
and new averaging. Freeze new seed roles, alternating order and complete inputs
before training. Do not reuse prior diagnostic streams or pool their scores.

Each pair gets64 fresh training windows excluded from its update source ranges.
Evaluate both distinct starting states, then updates1/10/25/50/100 with the same
qualified scorer. Total600 updates and36 evaluations. Keep all first window
scores. No validation, test, generation or checkpoint selection is admitted.

Primary diagnostic contrast at100 updates is
(RAW final CE − RAW initial CE) − (EMA final CE − EMA initial CE).
Negative values in all three pairs support a contribution from initial weight
choice to the observed drift under this fixed continuation policy. Mixed signs
or nonnegative values oppose that account. Always report both absolute endpoint
CEs and gaps to the unchanged EMA parent. Lower relative drift from a worse
initial model is not improvement over the selected parent. The comparison cannot
identify optimizer-state effects, which are shared. No model adoption follows.

This is a new controlled starting-point question, not another decay coefficient
search. Either result narrows the explanation before a longer training proposal.
If all endpoints still trail the selected parents, retain those parents and do
not promote the least damaging continuation into a quality winner.

## Minimal engineering and cost

Reuse the now-qualified no-decay optimizer, full state, clock, sampler and scorer.
Only the RAW starting-weight loader is new: bind its original receipt and plan,
verify model/buffer identity, and compare every initialized parameter exactly to
the retained CPU payload before any update. Record the starting-weight identity
in configuration. No model/optimizer/restart prerequisite repetition is needed
without a demonstrated new requirement. Preserve all frozen source namespaces.

Provisional cost uses the measured slowest0.359second update,1.70second scoring
allowance and90seconds setup/evidence:600×0.359+36×1.70+90≈366.6seconds. Freeze
measured admission under a600-second ceiling, RSS4GiB and MPS6GiB. Retain64MiB
evidence with64MiB overlap,10GiB reserve and2GiB extra headroom. No new model
payloads or scientific confirmation panel are admitted by this design. Recheck
actual worker state, the nonblocking shared lock and complete free-space forecast
before execution. Original first results and all selected/fresh weights stay kept.
