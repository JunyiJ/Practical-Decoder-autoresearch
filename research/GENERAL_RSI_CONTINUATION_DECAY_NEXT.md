# Continuation decay attribution: next bounded diagnostic

## Evidence and alternatives

`GENERAL_RSI_CE_DRIFT_RESULTS.md` establishes persistent CE-only continuation
regression from selected averaged parents. Independent per-step decay stays
0.00006 as the continuation learning rate cools. The new retained-weight test,
`GENERAL_RSI_NORM_RESTORATION_RESULTS.md`, finds readout scaling helps all six
endpoints but leaves positive parent-relative gaps. Scaling other parameters
harms every endpoint, with adverse interactions. Static norm restoration cannot
identify the effect of shrinkage during learning.

The contrary fixed-LR Shakespeare zero-decay result remains closed: all three
fresh seeds lost, mean+0.009165nats/token. See the original
`tokenizer-evidence/decay-preparation-001/quality001-closure.json`. Earlier weak/
strong-decay results also retained their incumbent. The changed question is
short, low-LR continuation from trained averaged general-corpus parents, with
readout and other-parameter decay separated. It is not a replay of fresh
Shakespeare training or permission to adopt zero decay.

Loshchilov and Hutter, [arXiv:1711.05101v3](https://arxiv.org/html/1711.05101v3),
section2/Algorithm2 and AppendixB.1, separate decay from gradient updates and
couple the schedule multiplier to both. These previously read sections support
a direct shrinkage control. Their Adam/SGD image experiments do not prescribe
our Muon coefficient or predict a local gain. Izmailov et al.,
[arXiv:1803.05407v3](https://arxiv.org/html/1803.05407v3), sections3.1–3.2, supply
averaged-point departure as an alternative explanation. Their equal-weight SWA
and image setting differ from our time EMA and text model.

## Hypothesis and discriminating design

Hypothesis: readout shrinkage contributes to early continuation CE drift;
other-parameter shrinkage may have a different effect. Compare the full2×2
factorial of readout decay{0,0.00006} and other-parameter decay{0,0.00006}.
The unchanged all-decay arm is the paired control. Neither coefficient is tuned.
Readout means exactly `lm_head.weight`; every other trainable parameter belongs
to the other group. All child optimizer settings and gradient processing stay
unchanged. Apply any enabled shrinkage before the optimizer update as before.

Use three preserved parents with fresh paired streams. Each of four arms starts
from the same parent, fresh optimizer state and identical100 natural CE updates.
Keep peak LR0.00025, clipping1, batch policy and1200-second schedule unchanged.
All100 updates must precede cooldown/EMA, as in the qualified short diagnostic.
Freeze roles, rotated arm order, source hashes and all inputs before launch.
Use64 fresh training windows per parent, disjoint from its update-source ranges
and common to its four arms. Score the parent and updates1/10/25/50/100 with the
qualified scorer. Total1,200 updates and63 evaluations. No pseudo objective,
validation, test, generation, endpoint weight selection or persistent trained
checkpoint is part of this diagnostic.

Report per-parent main effects and interactions, each arm's parent-relative drift,
clipping and separate readout/other norm changes. The primary diagnostic contrast
is removing readout decay with other decay retained, at update100. Consistently
lower CE and smaller parent-relative drift supports a readout contribution in
this setting; mixed signs or no benefit oppose it. Report magnitudes without
calling three parents population confidence. The other two contrasts distinguish
broader shrinkage and interaction. No diagnostic sign rule constitutes adoption.
If all arms still drift, do not call the best arm an improvement over the parent.

Rank this ahead of optimizer-restart or new objective changes because the current
factorial revealed group-dependent effects but did not intervene on training.
A negative result retires this explanation at the short horizon. A positive
result only motivates a separately frozen longer, fresh confirmation with full
validation and fixed generation evaluation. No such quality panel is admitted.

## New integration and complete provisional cost

Keep frozen optimizer sources unchanged. Implement the smallest separate wrapper
for group-specific decay, retaining canonical children, grouping and validation.
Check the all-decay path against the existing implementation on the same state,
verify the exact two group policies, and qualify the new policy's saved
configuration and full restart/replay. Reuse completed model, scorer and sampler
qualifications; do not repeat them. Freeze a bounded qualification attempt with
actual payload measurements before admitting scientific trajectories.

Provisional diagnostic timing uses measured0.3433seconds/update and approximately
1.65seconds/window-panel score:1,200×0.3433+63×1.65≈516seconds, plus144seconds
for parent/context setup, state checks and evidence. A900-second diagnostic ceiling
must be checked against new measured timing. A separate new-policy qualification
has a600-second ceiling. These are prospective budgets, not launch receipts.

Forecast four complete qualification restart payloads using the measured largest
445,039,285-byte precedent, one active serialization overlap,512MiB failed-write
allowance and128MiB evidence. That is approximately2.90GB incremental, plus10GiB
reserve and2GiB extra headroom. The subsequent diagnostic writes no model payloads.
Replace payload estimates with measured sizes and account for all retained first
attempts before admission. Preserve all existing selected/fresh weights and
original failures. Shared nonblocking lock and one serial worker remain required.
