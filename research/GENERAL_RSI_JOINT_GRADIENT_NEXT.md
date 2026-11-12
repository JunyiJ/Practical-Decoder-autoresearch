# Joint-gradient integration and diagnostic, 2026-11-12

The next hypothesis is that combining CE and pseudo gradients before clipping
and one optimizer advance can reduce the damage from separate updates. The
accumulated-state diagnostic found both an extra-advance effect and an additional
pseudo-gradient effect. Neither finding establishes that joint gradients help.
All900 CE gradients were clipped, while only5/297 pseudo gradients were clipped;
loss scaling and clipping therefore need explicit controls.

Local evidence is `GENERAL_RSI_PSEUDO_ACCUMULATED_RESULTS.md` and
`tokenizer-evidence/general-rsi-pseudo-accumulated-001/analysis.json`. The closed
quality panel still favors ordinary CE by0.144615nats/token. The earlier one-step
diagnostic had mixed pseudo-minus-zero signs, and ordinary CE itself drifted
upward from selected parents in the100-update diagnostic. These contrary findings
limit a simple mechanism claim and argue for a bounded test before confirmation.

## Literature and changed implementation

The inspected [DITTO trainer](https://github.com/Jxu-Thu/DITTO/blob/35f5a22cd94a1b35c0a57b1ccb1b74186d2bab1d/fairseq/trainer.py)
accumulates backward passes before one normalized/clipped optimizer advance.
Its released script uses six minibatches, NAG and a random CE/pseudo choice.
Exact source hashes and line receipts are retained in
`general-rsi-pseudo-accumulated-001/released-implementation-review.json`.
This supplies a mechanism to examine; it is not evidence for a local gain.
Our deterministic two-component mixture, Muon/AdamW, context128, precision and
valid-target normalization differ from the released setting.

The [supplement, sectionC](https://proceedings.neurips.cc/paper_files/paper/2022/file/148c0aeea1c5da82f4fa86a09d4190da-Supplemental-Conference.pdf)
reports weaker alternative penalties and discusses over-penalizing necessary
repetition. Preserve the current pseudo loss and its0.5 previous-copy coefficient;
do not add an alternative-loss or coefficient sweep. This new recipe does not
change the scientific status of any earlier negative result.

## Fixed three-arm definition

- `ce_full`: one fresh natural batch, ordinary mean CE, one clip and optimizer step.
- `ce_half`: the same natural-batch stream with0.5×mean CE, one clip and step.
- `ce_pseudo_mean`:0.5×mean CE plus0.5×mean pseudo loss. Different fresh natural
  and synthetic batches, evaluated at the same weights. Accumulate both backward
  passes, then clip once and advance the same optimizer once.

CE averages its1,024 targets. Pseudo averages its qualified eligible target mask.
The halves weight component means, not a pooled count of their unequal target
totals. `ce_half` controls the global scaling change; joint-minus-half isolates
adding the pseudo component under this one-step policy, while joint-minus-full
measures the complete recipe. Report the half-minus-full clipping effect as well.
Neither contrast isolates every nonlinear optimizer interaction.

All arms preserve the parent loader, model, LR0.00025, clipping1.0, optimizer
grouping, independent decay0.00006 and original samplers. No CE minibatch is
reused for multiple updates. The joint arm consumes a new pseudo batch each step.
No batch-reuse optimization arm is introduced.

## New bounded integration qualification

The new requirements are weighted accumulation before clipping and persistence
of both component counts/samplers per update. Existing model loading, scalar
pseudo loss, document scoring and original optimizer qualification are reused.
No old fixture is rerun as a new scientific attempt.

1. At one fresh parent state, independently differentiate unweighted CE and
   pseudo components. Compare each production arm's pre-clip gradient to its
   specified weighted sum for every parameter. Freeze rtol1e−5, atol1e−7 for
   this integration check; save exact first failing tensors if it fails. This
   checks the new accumulation wiring. It is not the old FP64 arithmetic test
   and is not a quality-adoption threshold.
2. Run24 synchronized training seconds per arm with a short time schedule and
   EMA half-life1s. Save the first averaged complete state before the budget
   endpoint, continue, then restore into a fresh context. Replay the original
   recorded update durations. Require exact complete final state agreement.
   Include model, both optimizer children, CPU/MPS RNG, data/pseudo samplers,
   histories, component counts, clock and EMA. Reject corrupted component counts
   before restoring state. These are checks of the new execution path.
3. Record finite gradients/loss/state, shared natural input prefixes, actual
   timing and serialized sizes. Do not select an arm from qualification losses.

One local worker,600s whole-worker ceiling, RSS4GiB, sampled MPS6GiB and10GiB
reserve. Plan3 retained restart files capped512MiB each, one256MiB possible
gradient-failure payload,512MiB serialization overlap and64MiB metadata. Require
another2GiB free headroom at admission. The measured prior complete restart was
about445MB; the new count/weight fields add only metadata, but actual sizes must
be recorded. Keep failures, original sources and all qualification outputs.
No validation, test or generation scoring is part of qualification.

## Next scientific diagnostic, not yet admitted

After integration passes, freeze fresh roles for all three selected parents and
all three arms. Use the100-natural-update horizon supported by the original onset
logs, fixed disjoint training windows, and explicit repetition measurements to
check whether improved natural loss merely removes the intended penalty effect.
Report all paired curves and clipping behavior. Joint improvement versus half CE
would support a benefit beyond scaling; a joint regression or loss of repetition
suppression would oppose the proposed mechanism. Mixed outcomes remain unresolved.

Use the measured new update/serialization costs to admit the complete diagnostic.
No additional quality panel or new model adoption follows from qualification.
Any later quality claim requires a separate frozen fresh confirmation and fixed
generation evaluation. Do not pool old scores into new confirmation or change
controls after launch.
