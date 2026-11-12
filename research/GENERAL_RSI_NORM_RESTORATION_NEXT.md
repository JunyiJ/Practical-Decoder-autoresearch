# Retained-weight norm-restoration diagnostic

## Evidence and scope

`GENERAL_RSI_CE_DRIFT_RESULTS.md` finds persistent CE-only continuation damage
against the starting averaged parents, including full validation and both sources.
Raw group norms end near83.3–83.5% of their parent norms. This identifies a
radial displacement worth testing; it does not establish decay as its cause.
Gradient updates, optimizer restart and movement away from an averaged point
remain alternatives. The original continuation recipe is not an implementation
failure. Preserve every closed comparison and original coefficient.

Contrary local evidence is important. The later fixed-LR Shakespeare decay study,
`tokenizer-evidence/decay-preparation-001/quality001-closure.json`, found zero-minus-
decay CE+0.009165 in all three seeds and retained0.00006. The earlier weak/strong
study also retained its incumbent. Those were fresh training recipes, not this
low-LR continuation from already-trained averaged general-corpus weights. Neither
justifies repeating a closed decay panel or assuming zero decay would help now.

Loshchilov and Hutter, [arXiv:1711.05101v3](https://arxiv.org/html/1711.05101v3),
section2/Algorithm2 schedules both the gradient and decay terms; AppendixB.1
notes budget-dependent decay and cautions that its normalization is only one
possibility. This motivates auditing our fixed shrinkage under cooldown, not
importing their image-classification gains or identifying an optimal coefficient.

Izmailov et al., [arXiv:1803.05407v3](https://arxiv.org/html/1803.05407v3),
sections3.1–3.2 distinguish averaged weights from individual trajectory points.
Their SGD/SWA image setting differs from our Muon/AdamW and time EMA. It supplies
an alternative explanation, not proof that averaging explains our regression.

## Next minimal test

Use the three selected parent inference weights and all six retained CE-only
continuation endpoints (RAW and time EMA). Load read-only CPU copies, hash-check
receipts and preserve immutable originals. Split each endpoint's parameters into
`lm_head.weight` and all other trainable parameters. The readout is bias-free.
For each group compute s = parent group L2 norm / endpoint group L2 norm in CPU
FP64. These six pairs of positive factors are mechanical measurements from
weights, not values chosen using loss. Freeze them before scoring.

Evaluate four conditions for each endpoint: original, readout-only rescaled,
other-parameters-only rescaled, and both groups rescaled. Restore the exact
original tensors before each condition; never compound rescalings. Apply a
uniform group factor, with no per-tensor fitting or search. Add the unchanged
parent reference. Total27 model-condition scoring passes, serial execution.
Changing a bias-free readout by a positive scalar changes logit scale and preserves
ideal-arithmetic argmax at a fixed hidden state. Thus a readout-only CE effect is
a calibration clue, not evidence of better greedy repetition or hidden features.

Before any scores, freeze64 fresh document-local128-target training windows and
all27 conditions, endpoint bindings, factors, order and aggregation. Use the same
8,192 targets for all conditions/parents, the qualified corrected-FP32 scorer,
and retain all first window losses. These are previously trained-on data, not a
holdout. Do not score validation/test or generate new text in this diagnostic.

Prediction: if radial/readout change contributes to the endpoint loss penalty,
mechanical norm restoration should lower CE relative to that endpoint. Separate
readout, other-parameter and interaction effects in every parent and endpoint.
A readout-only gain supports calibration as a component; an other-parameter gain
supports a broader functional effect. No gain, inconsistent signs, or worse loss
opposes the simple norm-restoration account. Report parent-relative residual gaps.
No result proves decay caused the original displacement. No best-factor choice,
training-recipe change or model adoption follows from this diagnostic.

Rank this ahead of a fresh decay training intervention because it tests functional
sensitivity directly from retained artifacts at much lower cost. Defer optimizer-
restart/averaging interventions until this alternative is discriminated. A later
training comparison needs its own fresh roles, controls, validation/generation
rules and complete confirmation forecast; none is admitted here.

## Qualification and cost before admission

Only new grouping/factor application and exact original-weight restoration need
small checks. Reuse qualified model loading and scoring. Verify the readout-only
logit-scaling relation on one frozen training case, finite states and unchanged
CPU source hashes. Do not repeat model, optimizer or restart qualifications.
Freeze full factors/input inventory and runtime/source hashes before launch.

Provisional cost:27×8,192/100,416×22.064≈48.6s of measured natural scoring cost,
90s loading allowance,30s scaling/restoration/checks and20s binding/log overhead.
Use a600s ceiling after measured admission, RSS4GiB and sampled MPS6GiB. No
checkpoint outputs;64MiB retained evidence plus64MiB overlap,10GiB reserve and
2GiB extra admission headroom. Recheck actual free space and the nonblocking
shared lock; one worker only. This is a prospective design, not yet admitted.
