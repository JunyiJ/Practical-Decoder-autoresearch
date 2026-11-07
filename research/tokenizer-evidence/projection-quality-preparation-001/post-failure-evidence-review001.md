# Incomplete projection panel: forward-validation evidence boundary

The original QUALITY001 is CLOSED_FAILED_INCOMPLETE. Read failure-closure001/
review.json, guard.json, STOP and original/ for the exact evidence. Controller
48758/session2476 exited1; worker50942 completed slot07 linear_projection1 at
LR.0025,1200updates/1228800targets/41EMA. Only slots00-06 were independently
accepted. Slots08-17 were never attempted. No LR selection, fresh confirmation,
serving panel, bootstrap or adoption exists. Native incumbent/holdout unchanged.
All1164 frozen sources and16originalRAW/H200 files4520794083B are preserved.
Closure2.049s/RSS25100288B used only metadata and streaming hashes; no tensors,
models, scoring, training, random draws, deletion or checkpoint copies.

## Local observation and missing evidence

Original validation.py computes both RAW full-native score reports, then at line89
requires abs(CPU.CE-MPS.CE)<=3e-5 AND equal target contracts. The parent guard records
this exact exception after the CPU RAW guard finished. The two reports are local
variables, and dispatch publishes accepted.json only after acceptance returns.
Consequently neither report nor the exact failing operand was persisted. H200
scoring and cache acceptance had not started. No rescoring or model replay is
authorized to reconstruct those missing values in this closed command.

Source scoring.target_contract derives coverage from the same CPU IDs and config
without a device field, suggesting CE disagreement is more likely than coverage
disagreement. This is an inference, not a measured failing CE or a proved cause.
The initial alert's claim that the numerical limit was exceeded was too specific;
the durable finding is failure of the combined CE/coverage predicate.

Accepted metadata shows max RAW/H200 CPU-MPS CE differences at nativeLR.0025
slot06 of2.8386214446562974e-5/2.801248809358725e-5, both still below3e-5.
LinearLR.00125 slot04 differences2.363060375376591e-5/2.2289254183682772e-5.
All seven accepted slots passed; these observations cannot identify an operator,
conditioning, precision, clipping or training mechanism. Slot07's462clipped
updates and five sampled CE points are descriptive, not causal evidence. An
incomplete runtime panel is not a negative nonlinear-head quality result.

## Primary documentation and applicability

Actually retrieved PyTorch2.14
[Numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html),
opening paragraphs and Batched computations or slice computations, and
[Reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html),
opening paragraphs. Different backends/reduction orders can produce different
floating point results despite identical inputs and controlled RNG. These are
general framework limits, not evidence of an MPS bug, a specific failing operator,
an acceptable error budget, or permission to weaken the frozen threshold. CUDA
TF32 and FP16 recipes are not an explanation of this FP32 MPS case.

Architecture/attention roadmaps, original protocol and actual model/scorer sources
were inspected. The old failed CPU-MPS optimizer-response proof remains unrelated
and closed; this failure was forward validation after an MPS-trained worker.
No optimizer, depth, LR, initializer, EMA or horizon rescue follows.

## Bounded next work and discriminating test

Select the demonstrated reporting defect ahead of any new architecture panel.
New additive research/projection_validation_pair_v1.py is future-only: persist
both caller-validated finite reports and separate CE/target/coverage booleans to a
new exclusive JSON file, fsync, then enforce the unchanged3e-5 gate. No frozen
module imports it. It does not implement scoring, physical codec validation,
model construction, checkpoint acceptance or a new training campaign.

Falsifiable prediction: NEW literal report pairs that fail CE or coverage leave
both original inputs durably recorded before rejection; equal/within-bound pairs
pass; boundary exactly3e-5 passes; any overwrite/oversized/invalid-finite request
rejects without changing an existing file or caller inputs. Independent standard
library assertions inspect emitted JSON; no old fixture or scores are inputs.
Passing fixes only future observability. Failing closes its original command;
neither outcome diagnoses or resumes QUALITY001.

Next proof scope is CPU-only literal metadata, zero Torch imports/models/tensors/
corpus/checkpoints/seeds/RNG calls. Pin source and exact plan SHA before one run;
120s/4GiBRSS/8GiBreserve/2MiB totalmetadata, each pair<=512KiB; predicted<1s/<64MiB.
No new weights or serialization overlap. Current~42.2GBfree covers this small
scope+reserve+2GiB; no storage blocker or cleanup action. After that, choose a
new evidence-qualified forward-numerics question from sources and genuinely new
generated inputs, with fresh role/cost gates before actual models. No old-score
reconstruction, tolerance change or replacement scientific panel is admitted.
Broad/interaction allocation stays carried forward; this correctness work is
not numerical exploration or quality improvement evidence.
