# Active research: deterministic byte-embedding recovery

The human authorized autonomous research and recovery on November 7, 2026.
The native incumbent is unchanged. Byte composition has no measured real-data
quality gain and has not been adopted.

The prior model-v2 finite integration proof remains accepted: six full-size
B8T128/V50257 CPU/MPS cases, 357 rejection transactions and 1,412,427,220 checked
coordinates. Its [saved review](tokenizer-evidence/byte-composition-boundary-001/model-saved-review004/review.json)
and all original sources/evidence remain closed and preserved.

## New checkpoint evidence and failure

Checkpoint attempt001 completed native CPU, then stopped while constructing a
raw-byte alias-rejection fixture: the smallest composer parameter has no
same-shaped peer. The adapter was not called for that failing fixture. A
[saved-inventory diagnosis](tokenizer-evidence/byte-composition-boundary-001/state-fixture-diagnosis001/diagnosis.json)
justified selecting an actual same-shaped pair. Separately identified attempt002
used fresh roles, unchanged model/state sources, all six cases, all 234 rejection
requirements and exact resume gates.

Attempt002 completed native/raw-byte/nonlexical CPU and native MPS. In raw-byte
MPS, saved parameters, optimizer state, training mode, LR and RNG restored
exactly. Next RNG draws, logits and loss also matched. Exactly one next-gradient
tensor differed: `composer.byte_weight`. The exact gate failed and the worker
stopped after 187.352 seconds; peak RSS was 3,349,708,800 bytes under 4 GiB.
Nonlexical MPS was not reached. This is not full checkpoint acceptance and is
not a negative scientific quality result. All failure snapshots and archives
remain retained. Original001 and002 must never be replayed.

## Isolated diagnosis and implemented recovery

A new lookup-only primitive diagnosis used retained weights and explicit
upstream arrays, without calling a historical model, backward fixture or
optimizer. The first small binary-fraction upstream had exactly representable
sums and was repeatable in both paths; it is retained as an exact-sum control.
A separately identified mixed-magnitude upstream reproduced native MPS lookup
nondeterminism: 251 and 259 of 3,885 gradient coordinates differed across repeats,
with maximum repeat difference 5.23869e-9. All remained within the numerical
reference-error bound; the failure concerns exact continuation, not that bound.
The historical model-gradient error magnitude is unknown because that proof
saved full gradient digests rather than gradient values.

A fixed-order grouped reduction was exactly repeatable in the isolated test.
The new [embedding primitive](byte_embedding_v1.py) keeps native forward/CPU
semantics and reduces MPS gradients in ordered 2048-position chunks. Its own
CPU/MPS autograd proof passed two upstream cases, three repeats per backend,
exact forward and repeated-gradient checks, 14 invalid-input transactions and
all 46,620 gradient coordinates against independently reconstructed float64 sums.
Maximum mixed-case MPS reference error was 3.59649e-9. The original primitive
proof001 failed constructing an unsupported MPS FP64 rejection input; proof002
uses supported FP16 to test the same FP32-only requirement and preserves001.

[Independent saved-array and source-delta review](tokenizer-evidence/byte-composition-boundary-001/embedding-recovery-review001/review.json)
accepted the primitive and the exact integration edits. New
[composer-v2](byte_composer_v2.py), [model-v3](byte_input_model_v3/model.py) and
[state-v2](byte_input_state_v2.py) sources use the primitive with distinct
identities. Parameter initialization, shapes, codebooks, native body, readout,
loss and optimizer arithmetic are unchanged. These integrated sources have not
been numerically imported or called. Primitive acceptance does not establish
full model-v3 checkpoint correctness.

## Next executable action

Prepare a fresh checkpoint proof for model-v3/state-v2, bound to the accepted
primitive/source-composition review. Preserve six B8T128 CPU/MPS cases, 234
rejections, empty-state and changed-LR/mode coverage, complete archives, and
exact next forward/gradient/update/RNG equality. Freeze source/runtime/assets,
fresh exclusive roles and complete cost, admit statically, then execute once
and review independently. Do not reopen original proofs or relax exact gates.

No worker is active at this checkpoint. The active autonomous goal owns the
next continuation; paused research automations remain unchanged. About 22.22 GB
is free. The next checkpoint envelope of 8 GiB output + 8 GiB reserve + 2 GiB headroom
requires 19,327,352,832 bytes and is covered. Refresh the memory forecast for the
~2.2 MiB grouped-reduction workspace. No current storage blocker exists. A future
full training/quality panel remains unfrozen and needs its own complete forecast
and authorized capacity recovery if necessary.

Full training resume still needs sampler/data identity, schedule, clipping
history, H200 averaging and worker integration. Quality and adoption remain
unproven. The [current progress receipt](tokenizer-evidence/byte-composition-boundary-001/checkpoint-recovery-handoff001/progress.json)
and [local payload manifest](tokenizer-evidence/byte-composition-boundary-001/checkpoint-recovery-handoff001/local-payload-manifest.json)
bind the retained checkpoint/diagnostic files. Git contains source and evidence
metadata; binary checkpoints and diagnostic arrays remain local. A clone alone
cannot perform every saved-array review.
