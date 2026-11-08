# Output-anchored table sharing, new MLA question

Human authorization: November 8, 2026, “would a model with shared table be helpful
for quality? let's try out?” This is new work, not a reopening of table-001.

## Evidence and hypothesis

Local evidence: `TOKENIZER_TABLE_RESULTS.md`, `tokenizer-evidence/table-001/`
`final-review.json`, `selection.json`, and original seed3311 metrics show that
the old GQA matched control lost0.102667 nat and tying added0.030508 nat.
All three fresh seeds lost; the closed negative result stands. Those matched/tied
recipes replaced the native output initialization with unit-normal input weights
and multiplied the output by k=1/sqrt(192). They used Adam on those coordinates.
Scaling a parameter under Adam is not generally equivalent to scaling its
gradient: effective output updates, epsilon and shared input dynamics differ.
This is a possible explanation, not an established cause of that loss.

The current four-block MLA native initializer has separate50,257x256 tables:
28,492,416 parameters total. Sharing leaves15,626,624. The adopted byte composer
has16,351,859 parameters and remains incumbent. Its saved scores will not be
pooled into this fresh comparison or treated as a paired baseline.

Primary literature retrieved November8: Press and Wolf, EACL2017 proceedings
version, https://aclanthology.org/E17-2025.pdf, section3 derives the combined
input/output gradient and section4.1/Tables2–3 includes contrary skip-gram results.
Inan et al., arXiv:1611.01462v3, March11,2017,
https://arxiv.org/pdf/1611.01462, section6.3/Table1 reports benefits of reused
embeddings in LSTMs on PTB/Wikitext-2. These motivate shared learning but do not
establish a gain for short-run GPT2-tokenized MLA, Muon/AdamW or this corpus.
Neither paper establishes our output-coordinate hypothesis.

Prediction: preserving native output initialization/coordinates while sharing
may improve validation CE relative to an identically initialized separate-table
control. Failure against that control falsifies the benefit for this fixed recipe.
The native control determines whether the whole recipe helps; the matched control
separates sharing from the changed input initialization/coordinates. Even success
cannot isolate which historical difference explains the old GQA failure.

## Three fixed arms

All constructors consume the full native initialization in the same order.
Let W be its native output table (std k=1/sqrt192), E its input table, h the
Transformer output, and width multiplier4.

- native: input E[token]; output h W^T /4.
- matched: initialize independent Q=W.clone(); input Q[token]/k;
  output h W^T /4. Q and W have independent Adam states.
- tied: initialize Q=W.clone(); input Q[token]/k; output h Q^T /4.
  One canonical Q, one decay, one optimizer state and one averaging shadow.

The last two have identical initial functions. Tying couples gradients and
optimizer history and constrains representational capacity. Input scaling stays
fixed throughout training; it is not a normalized or learned table. Relative to
native, their unit-variance input distribution is uniform rather than Gaussian,
correlated with output, and has different effective input updates. These are
declared recipe differences, not hidden controls.

Keep native MLA body, tokenizer,80/10/10 raw-text split, B8T128 paired512 sampling,
ordinary CE, Muon .5*base LR for44 body matrices; AdamW beta(.9,.999), eps1e-6
for remaining parameters, independent decay.00006 once per unique parameter,
global clipping1.0, no dropout/warmup. Fixed base LR.00125 for all arms, inherited
from native; no tuning or selection. This answers a fixed-recipe question, not
the best possible tied model question.

Nine trajectories: three fresh seed bundles x three arms,1200updates each,
cooldown800–1200 to.1; H200 averaging800..1200 every10=41 snapshots.
Order rotates native/matched/tied, matched/tied/native, tied/native/matched.
Fresh pilot bundle62101700001..5 and scientific bundles62101800001..5,
62101900001..5,62102000001..5; five roles are native initializer, reserved composer,
reserved code key, train sampler, validation sampler. Paired roles within bundle.
Check previous evidence for reuse before freezing; never substitute a losing seed.

## Necessary preflight and fixed acceptance

A single bounded CPU/MPS pilot verifies native outputs/gradients against its
original implementation, matched/tied initial outputs and shared-gradient sum
(atol2e-6,rtol2e-4), unique parameter/optimizer coverage, four updates and H200,
saved-weight roundtrip and shared identity. It does not select hyperparameters
or evaluate validation quality. No unrelated full test suite.

Primary outcome: complete H200 validation CE,34278targets, context128 resets
and unpadded tail; FP32 CE and FP64 aggregation. Save every target loss. Holdout
closed. Diagnostics at300/600/900/1200 use the same fixed full-validation scorer
on online weights; no early stopping or decisions from curves.

For tied-minus-native and tied-minus-matched, report allthree seed deltas and
mean; conditional text bootstrap,2048/4096/8192 contiguous blocks,2000 joint
draws, PCG64 seed62102000006, upper quantile1-.05/6. Quality success requires
mean<-.02, every seed<0 and all upper bounds<0 against BOTH controls. Report
matched-minus-native descriptively. No memory-only adoption and no claim about
the seed population or untouched holdout. A win only qualifies a later direct
comparison with the byte incumbent; this campaign cannot replace it.

## Cost, persistence and recovery

10800 scientific updates,11,059,200 targets;24 pilot updates (4x3x2).
Planning estimate45–90minutes, conservative per-trajectory cap7200seconds.
One worker under shared nonblocking lock. Strict FP32 MPS,4CPUthreads, no fallback,
4GiB RSS/6GiB MPS driver caps. Retain final raw WEIGHTS and H200 weights for
every trajectory, configs, curves, full losses, source/runtime/data and hashes.
No resume is promised; optimizer checkpoints are unnecessary for this completed
fixed-horizon question. Do not delete selected or fresh weights after scoring.

Conservative complete retained allocation2GiB, serialization overlap128MiB,
metadata128MiB, pilot32MiB =2.28125GiB; plus8GiB reserve and2GiB headroom
requires12.28125GiB before dispatch. Native/matched pair weights~228MB and tied
~125MB; allnine~1.74GB. Enforce total output cap2.28125GiB throughout. Source,
protocol, runtime, data bindings and gates frozen before pilot/scientific work.
Preserve failed actions, diagnose setup/harness defects, make separately identified
recovery attempts with unchanged scientific cases/gates. Do not replay closed
scientific outcomes. Complete saved-artifact and arithmetic review before reporting
quality success. Current byte recipe stays unchanged.
