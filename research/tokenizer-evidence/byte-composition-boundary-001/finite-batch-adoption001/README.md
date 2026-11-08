# Batched finite validation: equivalent implementation adopted

The active recipe is [recipe.json](recipe.json), with public loader
`research.byte_input_checkpoint_consumer_v3.load`. It takes the recipe's bound
implementation, original plan and original acceptance receipts plus CPU/MPS backend.
Callers retain the shared research lock and resource guards. The same first-fresh
H200 checkpoint,16,351,859 parameters, mathematical model, artifact formats and
scientific quality evidence are retained. Frozen originals and prior recipes are
unchanged. This is implementation adoption, not a new quality gain.

## Fixed correctness and validator timing

The finite-batch-runtime001 comparison preserved valid-model acceptance and20
fixed invalid-state rejections/restorations: NaN/+Inf/-Inf at canonical parameter
indices0/52/104, a frozen final parameter, and changed codebook content, each for
original/candidate on CPU/MPS. All parameter bytes and global RNG were unchanged.
The exact source map retains every structural, ownership, content and finite check;
all105 finite predicates feed the conjunction, rather than dropping checks.

Complete-validator medians over five alternating pairs were26.639ms original
versus26.864ms candidate on CPU (no useful CPU speed claim), and68.080ms versus
46.628ms on MPS. All five MPS pairs were faster and the prospective10% median
validator threshold passed. This is a local timing filter, not a confidence test.

## Integrated equivalence and contrary timing evidence

The new model-v5/state-v4/training-v4/inference-v3/cache-v3 sources passed exact
CPU/MPS B1/B8 cached logits/reset, B8T128 full logits/loss/all105 gradients, and
initial/step1/step2 complete captured/validated training states, including the
first averaging snapshot. Eight numerical updates used fresh sampler roles
62100800001/2. Original broader model/state/resume/cache qualifications transfer
through the exact source map; they were not replayed.

The sequential two-update comparison showed no MPS training gain: original
1.0905/1.1740seconds versus candidate1.1914/1.0860seconds, total speedup0.9944x.
That result is retained in finite-batch-integration-runtime001 and was not
relabeled as a failure, discarded, or pooled into the later timing result.

## Prospective warmed, interleaved timing decision

A separately frozen question measured warmed execution with alternating per-pair
order. Two simultaneous contexts started from the same accepted H200 with fresh
sampler roles62100900001/2 and the unchanged20-update numerical recipe. All20
pairs completed. Four pairs were declared warmups;16 were measured. No timing
trial or seed was replaced after observing this result.

The predeclared gate required median candidate/reference latency<=.95, at least
12/16 candidate wins, and no median regression in either pre-averaging or averaging
phase. All gates passed:

- Median ratio0.919603: **8.04% lower median paired update latency**.
- Candidate faster in15/16 measured pairs.
- Phase medians0.930725 before averaging and0.906318 with averaging.
- Measured totals18.685486seconds original versus17.182623seconds candidate.
- Every paired update history and the complete20-update endpoint matched exactly,
  including parameters, optimizer state, eight averages, scheduler, samplers and RNG.

This is one local paired trajectory and a short numerical recipe, not independent
trained seeds, a formal throughput confidence interval or a1200-step speed claim.
The averaging cadence differs from full quality training. Do not multiply this
ratio with an earlier short-run speedup to claim an unmeasured combined speedup.
No quality or holdout scoring and no new weight archives were performed.

## Publication and next work

A separate saved-evidence audit verified case hashes, all exact comparisons,
20-step endpoint equality and every prospective timing gate before publication.
The public v3 loader then loaded the same representative on MPS in1.3249seconds,
with unchanged CPU/MPS RNG. Its guard finished successfully. The validation,
integration, throughput and loading guards respectively completed in5.8860,
32.6764,52.2936 and1.9421seconds, all within their declared limits. No scientific
quality panel was repeated. All sampler roles above are closed/reserved, not
available for future reuse.

Next assess the existing profile's optimizer-state validation path. Its repeated
finite/nonnegative checks remain unchanged in this implementation. Any candidate
must preserve complete state/ownership and failure-before-mutation behavior and
show a prospectively defined useful gain. Padding cropping remains closed not
adopted because of its numerical differences; its gate must not be weakened.
