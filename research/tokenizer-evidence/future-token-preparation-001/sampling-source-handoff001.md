# Independent future-token data-path source preparation

Read original block_norm_v1/runtime/data/loaders.py:_get_batch, accepted
future_token_objective_v1.py:align, evidence-review001.md and the closed original
model-proof001 failure review. New research/future_token_sampling_v1.py is
SOURCE ONLY. AST parse/compile was performed without importing it or executing
any function, Torch, sampler, generator or model. No sampler seed is assigned.
The closed model proof and all its source/predicates/roles remain unchanged.

## Consumer need and source design

The new objective needs TWO right targets for each of1024 shared input positions.
The inherited paired512 loader samples two512-input blocks with one right target;
it cannot produce offset2 by simply reshaping its ordinary512+1 target view.
The new source instead draws two anchors and constructs four130-ID rows per
anchor, with row starts0/128/256/384. Each row contains128 inputs plus its two
right targets. Overlap of the last two IDs across adjacent rows is intentional;
no input position is discarded, wrapped, padded or taken from another split.

For train lengthL, a514-ID chunk requires0<=anchor<=L-514, countL-513. The
old next-token-only loader has one extra eligible endpoint. This is a declared
sampling-support change for ALL three fresh future-token arms, including native;
it is not exact equality to the old stream or reuse of an old trajectory.
Validation training-evaluation batches requireL>=130 and eight independent
starts0<=start<=L-130. Physical full34278-target scoring remains separate and
must NOT be replaced with these private mini-batches.

The pure positions(length,starts,split=...) function validates exact integer
bounds and produces8x130 absolute indices. The CPU-owned sampler clones its
train/val inputs, binds token-byte hashes/lengths/roles, and uses private CPU
train/eval generators. next() draws on a copied generator, constructs and checks
the complete output, then commits its generator/counter. State loading validates
BOTH supplied private generator states/seed identities before destination changes.
This code has not run. Claimed transactional properties need new proofs; they
are not accepted merely from reading the source. No optimizer-ledger extension.

## Next executable action

Prepare and review a NEW finite deterministic index/layout oracle with independent
literal expected offsets and split boundaries. Include minimal eligible train514/
val130, last permitted anchors, adjacent rows/two-target overlap, two distinct
anchors, all1024 positions, forbidden holdout/wrap/padding/overflow/bool/type/
length/start inventory cases. These are new literal data-index questions, not
replacement model/proof cases. No Torch/model/RNG calls are needed to test the
pure integer layout; if importing the module would import Torch, inspect or
extract ONLY this pure function under a prospectively pinned source plan.

Then separately prepare full CPU sampler draw/state/transaction/RNG preservation
cases and working/time/resource gates. Census ONLY newly proposed train/eval
roles before actual private generators. The failed model initializer cannot be
reused, and no scientific cohort/bootstrap/global roles should be invented.
Keep the actual model proof CLOSED_FAILED; no data-path success promotes its
partial records, admits a new model replay or qualifies real training.

This source work follows the already reviewed supervision mechanism and its
exact prospective data requirement. It is not a new quality result or another
CE diagnostic. No corpus/tokenizer/holdout is loaded, no model is run, no weight
or evidence is deleted, and no gate is weakened. Later source/design decisions
must read actual original evidence and use evidence-grounded-research as needed.
