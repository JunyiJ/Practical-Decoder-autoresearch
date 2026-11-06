# Input embedding feature-dropout preparation

Status: quality001 is now frozen and RUNNING after the direct human old-checkpoint
cleanup request resolved capacity. Both prerequisite workers/resource pilots and
finite source contracts passed. Initial slot00 is the control at LR.000625,
screen seed200090002; follow actual controller/guards/accepted anchors. No quality
result or adoption yet. Keep incumbent. Earlier capacity-wait sections are history.

The [evidence review](tokenizer-evidence/generalization-boundary-001/evidence-review.md)
separates measured local facts, literature and inference. Frequency controls
show unigram gaps .153–.173 versus the model gap 1.078; they are not a causal
loss decomposition. Earlier attention/residual dropout experiments mostly lost.
The new hypothesis therefore isolates the input lookup feature site, with one
literature-anchored probability .1 versus .0, rather than repeating those sites
or sweeping rates. It is local regularization/exploitation, not quota-driven
structural exploration. The proposed portfolio allocation stays proposed.

## Correctness evidence

- `correctness001-cpu/mps.json`: exact .0 incumbent initialization, RNG, logits,
  gradients and two optimizer steps; direct stochastic mask equations/gradients
  and device-local RNG repeat; own-policy checkpoint/optimizer/cache rejection;
  full/cache/reset/causality at batches 1/8. Parameters remain 28,492,416 across
  88 tensors, 44 Muon and 44 AdamW groups. Cache remains 196,608 bytes/item.
  Only input features are masked during training. Eval equations are unchanged;
  no attention/residual/FFN dropout, vocabulary, norm, scale or bias change.
- `resume001.json`: both arms passed exact four-update real-data fresh repeat and
  serialized 2+2 resume. Dtype/shape/SHA256 tensor-byte and scalar identities cover
  model/optimizer/EMA/clipping/samplers/CPU+MPS RNG/raw/H200/logits. All 88 EMA
  tensors and three snapshots agree. Serialized split payload is 445,080,807 bytes
  per arm; it lived in memory only, with zero weight files. RSS/MPS peaks were
  1,960,198,144/2,510,094,336 bytes. No comparative memory-saving claim.
- `resume-metadata001.json`: a separate zero-update adversarial proof found that
  inherited validation accepted a short MPS RNG tensor (the invalid resume itself
  was never called). New additive preflight rejects ten RNG/layout/sampler/copied
  configuration forgeries before model/optimizer/EMA/live-RNG mutation. Valid
  numerical transport and its original receipts remain unchanged.
- `worker-source-review001.json`: all 18 finite probability/horizon/LR configs,
  module routes and actual B8/T128 model RNG versus parent mask-only replay pass.
  This is not an independent full worker/validation acceptance yet. The new parent
  must account for mask RNG consumption; the old dropout-free RNG assertion is
  not reused. CPU/MPS full-validation CE agreement awaits actual worker artifacts.

Source sets are additive: 947 component-proof bindings, 950 transport, 952 RNG
preflight, 958 original worker bindings, 963 corrected scorer/parent bindings,
966 resource-pilot bindings and 972 total quality-source bindings. All original
939 scientific and six prior diagnostic sources are preserved. These namespaces
are immutable after their completed reviews. New fixes need additive sources.

## Actual worker acceptance and original failure

The original `worker001-build-run001.py` was executed once. Its control worker
completed four updates, then parent scoring failed because an inherited scorer
expected an attribute configuration while the new model has an explicit policy
dictionary. Its STOP/source/log/guard remain CLOSED_FAILED; this is an interface
failure, not a negative quality result. Never rerun it or repeat those updates.

The additive five-file `embedding_input_dropout_worker_v2` scorer adapts the
policy guard while preserving numerical scoring and acceptance arithmetic.
`worker002-import001.json` independently accepted the existing control artifacts
with zero retraining or weight copies. The separate worker002 context imported
only that metadata and trained only the previously unstarted .1 arm. Both passed
RAW/H200/optimizer/EMA/RNG/samplers/full clipping/full 34,278-target validation and
cache/reset batches1/8. CPU/MPS CE differences were below2.75e-7. Worker002 is
terminally complete; original worker001 STOP remains unchanged.

`pilot-integration001.json` closed actual generated tensor dependencies after
12 finite-config/forgery checks. The NEW exact `generated-retirement001` removed
four obsolete payloads, 1,118,325,460 logical /1,118,330,880 allocated bytes, after
full consumer, identity and47,759-survivor checks. Original receipts resolve the
absent tensors through that later manifest; never restore or rerun them.

## Actual resource pilots and finite design contracts

Both original20-update pilots are independently accepted, with20,480 targets,
eight H200 snapshots and88 EMA tensors each. Full checkpoint/optimizer/RNG/
clipping/validation/cache checks passed. CPU/MPS CE differences were below2.82e-7.
Conservative full-run projections are1,083.22 seconds for .0 and1,118.07 for .1,
under the original7,200-second limit. These feasibility measurements are not a
scientific quality result. The pilot root terminal STOP closes all pilot commands.

`quality-contracts001.json` passed exact six-screen/six-fresh configuration,
minimum/tie LR selection, partial/nonfinite/extra-weight-credit rejection, exact
pilot/loser scope and selected/fresh/duplicate rejection. Both complete1,200-mask
RNG histories match an independent transactional replay. Independent nonconstant
synthetic grouping/bootstrap/quantiles agree within1e-12. Actual accepted pilot
RAW/H200/optimizer/EMA/RNG/clipping reports closed pilot tensor dependencies,
without new training or model rescoring. This proof uses no quality seeds.

## Prospective scientific comparison and complete budget

Prepared runner: `research.embedding_input_dropout_quality_v1.execution`.
Future root: `research_runs/tokenizer-embedding-input-dropout-quality-v1-001`.
Only after actual source/proof/environment/data/process/nonuse checks AND the
complete live capacity pass, invoke `prepare` once, read the resulting new freeze,
then `run-all` once. If any root/plan/freeze/attempt exists, inspect its actual
state instead of rerunning preparation or an attempted slot.

The design is exactly six1200-update screens: both arms at nominalLR .000625,
.00125 and.0025, one common screen seed200090002. Select minimum complete H200
validation CE independently per arm, lower LR on exact ties. Then run exactly
three shared fresh seed pairs200090011/012/013 with rotated arm order, same
1,228,800 sampled targets and41 H200 snapshots each. Pilot200090001 and
evaluation200090003 are used; screen/fresh/bootstrap200090004 remain unused.
No extra rates/sites, rescue, replacement seed, old-score pooling or partial panel.

Adoption requires candidate-minus-control mean CE<−.02, every paired seed delta
negative, and negative simultaneous conditional-text upper bounds at blocks2048,
4096 and8192 (2,000PCG64 draws, linear quantile1−.05/3). Only after all12accepted
endpoints execute all-six-fresh serving/cache review and independent final
statistics, then a separate adoption review if all gates pass. These intervals
condition on three trained seeds and reused validation text; they are not
population-seed or holdout confidence. Holdout remains closed.

The tested stage high-water is8 inherited560MiB RAW/H200 pair caps:2pilots+
6screens, or after exact stage retirement2selected+6fresh. It includes256MiB
metadata,64MiB portable sidecars,8GiB reserve,1,267,904,512B measured external
headroom and1,458,176B entry headroom. Full floor **14,892,462,080B**; cleanup
target **17,039,945,728B**. A further measured entry margin makes the first-launch
wake **14,893,920,256B** after pilot retirement. Subtract only actual retained
accepted current-panel weights once. No planned retirement/hardlink credit.
This tested stage contract supersedes the earlier rough16-pair19,590,082,560B
all-output planning cap; it does not relax any original resource guard.

The standalone NEW exact `quality001-pilot-retirement001` closes and retires only
the four audited successful pilot payloads. Its manifest/completion and later
capacity record govern actual deletion, survivor checks and current wake. The
original pilot STOP is preserved. All selected/fresh historical models remain
protected. The only future eligible scientific deletions are the four actual
losing screen pairs after immutable complete LR selection and full audit.

Current inline recovery includes newly regenerated compiler caches and newly
closed metadata, under coordinated locks and exact journals. Original failed
cleanup002 and all completed cleanup/proof namespaces stay closed. No app or
worker is stopped, no permission bypass, and selected/fresh/original evidence is
preserved. After eligible recovery and independent work are exhausted, only one
silent hourly wake check remains for an unchanged deficit. New eligible proof,
access, dependency closure, changed requirement or added storage triggers inline
recovery. A supplied external APFS volume can support verified protected-archive
relocation; derive its exact required capacity from the current deficit, rather
than reusing the obsolete16GiB option. Never delete protected archive bytes.


## Current reconciled capacity

`quality001-ready-capacity001.json` verifies all972source hashes,225saved numerical
configs with no screen/fresh seed use, no active worker and absent quality
root/plan/freeze. After exact pilot retirement (47,898survivors unchanged), new
external004/005 and lossless storage040, actual observed free was11,109,249,024B.
Launch wake14,893,920,256B leaves **3,784,671,232B**; cleanup target17,039,945,728B
leaves5,930,696,704B. This is an observation, not a guarantee that free space stays
constant. Storage040 compressed237closed metadata files, reclaimed38,359,040B and
reverified48protected model hashes, preserving all readable bytes and paths.
Total exact allocation reclaimed this continuation is2,304,946,176B, including
both generated/pilot retirements; it is not the observed free-space delta.

All currently authorized eligible classes are exhausted. Normal root-owned npm/
unused-Keynote access blockers, live app caches/updaters/shared runtimes/containers,
unmatched downloads and personal data remain preserved. No additional obsolete
research payload has a closed dependency beyond the exact retired allowlists.
One short silent hourly check suffices while unchanged. Additional local space or
a human-provided external APFS volume with at least8GiBfree enables verified
protected-archive relocation toward complete budget+2GiB. No extra audit/closed
probe reruns or below-budget quality freeze. The old11.05GBprerequisite wake is
resolved and must not launch any closed builder. Current next action is the new
quality `prepare` then `run-all` only after the full live gate passes.


## Human cleanup resolved capacity; quality001 launched

The latest direct permission to delete old checkpoints authorizes the exact
superseded historical202RAW archives+202H200 and3H100 retirement. All original
acceptance/failure/terminal/config/log/result records remain. Recent body24 and
final-scale12 payloads, including all three incumbent models, remain unchanged.
First retirement audit001 closed before any deletion on a cross-root acceptance
path mismatch; separate002 verified the exact imported original hashes and retired
404payloads, with47,576survivors/36recent payloads/972sources unchanged. Separate003
retired3obsolete H100s from the closed EMA study with47,755survivors unchanged.
Never restore those weights or rerun any closed audit/study/materializer.

With45disposable cache files and lossless197-file metadata compression, this
request reclaimed95,481,438,208allocatedB. Observed free exceeded106GB; allocation
reclaimed is not the observed free-space delta. See TOKENIZER_CHECKPOINT_RETENTION.md
for the narrow current policy exception and exact manifests. Earlier external-APFS
and3.78GBdeficit wake instructions are resolved, not current blockers.

`quality001-preflight001.json` verifies all972bindings, original corpus/tokenizer/
train/validation IDs with closed holdout, actual prerequisite hashes,225saved
configs without scientific quality-seed use and complete live budget plus2GiB.
Scientific prepare executed ONCE, producing freeze
`d8bea1ca11ab6bb4963421100513aa0e39fd967e2af4198367c5d432b9bd9ac4`.
`quality001-launch001.json` records controller84058/worker84081/session95353 and
first completed update. Snapshots can stale: inspect actual processes/control/
guards/STOP/accepted prefix; never duplicate a live controller or attempted slot.
The frozen two-arm/six-screen/three-fresh-pair design and all original gates stay
unchanged. Screen200090002 is now USED; fresh200090011/012/013 and bootstrap200090004
remain unused until their phases. Controller serially completes the study; only
complete-panel inference/statistics can support a separate adoption review. After
closure continue evidence-grounded model improvement under human authorization.
