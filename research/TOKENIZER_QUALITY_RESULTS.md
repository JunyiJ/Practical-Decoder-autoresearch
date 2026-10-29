# GPT-2 tokenizer quality results

Updated October26,2026 (Los Angeles). The new campaign is running under the
human's instruction to continue until interruption. The first finite round is complete: LR0.0025 is adopted under the frozen validation rule. Follow the [frozen protocol](TOKENIZER_QUALITY_PROTOCOL.md).

## Completed LR transfer round: adopt LR0.0025

All18 trajectories and their raw/H200 checkpoints passed saved-artifact review.
Both controller and worker exited. The final review verified unchanged source,
runtime and corpus, exact saved results/requests/checkpoint hashes, frozen
selection, and separately recomputed all six conditional text-block intervals.
This is same-agent review, not an independent reviewer or test evaluation.

| Width | Baseline LR0.01 mean H200 CE | Adopted LR0.0025 mean H200 CE | Paired mean difference | 4096-token conditional interval |
|---|---:|---:|---:|---|
| 64 | 5.562136 | 5.214648 | -0.347488 | [-0.397065, -0.304287] |
| 256 | 7.082105 | 6.309554 | -0.772551 | [-0.821903, -0.726952] |

All three fresh seeds favor the candidate at each width. All predeclared
2048/4096/8192-token interval upper bounds are negative. Adopt only the LR
change; retain the inherited architecture,3000 updates and H200. This is a
local adaptive-validation result, not a general optimum or holdout improvement.
The larger model's worse absolute endpoint and late validation degradation
remain unresolved; lower LR does not solve that pattern. Training totaled
6170.17seconds across18 trials. Checkpoints and failure evidence are preserved.

Next: prepare a finite GQA-versus-MHA comparison with the same explicit0.5 score
multiplier, equal bounded LR tuning, fresh confirmation and the predeclared
quality-neutral inference-memory route. Keep the horizon/schedule hypothesis
active for a subsequent controlled comparison; architecture and HP research both
remain priorities. No next training launches before correctness/resource gates
and a frozen plan. [Final review](tokenizer-evidence/quality-001/final-review.json)
and [adopted recipe](tokenizer-evidence/quality-001/adopted-recipe.json).

Earlier progress sections below are a chronological record of this round.

The GQA follow-up has passed CPU, serial MPS and resource prerequisites. See
[GQA results and current progress](TOKENIZER_GQA_RESULTS.md) and its
[finite protocol](TOKENIZER_GQA_PROTOCOL.md). Both architecture and HP hypotheses
remain in the campaign agenda.

## Synced baseline and implementation

GitHub was fast-forwarded from `63d3970` to `ce274e5` (28 commits). The final
reviewed character recipe—not the retrospectively lowest individual seed—is the
starting architecture/training baseline. Tokenizer implementation and protocol
are in `84d0dba` on `codex/tokenizer-quality`, pushed to GitHub.

The default training configuration now uses GPT-2 BPE with50,257 vocabulary
entries. Legacy configs without a tokenizer field retain character behavior.
Training, resume, checkpoint evaluation, generation and inference comparison
use the saved tokenizer; raw and H200 checkpoints bind its identity. The new
no-output-bias binding removes the obsolete65-character prior while preserving
all inherited architecture policies. Character weights are not loaded into
incompatible vocabulary tables.

Raw-text80/10/10 boundaries are unchanged; each split is encoded separately:

| Split | Raw UTF-8 bytes | GPT-2 tokens |
|---|---:|---:|
| Training | 892,315 | 267,688 |
| Validation | 111,539 | 34,279 |
| Reserved test | 111,540 | 36,059 |

Tokenization counts are not test evaluation. Test loss remains closed. Full
validation scores34,278 targets, excluding the first token and its3 bytes,
covering111,536 target bytes. At3,000 updates×1,024 targets, the nominal token
exposure is11.476 training-corpus passes. Random sampling can overlap; this is
not a claim of uniform coverage. Subword CE cannot be compared with character CE.

## Correctness and resource pilots

333 targeted CPU tests passed in172.65s. These cover raw-text boundaries,
Unicode/literal-special-marker round trips, private sampler RNG, legacy paths,
actual training, exact CPU resume, tokenizer mismatch rejection, generation,
EMA exports and affected model/evaluator/cache contracts. This was a targeted
regression run, not the entire historical repository suite.

The synced archive verified3,893 metadata files and1,940 knowledge records.
Two serial40-update MPS pilots on source `84d0dba` completed. Each exported and
loaded raw/H200 checkpoints, evaluated every declared validation target and
produced a fixed-prompt generation sample.

| Width | Parameters | Training time | End-to-end worker time | Raw + H200 bytes | Recorded training RSS |
|---|---:|---:|---:|---:|---:|
| 64 | 6,629,120 | 4.716s | 11.648s | 132,051,020 | 468.8MiB |
| 256 | 28,851,200 | 7.237s | 15.654s | 564,819,660 | 587.8MiB |

RSS is the trainer's process-lifetime peak observation at its final logging
point; export/scoring can occur later. The sampled worker watchdog remained
active. Pilot schedules/EMA tails are shortened correctness settings; their CE
values and samples are excluded from quality rankings.

A saved-artifact review verified request/checkpoint hashes, exact config and
runtime/tokenizer contracts,40-step private training-sampler replay, untouched
test RNG,21 EMA snapshots, finite raw/H200 tensors, parent/export lineage,
validation coverage and CE/BPB arithmetic. This was a same-agent review, not
an independent reviewer or a training/scoring rerun.

Observed checkpoint sizes project about4.63GiB for the maximum18 full trials,
or6.95GiB with50% headroom. Scaling observed pilot training times projects about
2.08 training hours for18 trials, before full-run overhead; these are forecasts.
The round has an8GiB free-space reserve,12GiB output cap, and per-worker memory
and time guards. Pilots do not establish3,000-update stability.

## Preserved failures and lessons

- Initial CPU pass:253 passed, one test-fixture failure. A character-length
  fixture became too short for a128-token validation window after BPE. The
  fixture was enlarged; sampler boundary checks were retained.
- Intermediate CPU pass:331 passed, one import failure. Source was edited while
  the process had cached an earlier module version. The final333-case run used
  unchanged source and passed. Source stays frozen during quality training.
- An append-only knowledge record was successfully inserted, then its ID was
  mistakenly treated as an object by the recording script. The existing record
  was read back without duplicate insertion or altering evidence.

These are implementation/validation failures, not failed quality comparisons.
Original logs and metadata are checked in under [pilot evidence](tokenizer-evidence/pilots-001/).
Binary training checkpoints remain in local ignored `research_runs/`.

## Updated research direction

The human has directed inference-aware GQA/MLA adoption: better or neutral
quality can qualify when measured inference cost improves. The follow-up also
prioritizes Kimi-style KDA/KDA/KDA/MLA with bounded tuning, rather than treating
the historical all-KDA result as a hybrid test. See the checked
[attention roadmap and primary-paper findings](TOKENIZER_ATTENTION_ROADMAP.md).
The active LR experiment and its selection rule are unchanged.

The latest human steering makes architecture a central research objective:
attention/hybrids, sparse and shared experts, activations, normalization,
residual topology/scaling, depth/width allocation and embedding/readout design.
The [architecture agenda](TOKENIZER_ARCHITECTURE_ROADMAP.md) records structural
hypotheses, comparison controls and scheduling. The human clarified that this
means architecture exploration alongside hyperparameter tuning, not replacing
tuning or imposing architecture-only ordering. Both remain first-class research
directions; evidence determines the next finite round. These are planned
comparisons, not measured gains.

## Quality round execution (historical)

`research_runs/tokenizer-quality-001` ran one MPS worker at a time. Its
[frozen plan](tokenizer-evidence/quality-001/plan.json) compares peak LRs0.01,
0.005 and0.0025 with paired screen seeds2601/2602 at width64,3,000 updates each.
The selected recipe is frozen before fresh-seed confirmation at widths64/256.
The inherited0.01 recipe remains the baseline until the full adoption rule passes.

The current chat's five-minute heartbeat continues review, records learnings,
checks in results and advances to new justified finite rounds until human stop.
It reports substantive changes or failures and stays quiet when nothing changes.
The old character/null heartbeat remains paused, and campaign006 remains stopped.

## First completed baseline: screening seed2601

The width64 inherited LR0.01 trajectory completed all3,000 updates. It passed
the saved-artifact audit: config/tokenizer/request/checkpoint hashes, finite
model and optimizer tensors, optimizer step counts and independent decay,
101 EMA snapshots, training sampler replay, untouched test RNG, and exact
validation coverage/CE/BPB arithmetic. No training or scoring was rerun.

| Recipe | Raw validation CE | H200 validation CE | H200 bits/byte | Training time | Worker wall time |
|---|---:|---:|---:|---:|---:|
| Width64, seed2601, LR0.01 | 5.859556 | 5.610844 | 2.487729 | 254.74s | 262.21s |

These are subword-token losses from one screening seed, not a final quality
comparison. H200 improves this trajectory's raw endpoint by0.248712 CE, a
correlated diagnostic rather than an independent experiment. The fixed sample
is readable Shakespeare-like dialogue but has grammatical and semantic defects;
it is not a quality selector.

On the separate fixed sampled-validation panel, CE fell to4.8580 at update600
then rose to5.6499 at update2999 while training-minibatch loss fell. This is a
validation-degradation warning under the inherited budget. It does not isolate
overfitting, LR, schedule or repeated text exposure as the cause. A shorter
horizon is a justified candidate for a subsequent finite experiment after this
LR round; no early-checkpoint selection or retrospective budget change is made.

The first baseline was followed by the lower-LR trials and the second paired
seed. The current reviewed partial comparison appears below.

## Width64 fresh-seed confirmation complete

All three fresh width64 pairs passed saved-artifact audits. LR0.0025 mean H200
CE is 5.214648, versus 5.562136 for LR0.01: mean difference
-0.347488, with all three seeds favoring the candidate. The ratio of
perplexities computed from mean CE is 0.706461; this is not
an average of per-seed perplexities. No screening seeds are pooled into these
confirmation means. Width256 confirmation is now running serially. Adoption
still requires those results and the full frozen conditional-text checks; the
small-width result alone does not establish transfer. Test loss remains closed.

## First width256 trajectory: validation degradation warning

The completed LR0.0025 seed2611 trajectory passed the saved-artifact audit,
including finite model/optimizer tensors, schedule, sampler and checkpoint
contracts. Its full-validation H200 CE is 6.322959;
raw endpoint CE is 6.426976. Its paired LR0.01 baseline has now completed
and passed audit at H200 CE7.094273. The paired difference is -0.771314 CE,
favoring LR0.0025 on this seed. The second width256 pair also passed audit:
candidate H200 CE6.305714 versus baseline7.099229, difference -0.793514.
One more width256 pair and the frozen interval checks remain outstanding;
this is not yet a width256 adoption conclusion.

On the fixed sampled-validation diagnostic, CE reached 4.687503
at step600 and rose to 6.205675 at step2999;
the last training minibatch CE was 0.309860. This strengthens
the warning about the inherited training horizon. It is consistent with
an overfitting pattern, but does not isolate horizon, schedule, capacity or
regularization as the cause. Sampled validation differs from the full endpoint
panel, and its minimum is not an eligible checkpoint selector. Finish the
frozen comparison, then weigh a prospective shorter-horizon/schedule experiment
alongside the architecture agenda. No active trial or adoption rule changes.

## Reviewed round progress

Only completed, saved-artifact-reviewed trajectories appear below. All use
3,000 updates, the same GPT-2 tokenizer and full-validation H200 endpoint.
The full adaptive-validation comparison passed the frozen rule; LR0.0025 is
adopted. Historical intermediate interpretations above remain scoped to their dates.

| Stage | Width | Seed | Peak LR | Raw CE | H200 CE | H200 bits/byte | Training seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| screen | 64 | 2601 | 0.01 | 5.859556 | 5.610844 | 2.487729 | 254.74 |
| screen | 64 | 2601 | 0.005 | 5.583362 | 5.442269 | 2.412987 | 254.96 |
| screen | 64 | 2601 | 0.0025 | 5.276655 | 5.220207 | 2.314529 | 256.24 |
| screen | 64 | 2602 | 0.0025 | 5.284148 | 5.232680 | 2.320060 | 256.35 |
| screen | 64 | 2602 | 0.005 | 5.572239 | 5.458199 | 2.420050 | 256.87 |
| screen | 64 | 2602 | 0.01 | 5.754487 | 5.554708 | 2.462840 | 260.03 |
| confirm | 64 | 2611 | 0.0025 | 5.261932 | 5.204370 | 2.307507 | 258.30 |
| confirm | 64 | 2611 | 0.01 | 5.785170 | 5.558275 | 2.464422 | 257.61 |
| confirm | 64 | 2612 | 0.01 | 5.759009 | 5.564940 | 2.467377 | 258.32 |
| confirm | 64 | 2612 | 0.0025 | 5.279965 | 5.229425 | 2.318616 | 258.04 |
| confirm | 64 | 2613 | 0.0025 | 5.260866 | 5.210149 | 2.310070 | 257.00 |
| confirm | 64 | 2613 | 0.01 | 5.761821 | 5.563192 | 2.466602 | 256.72 |
| confirm | 256 | 2611 | 0.0025 | 6.426976 | 6.322959 | 2.803466 | 519.92 |
| confirm | 256 | 2611 | 0.01 | 7.410883 | 7.094273 | 3.145450 | 513.40 |
| confirm | 256 | 2612 | 0.01 | 7.355854 | 7.099229 | 3.147648 | 517.47 |
| confirm | 256 | 2612 | 0.0025 | 6.404070 | 6.305714 | 2.795820 | 508.44 |
| confirm | 256 | 2613 | 0.0025 | 6.402215 | 6.299989 | 2.793282 | 510.67 |
| confirm | 256 | 2613 | 0.01 | 7.310382 | 7.052812 | 3.127068 | 515.06 |

Paired H200 differences versus LR0.01 within the same stage, width and seed:

- screen, width64, seed2601, LR0.005: -0.168575 CE.
- screen, width64, seed2601, LR0.0025: -0.390637 CE.
- screen, width64, seed2602, LR0.0025: -0.322028 CE.
- screen, width64, seed2602, LR0.005: -0.096509 CE.
- confirm, width64, seed2611, LR0.0025: -0.353905 CE.
- confirm, width64, seed2612, LR0.0025: -0.335515 CE.
- confirm, width64, seed2613, LR0.0025: -0.353043 CE.
- confirm, width256, seed2611, LR0.0025: -0.771314 CE.
- confirm, width256, seed2612, LR0.0025: -0.793514 CE.
- confirm, width256, seed2613, LR0.0025: -0.752824 CE.

Frozen screening choice: LR0.0025. Confirmation results cannot change this choice.

Single-seed signs are preliminary. The controller remains serial, the source
and data contract are frozen, and test loss remains closed.

## October 28 proactive retention update

The human authorized non-critical checkpoint cleanup going forward. Cleanup002
removed 42 files from successful closed pilots and losing table/wide screens,
freeing 13,438,900,670 logical bytes (12.516 GiB). All 148 other tokenizer weights,
including selected screens and all fresh confirmation raw/H200 pairs, and 659
non-checkpoint run files were hash-verified unchanged. Metrics, failures and
original audits remain; deleted weights cannot be reloaded or audited again.
See `tokenizer-evidence/checkpoint-cleanup-002/README.md` and `manifest.json`.

Prospective audited stage-boundary retention reduces the complete 27-trajectory
width512 hybrid high-water storage requirement to 30,631,281,412 bytes (28.5276
GiB), with 8 GiB reserve and 256 MiB round metadata. This fits the post-cleanup
observation of about 28.984 GiB, subject to rechecking current disk. This is a
retention/accounting change only; no new quality evidence or launch. Versioned
retention-aware runners and outstanding numerical/pilot checks still precede
freeze. Follow `TOKENIZER_CHECKPOINT_RETENTION.md`; the current v1 storage guards
remain intact. Historical closed controllers and the failed CPU fixture stay closed.

## October 28 wider-hybrid pilot preparation

The v2 pilot resource recorder and complete three-pilot evidence binder passed
204 synthetic tests. Pilot completion now binds the completed tensor/cache audit
inside its600-second resource record; quality trials retain their separate7,200s
contract. This is preparation, not new model-quality or real numerical/pilot
passing evidence. No training launched or checkpoint deletion occurred. Actual
numerical prerequisites, measured pilots and complete v2 producer/controller
integration remain outstanding. See `TOKENIZER_WIDE_HYBRID_PREPARATION.md` and
`tokenizer-evidence/wide-hybrid-preparation-001/pilot-audit-v2-component-review.json`.

## October 28 wider-hybrid quality handoff preparation

An immutable idle-readiness seal now binds the complete first/later LR
predispatch review, evidence/source/control hashes, remaining storage requirement
and original output allowance. All134 synthetic readiness/handoff/startup tests
passed. This is controller preparation only: child consumption and dispatch
integration, actual numerical/input cases and measured pilots remain unfinished.
No new model-quality result, research training or checkpoint deletion occurred.
See `tokenizer-evidence/wide-hybrid-preparation-001/quality-readiness-v2-component-review.json`.

The next preparation increment adds same-parent, readiness-bound child checks
and a guarded callable quality body. It rechecks immutable evidence, frozen
prerequisites and actual remaining storage before the producer, and keeps the
parent's metadata writes available during source checks. Final172 synthetic
tests passed; one corrected test-fixture failure is preserved with its log.
There is no child CLI or parent dispatcher yet and no new quality result.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/quality-entry-v2-component-review.json`.

The child-only CLI and independent parent acceptance component now pass87
synthetic tests. Acceptance requires the exact launch/terminal/entry-guard
identity and repeats completed-prefix, pilot-history and remaining-storage audits.
Both fixture-failure runs are preserved. Parent orchestration and actual numerical/
pilot prerequisites remain unfinished; no new model-quality evidence was produced.
See `tokenizer-evidence/wide-hybrid-preparation-001/quality-acceptance-v2-component-review.json`.


## October 28 serial LR parent transition

The new callable parent controller prepares, seals, dispatches and independently
accepts exactly one LR slot. Tests exercise two-slot and all-nine-slot sequences,
retain prior evidence unchanged, and stop before selection or cleanup. Historical
completion receipts bind ordered predecessors, saved parent/child identities,
startup/readiness/launch/terminal/worker hashes, finished parent and child guards,
original limits and actual trial/log byte counts. A later parent can have a
different PID or inherited descriptors; the next startup still independently
replays the complete scientific prefix, pilot cleanup history and future storage.

Preparation and post-child review have separate1800s guards. Preparation finishes
before sealing. During the child, a cooperative parent resource callback raises
through the7200s process supervisor, which terminates and reaps its owned group.
No hard-exit parent watchdog surrounds a live child. Tiny real subprocess tests
verify this failure path. The child time/RSS/MPS, full storage and8GiB reserve
gates are unchanged. Failures and partial/completed attempts cannot retry.

Initial99 tests passed; one sequential fixture failed because its later-slot
module still referenced the real projection instead of the synthetic frozen
copy. Correcting that fixture yielded70 passing controller/process tests in40.88s.
After strengthening historical terminal-limit/census checks, all30 controller
tests passed in43.67s. The33 independent acceptance checks passed in the initial
run. All logs and240s/4GiB/8GiB actual CPU/disk guard observations, including the
fixture failure, are retained. Scientific child execution and PIDs are mocked;
this establishes controller behavior, not model correctness or quality.

Next implement saved LR/rank selection transitions and their audited cleanup,
rank/fresh dispatch and inference orchestration. Actual nine numerical/input
cases and measured pilots remain unresolved; the failed combined fixture stays
closed. No research training/MPS launch, checkpoint deletion, closed-controller
rerun or holdout evaluation occurred. The27-trajectory campaign remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/quality-dispatch-v2-component-review.json`.


## October 28 audited LR selection transition

The new versioned selection component requires all nine completed parent/child
receipts and independently replays frozen requests, every producer/tensor/resource/
validation audit, pilot cleanup history and complete future storage before saving
rates. It preserves a preselection review, writes the minimum-H200-CE choice with
the existing lower-LR tie break, then repeats full stage and storage audits before
publishing a finished-guard-bound completion receipt. A partial choice publication
blocks continuation and cannot be retried.

The read-only pre-cleanup verifier recomputes the complete scientific stage,
choice and remaining storage, binds original attempt/review/guard provenance and
checks the saved manifest. The exact idle inventory is required; later deleted
weights will need a separate reviewed history consumer. The synthetic choice
identifies12 candidate losing-screen files, but all18 screening weights remain
unchanged. Selection grants no deletion, rank dispatch, adoption or next-stage
credit.

All14 negative tests passed. The positive test reached its final assertion and
failed on a test-only storage field name. After correcting that assertion, the
positive case and rule/stage-audit dependencies passed66 tests in15.79s. The initial
176.15s run and focused rerun retain their raw logs and240s/120s,4GiB,8GiB actual
CPU/disk guard observations. Production code did not change between runs. These
fixtures use synthetic scientific evidence and mocked children; they are not
model-quality or numerical-correctness results.

Observed synthetic metadata occupancy was941073bytes against the2MiB group cap.
Full frozen configs/source maps/receipts still require exact metadata accounting;
the simplified fixture cannot establish their fit. Next integrate audited LR
cleanup and post-cleanup history, then rank/fresh transitions and inference.
Actual nine numerical/input cases and measured pilots remain unresolved. No
research training/MPS launch, checkpoint deletion, failed combined numerical
retry, closed-controller rerun or holdout evaluation occurred. The27-trajectory
campaign remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/lr-selection-v2-component-review.json`.


## October 28 LR cleanup and idle post-cleanup verification

A versioned LR boundary controller now requires the complete saved selection,
all nine parent/child receipts and independent scientific audits before invoking
the existing bounded deletion executor. It captures original checkpoint pairs,
full metadata inventory and all prior control hashes, then repeats the full
stage and future-storage audit immediately before deletion. Only the12 selected
losing-screen files enter the exact path/size/SHA manifest. The executor journals
each deletion and verifies every protected file; the parent then checks the
complete pilot/LR chain and actual remaining storage before publishing completion.

The post-cleanup verifier rebinds frozen source/runtime/plan, original selection,
pre-deletion audits and guards, surviving weights, preserved metadata and the
full deletion journal chain. It reconstructs the manifest without reloading
intentionally removed tensors. This is an exact idle-boundary verifier before
any rank request; an append-aware history consumer remains to be implemented.

All10 integration tests passed in152.39s under240s/4GiB/8GiB actual CPU/disk
guards. The successful synthetic case deletes exactly12 temporary byte files,
keeps all6 selected raw/H200 files and preserves every prior non-checkpoint byte.
Corrupted preconditions block deletion; changed journals, restored losers,
missing winners and altered controls block verification. An injected second
unlink failure retains one confirmed deletion, an intent/deleted/intent journal,
17 remaining files and permanent failure markers; it cannot retry or publish a
successful receipt. These are disposable fixtures, not research checkpoints.

Synthetic post-cleanup metadata occupied1082303bytes of the2MiB group allowance.
The full frozen configuration/source/receipt footprint still needs exact
accounting. Next add append-aware LR history for rank preparation, rank/fresh
transitions and final selection/cleanup/inference. Actual nine numerical/input
cases and measured pilots remain unresolved. No research training/MPS launch,
research checkpoint deletion, failed combined numerical retry, closed-controller
rerun or holdout evaluation occurred. The27-trajectory campaign remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/lr-boundary-v2-component-review.json`.


## October 28 LR history across rank preparation

A separate read-only history consumer now verifies completed LR cleanup after
declared rank files are appended. It keeps the original cleanup context identity
and checks saved selection, pre-deletion audits, finished guards, frozen sources,
protected controls, surviving weights and the entire pilot/LR deletion chain.
The complete supplied inventory is checked before and after verification. The
strict idle-boundary verifier still rejects appended files.

Only slots09 through14, declared producer outputs and explicit prospective rank
controller/selection metadata may be appended. Raw names match the production
final_001200_timestamp format. Temporary/failure files, unknown paths, fresh
confirmation slots and rank-cleanup attempts are rejected. All selected LR
anchors must remain present. New rank bytes receive no scientific, selection,
storage or dispatch credit; independent rank audits are still mandatory.

All44 tests passed in207.43s under300s/4GiB/8GiB actual CPU/disk guards, including
the10 existing cleanup integration tests. Tests cover unchanged protected
history, stale inventories, changed metadata/controls/journals, recreated losers,
unauthorized appends and mutation during verification. Deliberately invalid
scientific rank payloads demonstrate that history verification cannot grant
rank-audit credit. All fixtures are disposable and no research checkpoint was
used or deleted.

Next implement rank-prefix audits using independent conditional requests, all
three surviving LR anchors and completed rank members, then connect preparation
to the complete future storage gate. Fresh dispatch, final selection/cleanup and
inference remain unfinished. Full frozen metadata accounting and actual nine
numerical/input/measured-pilot prerequisites remain unresolved. No MPS/training,
failed combined fixture retry, closed-controller rerun or holdout evaluation
occurred; the27-trajectory campaign remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/lr-history-v2-component-review.json`.


## October 28 live LR-anchor and rank-prefix audits

The rank-prefix auditor now rechecks all three retained LR anchors plus zero
through six completed rank members. It requires independently supplied frozen
requests, exact conditional rank order, nine historical LR records and persisted
LR choices. Every live member must pass producer/tensor/export, training/EMA,
resource/process, validation and corpus/input checks. An optional next request
must be the only file in its exact pending slot and receives no completion credit.

The report distinguishes live audited indices from the six historical losers.
It grants no LR-history, frozen-plan, global-inventory, storage, final-selection,
deletion or dispatch credit. The controller must still bind historical rows,
rates and original pairs through the complete LR cleanup receipt chain, then
independently reconstruct conditional options and verify the full remaining
storage requirement before rank preparation or dispatch.

All173 tests passed in7.36s under180s/4GiB/8GiB actual CPU/disk guards:87 new
rank-prefix cases and86 existing LR-prefix/quality-stage dependency cases. They
cover all prefix/pending positions, malformed independent requests and choices,
missing live-member provenance, changed/absent/aliased weights, failed/partial
outputs, EMA/CE/holdout changes and numerical/input/corpus tampering. These are
synthetic fixtures, not model-quality or numerical results.

Next integrate bounded rank request preparation with verified LR history, frozen
conditional options and full future storage. Rank/fresh orchestration, final
selection/cleanup/inference, full metadata accounting and actual nine numerical/
input/measured-pilot prerequisites remain unfinished. No research training/MPS,
checkpoint deletion, failed combined fixture retry, closed-controller rerun or
holdout evaluation occurred. The27-trajectory campaign remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/rank-prefix-v2-component-review.json`.


## October 28 rank request preparation with verified history and storage

A new bounded API prepares rank slots9 through14 only after complete pilot/LR
cleanup history, unique independently frozen conditional requests, all live LR
anchor/rank audits and full future storage pass. Original deleted pairs remain
bound to their receipts; actual live occupied capacity is counted. No next-slot
directory exists until those gates succeed. Immutable attempts, requests and
finished-guard-bound reports retain the evidence; failures keep any partial
request and permanently block retry. Frozen inputs, full current inventory,
historical integrity, locks/STOP and free space are rechecked before completion.

The initial CPU run passed100 tests and failed3 positive cases in182.98s. One
production comparison incorrectly required the dynamic current-inventory digest
to stay identical after intentionally adding a request. It now compares unchanged
historical fields while still verifying the entire new inventory and chain. Two
later-prefix fixtures had preserved an intermediate synthetic tensor record;
the helper now preserves only files that existed before fixture construction.
The three repaired positives and source-change-after-publication case passed4
focused tests in76.18s. Both runs retain logs and successful300s/180s,4GiB,8GiB
actual CPU/disk guards. Previously passing cases were not broadly repeated.

Preparation grants no process/dispatch, selection or deletion approval. Next
implement read-only rank predispatch, then readiness/child entry and serial parent
dispatch with all prior transport receipts. Fresh transitions, final selection/
cleanup/inference, full metadata accounting and actual nine numerical/input/
measured-pilot prerequisites remain unfinished. Only synthetic temporary byte
fixtures were used; no research training/MPS, checkpoint deletion, failed combined
numerical retry, closed-controller rerun or holdout evaluation occurred. The
27-trajectory campaign remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/rank-startup-v2-component-review.json`.


## October 28 read-only rank predispatch verification

Prepared rank slots9 through14 now have a read-only predispatch check. It binds
startup/attempt/request/finished guard, verifies complete pilot and LR deletion
history, and re-audits every live LR anchor and completed rank member against
independently frozen requests. The pending request receives no completed-trial
or checkpoint credit. Original deleted pairs remain traceable to their receipts.

The check independently reconstructs complete future storage using actual live
occupied capacity, the unchanged8GiB reserve and full256MiB metadata allowance.
It rejects unexpected process/weight/future/failure/partial artifacts, then repeats
frozen inputs, historical evidence, locks, checkpoint stamps, census and actual
free space before returning. Its report explicitly leaves dispatch unauthorized.

All12 synthetic tests passed in139.85s, including28 saved-report mutations,
16 actual evidence/active/failure mutations, exact-capacity and late-space-loss
cases, late history/weight/lock changes, and first/last rank-slot positives. The
360s/4GiB/8GiB actual CPU/disk guard finished normally; peak RSS241,418,240 bytes.
Raw logs and guard evidence are retained. No scientific worker or numerical
fixture was run, and no research weights or closed-holdout data were touched.

Next integrate rank readiness/child entry and serial parent transport verification.
Fresh transitions, final rank selection/cleanup/inference, full metadata accounting
and actual nine numerical/input/measured-pilot prerequisites remain unresolved.
The27-trajectory campaign remains unfrozen and the failed combined numerical
fixture remains closed.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/rank-predispatch-v2-component-review.json`.


## October 28 rank readiness receipts

The v2 readiness seal now supports rank slots9 through14 as well as the existing
LR slots. It binds the unique conditional frozen request and output allowance,
requires complete pilot/LR history and live rank-prefix predispatch verification,
and records the entire immutable evidence/control inventory, frozen context and
full remaining storage requirement. Post-review and post-guard checks preserve
source, controls, locks and free-space requirements. A failed or partially written
seal remains permanent evidence and cannot be retried. No dispatch or deletion
approval is granted; the child CLI still accepts LR slots only.

An initial eager import produced two collection errors before any test executed:
rank history already depends on readiness helpers. Rank-specific imports now
occur after module initialization when binding/verifying rank requests. The
original failure log and guard remain preserved. The repaired full suite passed
37 tests in202.28s:15 new rank tests and22 existing LR readiness tests. Both420s,
4GiB,8GiB actual CPU/disk guards finished normally; repaired peak RSS242,548,736
bytes. This import repair did not retry the closed combined numerical fixture.

Next implement rank child live-handoff/storage/entry and parent acceptance/serial
dispatch with every prior rank transport receipt. Fresh transitions, final rank
selection/cleanup/inference, full metadata accounting and actual nine numerical/
input/measured-pilot gates remain unresolved. No research training/MPS, checkpoint
deletion, closed-controller rerun or holdout evaluation occurred. The27-trajectory
campaign remains unfrozen and all scientific choices/guards remain unchanged.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/rank-readiness-v2-component-review.json`.


## October 28 rank child handoff and entry

Canonical sealed child argv and CLI now accept rank slots9 through14, requiring
the exact rank-startup and readiness references. Unsealed rank transport and
fresh confirmation slots remain rejected. Child entry inherits the live parent's
full idle audit only through its exact immutable seal hash, then verifies all
sealed evidence/control bytes, frozen numerical/source/request bindings, parent
identity, locks and bounded append-only worker log before producer import.

Rank storage rebinds the original pilot/LR cleanup receipts, original deleted and
retained pairs, and saved LR choice. It derives the three live LR anchors plus
completed rank members and counts only actual live occupied capacity. Deleted
weights remain historical evidence and cannot supply capacity credit. Complete
future storage,8GiB reserve,256MiB metadata allowance and the7200s child guard
remain unchanged; final source/evidence/control/log/parent/STOP/free-space checks
still run. Entry failure permanently blocks retry.

All94 tests passed in197.50s:10 new rank tests,40 LR entry regressions and44
transport regressions. The420s/4GiB/8GiB actual CPU/disk guard finished normally,
with peak RSS242,876,416 bytes. Tests used disposable byte fixtures and a
synthetic producer; no research worker, checkpoint deletion, failed combined
numerical retry, closed-controller rerun or holdout evaluation occurred.

Next implement rank parent acceptance and serial dispatch with every prior rank
transport receipt. Existing LR controllers are not rank launchers. Fresh
transitions, final rank selection/cleanup/inference, exact full metadata
accounting and actual nine numerical/input/measured-pilot gates remain unresolved.
The27-trajectory campaign remains unfrozen; the child CLI grants no campaign
launch authorization or quality claim.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/rank-entry-v2-component-review.json`.


## October 28 parent acceptance of completed rank children

A separate exact-slot rank parent acceptance API now shares the strict completed
process checks with LR acceptance. It verifies the complete same-parent sealed
predispatch review, immutable evidence/controls, frozen request/output allowance,
canonical launch and terminal records, exact output/log census and successful
7200s child guard. A successful exit alone cannot establish acceptance.

Rank review independently reconstructs all frozen LR/conditional rank requests,
verifies original pilot/LR cleanup history and saved rates, and re-audits every
live selected LR anchor and completed rank member including the new child.
Original records and anchor pairs must agree with preserved history. Full storage
receives all original pilot/LR pairs, including deleted losers, and live rank
pairs with both verified boundaries. Final source/inventory/guard/control/lock/
free-space checks remain. Even after the last rank screen, selection staysNone;
adoption, dispatch, deletion, cleanup and next-stage permission all stay false.

All50 tests passed in175.15s:17 new rank acceptance tests and33 LR regressions.
The420s/4GiB/8GiB actual CPU/disk guard finished normally with peak RSS241,729,536
bytes. First/middle/last rank cases, altered evidence/process records, independent
full audits/storage and late failures were tested with disposable synthetic
fixtures. No research child/model/MPS, checkpoint deletion, failed combined
numerical retry, closed-controller rerun or holdout evaluation occurred.

Next implement serial rank dispatch with complete verification of all earlier
rank parent/child transport receipts. Fresh transitions, final rank selection/
cleanup/inference, exact full metadata accounting and actual nine numerical/
input/measured-pilot gates remain unresolved. The27-trajectory campaign remains
unfrozen and all scientific choices and resource guards remain unchanged.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/rank-acceptance-v2-component-review.json`.


## October 28 serial rank dispatch and metadata-cap finding

The v2 controller now has separate exact-slot LR and rank dispatch APIs. Each
rank transition verifies original LR/pilot cleanup history, every earlier rank
parent/child receipt and guard, and all controls bound by earlier readiness
seals. It then prepares one frozen conditional request, supervises one child and
independently re-audits the live scientific prefix and full remaining storage.
Failures remain terminal. The final rank screen grants no selection, cleanup,
adoption or fresh-confirmation permission.

The first six-screen synthetic integration test exposed a metadata-cap failure:
four rank screens completed, then slot13 readiness publication reached the
unchanged2MiB evidence/meta cap. The run finished with46 passes and1 failure in
275.89s; its CPU resource guard finished normally. Raw failure output and the
complete synthetic file-size/hash census are preserved. This is separate from
the closed failed numerical fixture.

New metadata publications now use compact sorted ASCII JSON, preserving all
fields, streaming checks and the trailing newline. Existing evidence is never
rewritten. No cap, per-trajectory allowance,256MiB total or8GiB reserve changed.
With this source fix, all82 rank/LR/metadata-writer tests passed in287.28s.
A final historical-control tightening passed four focused regressions in89.83s,
including all six rank screens and all nine LR screens. Peak RSS was245,760,000
bytes across the passing runs; all outer CPU/disk guards finished normally.

The successful synthetic six-screen prefix uses1,985,851 metadata bytes,
leaving111,301 bytes under the2MiB sub-cap. This is **not** proof that final
selection, cleanup, nine fresh confirmations and inference metadata fit. Next
resolve their complete bounded representation/accounting and implement final
rank selection and cleanup with full reviews. Fresh transitions, inference and
actual nine numerical/input/measured-pilot prerequisites remain unresolved.
No research training/MPS, real checkpoint deletion or holdout evaluation occurred.
The27-trajectory campaign remains unfrozen and no research worker is active.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/rank-dispatch-v2-component-review.json`.


## October 28 lossless inventory references and metadata sizing

New readiness and rank-startup receipts use the explicitly versioned
`wide_hybrid_inventory_delta_v1` format. A receipt references an earlier immutable
inventory by document SHA256 and stores its changes plus the complete inventory
count/hash. Readers reconstruct the full map before existing provenance,
scientific, cleanup and storage audits. Old full-inventory documents remain
compatible and are never rewritten.

Decoding checks campaign identity, canonical paths and exact row types, both
physical hashes and the reconstructed digest. It rejects malformed or redundant
deltas, symlinks, traversal and unknown versions, and bounds rows, decoded bytes,
reference depth and total referenced bytes. The prospective plan explicitly
binds these limits. A delta removal only describes a map difference; reviewed
cleanup receipts and protected-evidence checks still govern actual weight removal.

All135 integration/provenance tests passed in470.61s, including the complete six
rank-screen sequence. The subsequent explicit-layout declaration passed75
codec/plan tests in2.04s. Both bounded CPU/disk guards finished normally; maximum
RSS was243,892,224 bytes. The measured synthetic prefix shrank from1,985,851 to
1,403,111 metadata bytes (582,740 bytes,29.34%) under the unchanged2MiB cap.
No per-group cap,256MiB total,8GiB reserve or scientific choice changed.

Actual read-only configuration sizing found355,839 compact bytes for the known
request/source/runtime/layout plan fields alone:3 pilot options and108 conditional
quality options still produce27 selected trajectories. This is a lower bound,
not a complete plan or campaign fit proof. Final rank selection/cleanup, fresh
confirmations, inference and actual numerical/input evidence still need complete
metadata accounting and compact immutable receipts before freeze. No research
worker, MPS training, real checkpoint cleanup or holdout evaluation ran.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/inventory-delta-v1-component-review.json`.


## October 28 final rank selection boundary

The new v2 rank selection component independently reviews the three live LR
anchors and all six rank screens, binds original pilot/LR cleanup history and
serial transport, and persists the prescribed minimum-H200-CE choice with the
larger-divisor tie rule. Complete remaining storage is checked before and after
selection, without credit for an intended rank cleanup. Immutable compact
receipts preserve records and original checkpoint pairs; read-only verification
reconstructs the full stage and manifest against the exact saved inventory.
Source and historical controls are rechecked immediately before completion.

All five initial tests passed in256.06s. After the final publication recheck,
all seven tests passed in375.85s, including tampering, incomplete panels, three
partial-publication failures and two late source/control changes. The driver
then raised an AssertionError in its optional census-export glob because pytest
truncated a temporary directory name. Its output and fixture-exception guard
are retained; this was after all tests passed, with no resource-limit breach.
A separately guarded read-only census captured the completed fixture using
its observed directory, without rerunning tests or changing fixture evidence.

The synthetic campaign uses1,420,508 of2,097,152 metadata bytes; the four new
selection documents total16,405 bytes. All18 live screen checkpoint files remain.
This is not a full real-panel metadata fit proof. Rank cleanup/cumulative history,
fresh transitions, inference and actual numerical/input/measured-pilot gates
remain unresolved. No research training/MPS, real checkpoint deletion or test
holdout evaluation occurred; the campaign remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/rank-selection-v2-component-review.json`.


## October 28 audited rank cleanup boundary

The new v2 boundary repeats the full saved-selection audit before cleanup and
binds an exact losing-weight manifest, original hashes and complete protected
inventory. It uses the locked, fsynced deletion executor. Compact immutable
receipts support strict idle verification of the cumulative pilot/LR/rank chain
without loading intentionally deleted weights. Frozen source, control files,
selected checkpoints, all non-checkpoint evidence and remaining capacity remain
mandatory. Canonical rank-cleanup review names are now admitted by the earlier
LR history reader as bytes only, with no deletion or dispatch credit.

All five full-boundary tests passed in319.35s and all36 affected LR-history
regressions passed in57.64s. Both CPU/disk guards finished normally; maximum
observed RSS was243,744,768 bytes. The equal-CE disposable fixture selected D/8
under the existing tie rule, removed12 losing files including former LR anchors,
and retained six selected files and every original non-checkpoint artifact.
The verifier checked30 cumulative deletions and rejected altered evidence or
recreated weights. Partial-unlink and completion-publication failures retained
their journals/failure evidence and blocked retry. No test/driver failures occurred.

Synthetic metadata is1,436,578/2,097,152 bytes, with111,636 additional bytes in
the separate5MiB rank-retention allowance. This does not prove complete real-panel
fit. Next implement retained rank history for fresh appends and fresh confirmation
preparation, then serial fresh/inference integration and complete metadata
accounting. Actual numerical/input/measured-pilot gates remain unresolved.
No research checkpoints were deleted, no model training/MPS or test-holdout
evaluation ran, and the campaign remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/rank-boundary-v2-component-review.json`.


## October 28 retained rank history and fresh confirmation audits

The new read-only rank-history reader permits declared fresh files while
reconstructing the original selection, execution identity, cumulative cleanup
chain and historical storage observation. It grants no current storage or fresh
scientific/dispatch credit. The existing strict idle verifier remains strict.
A separate prefix auditor checks the persisted LR/rank choices and independently
frozen requests, then reaudits all three selected screen anchors and every
completed fresh trial through producer, tensor, resource, validation and
numerical/input/corpus evidence. Pending requests receive no completed-trial
credit. Neither component selects again or adopts a recipe.

All156 confirmation/shared-fixture tests passed in8.25s; all40 history and
strict-boundary tests passed in160.47s. Both guards finished normally, with
maximum observed RSS245,481,472 bytes and no test/driver failures. The full
disposable chain retained all24 selected/fresh checkpoint files, kept30 reviewed
fixture deletions absent and preserved all original non-checkpoint evidence.
Corruption, undeclared files, resurrected weights and concurrent byte changes
were rejected. All zero-through-nine fresh prefixes were checked.

Synthetic metadata totals2,111,145 bytes, including1,436,485/2,097,152 bytes in
the separately capped meta group. Fresh parent-controller receipts and inference
are absent; this is not complete real-panel fit evidence. Next implement fresh
startup and its complete history/live-audit/transport/storage gates, then serial
fresh and inference integration and full actual metadata accounting. Actual
numerical/input/measured-pilot prerequisites remain unresolved. No research
weights were removed, no training/MPS or test-holdout evaluation ran, and the
campaign remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/confirmation-history-v2-component-review.json`.


## October 28 first fresh-confirmation startup

A new v2 component prepares only the first fresh request, slot15, after full
retained-history, selected-screen producer/tensor/resource/validation, frozen
request, numerical/input and complete remaining-storage checks. Its read-only
verifier reconstructs the pending request without completed-trial or checkpoint
credit, rechecks original identities and current capacity, and rejects changed
evidence or forged receipts. It launches no child and grants no dispatch credit.
Later fresh slots require earlier fresh transport receipts and remain unsupported
by this first-slot component.

All14 initial tests passed in295.65s, including insufficient space and source
changes before/after request creation. Failed attempts retain their evidence,
publish no successful receipt and cannot retry. The first synthetic receipt
was91,589 bytes because it repeated history. Replacing that repetition with
immutable reference-bound hashes reduced the receipt to5,073 bytes; all11
targeted post-compaction checks passed in89.63s. Verification reconstructs the
records, pairs, cleanup stages and frozen files and rejects forged hashes.
Both CPU/time/disk guards finished normally, maximum observed RSS243,744,768
bytes, with no test or driver failures. Both census versions are preserved.

The final disposable fixture has1,442,712/2,097,152 bytes in the meta group,
1,986,094 total metadata bytes and511 files. Six selected checkpoint files
remain; zero fresh trials completed. This is not a full real-panel fit proof.
Next integrate fresh readiness/child/parent transport and serial dispatch,
then inference and full actual metadata accounting. Actual numerical/input/
measured-pilot prerequisites remain unresolved. No research checkpoints were
deleted and no training/MPS or test-holdout evaluation ran; the campaign remains
unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/first-confirmation-startup-v2-component-review.json`.


## October 28 first fresh-confirmation readiness

Readiness sealing now accepts first-fresh slot15 through its exact startup
reference and frozen conditional request. It requires the complete idle audit,
original rank history and selected-screen checks, and independently reconstructed
remaining capacity. The seal binds the full evidence/control inventory, frozen
inputs, finished guard and output allowance. These are rechecked after the guard
finishes and before publication. No child transport or dispatch permission is
granted; later fresh slots remain unsupported.

All25 tests passed in417.23s, covering the new seal and failure cases plus
affected LR0/1/4/8 and rank9/14 readiness paths. Missing audit claims, late source
changes, late space loss and partial publication preserve failure/partial
evidence and block retry. The guard finished normally, maximum observed
RSS243,761,152 bytes, with no test or driver failure. Six selected fixture
checkpoints remain, with zero fresh trials completed; child transport still
rejects the new first-fresh seal.

The synthetic seal is16,358 bytes. The fixture uses1,458,831/2,097,152 bytes
in the meta group,2,001,503 total metadata bytes and513 files. This is not a
full real-panel fit proof. Next integrate first-fresh child entry, independent
parent acceptance and serial dispatch, then later fresh transport, inference
and full actual metadata accounting. Actual numerical/input/measured-pilot
prerequisites remain unresolved. No research checkpoint deletion, training/MPS
or test-holdout evaluation occurred; the campaign remains unfrozen.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/first-confirmation-readiness-v2-component-review.json`.


## October 28 continuation and first fresh-confirmation integration

The old tokenizer heartbeat is paused and a new five-minute continuation heartbeat
is active in this chat. Process inspection works again: no research workers were
found at entry or completion, and none were killed. The two initial pending commits
were pushed. First-fresh slot15 now has tested child entry, independent parent
acceptance and a single-slot serial dispatcher; it cannot advance to slot16.

Validation passed 7 child tests, 15 parent tests, 97 integration regressions and
96 parent/CLI regressions. A repeated-audit metadata issue was corrected: the
synthetic receipt shrank from 86,122 to 4,397 bytes using versioned audit digests
and original evidence references. The final synthetic meta group is
1,458,935/2,097,152 bytes; full real-panel fit remains unproven. The initial test
harness's mock-contaminated disk observations are preserved and disclosed; later
runs used the captured actual disk reader under unchanged resource guards.

This is controller preparation, not model-quality evidence. Later fresh transport,
inference, full metadata accounting and actual numerical/input/measured-pilot
prerequisites remain before freeze. No training, research checkpoint deletion,
closed numerical-fixture retry or holdout evaluation occurred. See
`tokenizer-evidence/wide-hybrid-preparation-001/first-confirmation-integration-v2-component-review.json`.


## October 28 first-fresh historical receipt verification

The compact first-confirmation receipt now has an independent read-only consumer
that reconstructs full scientific, cleanup, transport and capacity audits under
its original parent identity. A new controller PID/FD pair is allowed only on
this historical path; immediate parent acceptance stays strict. All 8 new tests
and 51 affected parent regressions passed under actual CPU/disk guards. Evidence
and all eight selected/fresh fixture weights remain unchanged.

Later fresh appends/slots, inference, full real metadata accounting and actual
numerical/input/measured-pilot gates remain before freeze. No model training,
quality result, research cleanup, closed-fixture retry or holdout evaluation.
See `tokenizer-evidence/wide-hybrid-preparation-001/first-confirmation-history-v2-component-review.json`.


## October 28 supervisor preparation defects found before launch

Realistic child evidence exposed two launch-blocking defects hidden by earlier
synthetic fixtures: fresh history rejected the real supervisor's control record,
and process output censuses counted pretty JSON despite compact publication.
Both are corrected; 84 focused checks and 72 affected regressions passed under
actual CPU/disk guards. Parent acceptance now verifies and retains the supervisor
hash, including its exact pre-terminal byte census; serial transport and historical
replay enforce that binding. Original reproductions and failed checks are retained.

This is preparation, with no new model-quality result. Later-fresh slots, inference,
full metadata accounting and actual numerical/input/measured-pilot gates remain
before freeze. No training, protected checkpoint deletion, closed-fixture retry
or holdout evaluation occurred. See
`tokenizer-evidence/wide-hybrid-preparation-001/supervisor-control-v2-component-review.json`.


## October 28 bounded later-fresh preparation history

A separate read-only contract now verifies completed slot15 history alongside
exactly a bounded slot16 preparation attempt and optional independently frozen
request. It preserves the original strict idle API and rejects undeclared
controls, process output and later slots. The pending request receives no
completed-trial or checkpoint credit and does not reduce the full future-space
requirement. All 60 focused checks and 34 separately bounded parent-acceptance
checks passed. The broader regression batch hit its unchanged 600-second cap
after 39 pass markers; that incomplete verdict and failed guard are retained.
This is controller preparation, with no new model-quality result.

Slot16 startup publication/predispatch, later child/parent dispatch, inference,
full metadata accounting and actual numerical/input/measured-pilot prerequisites
remain before freeze. No training, protected checkpoint cleanup, closed-fixture
retry or holdout evaluation occurred. Evidence:
`tokenizer-evidence/wide-hybrid-preparation-001/confirmation-append-v2-component-review.json`.


## October 28 slot16 preparation receipt

Slot16 now has exclusive request/compact receipt publication and an independent
predispatch recheck. Both bind completed slot15 transport/scientific history,
selected/fresh checkpoints, frozen choices, finished controls and full future
storage. Failed attempts retain evidence and cannot be retried; the earlier
public readers keep their stricter append contracts. This is preparation and
provides no new model-quality result.

Readiness/entry/parent dispatch and later fresh slots remain unfinished. Actual
numerical/input/measured-pilot gates, inference and complete real metadata fit
still precede freeze. No training, protected checkpoint cleanup or holdout use.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot16-startup-v2-component-review.json`.


## October 28 slot16 readiness and child-entry preparation

Slot16 now seals its full idle review with an explicitly bound readiness guard,
then validates a live child handoff against the compact startup receipt. Child
entry reconstructs16 completed records, three cleanup boundaries and all four
live selected/fresh pairs before recomputing full remaining storage. Existing
idle readers remain strict, and the bounded entry cannot be retried. This is
controller preparation and supplies no new model-quality result.

Slot16 parent acceptance/serial dispatch, later fresh transport, inference,
complete real metadata fit and actual numerical/input/measured-pilot gates still
precede freeze. Training has not launched; protected checkpoints and the closed
holdout are unchanged. Evidence and test verdicts:
`tokenizer-evidence/wide-hybrid-preparation-001/slot16-readiness-entry-v2-component-review.json`.


## October 28 slot16 parent acceptance and serial-transition preparation

The slot16 parent now audits17 completed quality records, both fresh checkpoint
pairs, selected screens, prior/current process records and complete remaining
capacity. A separate controller consumes the existing same-parent readiness seal,
repeats full guarded predispatch verification, supervises one child and publishes
a compact receipt only after independent acceptance and final integrity checks.
It stops before slot17; startup success and readiness flags alone do not dispatch.

This is synthetic controller preparation, with no new model-quality result.
Later fresh transport, inference, complete real metadata fit and actual numerical/
input/measured-pilot prerequisites remain before freeze. No training or holdout
use occurred. Evidence and bounded test verdicts:
`tokenizer-evidence/wide-hybrid-preparation-001/slot16-parent-integration-v2-component-review.json`.


## October 28 compact slot16 receipt replay

The new idle verifier independently reconstructs slot16's compact acceptance,
including historical parent/child transport,17 completed records, all five live
selected/fresh pairs, three cleanup boundaries and full remaining storage. It
also recomputes pre-slot16 capacity rather than trusting copied storage claims.
Later artifacts require their own explicit contract; verification grants no
dispatch or cleanup permission. Immediate acceptance still requires its original
parent.

This is controller preparation with no new model-quality result. Slot17–23
integration, inference and actual numerical/input/measured-pilot and real metadata
prerequisites still precede freeze. Evidence:
`tokenizer-evidence/wide-hybrid-preparation-001/slot16-history-v2-component-review.json`.


## October 28 slot17 preparation append contract

Slot17 now has an explicit read-only attempt/guard/pending-request contract around
full slot16 history replay. The new attempt must bind its current parent, and
the request must match the independently frozen option. Completed history and
required storage remain unchanged: a pending request earns no trial or checkpoint
credit. Existing idle readers stay strict, and no child dispatch is enabled.

Compact startup publication and later serial integration remain, followed by
inference, real metadata fit and actual numerical/input/measured-pilot gates
before freeze. This is synthetic controller preparation with no new quality
result or training. Evidence:
`tokenizer-evidence/wide-hybrid-preparation-001/slot17-append-v2-component-review.json`.

## October 28 slot17 compact startup publication and replay

Slot17 now has an exclusive compact startup receipt and independent consumer.
The receipt binds the current parent, frozen pending request and complete prior
history; verification reconstructs retained science, original transport and
storage. Failures remain immutable and cannot be retried. The synthetic receipt
is3,084bytes and retains all10 checkpoint files, with no credit for the pending
trial. Actual full-panel metadata fit remains unproven.

All17 bounded CPU checks passed on unchanged source, including preserved failure
paths, late-change rejection and existing-reader compatibility. No resource
guard failed and final process inspection found no research workers.

This is controller preparation, with no new model-quality result or training.
Slot17 readiness/entry and parent dispatch, remaining fresh slots, inference and
real numerical/input/measured-pilot and metadata gates still precede freeze.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot17-startup-v2-component-review.json`.

## Slot17 readiness and child-entry preparation

Slot17 readiness and child entry now bind the current parent and complete prior
history, reconstructing17 completed records and preserving all10 live checkpoint
files. Entry checks immutable evidence, original transport, source, storage and
resource guards before invoking a synthetic producer. Parent acceptance and
serial dispatch for slot17 remain pending; these checks establish no model-quality
or numerical result and launch no training.

The synthetic seal is22,275bytes. Its inventory chain uses11 of16 allowed
documents; later fresh slots must use a stable earlier anchor to avoid exceeding
the cap at slot23. Full real metadata fit, inference and actual prerequisite
gates still precede plan freeze. Evidence:
`tokenizer-evidence/wide-hybrid-preparation-001/slot17-readiness-entry-v2-component-review.json`.

All17 bounded checks passed on unchanged source, including failure preservation,
late-change rejection and existing slot15/16 compatibility. No resource guard
failed and final process inspection found no research workers.

## Slot17 parent acceptance and serial-dispatch preparation

Slot17 now has a separate parent acceptance API and one sealed serial transition.
Acceptance rechecks both earlier fresh transports and all current/prior scientific
evidence and storage. The synthetic success preserves all12 live checkpoint
files, records18 completed trials and publishes a4,918-byte compact receipt,
then stops before slot18. Failure evidence remains immutable and cannot be retried.

This is controller preparation with no new quality or numerical result and no
training launch. Independent completed-slot17 replay, remaining fresh slots,
inference and real numerical/input/measured-pilot and metadata gates still precede
freeze. Later inventory seals need a stable earlier anchor to respect the depth
cap. Evidence:
`tokenizer-evidence/wide-hybrid-preparation-001/slot17-parent-integration-v2-component-review.json`.

All44 bounded checks passed on unchanged source, including integrity rejection,
failure preservation/no retry and slot16 compatibility. No test or resource
guard failed, and final inspection found no research workers.

## Slot17 independent compact-history replay

The completed compact slot17 receipt can now be independently replayed under a
different parent/descriptor identity. The reader reconstructs18 completed
records, six live pairs, both earlier fresh transports, all three retention
boundaries and prechild/current/future capacity. It preserves all evidence and
controls and grants no launch, cleanup or adoption permission. The public parent
acceptance API remains same-parent.

This is synthetic controller preparation with no new quality or numerical result
and no training launch. Remaining fresh slots18–23 need bounded shared integration
and a stable inventory anchor, followed by inference and the real prerequisite
and metadata gates before freeze. Evidence:
`tokenizer-evidence/wide-hybrid-preparation-001/slot17-history-v2-component-review.json`.

All19 bounded checks passed on unchanged source, including changed-parent replay,
false receipt/capacity rejection, integrity checks and slot16 compatibility.
No resource guard failed, and final inspection found no research workers.

## Finite remaining-fresh append and storage contracts

Slot18 preparation now replays complete slot17 history with an explicitly bound
attempt, current-parent guard and optional frozen pending request. Shared naming
and storage contracts cover the finite remaining positions, while full replay for
slots19–23 remains closed pending integration. The pending request changes none
of the18 completed records, six retained pairs or required future storage.

This is synthetic controller preparation with no new model-quality result or
training. Compact startup publication, remaining serial integration, inference,
real metadata fit and actual prerequisite gates still precede freeze. Evidence:
`tokenizer-evidence/wide-hybrid-preparation-001/remaining-append-v2-component-review.json`.

All101 bounded checks passed on unchanged source, including the full storage
suite, slot18 preparation, prior API compatibility and integrity rejection.
No resource guard failed, and final inspection found no research workers.


## Compact fourth-fresh startup preparation

Slot18 now has compact startup publication and independent same-parent receipt
replay, binding the exact slot17 dispatch, all18 completed records, six retained
pairs, complete evidence/control digests and full remaining capacity. The older
idle and append-only readers remain strict, and failures prohibit retry.

This is disposable synthetic controller preparation, with no new model-quality
result or training. Slot18 readiness/entry, remaining serial integration, inference,
real metadata fit and actual prerequisite gates still precede finite-plan freeze.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot18-startup-v2-component-review.json`.

All20 guarded checks passed on unchanged source, including preserved failures,
late integrity rejection and prior reader compatibility. No resource guard
failed; final process inspection found no research workers. Next integrate
slot18 readiness and child entry using a stable earlier inventory anchor.


## Fourth-fresh readiness and child-entry preparation

Slot18 readiness and child entry now bind the complete18-record prefix, six
protected pairs and full remaining capacity before a bounded producer. The
readiness inventory uses the authenticated original slot16 seal as a stable
anchor, keeping its reference chain at11 documents and231,748bytes without
relaxing caps. Ordinary startup readers remain strict.

This is synthetic controller preparation, with no new model-quality result or
training. Slot18 parent acceptance/serial dispatch, later fresh integration,
inference, real metadata fit and actual prerequisite gates remain before freeze.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot18-readiness-entry-v2-component-review.json`.

All26 guarded checks passed on unchanged source, including preserved failures,
child integrity/capacity rejection, canonical CLI and prior API compatibility.
No resource guard failed; final inspection found no research workers. Next add
slot18 parent acceptance and one guarded serial transition.


## Fourth-fresh parent acceptance and serial preparation

Slot18 now has separate same-parent acceptance and a single guarded serial
transition. It rechecks original slot15–17 transport and all scientific evidence,
retains seven selected/fresh pairs, and publishes a compact5,037-byte parent
receipt after full remaining-storage checks. The stable slot16 inventory anchor
is explicitly required. The controller stops before slot19 and prohibits retries.

This is synthetic controller preparation with no new model-quality result or
training. Independent completed-slot18 history replay, later fresh integration,
inference, real metadata fit and actual prerequisite gates still precede freeze.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot18-parent-integration-v2-component-review.json`.

All42 guarded checks passed on unchanged source, including preserved failures,
no retries, exact stable-anchor validation, late integrity rejection and prior
slot17 compatibility. No resource guard failed; final inspection found no
research workers. Next independently replay the completed slot18 receipt before
integrating slot19 preparation.


## Independent fourth-fresh history replay

The compact completed-slot18 receipt now has strict independent replay under a
verified historical parent and descriptors. It reaudits all19 records, seven
retained pairs, original transport/retention history and current plus pre-child
capacity. All evidence remains unchanged; later artifacts and controls are
rejected, and pending slot19 requests remain closed.

This is synthetic controller preparation with no new model-quality result or
training. Slot19 preparation and remaining finite integration, inference, real
metadata fit and actual prerequisite gates still precede freeze.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot18-history-v2-component-review.json`.

All24 guarded checks passed on unchanged source, including historical capacity,
integrity and late-change rejection, exact API boundaries and slot17 compatibility.
No resource guard failed; final inspection found no research workers. Next add
explicit slot19 preparation while keeping the ordinary idle reader strict.


## Fifth-fresh preparation replay

Slot19 preparation now independently replays completed slot18 with an explicitly
bound attempt, current-parent guard and optional frozen request. All19 completed
records, seven retained pairs and full remaining-storage requirements stay
unchanged. Ordinary idle replay and immediate-parent acceptance remain strict.

This is synthetic controller preparation with no model-quality result or
training. Compact slot19 startup publication, remaining finite integration,
inference, real metadata fit and actual prerequisite gates still precede freeze.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot19-append-v2-component-review.json`.

All37 guarded checks passed on unchanged source, including capacity preservation,
API boundaries, integrity/late-change rejection and slot18 compatibility. No
resource guard failed; final inspection found no research workers. Next publish
and independently consume compact slot19 startup.


## Fifth-fresh startup publication and replay

Compact slot19 startup receipts now independently bind the complete completed
slot18 history, all19 records, seven protected pairs, original transport/retention
lineage, current parent and full remaining capacity. Failures remain preserved
without retries; ordinary readers stay strict and slot19 readiness stays closed.

All21 guarded checks passed on unchanged source, including tamper/late-change
rejection, failure preservation and slot18 compatibility. No resource guard failed;
final inspection found no research workers. This is synthetic controller
preparation, with no training or new model-quality result. Next complete slot19
readiness/entry using the stable slot16 anchor. Remaining integration, inference,
real metadata fit and actual prerequisite gates still precede finite-plan freeze.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot19-startup-v2-component-review.json`.


## Fifth-fresh readiness and sealed child entry

Slot19 readiness and child entry now bind all 19 completed records, seven
protected pairs, original transport/retention history and full remaining capacity.
The readiness inventory keeps the stable original slot16 anchor and unchanged
chain limits. Ordinary idle readers stay strict; parent dispatch still rejects
slot19. This is synthetic controller preparation, with no new model-quality result.

All 38 guarded checks passed on unchanged source, including protected-evidence
and late-change rejection, preserved failures/no retry, canonical child arguments
and slot18/slot19-startup compatibility. No resource guard failed; final inspection
found no research workers. Next finish slot19 parent acceptance and serial dispatch.
Remaining integration, inference, real metadata fit and actual prerequisite gates
still precede finite-plan freeze; training remains unlaunched.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot19-readiness-entry-v2-component-review.json`.


## Fifth-fresh parent acceptance and serial dispatch

Slot19 now has same-parent full acceptance and a bounded one-child serial
transition from its existing readiness seal. Acceptance reaudits all four prior
fresh transports and binds 20 completed records, eight protected pairs and full
remaining capacity. The compact receipt grants no automatic later launch or
adoption. The exact stable slot16 inventory anchor and existing caps remain.

All 40 guarded checks passed on unchanged source, including provenance/integrity,
late-change rejection, preserved failures/no retry, pre-child seal refusal and
slot18 compatibility. No resource guard failed; final inspection found no research
workers. This is synthetic preparation, with no new model-quality result or training.
Next add independent completed-slot19 replay, then remaining finite integration.
Inference, real metadata fit and actual prerequisite gates still precede freeze.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot19-parent-integration-v2-component-review.json`.


## Independent fifth-fresh history replay

Completed slot19 now has strict independent replay under verified historical
parent/descriptors, reauditing all 20 records, eight protected pairs, original
transport/retention history and current plus pre-child capacity. All evidence
remains unchanged; pending slot20 artifacts and controls remain closed.

All 25 guarded checks passed on unchanged source, including historical-capacity
and receipt forgery rejection, protected/late-change checks, exact identity/API
boundaries and slot18 compatibility. No resource guard failed; final inspection
found no research workers. This is synthetic preparation with no new model-quality
result or training. Next add explicit slot20 preparation. Remaining integration,
inference, real metadata fit and actual prerequisite gates still precede freeze.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot19-history-v2-component-review.json`.


## Sixth-fresh preparation replay

Slot20 preparation now independently replays completed slot19 with an explicitly
bound attempt, current-parent guard and optional frozen request. All 20 completed
records, eight protected pairs and full remaining-storage requirements stay
unchanged. Ordinary idle replay and immediate-parent acceptance remain strict.

All 37 guarded checks passed on unchanged source, including capacity preservation,
API boundaries, integrity/late-change rejection and slot19 compatibility. No
resource guard failed; final inspection found no research workers. This is
synthetic preparation with no model-quality result or training. Next publish and
independently consume compact slot20 startup. Remaining integration, inference,
real metadata fit and actual prerequisite gates still precede finite-plan freeze.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot20-append-v2-component-review.json`.


## Compact sixth-fresh startup receipt

Slot20 now has exclusive startup publication and independent compact-receipt
verification. Replay binds all20 completed records, eight protected pairs,
original slot15..19 transport/retention history and full remaining capacity.
Failures are preserved without retries; ordinary history readers stay strict.

All43 guarded checks passed on unchanged source, including publication/replay,
false claims and unreviewed outputs, preserved failures, late mutations/STOP,
API boundaries and slot18/19 compatibility. No resource guard failed; final
inspection found no research workers. This is synthetic preparation, with no
model-quality result or training. Next integrate slot20 readiness/child entry.
Remaining integration, inference, real metadata fit and actual prerequisite gates
still precede finite-plan freeze.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot20-startup-v2-component-review.json`.


## Sixth-fresh readiness and child entry

Slot20 now has an independently audited readiness seal and guarded child entry.
The stable slot16 inventory anchor preserves the existing chain caps. Entry
reconstructs all20 prior records and eight protected pairs, binds original
transport/retention history and verifies actual retained and remaining capacity.
Parent acceptance and serial dispatch remain closed.

All58 guarded checks passed on unchanged source, including protected evidence,
transport, late changes, preserved failures/no retry, exact CLI/API boundaries
and slot18/19 compatibility. No resource guard failed; no research workers
remain. This is synthetic preparation, with no new model-quality result or
training. Next implement slot20 parent acceptance and serial dispatch. Remaining
integration, inference, real metadata fit and actual prerequisite gates still
precede finite-plan freeze.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot20-readiness-entry-v2-component-review.json`.


## Sixth-fresh parent acceptance and serial dispatch

Slot20 now has same-parent full acceptance and a bounded one-child serial
transition from its existing readiness seal. Acceptance reaudits all five prior
fresh transports and binds21 completed records, nine protected pairs and full
remaining capacity. The compact receipt grants no automatic later launch or
adoption. The exact stable slot16 inventory anchor and existing caps remain.

All 90 guarded checks passed on unchanged source, including provenance/integrity,
late-change rejection, preserved failures/no retry, pre-child seal refusal and
compatibility. No resource guard failed; no research workers remain. This is
synthetic preparation with no new model-quality result or training. Next add
independent completed-slot20 replay, then remaining finite integration. Inference,
real metadata fit and actual prerequisite gates still precede freeze.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot20-parent-integration-v2-component-review.json`.


## Independent sixth-fresh history replay

Completed slot20 now has strict independent replay under verified historical
parent/descriptors, reauditing all 21 records, nine protected pairs, original
transport/retention history and current plus pre-child capacity. All evidence
remains unchanged; pending slot21 artifacts and controls remain closed.

All 25 guarded checks passed on unchanged source, including historical-capacity
and receipt forgery rejection, protected/late-change checks, exact identity/API
boundaries and slot19 compatibility. No resource guard failed; no research
workers remain. This is synthetic preparation with no new model-quality result
or training. Next add explicit slot21 preparation. Remaining integration,
inference, real metadata fit and actual prerequisite gates still precede freeze.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot20-history-v2-component-review.json`.


## Explicit seventh-fresh preparation

Slot21 attempt/guard/pending-request preparation now reaudits completed slot20:
21 records, nine protected checkpoint pairs, original transport/retention history
and current plus historical capacity. The pending request receives no completed
record, checkpoint or occupied-capacity credit. Ordinary replay remains strict.

All 39 guarded checks passed on unchanged source, including false-claim,
undeclared-artifact and late-change rejection, plus slot20 preparation/history
compatibility. No guard failed; no research workers remain. This is synthetic
preparation with no new model-quality result or training. Next add compact slot21
startup publication/reading. Remaining integrations, inference, real metadata fit
and actual prerequisite gates still precede freeze.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot21-append-v2-component-review.json`.


## Compact seventh-fresh startup receipt

Slot21 now has exclusive compact startup publication and independent full-history
verification. It binds completed slot20, all 21 records and nine protected pairs,
original transport/retention lineage, frozen identities and current plus
historical capacity. Failures remain preserved, retries are rejected and slot21
readiness/child dispatch remain closed.

All 46 guarded checks passed on unchanged source, including false-claim,
undeclared-artifact, publication-failure and late-change rejection, plus slot20
startup/history compatibility. No resource guard failed; no research workers
remain. This is synthetic preparation with no new model-quality result or
training. Next integrate slot21 readiness/entry. Remaining finite integrations,
inference, real metadata fit and actual prerequisite gates still precede freeze.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot21-startup-v2-component-review.json`.


## Seventh-fresh readiness and guarded child entry

Slot21 now has an immutable readiness seal and guarded child entry, binding
21 completed records, nine protected pairs, original transport/retention lineage,
frozen identities and full capacity. Its inventory chain retains the original
slot16 anchor and unchanged caps. The producer is mocked; no training launched.

All 50 guarded checks passed on unchanged source, covering integrity, capacity,
parent/locks/argv/STOP, late changes, preserved failures, retry rejection and
earlier-slot compatibility. No resource guard failed; no research workers remain.
Next integrate slot21 parent acceptance/serial dispatch. Remaining finite
integrations, inference, real metadata fit and actual prerequisite gates still
precede freeze; there is no new model-quality result.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot21-readiness-entry-v2-component-review.json`.


## Seventh-fresh parent integration and shared transport checks

Slot21 same-parent acceptance and single-child dispatch now bind all prior
transport, retained science, retention lineage and full capacity before a compact
success receipt. Three duplicated prior-transport checks now share one bounded
implementation for slots18..21; production audits remain independent and uncached.

All 87 pytest cases passed in two serial guarded batches, including
79 sequential negative scenarios on one disposable prefix, partial-publication
failure preservation, exact API/type gates and different-parent completed20 replay.
No resource guard failed. Synthetic completed state has 22 records and 20 protected
checkpoint files; its meta group uses 1,699,986 of 2,097,152 bytes.
Full real-panel metadata fit remains unproven.

No research training or new quality result. Next consolidate independent completed21
history and finite slots22..23, then inference and actual prerequisites before freeze.
Closed campaigns, failed numerical fixture and holdout remain closed.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/slot21-parent-integration-v2-component-review.json`.


## Complete finite confirmation integration through slot23

Slots22..23 now share startup, readiness, child entry and parent dispatch, with
one bounded history reader for completed21..23. It reconstructs all original
transport, retained science, retention lineage and current/historical capacity.
Different-parent replay is tested; immediate dispatch remains same-parent and
single-child. Slot23 rejects all later quality appends. The original16 inventory
anchor, chain caps, resource guards and every fresh pair remain protected.

All 181 checks passed in three serial bounded batches, including
20 sequential history mutations, new entry/API/CLI boundaries and legacy
compatibility. One initial combined sequence hit its600s wall-time guard; its
source and evidence are preserved. Removing a redundant test-only inventory
rescan allowed the fresh bounded run to pass without changing production audits
or limits. A separate negative-test path typo was corrected with its evidence
retained. Production code and the completed A2 checks were unchanged. No research
worker remains.

The synthetic final state has24 quality records and24 checkpoint files: three
selected screen pairs and all nine fresh pairs. Meta-group usage is
1,801,738/2,097,152 bytes; total metadata is 2,492,850 bytes.
This does not establish real-panel fit. No training or model-quality result.
Next complete v2 inference, real numerical/input/resource prerequisites and finite
pilot/quality freezes; the closed numerical fixture and holdout remain closed.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/final-confirmation-v2-component-review.json`.


### Retention-aware inference provenance for all three fresh seeds

The v2 idle inference gate now replays completed slot23 and binds all 24 quality
records, the three selected screening pairs, all nine fresh raw/H200 pairs and
original transport/retention lineage. All three seed triples are constructed in
one call with exact frozen requests, result hashes, tokenizer/input identity and
the existing export loader's binding schema. Deleted losing checkpoints are
proven through retention history rather than required to exist again.

Validation passed **5 tests plus six sequential adapter-negative scenarios** in
527.08 seconds under the unchanged 600-second, 4-GiB RSS and 8-GiB reserve guard.
A single completed synthetic prefix served all nine bindings. Production history
was not cached or bypassed; frozen scientific inputs and training contracts were
mocked as stated in `inference-binding-v2-component-review.json`. All
331 source/test hashes remained unchanged during validation.

This gate is read-only and returns no dispatch, adoption or deletion authority.
The strict idle final-history API remains unchanged. No receipt containing the
large live inventory is written; bindings remain in memory. Next implement one
shared, finite inference append/serial controller for seeds3511/3512/3513 using
compact references, exact output/control namespaces and existing loading/cache/
timing audits. Then audit real metadata/high-water fit, complete the separate
nine-case numerical/input prerequisites, freeze and measure resource pilots,
and freeze quality training. Training and inference have not launched; closed
campaigns, the closed numerical fixture and the test holdout remain closed.


### Real request-table metadata overflow and compact identity

The real 3-pilot/108-conditional-request table exposed a concrete metadata issue:
substituting only that table for the tiny synthetic plan raises the completed
meta group to **2,119,268 bytes**, above its **2,097,152-byte** cap by22,116 bytes.
This lower bound omits the rest of the real plan and prerequisite trace growth.
Evidence: `inference-plan-metadata-lower-bound.json`.

The plan gate now also accepts an exact, versioned request-table SHA256 plus
pilot and per-slot counts. It reconstructs and checks every request under frozen
source/runtime and returns the same full request table to downstream components.
Full-document compatibility remains tested. This reduces the request field from
317,940 to **229 bytes**, saving **317711 bytes**, without changing the
finite grid, resource caps or required audits. The real complete metadata and
transient high-water census remains a prerequisite; this is not a fit claim.

All **59 plan tests passed** in1.84 seconds, including compact/full equality,
729 selection combinations and tampered identity/reconstructed-config rejection.
Together with the earlier five inference binding tests, this iteration passed64
tests plus six sequential adapter negatives. Only the plan implementation/test
changed after the full inference-prefix run; both source manifests and the scope
of that validation are preserved in `inference-compact-plan-component-review.json`.
Use the compact request identity in the future frozen plan. Continue the shared
v2 inference controller, then real metadata/numerical/input/pilot prerequisites;
training, inference, protected deletion and closed-fixture retry remain unlaunched.


### Shared inference preparation and reusable synthetic replay

One immutable request now binds the complete fresh inference panel (seeds3511,
3512,3513). Preparation performs the full idle audit, appends only the request
and its bounded guard, then repeats the retained science/transport audit through
an explicit scope. Final quality, rank and pilot history remain strict by default;
only the independently validated inference reference admits these two files.
No worker dispatch, measurement, adoption or deletion authority is granted.

The initial integration test failed because the scoped reference was not yet
propagated to rank/pilot history. Its request, failed guard, failure receipt,
raw log and source versions remain intact. The correction was validated on
fresh attempts in two other completed disposable synthetic prefixes; the failed
prefix and closed numerical fixture were not retried. No resource guard failed.

The final guarded batch passed **29 tests**, including **19 sequential negative
scenarios**, completed replay from a different process/new lock descriptors,
late STOP/capacity failures and strict API compatibility. The reusable test
fixture pins every original file and reinstalls only synthetic input mocks;
all production audits execute again. Preparation plus negative scenarios took
60.57s instead of rebuilding the entire synthetic sequence for aboutnine minutes.
The final combined batch took143.29s under180s,4-GiB RSS and8-GiB reserve limits.

The request lives in the existing inference3511 metadata allowance and adds no
file to the2-MiB meta group. Every original fixture byte and all24 selected/fresh
checkpoint files were preserved. Real-panel metadata/high-water fit remains
unproven. Next integrate the shared serial child controller, entry/measurement
and completed-result acceptance against this request, including the active
control/log lifecycle; keep ordinary idle history strict. Then complete real
numerical/input/resource prerequisites and finite pilot/quality freezes before
training. Evidence: `inference-preparation-v2-component-review.json`.


### Bounded inference worker body and independent artifact review

A shared worker body now supports all three fresh seeds using the existing
export loader, loaded-model identity check, cache callbacks, timing sequence and
saved-result audit. It requires an independently implemented live entry verifier
at entry, after loading, after measurement and after publication. The controller
must still provide that verifier and supervise exactly one child at a time.
There is no new launch CLI or dispatch authority.

Independent saved-output review binds the attempt, selected raw/H200 pairs,
model metadata, timing arithmetic and finished worker guard. It rejects rehashed
corruption, failed/partial artifacts and changed resource limits. It explicitly
grants no transport, full-quality-history or next-seed acceptance; those remain
mandatory controller work. The600s worker,4-GiB RSS,6-GiB MPS and8-GiB reserve
limits remain, and the4-MiB output cap applies to the whole inference panel.

All **58 final tests passed in1.66s**, using mocked loading/MPS/live entry with
real bounded metadata producers and independent artifact/timing review. A
refactor regression that called entry verification before the initial panel-cap
check was corrected; its raw failure and source are preserved. No loading or
measurement ran in that failing case. No resource guard failed.

Next implement the actual live entry verifier, readiness/active-output scope,
supervisor transport acceptance and serial controller against these contracts.
Then complete real metadata/numerical/input/resource prerequisites and finite
freezes before training. No real inference/training, protected deletion,
closed-fixture retry or holdout use occurred.
Evidence: `inference-outputs-v2-component-review.json`.


### Compact first inference readiness and live entry verification

The first fresh inference triple now has a real live entry verifier composed
with the bounded worker body. Before readiness publication, the parent reruns
the complete retained quality, cleanup and historical transport audits. The
child reconstructs all nine fresh bindings from the existing immutable evidence
and validates their digest against the shared request, frozen source/runtime,
numerical/input contracts, original controls and selected/fresh checkpoints.
The readiness record is 996 bytes; no copied history manifest consumes the
per-seed metadata allowance.

The live verifier checks canonical argv and inherited lock descriptors, current
parent/child/process-group identity, exact phase-specific files, bounded parent
supervisor and worker guards. It permits the supervisor and log to advance
without treating those mutable files as frozen quality evidence. It rejects
replaced log prefixes, extra/future/partial outputs, changed protected bytes,
source/input bindings, STOP and resource-limit changes. Existing ordinary idle
and prepared-history scopes remain strict.

The final guarded batch passed **55 tests in32.55s**, including a full reused
completed synthetic history and all four composed worker boundaries. All 785
original prefix files and24 retained checkpoint files stayed unchanged. The
only append to that prefix was the small readiness record. One earlier test
incorrectly expected a new failure marker for an already-existing partial
output; the existing no-retry gate correctly refuses before creating a worker.
The test was corrected; failed raw evidence and original test source remain.

There is no launch CLI or serial controller yet. Real process creation, model
loading and MPS inference were not exercised; unit process identity and model
work remain synthetic. The first seed's live entry is implemented, while
seeds3512/3513 deliberately remain blocked pending completed transport review
and full idle-history replays between seeds. Implement that shared serial
integration next, then complete actual metadata/high-water, numerical/input and
resource prerequisites and finite freezes before training. No protected cleanup,
closed fixture retry or holdout access occurred. Evidence:
`inference-entry-v2-component-review.json`.


### Three-seed idle inference acceptance and full-history replay

A shared acceptance component now validates exact readiness, completed-output
and accepted-result states for seeds3511/3512/3513. Each accepted seed binds
its worker artifacts, original launch, canonical historical descriptors,
self-inclusive terminal output bytes and final supervisor observations. New
acceptance requires the same current parent and descriptors and a reaped owned
process group. Historical read-only replay supports new lock descriptors.

Acceptance and next readiness independently replay the complete retained
quality, cleanup and original transport history. Later readiness binds every
earlier accepted inference file/control without copying those histories. Only
explicit, fully checked idle inference references admit appended files into
lower quality-history readers; ordinary idle/prepared readers stay strict.
No process-success flag alone grants next-seed credit.

**85 tests passed** across83 focused cases, one isolated fixture-adapter test
and one full three-seed integration. The final integration executed ten full
history replays in328.76s under420s,4-GiB RSS and8-GiB reserve guards; observed
peak RSS was257,835,008bytes. All783 original successful-prefix files and24
retained checkpoint files remained unchanged. Its complete synthetic inference
metadata occupied71544bytes; this is not real-panel or transient high-water proof.

Two test-fixture integration failures are preserved. The old completed fixture
uses a placeholder token digest, so a range(96) input assertion first stopped
before any append. A second attempt exposed a missing adapter in the independent
timing reviewer, which correctly rejected that synthetic report and preserved
a failed worker. Both prefixes are closed to further worker appends. The exact
test-only token adapter was independently checked before a third, untouched
successful prefix completed the panel. Production token hashing and scientific
gates were unchanged; no resource guard failed or closed numerical fixture ran.

OS process identity, loading and timings in the panel integration remain
synthetic. The live entry is still intentionally first-seed-only and no launch
CLI or serial controller is exposed. Next integrate the shared later-seed live
namespace and actual serial supervision/cancellation, then finish real metadata,
numerical/input/resource prerequisites and finite freezes before training.
Evidence: `inference-acceptance-v2-component-review.json`.


### Shared three-seed inference CLI and guarded serial controller

The live entry now supports all three seeds through one finite namespace and a
canonical inherited-FD child CLI. Each later readiness binds the prior accepted
inference files and controller record. Metadata censuses during live entry
serialize against atomic supervisor writes; retained training weights are
hashed outside that lock.

The controller creates only one fresh child at a time. Preparation/review have
separate1800s CPU watchdogs, and the live600s phase uses cooperative parent
resource checks so STOP, time/RSS/disk/output failures and interrupts terminate
and reap the owned process group. Parent phases retain4-GiB RSS and8-GiB reserve;
the worker retains its6-GiB MPS guard and the whole inference panel retains its
4-MiB cap. SIGINT/SIGTERM record STOP/failure and restore prior handlers. No
failed or already-attempted seed is resumed. Exact successful parent guard
records append after acceptance and are independently checked on later replay.

**99 distinct tests passed** across112 executions, with no unexpected failure.
The final batch passed16 tests in47.70s and includes12 real child processes:
three canonical serial CLI children, controlled cancellation/resource/signal
failures, a second-seed loading failure and idle watchdog hard exits. Every
owned process group was reaped, prior successful evidence was retained, and no
third seed launched after the second failed. Loading, MPS and the scientific
inputs in these process tests are synthetic; CPU guards, argv, descriptors,
PIDs/groups, writers and transport checks are real. These are implementation
tests, not model-quality or inference-performance evidence.

A separate read-only test reran the complete retained science, cleanup and
transport history of the previously completed synthetic three-seed panel. All
818 original files and24 retained checkpoint files stayed unchanged. Its old
receipts remain readable; the new controller advances only from a preceding
guarded controller completion.

The controller returns three reviewed seed receipts pending the frozen
quality/cost decision; it does not adopt a recipe. Next audit actual complete
metadata/transient high-water fit, finish the separately reviewed numerical
and exact-input prerequisites, and freeze/measure resource pilots before the
quality-plan freeze and training. Real research inference/training remain
unlaunched; no protected cleanup, closed numerical retry or holdout access
occurred. Evidence: `inference-dispatch-v2-component-review.json` and
`inference-dispatch-v2-process-review.json`.

The final consolidated controller check passed **159 tests in65.98s** under the
180s guard. A source review narrowed the live frozen-identity lock to the
metadata census: hashing source/prerequisite files must not delay the
supervisor's bounded pipe/control publication. An independent lock test confirms
that snapshot reading is exclusive, file hashing leaves the lock available,
and only the exact campaign's MetadataWriter may request this live path.
Default idle identity verification is unchanged. The consolidated batch reran
real subprocess/cancellation tests and the complete old-panel replay; all818
historical files and24 checkpoint files remain unchanged. Final evidence:
`inference-dispatch-v2-final-review.json` and
`inference-dispatch-v2-process-review-final.json`.


### Actual training-input provenance and remaining metadata capacity

The first96 GPT-2 training tokens are now verified against the real frozen
Shakespeare corpus and inherited tokenizer contract. Only the80% training
text was tokenized; validation/test text was not encoded, and no model or
checkpoint was loaded. Ordinary-text encoding matched the independent
`encode_ordinary` entry point and round-tripped the entire training split.
The canonical uint32 little-endian input digest is
`0e77bdca8368d30a983e0b1ef033bd141411c35ff00c8f9e7cc1e05c6a276812`.
The three immutable input artifacts occupy27,838bytes and bind current
source/runtime. A separate review independently recomputed the input digest
and rechecked all artifact hashes. The180s/4GiB/8GiB guard finished in0.19s,
with262,733,824bytes peak RSS. No numerical optimizer test ran.

The initial input-only attempt failed before encoding because the default
GPT-2 cache was absent and sandbox DNS blocked the implicit download. Its
guard and original script remain preserved and closed. The two public assets
were restored into ignored `research_state/tiktoken-cache`, verified against
the installed tiktoken SHA256 pins, and a new fresh input-only attempt passed.
The revised procedure refuses implicit downloads and requires a verified cache.
Future workers must receive that explicit cache path. This did not retry the
closed failed combined continuation fixture.

A guarded read-only census verified all818 retained synthetic-prefix files,
including24 checkpoint files, unchanged. Substituting the known real plan/input
fields and propagating the known frozen-file maps through34 occurrences gives
a **partial** meta-group projection of1,900,578bytes, leaving196,574bytes of
the2,097,152-byte cap. Nine additional numerical-artifact references leave
134,987bytes;18 leave73,400bytes, before their payloads and other unmodeled
growth. These figures are sensitivity calculations, not a complete upper bound
or fit claim. The actual numerical payloads, inventory propagation, full-duration
resource records, final controller/decision receipts and transient overlap
still require a complete audit. Existing metadata/resource limits are unchanged.

Evidence: `training-input-and-metadata-review.json`,
`training-input-v2-002/meta/training-input.json`, and
`prerequisite-metadata-v2-review.json`, under the wider-hybrid preparation
evidence directory. Next implement and review the separate serial nine-case
rank-policy numerical procedure with bounded disk-backed checkpoint lifetimes,
then execute its fresh cases once. Do not rerun the closed combined fixture.
Complete actual metadata and quality/cost aggregation review, freeze/measure
resource pilots, and freeze the finite quality plan before research training.


### Separate frozen nine-case CPU numerical procedure begins

The separately reviewed rank-policy procedure is frozen in
`rank-numerical-v2-frozen-plan.json`; its executable run plan hashes to
`d89c3e501c80f8391da471dafde93884e5f635df5f93048a576c69e9729e32a1`.
Five small streaming-hash/config/freeze/STOP/provenance orchestration tests
passed in0.50s under the180s/4GiB/8GiB CPU guard. The prior four-test batch
also passed and is retained. No closed continuation test is imported or called.

This fresh prerequisite uses nine width512 family/rank cases, with baseline,
partial-save and resume isolated in27 serial CPU processes. The existing trainer,
model, sampler, optimizer, checkpoint reader and EMA math remain unchanged.
The partial checkpoint is written directly to disk; baseline and resumed
end states are streamed into complete hashes without retained serialized
buffers. Each phase has600s/4GiB/8GiB guards, inherited global/run locks and
owned-process cleanup. Each case has a2GiB output cap; the initial complete
reservation is26GiB versus roughly51GiB observed free. Failures stop the panel
and preserve all artifacts. This is four-update correctness validation with
H200 snapshots1–4, not a resource pilot or quality trajectory.

The first complete case, hybrid D/8, passed all three phases. Full model,
optimizer, EMA and torch/loader RNG hashes exactly matched baseline versus
resume. Independent scalar EMA recurrence checks sampled the first/middle/last
values of every parameter; complete EMA state equality complements that sample.
The inference-only EMA load matched every stored parameter, preserved buffers
and correctly rejected optimizer resume. Phase times were74.58/21.31/58.44s;
maximum RSS was3,855,155,200bytes (3.59GiB). Its981,866,923-byte partial checkpoint
is retained. The remaining eight cases are still pending/running serially.

Evidence: `rank-numerical-v2-procedure.md`,
`rank-numerical-v2-procedure-review.json`, and
`rank-numerical-v2-first-case-review.json`. Actual running evidence is under
`research_runs/tokenizer-wide-hybrid-numerical-v2-001`. Monitor this controller;
do not start a competing worker or retry its existing directory. After all27
phases finish, independently review original guards/process/receipt/checkpoint
hashes and export the compact numerical trace/review. No nine-case numerical
pass, MPS pilot, quality-plan freeze or adoption is claimed yet. The closed
failed combined fixture and test holdout remain closed.


### Nine-case rank-policy numerical prerequisite completed

All nine width512 cases (hybrid, gated-MLA and KDA at divisors8/4/2)
completed27 isolated CPU phases under the frozen separate procedure. For each
case, four uninterrupted updates exactly matched one disk-checkpointed update
plus three resumed updates in complete model, optimizer, EMA, torch RNG and
loader RNG hashes. Every saved floating tensor was checked finite. Independent
scalar EMA recurrences sampled first/middle/last values of every parameter;
complete EMA equality, inference-only loading, persistent-buffer preservation
and optimizer-resume rejection also passed. These are four-update correctness
checks, not1,200-step quality, general stability or MPS measurements.

Every600s/4GiB-RSS/8GiB-reserve guard finished successfully. Peak RSS was
3,890,331,648bytes (3.62GiB); the longest phase was87.57s and total worker
time was1,395.45s. All nine partial checkpoints remain retained, totaling
8,858,999,191bytes. No failed artifact was replaced or cleaned up, and the
old failed combined continuation fixture was never called. No worker remains.

The independent terminal reviewer verified all121 original files, all27
process/guard/receipt chains, exact argv/config/plan identities and original
checkpoint hashes before and after review. Eight small reviewer checks passed,
including rejection of failed guards, mismatched resume/checkpoint hashes,
extra/missing files, nonfinite recurrence diagnostics and wrong argv. Together
with the five earlier procedure checks, these are13 distinct small tests; the
27 actual CPU phases are reported separately. The reviewer strengthened exact
file-set/hash/finite checks before this first terminal review; it did not change
the frozen worker. Geometry/cache coverage reuses the unchanged, hash-verified
prior rank evidence, rather than claiming new cache runs.

The compact numerical package contains only two meta files, totaling65,396bytes,
and passes the production `numerical_review` binder. Original metadata for112
non-checkpoint files is also archived byte-for-byte in the portable terminal
review; original checkpoints and run records remain at their recorded paths.
Evidence: `rank-numerical-v2-reviewed-001/meta/numerical.json`,
`rank-numerical-v2-terminal-review.json`, and the terminal-review guard/test logs.

A fresh disposable identity plan then exercised `bind_campaign` without mocks,
using the actual numerical package, the27,838-byte train-only input package,
current source/runtime/corpus and the complete3-pilot/108-conditional-request
table represented by its compact exact identity. It passed with11 bound files,
a40,760-byte plan and142,859 initial metadata bytes. The gate correctly returned
false for dispatch authority, complete storage fit and measured pilots. This
identity-only plan is not the resource-pilot or quality campaign freeze, and
its initial census does not prove future metadata/transient high-water fit.
Evidence: `actual-prerequisite-binding-v2-review.json` and guard.

Next finish the complete real metadata/inventory/transient envelope and final
quality/cost aggregation review, then freeze and measure the serial resource
pilots and freeze the finite quality plan before training. Source/runtime
changes require explicit prerequisite review; do not silently reuse these
source-bound passes. MPS pilots, research inference and quality training remain
unlaunched. All selected/fresh evidence and the test holdout remain protected.


### Finite wider-hybrid freeze and final decision wiring — October 29, 2026

The real numerical/input prerequisites are now frozen for `/Users/junyi/Documents/ChatGPT/LLM-pretraining-RSI/research_runs/tokenizer-wide-hybrid-001`;
plan SHA256 `bb44ea3b55db81b6808072f36d09a9eb8c5d263aa18bc88d06acc88b4ffc19d5`. The plan fixes three resource pilots and
24 quality trajectories (108 conditional request options), with the existing
serial worker, retention, memory, time, storage and closed-holdout constraints.
Quality remains gated on measured and independently accepted pilots.

The first virtual metadata estimate missed552 occurrences of synthetic
`raw.pt` names. Corrected accounting reserves real timestamped filenames and
binds both the preparation and final-decision sources. The numerical trace is
now a lossless gzip attachment:63,746 original bytes become12,576 bytes;
exact decompression and the unmodified production identity gate pass. Original
reports, traces and every numerical checkpoint remain untouched. No numerical
or synthetic training fixture was repeated. Corrected meta bound is
2,058,492 of2,097,152 bytes, leaving
38,660 bytes. Every other group fits its
existing cap; the conservative count is854 of2048 files. Full-duration quality
resource reports/events are bounded at1,282,827/1,106,155 bytes before remaining
fixed-schema padding; metrics, sample text, controller receipts and control
replacement overlap are included. Success-schema estimates remain subject to
the actual bounded writers and watchdogs; unexpected output stops the run.

`review_final_campaign_v2.py` now composes strict quality OR (neutral quality
AND measured memory), after full production acceptance replay of all fresh
triples. It checks its own frozen source hash, requires all three controller
receipts, binds result hashes, and rechecks protected files. It preserves the
conditional-bootstrap caveat and reports latency without a speed gate; it does
not itself modify an adopted recipe. Seventeen new small tests passed across
two bounded runs (14 arithmetic/boundary cases,3 file-binding wrapper cases).
No model or timing worker was used in these tests. The final external review
has a128KiB cap, additionally reserved at freeze.

Evidence: `finite-campaign-freeze-v2-review.json`, its finished guard, the exact
`finite-campaign-v2-frozen/meta/` archive, and the two final-decision test logs.
The earlier `full-metadata-v2-review.json` is retained as an explicitly
superseded estimate. Source/runtime hashes match the completed numerical and
train-only input passes. At freeze no resource pilot or quality worker has
launched. Next: serial resource pilots through the existing production
`run_pilot_once`, then the independent pilot boundary and quality readiness.


### Real MPS resource pilots accepted — October 29, 2026

All three width512/divisor2 resource pilots completed and passed independent
production acceptance: hybrid49.04s, gated MLA41.69s, KDA56.64s. Each used
20 updates. Maximum observed RSS was3,233,792,000bytes (3.01GiB), and maximum
MPS driver allocation was4,112,433,152bytes (3.83GiB), below4/6GiB guards.
Raw/H200 prefill/decode/reset cache checks passed at both batches; maximum
absolute error was4.65e-6. Tensor finiteness and exact EMA-export checks passed.
These establish measured local feasibility, not1200-step stability or quality.

The authorized pilot retention boundary independently re-audited all receipts,
executed its exact six-file pilot-checkpoint manifest, verified the deletion
journal and protected noncheckpoint evidence, and passed remaining full-campaign
storage. Numerical partials, selected checkpoints and fresh confirmations were
outside that manifest. All original pilot metadata bytes plus the subsequent
boundary changes are archived in `resource-pilot-reviewed-v2.json` and
`resource-pilot-boundary-v2-archive.json`. The complete boundary ref is
`meta/pilot-boundary-complete.json`, SHA256
`455b3f962b50b7e9f6bd72fa1e81707ff23cbcdf412ba946893d04564a6c6afc`.

The finite quality grid was fixed before pilots. Next is its first1200-step LR
trial through the existing production readiness/serial-dispatch path, using
this accepted boundary and the same frozen source/runtime. No research quality
result or adoption is claimed by these pilot observations.


### First quality readiness failure and narrow correction — October 29, 2026

Campaign `research_runs/tokenizer-wide-hybrid-001` is now closed after its first
quality request failed before a process launch. The error was
`Readiness inputs changed after guard completion`; quality slot00 contains only
its request, with no process directory or training output. All pilot successes,
audited retention receipts and the failed readiness/dispatch evidence are
preserved. This campaign and failed attempt must never be retried.

The production binder returns live `metadata_bytes` alongside frozen identity.
Readiness compared the entire returned object across its own guard's completion,
although the legitimate final guard write can change byte length. Prior small
fixtures omitted that live census field. Read-only forensics reproduced the
comparison sensitivity using the real binder result; the exact starting-guard
bytes were not retained, so no exact historical byte delta is claimed.

The correction excludes only that census counter from the post-guard frozen
comparison, checks its range, and retains current census enforcement, finished
guard validation, immutable inventory, every other control, full storage and
lock checks. All26 focused readiness tests passed, including actual byte-count
growth from a valid finished guard and rejection of changed frozen files,
source context and other controls. No closed fixture or campaign was rerun.

`readiness-fix-source-applicability-v2-review.json` verifies that this is the only
changed source file and runtime is identical. Trainer/model/optimizer/EMA/cache
and input/tokenizer code are unchanged, so the nine-case numerical and exact
train-input evidence remain applicable by explicit review, without claiming
those old runs executed under the new hash. A new package and finite freeze
must bind both source identities, that applicability review and original
evidence. No quality training has started; replacement campaign gates remain.


### Replacement campaign002 frozen — October29,2026

The pre-freeze audit found the live-census comparison in child entry as well
as readiness. Entry now excludes only that counter while preserving all
append-only log, immutable inventory, control, transport and storage checks.
All40 distinct entry cases pass:39 passed initially, with an overflow error
message mismatch because the real census rejected it earlier; all5 log cases
passed after updating that test expectation in a fresh small fixture. Initial
log/source preserved. The unchanged readiness fix retains its26-case pass.

The explicit source-applicability review verifies only these two control files
changed; the other226 source hashes and runtime match the actually executed
numerical/input evidence. Campaign002 binds a lossless attachment containing
the original reports and applicability review. No numerical/input rerun and
no transplant of old pilot/process/cleanup receipts occurred.

Plan SHA256 `dba216b378680ecd56641051c00c5077cb99006f45858ddc03bb23f338cfc769` fixes3 pilots,24 quality trajectories and108
conditional requests under the same scientific/resource/retention rules.
Meta envelope: 2,073,259 of2,097,152bytes; 23,893bytes
remaining. Other groups fit; file peak855. The unmodified production binder
passes. The freeze/evidence commit is e04f4a0. Campaign001 remains closed.
The serial starter will run3 new guarded pilots, verify their retention
boundary, then admit only quality slot0 through production readiness.


### Campaign002 resource acceptance and first quality training — October29,2026

All3 new resource pilots passed independent acceptance with raw/H200 cache
checks and tensor audits. Durations were47.69s, 41.76s, 56.53s
for hybrid, gated MLA and KDA. Peak RSS was3,233,333,248
bytes; peak MPS driver memory was4,112,433,152
bytes, within the original4/6GiB limits. The exact authorized pilot-retention
manifest completed and its boundary was independently verified. Original
pilot metadata and boundary files are archived; no unrelated or protected
weights were targeted.

Quality slot0 has passed the corrected readiness and live-child handoff and
started real MPS training: width512 hybrid, divisor4, LR0.000625, seed3501,
1200steps. Parent PID3352, child PID3463 at this observation.
Only one training worker is active. The starter stops after this one quality
trial; future LR slots require their production gates and explicit next
continuation. No quality result, selection or adoption is claimed yet.
Follow actual process/events and `campaign002-terminal-v2.json`; never launch
a duplicate while it runs, and preserve/close any failure without retry.


### Campaign002 first quality result independently verified — October29,2026

Slot0 completed1200 MPS updates at width512: hybrid, divisor4,
LR0.000625, seed3501. Full-validation H200 CE is4.961644521
and raw endpoint CE is4.988754896, over34278 targets
and111536 target bytes. This is one LR-screening cell, not a family comparison,
selection or adoption. Sampled online validation did not choose the checkpoint;
the closed test remains unused.

Child wall time was789.85s. Sampled peak RSS was2702704640bytes
and MPS driver memory was3969826816bytes, below the unchanged4/6GiB
limits. The completed entry guard also passed. Independent read-only replay
verified the original dispatch transport, all scientific LR-prefix artifacts,
raw/H200 hashes, pilot-retention lineage, frozen source/input bindings and
remaining full-panel storage. About43.30GB was free against29.38GB required.
The exact non-weight campaign bytes and all file hashes are preserved in
`tokenizer-evidence/wide-hybrid-preparation-001/campaign002-lr-prefix-01-v2-review.json`;
the independent review guard finished successfully. Both slot0 checkpoints remain.

The one-slot continuation wrapper calls the unchanged production `run_lr_once`
for a reviewed prefix and stops after that slot. Next is frozen slot1:
all-gated-MLA, divisor4, LR0.000625, seed3501,1200updates. All9 LR
cells must complete before rate selection or eligible screen cleanup. No
training implementation, frozen plan, guard or statistical rule changed.

Slot0 review/evidence was committed and pushed as `cd46ebc`. Frozen LR slot1
started through production gates with parent4486 and child4497. The wrapper
stops after this slot; later slots require fresh prefix review and serial dispatch.


### Campaign002 LR slot1 independently verified — October29,2026

All-gated-MLA at width512/divisor4, LR0.000625, seed3501 completed1200updates.
Full-validation H200 CE is4.953339260; raw endpoint CE is4.920094199.
Hybrid minus gated-MLA H200 CE at this one LR/seed is+0.008305262;
this incomplete screening comparison supports no selection or adoption.
H200 remains the declared selector even when the raw endpoint scores lower.
Child wall time was467.00s, sampled peak RSS3,169,337,344bytes and
MPS driver3,567,042,560bytes. All guards passed; test holdout stays closed.

Independent replay of both completed LR cells passed, including original
transport/control receipts, checkpoint hashes, scientific artifacts, pilot
retention history and frozen bindings. Exact non-weight bytes and file hashes
are archived in `tokenizer-evidence/wide-hybrid-preparation-001/campaign002-lr-prefix-02-v2-review.json`;
its guard finished successfully. Both raw/H200 pairs remain protected during
the incomplete screen. Remaining full-panel storage requires28,162,784,220bytes
against42,054,352,896bytes free. Next is frozen slot2: all-KDA at the same
LR/divisor/seed/width/budget, through the unchanged one-slot production wrapper.

The two-cell review was committed and pushed as `ce4d9e7`. Frozen LR slot2
started through production gates with parent5171 and child5184. Exactly one
training worker is active; the wrapper stops after this all-KDA trial.
