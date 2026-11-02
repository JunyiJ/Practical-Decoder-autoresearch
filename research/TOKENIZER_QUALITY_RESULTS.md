# GPT-2 tokenizer quality results

> **Current tokenizer continuation:** start with the [resume guide](TOKENIZER_RESUME.md) and [compact state index](TOKENIZER_RESUME_STATE.json). These distinguish active tokenizer work from historical stopped campaigns. Check live state before dispatch.

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


### Campaign002 first LR trio independently verified — October29,2026

All-KDA slot2 completed1200updates at width512/divisor4, LR0.000625,
seed3501. Full-validation H200 CE is4.947694237; raw CE is4.975362476.
The first LR trio is hybrid4.961644521, gated-MLA4.953339260 and
KDA4.947694237. These single-seed screening differences do not determine
family superiority, rate selection or adoption; the remaining six LR cells,
rank stage and fresh confirmations retain their frozen rules.

Child wall time was909.66s; sampled peak RSS2,600,484,864bytes and MPS
driver4,104,044,544bytes passed the unchanged guards. Independent replay
of all three completed cells passed the transport/scientific/checkpoint/input
and retention-history checks. Lossless non-weight bytes and all hashes are
archived in `tokenizer-evidence/wide-hybrid-preparation-001/campaign002-lr-prefix-03-v2-review.json`;
its review guard finished. All three raw/H200 pairs remain. Full remaining
storage requires26,907,661,636bytes against40,776,089,600bytes free.
Test holdout is unused. Next is frozen slot3: all-KDA, LR0.00125,
divisor4, seed3501, width512 and1200updates, through production serial gates.

The three-cell review was committed and pushed as `f04fcfc`. Frozen LR slot3
started through production gates with parent6128 and child6139. The sole
training worker runs all-KDA at LR0.00125; the wrapper stops after this trial.


### Campaign002 second-rate KDA result — October29,2026

Slot3 all-KDA at LR0.00125, width512/divisor4, seed3501 completed1200updates.
Full-validation H200 CE is4.980911690, worse by0.033217454 than its
LR0.000625 screen; raw endpoint CE is5.084563022. Faster early sampled
validation improvement did not predict the final H200 ranking. This is a
single-seed LR comparison, not final rate selection or evidence of a general
architecture disadvantage. The frozen grid is unchanged.

Child wall time was911.94s; sampled peak RSS2,629,910,528bytes and
MPS driver4,104,044,544bytes passed the original guards. Independent replay
of all four completed cells verified transport, scientific records, checkpoint
hashes, frozen source/input and retention history. The lossless archive is
`tokenizer-evidence/wide-hybrid-preparation-001/campaign002-lr-prefix-04-v2-review.json`;
its guard finished. Remaining full-panel storage requires25,652,539,052bytes
against39,483,129,856bytes free. All four raw/H200 pairs remain; no test use,
cleanup, selection or source change occurred. Next is slot4: hybrid at
LR0.00125, with the same divisor4/seed3501/width512/1200update budget.

The four-cell review was committed and pushed as `1a531c5`. Frozen LR slot4
started through production gates with parent7031 and child7063. The sole
training worker runs hybrid at LR0.00125; the wrapper stops after this trial.


### Campaign002 second-rate hybrid result — October29,2026

Slot4 hybrid at LR0.00125, width512/divisor4, seed3501 completed1200updates.
H200 full-validation CE is4.981121139, worse by0.019476618 than its
LR0.000625 screen; raw endpoint CE is5.083567067. Both hybrid and KDA
currently score worse at the middle rate than the lower rate on this screening
seed. The higher-rate cells, gated-MLA middle-rate cell and fresh confirmation
remain necessary; no grid change, rate selection or adoption follows.

Child wall time was806.65s; sampled peak RSS2,612,396,032bytes and MPS
driver3,969,826,816bytes passed the unchanged guards. Independent review of
all five completed cells passed original transport, scientific/checkpoint,
frozen-input and retention-history checks. Exact non-weight bytes and hashes
are archived in `tokenizer-evidence/wide-hybrid-preparation-001/campaign002-lr-prefix-05-v2-review.json`;
its guard finished. Full remaining storage requires24,405,766,076bytes
against38,221,434,880bytes free. All five raw/H200 pairs remain and test
holdout is closed. Next is frozen slot5: gated-MLA at LR0.00125,
divisor4, seed3501, width512 and1200updates.

The five-cell review was committed and pushed as `a1b0bf1`. Frozen LR slot5
started through production gates with parent7791 and child7818. The sole
training worker runs gated-MLA at LR0.00125; the wrapper stops after this trial.


### Campaign002 closed on slot5 checkpoint-lock timeout — October29,2026

Slot5 gated-MLA at LR0.00125 logged all1200updates but failed during raw
checkpoint output. The parent reported `Metadata producer lock timeout; no retry`
and terminated/reaped child7818 with returncode−15 after456.04s. No research
worker remained on process inspection. There is no complete raw/H200 pair,
training summary, full-validation result, tensor/resource audit or acceptance
for slot5. Its sampled training metrics cannot be used for selection. The failed
761,768,510-byte raw `.pt.tmp` is protected, SHA256
`2744ab5d95a972152be379d6adf49fc2125fecf8b942499b227a881a4c67ebb3`.

The forensic review verifies every file from the accepted five-cell archive is
unchanged and preserves all current non-weight bytes, hashes and relevant source
bytes in `tokenizer-evidence/wide-hybrid-preparation-001/campaign002-lock-failure-v2-review.json`.
Its independent guard finished. No cleanup or retry occurred. Last sampled RSS
and MPS peaks were616,054,784 and3,567,042,560bytes, respectively; sampled
resource limits did not trigger. These samples do not establish an instantaneous
peak after the final sample. Test holdout stays closed. Campaign002 is now closed;
slot6 must not launch, and the unsuccessful slot5 cannot be resumed or retried.

The shared metadata lock has a2s nonblocking acquisition deadline. Checkpoint
writes release it between1MiB chunks and immediately compete to reacquire it;
this provides no fairness guarantee. The evidence is consistent with starvation
or a slow critical section, but no lock trace distinguishes them. Existing tests
cover tiny chunk boundaries and a640-byte child producer with explicit sleeps;
they do not establish progress for unpaced sustained writes with a larger
campaign census. Do not relabel this infrastructure failure as a numerical or
quality regression.

Next prerequisite is a new isolated, bounded producer/supervisor contention
fixture: no model/corpus/training, one child, fresh paths, explicit source hashes,
wall/RSS/disk/output caps, actual lock-wait/hold durations, sustained unpaced
checkpoint chunks and simultaneous supervisor/resource publication. First retain
baseline evidence without changing production source; then design and test a
versioned progress fix, preserving fail-closed behavior, metadata/inode/byte
integrity, storage checks and independent resource supervision. No arbitrary
lock-timeout increase or disabled guard is justified by this failure. A fresh
source/evidence applicability review and finite campaign plan are required before
any later training; neither002 nor any earlier closed fixture is reopened.


### Checkpoint namespace handoff mitigation — October29,2026

A separately frozen nontraining baseline used one child, an800-file synthetic
metadata census and56MiB of unpaced checkpoint bytes. It preserved the original
2s lock deadline and1MiB chunks,180s parent/150s child limits,4GiB RSS,
8GiB reserve and64MiB total output cap. It completed in1.88s; the parent
waited up to0.40315s while the longest measured hold was0.02817s. Only one
resource sample occurred. This did not reproduce the original failure; its
larger census and much smaller checkpoint differ from002's250files/761MiB
failed partial. Instrumentation can also affect scheduling. Baseline plan,
script, original source and lossless metadata are retained.

Only the production checkpoint writer changes: after each accumulated MiB it
yields20ms outside the namespace lock, including across small serialization
callbacks. This gives other publishers an interval to acquire the lock before
more checkpoint output. It preserves the2s deadline and all byte/inode/hash,
capacity, disk, memory and supervision checks. It adds intentional output time;
it is a cooperative mitigation, not a guarantee of scheduler fairness.

All134 checkpoint/metadata/process tests pass under the bounded test guard.
New tests verify both supervisor and resource publishers finish during a
namespace-free handoff, and small callbacks accumulate yields without changing
bytes. Existing lock-timeout, partial retention, mutation, storage and process
failure checks still pass. A distinct changed-source fixture with the same
56MiB/800file workload completed in3.91s: parent maximum wait0.05603s,
maximum hold0.03078s and three resource samples. Both fixtures verified exact
output hashes; these single observations do not establish general performance
or prove the original failure is eliminated.

The independent report `tokenizer-evidence/wide-hybrid-preparation-001/checkpoint-handoff-v1-review.json`
archives the tests and changed-source fixture and verifies every closed002
file remains unchanged, including five accepted pairs and the failed partial.
No training, numerical panel, closed fixture or failed trial was rerun. An
initial report command used system Python and failed importing a dependency
before any review mutation; the report then completed under the repository
venv. Both original fixture attempts remain immutable.

Before further training: explicitly review source applicability of this
scheduling-only change, preserve original executed-source provenance, and
re-audit a fresh finite plan's complete storage/metadata envelope including
all002 protected weights. Old001/002 remain closed. Do not use the passed
synthetic fixture as a pilot, numerical or quality result, or reuse002 process
receipts as a new campaign's transport provenance.


### Campaign003 source applicability and finite freeze — October29,2026

The explicit applicability review compares against the original executed
numerical/input source. Exactly three production files differ: the two already
reviewed census checks and checkpoint-writer scheduling. The other225 source
hashes and runtime match. The CPU numerical procedure uses the unchanged
checkpoint save/read helpers; input encoding has no checkpoint-output dependency.
Serialization/guard coverage includes the134 passed tests. Original numerical
and input reports, hashes and executed-source identity remain in the lossless
applicability attachment; neither prerequisite was rerun. New real pilots must
still validate this source's actual raw/H200/export/resource behavior.

Campaign003 is a fresh complete panel under the handoff source, not a continuation
of002's failed attempt. The original candidate grid, seeds, budgets and selection
rules remain unchanged:3 pilots,9 LR cells,6 rank cells,9 fresh confirmations,
then3 inference seeds. The reused screening seed3501 is not independent
replication;002's five accepted cells remain descriptive and are excluded from
003 selection. No old process, pilot or cleanup receipts are transplanted.
Old001/002 stay closed and every retained/failed checkpoint remains protected.

Plan SHA256 `4a08476fe21b92ae8debc7d7b004dad8b0f082637e76e65ad7cdc33f22f85102`.
The unmodified production binder accepts the complete numerical/input bindings.
The conservative meta envelope is2,075,524 of2,097,152bytes
(21,628 remaining), all other groups fit, and the file peak is855.
Full future storage plus the external review allowance requires30,631,412,484bytes
against37,209,874,432bytes actually free. That free-space reading already includes
all6,987,283,846bytes of protected002 weights/partial; no speculative cleanup
credit is used. Reserve and all time/memory/metadata guards stay unchanged.
The freeze and applicability guards finished successfully.

After committing this freeze, the serial starter may run only the3 new resource
pilots, verify their exact authorized retention boundary, then admit quality
slot0 through production gates and stop. Any failure closes that new attempt;
no automatic retry, source change or later quality dispatch occurs.

Campaign003 freeze/evidence was committed and pushed as `bc8e49b`. The starter
is active with parent9293 and initial pilot0 child9294. Only one local MPS
worker is active. No quality trial has launched at this observation; actual
pilot acceptance and retention gates precede the starter’s sole quality slot0.


### Campaign003 pilot acceptance and first quality dispatch (2026-10-29)

All three fresh resource pilots completed and were accepted: hybrid89.5785s,
gatedMLA83.7171s, KDA100.3218s. These are20-step resource/export checks at
width512, divisor2, LR0.0025 and seed3491, not quality comparisons.
The immutable pilot archive contains82 file hashes and76 exact noncheckpoint
byte records. Archive byte hashes and the three dispatch references were checked.
The verified retention boundary removed exactly six authorized pilot weight
files totaling3,769,780,224bytes; their archived hashes match the deletion manifest.
No selected, confirmation, numerical or failed-attempt checkpoint was removed.

Boundary reference `meta/pilot-boundary-complete.json` SHA256
`711f3f95cb61c692bf27257e901cca26b625592e5043afe9984a6865a369c8b8`.
The standalone boundary/storage review is evidence only; its dispatch-authorized
flags remain false. The production quality gates separately admitted slot0,
hybrid/divisor4/LR0.000625/seed3501, with1200 planned updates.
At observation parent9293 and sole worker9618 were active, with no process error;
no quality acceptance or new quality finding is claimed yet. The starter stops
after this slot. Independently review its accepted prefix before any later slot.
Source and plan remain frozen. Campaigns001/002 remain closed, test holdout closed.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/campaign003-pilots-v2-archive.json`,
`campaign003-pilot-boundary-v2.json`, and `campaign003-quality0-start-v2.json`.
Read-only archive inspection initially resolved evidence-relative paths against
the campaign root and stopped with missing-path/key errors; resolving against
`evidence/` completed verification. No artifact or worker was changed by these checks.


### Campaign003 first LR cell independently reviewed (2026-10-29)

Slot0 completed1200 updates: hybrid, width512, divisor4, LR0.000625,
seed3501. Full-validation H200 CE is4.957330844412886 and raw CE
4.985172284648273, on34278 targets/111536 bytes across17 blocks.
The child completed successfully in832.5694s; its raw/H200 pair totaling
1,246,772,976bytes remains retained. This single screening cell establishes
no family comparison or adoption decision; closed002 results are not pooled.

The independent prefix1 review verified production prior transport, frozen
identity, pilot history, result/checkpoint bindings, all115 file hashes and
remaining storage, and archived exact nonweight bytes. Its CPU guard finished
in4.5116s with no reason/error and peak RSS220,086,272bytes. Available space
35,935,940,608bytes exceeded required29,384,508,436bytes. No cleanup occurred.
The external reviewer and one-slot continuation were adapted from002 solely
by changing campaign identity/path strings; source equivalence and syntax were
checked. Production source, finite plan and guards remain unchanged.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/campaign003-lr-prefix-01-v2-review.json`,
its guard, and `campaign003-terminal-v2.json`. After committing this evidence,
the sole next dispatch is slot1: gatedMLA, divisor4, LR0.000625, seed3501.
The continuation stops after that slot, preserves failures and never retries.
No selection, fresh confirmation, inference or closed-test result is claimed.


Prefix1 evidence and serial wrappers were committed/pushed as `5a3f5c0`.
The sandbox-only launch preflight could not see MPS and stopped at its availability
assertion before any launch receipt, campaign mutation or child. Host execution
then passed that same preflight with CPU fallback disabled. Slot1 controller
PID10515/session57898 is active through the original production gates; it may
run only this one frozen cell. This is not a retried training attempt.
Launch identity is preserved in `campaign003-lr-01-v2-launch.json`.


### Campaign003 second LR cell independently reviewed (2026-10-29)

Slot1 gatedMLA completed1200 updates at width512/divisor4/LR0.000625,
seed3501. Full-validation H200 CE4.953336207757286; raw CE4.92009252749222.
Child elapsed508.1716s. Relative to slot0 hybrid, H200 CE is lower by
0.003994636655600 on this screening seed; the incomplete adaptive panel
supports no superiority or adoption conclusion. Both raw/H200 pairs are retained,
2,468,497,192bytes total. No closed002 result is pooled.

Independent prefix2 review verified144 file hashes, losslessly archived all
nonweight evidence, and passed production prior/frozen/pilot/checkpoint/storage
bindings. The external guard finished in8.7815s, peak RSS220,069,888bytes,
with no error. Observed free34,695,581,696bytes exceeds required28,162,784,220.
No cleanup, selection or closed-test evaluation occurred. Evidence is in
`tokenizer-evidence/wide-hybrid-preparation-001/campaign003-lr-prefix-02-v2-review.json`,
its guard and `campaign003-lr-01-v2-terminal.json`.
After committing/pushing this review, dispatch only slot2 KDA at the same
LR/divisor/seed through the original one-slot controller. Frozen source and
all resource guards remain unchanged.


Prefix2 review committed/pushed as `ea6f4e9`. The sole slot2 KDA controller
PID11085/session26358 is active through production gates; its immutable
receipt is `campaign003-lr-02-v2-launch.json`. It stops after this cell.


### Campaign003 third LR cell independently reviewed (2026-10-30 UTC)

Slot2 KDA completed1200 updates at width512/divisor4/LR0.000625, seed3501.
Full-validation H200 CE4.941342525602911; raw CE4.968353925571475.
Child elapsed952.9232s. The first LR group is complete: H200 CE hybrid
4.957330844412886, gatedMLA4.953336207757286, KDA4.941342525602911.
KDA is lowest at this setting on one screening seed; the full nine-cell LR
panel must finish before family-specific LR selection. No superiority,
confirmation or adoption claim follows from this partial panel.

Independent prefix3 review verified173 file hashes and losslessly archived
nonweight evidence. Production prior/frozen/pilot/checkpoint/storage bindings
passed; external guard finished13.0889s with peak RSS224,231,424bytes and no
error. All three raw/H200 pairs remain retained,3,723,619,776bytes total.
Observed free33,422,503,936bytes exceeds required26,907,661,636bytes.
No cleanup or closed-test evaluation occurred; closed002 results are not pooled.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/campaign003-lr-prefix-03-v2-review.json`,
its guard and `campaign003-lr-02-v2-terminal.json`.
After commit/push, the sole next cell is slot3 KDA at LR0.00125 with unchanged
width/divisor/seed/budget and original production gates. Frozen source and
resource guards remain unchanged.


Prefix3 review committed/pushed as `427f1d1`. Slot3 controllerPID12027
(session88858) is active through production gates for the single frozen KDA
LR0.00125 cell. Launch receipt: `campaign003-lr-03-v2-launch.json`.


### Campaign003 fourth LR cell independently reviewed (2026-10-30 UTC)

Slot3 KDA completed1200 updates at width512/divisor4/LR0.00125, seed3501.
Full-validation H200 CE4.978608155903051; raw CE5.076564651378714.
Child elapsed954.1204s. Increasing KDA LR from0.000625 to0.00125 raised
H200 CE by0.037265630300140 on this screening seed. This descriptive result
favors the lower of these two rates, but selection remains deferred until all
nine frozen LR cells complete. No closed002 data are pooled and no fresh-seed
or adoption claim follows.

Independent prefix4 review verified202 file hashes and losslessly archived
nonweight evidence, with production prior/frozen/pilot/checkpoint/storage
bindings intact. Guard finished17.3942s, peak RSS224,575,488bytes, no error.
All four raw/H200 pairs remain retained,4,978,742,360bytes total. Observed
free32,149,258,240bytes exceeds required25,652,539,052bytes. No cleanup or
closed-test evaluation occurred. Evidence:
`tokenizer-evidence/wide-hybrid-preparation-001/campaign003-lr-prefix-04-v2-review.json`,
its guard and `campaign003-lr-03-v2-terminal.json`.
After commit/push, the next sole cell is slot4 hybrid at LR0.00125 with the
same width/divisor/seed/budget. Production source, plan and guards remain frozen.
Old heartbeat is confirmed paused and this continuation heartbeat active.


Prefix4 review committed/pushed as `3da4671`. Slot4 controllerPID12854
(session74741) is active through production gates for the single frozen hybrid
LR0.00125 cell. Launch receipt: `campaign003-lr-04-v2-launch.json`.


### Campaign003 fifth LR cell independently reviewed (2026-10-30 UTC)

Slot4 hybrid completed1200 updates at width512/divisor4/LR0.00125,
seed3501. Full-validation H200 CE4.975304379509017; raw CE5.07451207831503.
Child elapsed850.2786s. Hybrid H200 CE is0.017973535096131 higher than
its LR0.000625 cell on this screening seed. Five of nine LR cells are complete;
family-specific LR selection remains deferred. Closed002 results are not
pooled; no fresh-seed or adoption claim follows from this partial panel.

Independent prefix5 review verified231 file hashes and losslessly archived
nonweight evidence. Production prior/frozen/pilot/checkpoint/storage bindings
passed; guard finished21.7198s, peak RSS224,280,576bytes, no error. All five
raw/H200 pairs remain retained,6,225,515,336bytes total. Observed free
30,871,764,992bytes exceeds required24,405,766,076bytes. No cleanup or
closed-test evaluation occurred. Evidence:
`tokenizer-evidence/wide-hybrid-preparation-001/campaign003-lr-prefix-05-v2-review.json`,
its guard and `campaign003-lr-04-v2-terminal.json`.
After commit/push, the next sole cell is slot5 gatedMLA at LR0.00125 with
unchanged width/divisor/seed/budget. This is the originally frozen003 panel;
closed002's failed slot5 remains preserved and is not resumed or retried.
Production source, plan and guards remain frozen.


Prefix5 review committed/pushed as `714fd96`. Slot5 controllerPID13822
(session26123) is active through production gates for the single frozen gatedMLA
LR0.00125 cell. Launch receipt: `campaign003-lr-05-v2-launch.json`.


### Campaign003 sixth LR cell independently reviewed (2026-10-30 UTC)

Slot5 gatedMLA completed1200 updates at width512/divisor4/LR0.00125,
seed3501. Full-validation H200 CE4.821469271057781; raw CE4.824475110326953.
Child elapsed529.3563s. H200 CE improved by0.131866936699505 versus gatedMLA
at LR0.000625; hybrid and KDA instead worsened at this higher rate on the same
screening seed. GatedMLA is0.153835108451236 lower than hybrid at LR0.00125.
These descriptive comparisons motivate completing equal family-specific tuning;
six of nine frozen cells are complete, with no LR selection or adoption yet.
Fresh confirmations and final conditional criteria remain necessary.

The checkpoint/export/validation path completed successfully under the frozen003
handoff source. This does not prove the prior metadata timeout impossible;
closed002's failed cell and partial checkpoint remain preserved, not resumed.
Independent prefix6 review verified260 file hashes and losslessly archived
nonweight evidence; production prior/frozen/pilot/checkpoint/storage bindings
passed. Guard finished29.4563s, peak RSS224,624,640bytes, no error. All six
raw/H200 pairs remain retained,7,447,239,552bytes total. Observed free
29,623,996,416bytes exceeds required23,184,041,860bytes. No cleanup or
closed-test evaluation occurred; closed002 results are not pooled.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/campaign003-lr-prefix-06-v2-review.json`,
its guard and `campaign003-lr-05-v2-terminal.json`.
After commit/push, next is solely slot6 gatedMLA at LR0.0025, unchanged
width/divisor/seed/budget, original production gates, frozen source and guards.


Prefix6 review committed/pushed as `eb6c9f3`. Slot6 controllerPID14687
(session77627) is active through production gates for the single frozen gatedMLA
LR0.0025 cell. Launch receipt: `campaign003-lr-06-v2-launch.json`.


### Campaign003 seventh LR cell independently reviewed (2026-10-30 UTC)

Slot6 gatedMLA completed1200 updates at width512/divisor4/LR0.0025,
seed3501. Full-validation H200 CE4.85658606647173; raw CE4.919580532579774.
Child elapsed535.4267s. H200 CE is0.035116795413949 higher than gatedMLA
at LR0.00125. Its three-rate screen is complete:0.000625/0.00125/0.0025
produce4.953336207757286/4.821469271057781/4.85658606647173 respectively.
The middle rate is descriptively lowest; formal selection remains deferred
until the two remaining frozen hybrid/KDA cells complete. No adoption claim.

Independent prefix7 review verified289 file hashes and losslessly archived
nonweight evidence; production prior/frozen/pilot/checkpoint/storage bindings
passed. Guard finished35.0949s, peak RSS224,526,336bytes, no error. All seven
raw/H200 pairs remain retained,8,668,963,768bytes total. Observed free
28,387,024,896bytes exceeds required21,962,317,644bytes. No cleanup or
closed-test evaluation occurred; closed002 results are not pooled.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/campaign003-lr-prefix-07-v2-review.json`,
its guard and `campaign003-lr-06-v2-terminal.json`.
After commit/push next is solely slot7 KDA at LR0.0025 with unchanged
width/divisor/seed/budget, original production gates, frozen source and guards.


Prefix7 review committed/pushed as `4ad370f`. Slot7 controllerPID15590
(session89694) is active through production gates for the single frozen KDA
LR0.0025 cell. Launch receipt: `campaign003-lr-07-v2-launch.json`.


### Slot7 reporting-label correction (2026-10-30 UTC)

Prior prose incorrectly named slot7 hybrid. The unchanged frozen plan and actual
request specify slot7 KDA at LR0.0025; slot8 is hybrid. The production controller
correctly dispatched KDA, with four KDA layers, divisor4/width512/seed3501.
A read-only check matched the actual request spec to frozen panel slot7, verified
the original plan digest and unchanged LR-rule source hash. No execution or
selection change was needed. The erroneous label affected commentary and these
notes only; numerical results for completed slots0..6 remain unchanged.
The active KDA worker remains sole worker15757 under controller15590.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/campaign003-slot7-label-correction-v1.json`.


### Campaign003 eighth LR cell independently reviewed (2026-10-30 UTC)

Slot7 KDA completed1200 updates at width512/divisor4/LR0.0025, seed3501.
Full-validation H200 CE5.114296068829509; raw CE5.279939868603322.
Child elapsed975.6345s. H200 CE is0.172953543226598 above KDA at LR0.000625.
KDA's three-rate screen gives4.941342525602911/4.978608155903051/5.114296068829509
for ascending rates. The lowest rate is descriptively best on this seed;
formal family-specific selection still waits for the final hybrid cell.
Eight of nine LR cells are complete; no adoption or fresh-seed claim.

Independent prefix8 review verified318 file hashes and losslessly archived
nonweight evidence. Production prior/frozen/pilot/checkpoint/storage bindings
passed. Guard finished40.8564s, peak RSS224,641,024bytes, no error. All eight
raw/H200 pairs remain retained,9,924,086,352bytes total. Observed free
27,098,804,224bytes exceeds required20,707,195,060bytes. No cleanup or
closed-test evaluation occurred; closed002 results are not pooled.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/campaign003-lr-prefix-08-v2-review.json`,
its guard and `campaign003-lr-07-v2-terminal.json`.
The actual frozen panel identifies final slot8 as hybrid at LR0.0025,
width512/divisor4/seed3501/1200steps. Dispatch only that cell after commit/push.
After acceptance, use the original full-panel LR selection and retention
components with independent archive before authorized cleanup. The prefix
reviewer deliberately stops at8 completed cells; do not alter production gates.


Prefix8 review committed/pushed as `1b50d3c`. Slot8 controllerPID16764
(session51551) is active through production gates for the final frozen hybrid
LR0.0025 cell. Launch receipt: `campaign003-lr-08-v2-launch.json`.


### Campaign003 closed at final LR slot8: metadata lock timeout (2026-10-30 UTC)

Final hybrid/LR0.0025 slot8 failed with
`TimeoutError('Metadata producer lock timeout; no retry')`. Controller16764
stopped/reaped worker16918 (returncode-15), child elapsed811.4618s, wrapper
893.9771s. Actual process inspection confirms no remaining research worker.
The surfaced traceback places the parent at `run_process -> save_control ->
MetadataWriter.publish_json -> _serialized`. All1200 updates were logged, but
raw checkpoint publication did not finish. There is no completed raw/H200 pair,
training summary, training-outputs, result, tensor/resource audit, worker-outputs
or parent acceptance for slot8. Online metrics are not selection evidence.

The protected partial is
`research_runs/tokenizer-wide-hybrid-003/evidence/quality/08/checkpoints/final_001200_1793325864452219000.pt.tmp`,
780,502,912bytes, SHA256
`6a572d5c2a72198e56de2e188940e24161f2cbcea367af55743d0bcaef546f64`.
Independent failure audit verified all337 campaign file hashes twice and every
previously archived prefix8 file unchanged; nonweight bytes and relevant source
were archived losslessly. All eight accepted pairs plus partial remain protected,
10,704,589,264bytes total. Failure-review guard finished8.8228s with no error,
peak RSS228,900,864bytes. Failed worker sampled peaks were RSS634,847,232bytes
and MPS driver3,969,826,816bytes, under original caps; samples do not bound the
instant of failure. No cleanup, LR selection, rank dispatch or test evaluation
occurred. Campaign003 is closed; no retry/resume/replacement cell is authorized
by this evidence. Closed001/002 and the failed numerical fixture stay closed.

The20ms-per-MiB cooperative handoff did not eliminate the timeout. No recorded
lock-acquisition/hold trace distinguishes starvation, a slow filesystem operation,
a long census or other critical-section delay. The earlier56MiB synthetic checks
were not a reproduction of this failure and cannot establish safety at this
checkpoint size. Do not simply extend the2s deadline or repeat training.
Next prepare a separate finite nontraining diagnostic with actual-scale synthetic
checkpoint writes and bounded per-process/thread lock-wait/hold and operation
traces. Preserve original guards, freeze its script/source/workload and allow
one fresh attempt; no model/corpus/checkpoint load and no rerun of closed fixtures.
Only diagnose before choosing another versioned implementation change.

At this audit26,295,799,808bytes were free. The unchanged full-panel initial
requirement30,631,412,484bytes would exceed that by4,335,612,676bytes, before
new diagnostic outputs. Protected evidence gets no speculative cleanup credit.
A new full training panel therefore needs a separately verified storage solution
as well as an evidence-backed fix and new finite plan; it must not auto-launch.

Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/campaign003-lr-08-v2-terminal.json`,
`campaign003-lock-failure-v2-review.json`, its guard and
`review_campaign003_lock_failure_v2.py`. Previous eight cells remain descriptive
screening evidence, not a completed panel or adoption result.


### Prospective actual-scale contention trace v2 (2026-10-30 UTC)

Freeze one separate nontraining diagnostic after closed003's repeated timeout:
`tokenizer-evidence/wide-hybrid-preparation-001/run_metadata_contention_trace_v2.py`
and `metadata-contention-trace-v2-plan.json`. The new output root is
`research_runs/tokenizer-metadata-contention-trace-v2-001`; no existing fixture
or campaign is resumed. All228 production/runtime-input source hashes match003.
The synthetic CPU payload has four240MiB float32 tensors plus128 small tensors,
serialized through actual torch.save and the unchanged bounded checkpoint writer.
337 flat synthetic control files approximate the failed campaign's file count.

One child only, four CPU threads,150s child/180s parent limits, original4GiB RSS,
8GiB reserve,2s namespace-lock deadline and20ms-per-MiB handoff. Checkpoint cap
964MiB, total output cap992MiB, each external JSON cap4MiB, at most16384 trace
records per process. Both worker locks and original process supervisor remain.
Process-local observational wrappers time the original lock scopes, census,
write and fsync operations with monotonic clocks/PIDs/thread names. No source
mutation, model/corpus/training, existing checkpoint read or test-holdout use.

This is a new scale/serialization diagnostic, not a rerun of the closed56MiB
checks or failed numerical fixture. Instrumentation changes scheduling; flat
synthetic files and CPU tensors do not reproduce nested topology, tensor layout,
prior MPS/dirty-page state or machine load. A successful run cannot prove the
failure eliminated or authorize training. Preserve every outcome, independently
review traces/hashes and unchanged closed003 evidence, then choose further work
from observations. New full-panel storage remains blocked. Syntax and source
pins checked before execution; commit/push this finite plan before its sole run.


### Actual-scale trace v2: timeout reproduced and small-write starvation observed

Frozen plan committed/pushed as `010ff39`; exactly one synthetic attempt ran.
It failed closed with the same 2 s metadata producer-lock timeout. No worker
remains, no retry occurred, and no production source was changed. The new
synthetic partial is 1,006,654,080 bytes, SHA 256
`1c4b101ebf2e022be9035a6662b165bcb05b75e71cafd6972aaf571aa6c8ccfb`.
All 350 diagnostic files are preserved (1,011,820,960 bytes total, below 992 MiB).

During the parent's 2.002416167 s failed acquisition, the child made 154 checkpoint
writes totaling 11,504 bytes, with write sizes 4..256 bytes. 310 overlapping child
lock scopes occupied 1.961031242 s (97.9333%) of that interval. Maximum overlapping
hold was 0.012483916 s; maximum child operation starting in the interval was
0.007236292 s. Across the whole saved trace, parent/child maximum holds were
0.024327625/0.023709708 s. The last 1 MiB write preceded this small-write tail.
The child recorder completed 44 samples with no resource-recorder error.

This supports starvation in this instrumented synthetic run: many small
callbacks repeatedly acquire the namespace lock for census/write checks, while
only 11.5 KiB is accumulated during the parent's deadline. The byte-only 1 MiB
handoff therefore fails to schedule a gap in that interval. No individual long
critical section appears in the saved trace. This narrows a credible mechanism
but does not prove the uninstrumented 003 failure had the identical cause.

Instrumentation has a separate failure: parent trace reached its 16384-record
cap during later failure publication/shutdown. The original lock timeout is
record 16305; later records continue after its end. Per-JSON-piece metadata
writes consumed 15206 trace records. The final traced fsync raised the cap error,
and a partial process-control JSON remains. Child termination caused a secondary
torch.save unexpected-position error. Both fixture guards report exceptions;
this is a failed diagnostic with an incomplete final trace tail, not a passed
test. Saved trace sizes 4,074,367/1,023,669 bytes remain below their 4 MiB caps.

Independent read-only review verified frozen source/script pins, all diagnostic
hashes, original output limits and every one of closed 003's 337 files unchanged.
It archived all noncheckpoint bytes losslessly and hashed the partial without
loading it. Review guard finished 5.4562 s, peak RSS 264,224,768 bytes, no error.
Worker/parent peaks were 1,232,322,560/236,929,024 bytes, below 4 GiB. No cleanup,
training, corpus access or test-holdout use. Evidence:
`tokenizer-evidence/wide-hybrid-preparation-001/metadata-contention-trace-v2-review.json`,
its guard and `review_metadata_contention_trace_v2.py`.

Next design a handoff bounded by elapsed work as well as bytes, so small
callbacks cannot indefinitely delay the cooperative gap; retain the 2 s deadline,
all integrity checks and resource caps. Add targeted regression coverage for
sub-MiB small-call sequences and competing producers. Bound diagnostic operation
tracing by aggregation/thresholds so JSON piece counts cannot consume the event
budget; retain every lock wait/failure and meaningful operation maximum. A
changed-source comparison needs its own frozen plan and new single-attempt root;
never rerun this closed attempt. New training remains blocked pending a validated
fix, fresh evidence contract/finite plan and verified storage solution.


### Prospective elapsed-time checkpoint handoff and trace-v3 comparison

The checkpoint writer now yields outside the namespace lock after either 1 MiB
or 20 ms since its previous handoff. Checks run after preflight, each bounded
write and flush, including empty callbacks. The clock resets after the gap;
existing accumulated-byte behavior remains. The 20 ms gap, two-second namespace
deadline, integrity checks and resource caps are unchanged. A single operation
is not preempted: this is cooperative scheduling, not a hard latency guarantee.

140 checkpoint/metadata/process tests passed in 6.73 s under the unchanged
180 s CPU guard (peak RSS 240,599,040 bytes). Six added cases cover sub-MiB
callbacks with competing publishers, time spent in preflight/write/flush,
STOP during a gap and interrupted handoff preservation. The first command named
a nonexistent process test file, exited 4 and ran no tests; its log and guard
remain, alongside the corrected invocation. No numerical fixture was retried.
Three recorder tests also passed: per-JSON-piece aggregation, explicit overflow
accounting and preservation of original failures when the detail cap is reached.

Freeze one new nontraining trace-v3 comparison in
`research_runs/tokenizer-metadata-contention-trace-v3-001`. It uses the same
four 240 MiB tensors, 128 small tensors, 337 synthetic files, one CPU child,
four threads, 150/180 s child/parent bounds, 4 GiB RSS, 8 GiB reserve and
992 MiB total output allowance as trace-v2. All 228 source pins were compared;
only the checkpoint writer changed. The new recorder keeps detailed locks and
checkpoint writes, aggregates five fixed operation labels and retains bounded
failure records. Overflow marks the trace incomplete without masking an original
exception. All external traces remain capped at 4 MiB. Python version is checked.

The handoff and instrumentation overhead both change, so a pass cannot isolate
their effects or prove training reliability. Commit and push this finite plan
before its sole execution. Review hashes, guards, trace completeness and
unchanged closed003/trace-v2 evidence afterward. Preserve every outcome, no
cleanup or retry, no training or holdout use. Full training-panel storage remains
blocked. Plan, test logs/guards and component review are under
`tokenizer-evidence/wide-hybrid-preparation-001/` with `trace-v3` and
`checkpoint-time-handoff-v2` names.


### Elapsed-time handoff: actual-scale trace-v3 completed and independently reviewed

The prospective source/plan was committed and pushed as `19f65b1` before its
single run. The synthetic checkpoint completed: 1,006,700,215 bytes, SHA256
`853ba84e95525d097d26f2ab9aad58af4bca86ccc182c9cf7eb7cf4f660552db`.
Parent/child guards finished in 73.3748/71.9172 s; peak RSS was
220,463,104/1,230,225,408 bytes. All time, memory, disk and output caps held.
No research worker remains and no retry occurred.

Maximum parent lock wait was 0.0453204 s; child main/resource maxima were
0.0605946/0.0602781 s. Maximum parent/child holds were 0.0233675/0.0222068 s.
The child completed 1,650 checkpoint writes, including 690 sub-MiB writes,
and 70 resource samples with no recorder errors. The trace retained all 629
parent and 2,419 child lock scopes. There were no failures or dropped records.
Aggregation handled 25,337 parent JSON writes without spending detail capacity;
parent/child trace sizes were 161,451/1,048,335 bytes, below 4 MiB each.

Independent guarded review rehashed the output, verified source/plan pins and
all 337 closed003 plus 350 closed trace-v2 files unchanged, and archived every
noncheckpoint byte. Review guard finished in 5.8864 s, peak RSS 240,238,592 bytes.
Evidence: `tokenizer-evidence/wide-hybrid-preparation-001/metadata-contention-trace-v3-review.json`,
its guard and `review_metadata_contention_trace_v3.py`. The synthetic checkpoint
and all failed evidence remain preserved; no cleanup, model/corpus access or
holdout use occurred.

This supports the revised handoff under one actual-scale synthetic workload.
It does not prove that future training cannot time out, or isolate the handoff's
effect from the changed tracing overhead. Flat synthetic topology, CPU tensors
and absence of preceding MPS training remain limitations. Next audit source
applicability against the completed numerical/input prerequisites and prepare
an explicit new evidence contract. Never reopen campaigns001/002/003 or rerun
the closed numerical/trace fixtures. A new full panel still requires
30,631,412,484 bytes including reserve versus roughly 22.68 GB currently free;
resolve storage prospectively under retention policy or a justified new finite
scientific design before any training. Do not promote the incomplete old panel
or delete protected weights to fit it.


### Time-handoff applicability and retention audit: capacity is now the blocker

A read-only guarded audit compared all 228 source entries: 227 are unchanged
since campaign003, with only the checkpoint writer's scheduling changed. Relative
to the completed original numerical execution, 225 entries remain unchanged;
the two earlier census fixes and checkpoint scheduling are explicitly accounted
for. Runtime identity is unchanged. Original numerical/input reports, trace
artifacts and all numerical phase files/checkpoints were hash-verified without
reexecuting or loading them. The mathematical/state and input evidence remains
applicable, with original executed-source identity retained. New MPS pilot and
process/retention evidence will still be required for a new campaign.

The retention inventory verified all 189 tokenizer `.pt`/`.pt.tmp` files,
62,800,809,984 logical bytes: 148 previously protected selected/fresh pairs' files
(34,119,142,876 bytes), 28 incomplete-campaign/failure files (17,691,873,110),
nine numerical-prerequisite checkpoints (8,858,999,191) and four preserved
synthetic diagnostic weights (2,130,794,807). All 42 prior cleanup paths remain
absent. Closed002, closed003 and both actual-scale diagnostic inventories matched
their saved reviews. No eligible deletion was identified and nothing was removed.
Incomplete screens have no completed, persisted selection permitting cleanup;
synthetic diagnostic saves are not successful resource pilots or losing screens.

The unchanged full-panel requirement is 30,631,412,484 bytes including the 8 GiB
reserve. Observed free space was 22,849,273,856 bytes, a 7,782,138,628-byte shortfall.
For this design, provide additional local capacity or a new output volume with
at least 30.64 GB free (with practical headroom). Do not shrink a frozen guard,
promote an incomplete old panel, delete protected evidence or repeatedly poll
unchanged free space. No new campaign directory, plan freeze or worker was created.

Independent preparation produced six losslessly verified prerequisite files
(64,513 bytes) in `tokenizer-evidence/wide-hybrid-preparation-001/time-handoff-prerequisite-transport-v2/`.
The original reports and applicability review are embedded with hashes and exact
bytes; numerical traces decompress exactly, and the 96 train IDs are unchanged.
The numerical gate accepted the transported evidence. No holdout was accessed.
Replacing the six prior payloads adds 14,501 bytes to the last metadata envelope,
leaving only 7,127 bytes of its former 21,628-byte margin before other plan/path
changes. This is a static delta, not a new complete envelope proof. Once storage
is available, recheck the final destination, source and full metadata envelope,
then freeze the finite plan before any new pilot or training launch.

Evidence: `checkpoint-time-handoff-source-applicability-v2-review.json`,
`time-handoff-prerequisite-transport-v2-review.json`, their guarded procedures
and guard journals under `tokenizer-evidence/wide-hybrid-preparation-001/`.
The old heartbeat remains paused and this continuation heartbeat active.


### October 29: verified lossless archival recovers the required capacity

The human explicitly requested proactive archival/removal while preserving
selected models, fresh confirmation pairs and failure evidence. The earlier
finding of no eligible irreversible deletion still holds; cold archival retains
all checkpoint bytes and resolves storage without discarding those records.
The committed prospective plan (`f6f544b`) covered only completed numerical and
non-selected/non-fresh screen weights. No closed campaign was reopened.

35 loose files totaling 25,008,600,879 bytes were replaced by three protected local
Zstandard archives totaling 13,454,327,571 bytes. Net logical reclamation is
**11,554,273,308 bytes (11.55 GB)**. Each archive was fully decompressed and every
member's path, size and original SHA256 verified before any loose copy was
removed. Removal was journaled and archive hashes were rechecked afterward.

| Archive group | Files | Original bytes | Archive bytes | Net bytes reclaimed |
|---|---:|---:|---:|---:|
| Completed numerical prerequisites | 9 | 8,858,999,191 | 3,625,971,340 | 5,233,027,851 |
| Completed campaign002 screens | 10 | 6,225,515,336 | 3,791,160,683 | 2,434,354,653 |
| Completed campaign003 screens | 16 | 9,924,086,352 | 6,037,195,548 | 3,886,890,804 |

All **154 excluded loose checkpoints** were hash-verified unchanged, including
selected models, all fresh confirmation pairs and every failed partial. All
**3,740 non-checkpoint files** in tokenizer run roots were verified unchanged.
A separate read-only review reconciled the archives with all 250 original
campaign002, 337 campaign003 and 121 numerical-prerequisite file identities.
The original reports retain their executed-source hashes and byte contents;
there was no model load, numerical rerun, training or holdout access.

Observed free space after the operation was 33,493,016,576 bytes, exceeding the
unchanged 30,631,412,484-byte panel requirement by 2,861,604,092 bytes. Other system
activity can affect observed free space; attributable savings are the exact
logical figure above. One archival worker finished within the guarded bounds.
Initial filesystem-compression/sparse-copy probes saved no allocated blocks;
only their independently verified scratch duplicates were removed.

Archives remain protected under `research_archives/checkpoint-compaction-001/`
and are excluded from Git. Text manifests, archive/member hashes, removal
journal, restoration instructions and guard/review evidence are committed under
`tokenizer-evidence/checkpoint-compaction-001/`. Do not delete these archives as
caches. Restore exact bytes before invoking an original-path-only audit, or use
the explicit archive resolver; absent loose paths are intentional archival,
not lost evidence or permission to regenerate checkpoints.

Capacity is now sufficient at the observed snapshot. Next incorporate the
archival proof into prospective prerequisite provenance, audit the complete
metadata envelope at the final new destination, and freeze the finite plan.
Recheck actual capacity before dispatch and run new resource pilots first.
No training has launched under the changed source. Continue proactive bounded
storage management without waiting for another human cleanup request.


### Campaign004 finite freeze after elapsed-time handoff and archival

A fresh complete campaign is frozen at `research_runs/tokenizer-wide-hybrid-004`.
Plan SHA256:
`6e42949c51dcfab5766444a293763856d22e2913aa01ebbdc1eb8b7a1f6625c2`.
Source SHA256:
`0fc7f6b3440484700151a99f6a02f1b847914f7c8914612b4d684f54d058ccaa`.
The original nine-case numerical and train-only input reports, applicability
review and explicit archival resolver are carried losslessly, retaining original
executed-source identities. All three protected archive hashes were checked.
No old quality trajectories, process receipts or cleanup chains are reused.

The unchanged finite design has three resource pilots, nine LR cells, six rank
cells and nine fresh confirmation cells, followed by three inference workloads.
All 108 conditional quality request options are bound. Seed3501 is reused for
screening and is not independent replication; old incomplete screen results
remain excluded from selection. Fresh3511/3512/3513 have not been launched by
previous closed campaigns. No holdout use or quality/adoption claim follows
from this engineering change.

The full metadata envelope fits: 2,093,907 of 2,097,152 bytes for evidence/meta,
leaving 3,245 bytes, and 855 conservative files below the 2,048 cap. Other group
caps and producer guards remain unchanged. Full-panel storage requirement remains
30,631,412,484 bytes including reserve; freeze observed 33,491,800,064 bytes free.
The guarded preparation finished in 5.77 s. The actual plan binder accepted
numerical/input provenance and explicitly left dispatch unauthorized until
measured pilot gates pass. Originals and archival member identities are preserved.

`start_campaign004_v2.py` uses the existing production dispatchers to run the
three resource pilots serially, independently audit the completed pilot boundary,
perform only its policy-eligible six-weight cleanup after exact metadata archival,
and dispatch quality slot0 only if every prerequisite passes. It stops after
slot0 or any failure, with no retry or automatic later slots. Commit/push the
finite plan before launching this wrapper with local MPS access. One training
worker, all existing resource limits and the closed test remain enforced.

Campaign004 launch followed committed/pushed freeze `73f2507`: controller23108
started one MPS resource-pilot worker23109. Pilot0 was observed running under
its production process guard; no quality result exists yet. The bounded wrapper
may advance through the remaining serial pilots and qualityslot0 only after
all required reviews succeed. Launch/start receipts are recorded separately;
no other worker, later quality slot or retry was launched.


### Campaign004 resource pilots accepted; first quality cell running

All three 20-update width512 D/2 resource pilots completed under the frozen
elapsed-time-handoff source. Full/cache and raw/H200 audits, resource reviews
and parent acceptance passed for hybrid, gated MLA and KDA. Worker times were
99.94, 100.67 and 120.15 s. Largest sampled RSS was 3,228,827,648 bytes and MPS
driver allocation was 4,112,433,152 bytes, within the unchanged 4/6 GiB limits.
No resource violations were recorded; sampled maxima are not instantaneous bounds.

The complete pilot boundary passed its production audit. Exactly six successful
pilot weights (3,769,780,224 bytes) were removed under the frozen retention rule
after preserving exact non-weight bytes and hashes. The external archive retains
76 non-checkpoint files; a separate read-only check verified the immutable pilot
records, archive bytes, six-file manifest and saved execution-review reference.
Boundary reference SHA256:
`12a766f7c9a301d5176f4556d8512bebfe320af243925d0e9fdf08b1a9fa1f93`.

The existing serial wrapper then started quality slot0 (hybrid, LR0.000625,
seed3501, 1,200 updates) with one MPS worker23466 and controller23108. No quality
result exists yet. This is execution/correctness evidence, not a quality gain or
proof that the full training campaign cannot fail. The wrapper stops after
slot0 acceptance or failure; later slots require independently reviewed progress.
Evidence: `campaign004-pilots-reviewed-v2.json`, pilot archive/boundary receipts
and `campaign004-quality0-start-v2.json` under the preparation evidence directory.


### Campaign004 first quality cell accepted and independently reviewed

Slot0 completed all 1,200 updates: width512 hybrid, D/4, LR0.000625,
seed3501. Full-validation H200 CE is 4.963671529014013; raw endpoint CE is
4.991196894351007 over 34,278 targets / 111,536 bytes. Both checkpoint hashes,
request/result audits, parent acceptance and the completed pilot-cleanup chain
passed independent prefix1 review. The review guard finished in 4.52 s with
220,250,112-byte peak RSS. The controller and child exited; no research worker
remained before continuation.

The review measured 32,232,665,088 bytes free against a remaining-campaign
requirement of 29,384,508,436 bytes, including the unchanged 8 GiB reserve.
Both slot0 checkpoints remain protected. This is one of nine LR screening cells;
there is no family selection, fresh-seed claim or adoption, and the test holdout
remains closed. Closed-round results are not pooled with this panel.

The campaign004 prefix-review and one-slot continuation wrappers differ from the
reviewed campaign003 wrappers only in run-root/receipt identifiers; production
source and frozen plan are unchanged. Evidence: `campaign004-terminal-v2.json`,
`campaign004-lr-prefix-01-v2-review.json` and its guard in
`tokenizer-evidence/wide-hybrid-preparation-001/`. After committing/pushing this
review, the next authorized serial cell is slot1, gated MLA at LR0.000625,
D/4, seed3501, through all original production gates. No retry or automatic
later-slot dispatch is enabled.


Campaign004 slot1 was launched only after prefix1 review commit `36b061e` was
pushed. Controller24348 started the sole training child24360 through the existing
production gates: gated MLA, LR0.000625, D/4, seed3501, 1,200 updates. The launch
receipt is `campaign004-lr-01-v2-launch.json`; it binds the frozen plan and
independent prefix review. This wrapper stops on acceptance or failure and cannot
automatically dispatch slot2. No retry, selection or holdout evaluation occurred.


### Campaign004 second quality cell independently accepted

Slot1 gated MLA completed 1,200 updates at width512, D/4, LR0.000625,
seed3501. Full-validation H200 CE is 4.95333731332134 and raw endpoint CE is
4.920093629712963. Parent acceptance and the independent two-cell prefix review
passed, including both raw/H200 pairs and prior cleanup provenance. The review
guard finished in 8.73 s, peak RSS220,397,568 bytes. No research worker remained
before continuation. Remaining storage fits: 30,985,097,216 bytes free versus
28,162,784,220 required, including the unchanged reserve.

Only two of nine LR cells are complete. The observed H200 difference from hybrid
at this one LR/seed is descriptive; no selection, fresh confirmation or adoption
is available. The test holdout remains closed. Evidence is in
`campaign004-lr-01-v2-terminal.json`, `campaign004-lr-prefix-02-v2-review.json`
and its guard. Commit/push this review before the next serial cell, slot2 KDA,
LR0.000625, D/4, seed3501, through the unchanged production gates.


After prefix2 review was committed and pushed as `7cca48e`, slot2 KDA
(LR0.000625, D/4, seed3501) started through the original guarded dispatcher.
Controller25024 owns child25039. The launch receipt binds the frozen plan and
independent review: `campaign004-lr-02-v2-launch.json`. It permits this cell only;
slot3 requires a completed result and a fresh independent prefix review.


### Campaign004 first LR group completed; three-cell prefix accepted

Slot2 KDA completed 1,200 updates at width512, D/4, LR0.000625,
seed3501. Full-validation H200 CE is 4.94793046301675; raw endpoint CE is
4.97468234106922. Parent acceptance and independent prefix3 review passed,
including all three checkpoint pairs and prior cleanup provenance. The review
guard finished in 13.08 s with 224,100,352-byte peak RSS. No research worker
remained before continuation. Storage fits: 29,750,755,328 bytes free versus
26,907,661,636 required for the remaining campaign including the reserve.

The first LR group's H200 CE values are hybrid4.9636715290,
gated MLA4.9533373133 and KDA4.9479304630. These single-seed, single-LR
observations do not select an architecture or establish a fresh-seed benefit.
Three of nine LR cells are accepted; full tuning, rank probes, fresh confirmation
and inference measurements remain. The test holdout remains closed.
Evidence: `campaign004-lr-02-v2-terminal.json`,
`campaign004-lr-prefix-03-v2-review.json` and its guard. After committing/pushing
this review, the frozen next cell is slot3 KDA at LR0.00125, D/4, seed3501.


After prefix3 review was committed and pushed as `36ffb14`, slot3 KDA
(LR0.00125, D/4, seed3501) started through the unchanged guarded dispatcher.
Controller25964 owns child25983. The launch receipt is
`campaign004-lr-03-v2-launch.json`; it binds the frozen plan and independent
review and permits this cell only. No automatic later-slot dispatch or retry.


### Campaign004 fourth LR cell accepted

Slot3 KDA completed 1,200 updates at width512, D/4, LR0.00125, seed3501.
H200 full-validation CE is 4.977440113210341; raw CE is 5.072588973647136.
At this screening seed, H200 CE is 0.0295096502 higher than KDA at LR0.000625.
This is a within-panel descriptive LR effect; the remaining grid and fresh seeds
are still required, and no selection/adoption has been made.

Parent acceptance and independent prefix4 review passed for all four raw/H200
pairs and the existing cleanup chain. The review guard finished in 17.42 s with
224,608,256-byte peak RSS. No research worker remained before continuation.
Storage fits: 28,447,940,608 bytes free versus 25,652,539,052 required including
the unchanged reserve. All checkpoints remain protected and the test stays closed.
Evidence: `campaign004-lr-03-v2-terminal.json`,
`campaign004-lr-prefix-04-v2-review.json` and its guard. Commit/push this review
before slot4 hybrid at LR0.00125, D/4, seed3501, through the frozen serial gates.


After prefix4 review was committed and pushed as `57f5d8a`, slot4 hybrid
(LR0.00125, D/4, seed3501) started under controller26836 and the sole
training child26847. `campaign004-lr-04-v2-launch.json` binds the frozen plan
and independent review. The unchanged guarded wrapper permits this cell only;
no automatic later-slot dispatch or retry is enabled.


### Campaign004 fifth LR cell accepted

Slot4 hybrid completed 1,200 updates at width512, D/4, LR0.00125, seed3501.
H200 full-validation CE is 4.978503878216632; raw CE is 5.081234641065298.
H200 CE is 0.0148323492 higher than hybrid at LR0.000625 on this screening seed.
Like KDA's observed increase, this motivates finishing the frozen grid rather
than treating a larger LR as an improvement. It is not a fresh-seed conclusion,
family selection or adoption. Five of nine LR cells are accepted.

Parent acceptance and independent prefix5 review passed for all five raw/H200
pairs and prior cleanup provenance. The review guard finished in 24.69 s with
216,334,336-byte peak RSS. No research worker remained before continuation.
Storage fits: 27,188,117,504 bytes free versus 24,405,766,076 required including
the unchanged reserve. Protected weights and closed test remain unchanged.
Evidence: `campaign004-lr-04-v2-terminal.json`,
`campaign004-lr-prefix-05-v2-review.json` and its guard. Commit/push this review
before slot5 gated MLA at LR0.00125, D/4, seed3501, through the original gates.


After prefix5 review was committed and pushed as `acc6d1d`, slot5 gated MLA
(LR0.00125, D/4, seed3501) started under controller27909 and child27950.
`campaign004-lr-05-v2-launch.json` binds the frozen plan and independent review.
The original guarded wrapper permits this cell only, with no retry or automatic
later-slot dispatch.


### Campaign004 six LR cells accepted: gated MLA responds to LR0.00125

Slot5 gated MLA completed 1,200 updates at width512, D/4, LR0.00125,
seed3501. H200 full-validation CE is 4.821449135542395; raw CE is
4.824457830622191. Relative to its LR0.000625 result, H200 CE improved by
0.1318881778 nats/token on this screening seed. At LR0.00125 it is below
hybrid4.9785038782 and KDA4.9774401132. This is a descriptive result within
an incomplete single-seed screen, not a family selection or fresh-seed gain.
Three LR0.0025 cells, rank probes and fresh confirmations remain before adoption.

Parent acceptance and independent prefix6 review passed for all six raw/H200
pairs and prior cleanup provenance. The review guard finished in 29.90 s with
225,017,856-byte peak RSS. No research worker remained before the review.
Storage fits: 25,943,928,832 bytes free versus 23,184,041,860 required including
the unchanged reserve. All retained evidence and closed test remain protected.
Evidence: `campaign004-lr-05-v2-terminal.json`,
`campaign004-lr-prefix-06-v2-review.json` and its guard. Commit/push this review
before slot6 gated MLA at LR0.0025, D/4, seed3501, through the unchanged gates.


After prefix6 review was committed and pushed as `eecf9a1`, slot6 gated MLA
(LR0.0025, D/4, seed3501) started under controller28761 and child28783.
`campaign004-lr-06-v2-launch.json` binds the frozen plan and independent review.
The original guarded wrapper permits this cell only, with no retry or automatic
later-slot dispatch.


### Campaign004 seventh LR cell accepted

Slot6 gated MLA completed 1,200 updates at width512, D/4, LR0.0025, seed3501.
H200 full-validation CE is 4.856296313508867; raw CE is 4.919025578423119.
Its three observed H200 values are 4.9533373133 (LR0.000625), 4.8214491355
(LR0.00125), and 4.8562963135 (LR0.0025). The middle LR is currently best for
this family on the screening seed; selection remains deferred until all nine
LR cells pass. No fresh-seed or adoption claim is made.

Parent acceptance and independent prefix7 review passed for all seven raw/H200
pairs and prior cleanup provenance. The guard finished in 35.36 s with
225,394,688-byte peak RSS. No research worker remained before review.
Storage fits: 24,693,530,624 bytes free versus 21,962,317,644 required including
the unchanged reserve. Protected evidence and closed test remain unchanged.
Evidence: `campaign004-lr-06-v2-terminal.json`,
`campaign004-lr-prefix-07-v2-review.json` and its guard. Commit/push this review
before slot7 KDA at LR0.0025, D/4, seed3501, through the unchanged serial gates.


After prefix7 review was committed and pushed as `cfbe012`, slot7 KDA
(LR0.0025, D/4, seed3501) started under controller29612 and child29630.
`campaign004-lr-07-v2-launch.json` binds the frozen plan and independent review.
The original guarded wrapper permits this cell only, with no retry or automatic
later-slot dispatch.


### Campaign004 eighth LR cell accepted

Slot7 KDA completed 1,200 updates at width512, D/4, LR0.0025, seed3501.
H200 full-validation CE is 5.104937873943505; raw CE is 5.273170680299278.
Its H200 values increase across the tested LRs: 4.9479304630 at0.000625,
4.9774401132 at0.00125, and5.1049378739 at0.0025. The lowest LR is currently
best on this screening seed. This is descriptive; selection waits for the ninth
cell and the complete LR-stage audit. No adoption or fresh-seed claim is made.

Parent acceptance and independent prefix8 review passed for all eight raw/H200
pairs and prior cleanup provenance. The guard finished in 40.98 s with
224,673,792-byte peak RSS. The training child exited before review and the
parent acceptance completed before shared locks were acquired.
Storage fits: 23,404,253,184 bytes free versus 20,707,195,060 required including
the unchanged reserve. Protected evidence and closed test remain unchanged.
Evidence: `campaign004-lr-07-v2-terminal.json`,
`campaign004-lr-prefix-08-v2-review.json` and its guard. Commit/push this review
before slot8 hybrid at LR0.0025, D/4, seed3501, through the unchanged gates.
After all nine cells pass, use the production LR selection and cleanup boundary
before rank dispatch; the prefix wrapper intentionally supports only1–8.


After prefix8 review was committed and pushed as `fda2e76`, final LR slot8 hybrid
(LR0.0025, D/4, seed3501) started under controller30644 and child30676.
`campaign004-lr-08-v2-launch.json` binds the frozen plan and independent review.
The original wrapper stops after this cell. Complete LR selection, independent
verification and audited cleanup must precede any rank dispatch; no stage is
implicitly authorized by this launch receipt.


### Campaign004 complete LR screen and verified selection

All nine 1,200-update LR trials completed and passed the full production stage
audit and independent selection verifier. Width512, D/4, seed3501 H200 CE:

| Family | LR0.000625 | LR0.00125 | LR0.0025 | Selected LR |
|---|---:|---:|---:|---:|
| Hybrid | 4.9636715290 | 4.9785038782 | 5.1111339709 | 0.000625 |
| Gated MLA | 4.9533373133 | 4.8214491355 | 4.8562963135 | 0.00125 |
| KDA | 4.9479304630 | 4.9774401132 | 5.1049378739 | 0.000625 |

Gated MLA benefits from the middle LR in this screen; increasing LR worsened
hybrid and KDA at the tested settings. These are selected single-seed results,
not independent confirmation, universal architecture superiority or adoption.
The remaining frozen stages are six rank probes, nine fresh confirmations and
three inference evaluations. Test holdout remains closed.

The new external boundary wrapper calls the unchanged production selection and
cleanup APIs in separate phases under shared locks and a 1,800-second CPU guard.
All25 existing selection/cleanup component tests passed in360.95s; their outer
guard finished with240,238,592-byte peak RSS. Boundary wrapper syntax and first-rank
wrapper adaptation were inspected; no frozen production source changed.

Selection plus verification finished in172.97s, peak RSS231,391,232 bytes.
The external review preserves hashes/sizes for352 files and exact bytes for334
non-checkpoint files. All18 weights remain present at this point. The prospective
manifest retains slots0,2,5 (six selected raw/H200 files) and marks12 losing-screen
weights eligible for policy cleanup. Selected/fresh/failure evidence and the
lossless archives from closed rounds remain protected. Commit/push this complete
selection evidence before running cleanup; then independently verify the cleanup
chain and protected bytes before any rank dispatch.

Evidence: `campaign004-lr-select-v2-review.json`, its before archive/guard,
`campaign004-lr-boundary-tests-v2.log`, and `campaign004-lr-08-v2-terminal.json`.
Selection completion SHA256:
`100a5e54fa1e0d8d3dd3758b244f9e1e48970e92dfde120c3c5dd21b13f43ebc`.


### Campaign004 LR boundary cleanup verified; rank stage ready

After selection evidence was committed/pushed as `db62fa1`, the frozen production
cleanup re-audited all nine screens and the exact manifest under shared locks.
Exactly12 losing-screen weights totaling7,447,239,552 bytes were removed. Selected
raw/H200 pairs at slots0,2,5 remain byte-identical (3,723,619,776 bytes total).
All334 pre-cleanup non-checkpoint files retain exact archived bytes, and every
other preexisting file hash is unchanged. The post-cleanup external archive has
350 files, including344 non-checkpoint files. Original selection, metrics,
validation block sums, resource/tensor audits, logs, manifests and deletion journal
are preserved. No closed-round, fresh-confirmation or failure evidence was removed.
Deleted losing weights cannot be reloaded; their original audited evidence remains.

The independent production verifier accepted the two-boundary cleanup chain.
The outer guard finished in186.52s with237,715,456-byte peak RSS. Free space at
completion was29,654,528,000 bytes; the verifier's remaining-campaign requirement
is26,907,661,636 bytes including the unchanged8GiB reserve. The boundary does not
itself authorize dispatch: the next child must pass normal frozen rank readiness.

Evidence: `campaign004-lr-cleanup-v2-before.json`, review and guard. Completion
SHA256: `65923075d6c2e3861972831c3ba96c91c02cb859fd507f435e71b08b06039036`.
Commit/push this reviewed boundary before `start_campaign004_rank_v2.py --slot 9`:
first rank probe is hybrid D/8 at selected LR0.000625, seed3501, 1,200 updates.
The new wrapper is restricted to slot9 and calls the unchanged `run_rank_once`
production dispatcher; it stops after this one trial, with no retry or later slot.


### Campaign004 first rank trial launched

Reviewed LR cleanup was committed/pushed as `53e7e28` before dispatch.
The unchanged production rank startup and readiness gates admitted slot9:
hybrid width512, D/8, LR0.000625, seed3501, 1,200 updates. Controller35596
started child35615; process inspection confirmed exactly one training worker.
The running control reports no error or log truncation, about29.65GB free against
the26.91GB remaining-campaign requirement including8GiB reserve. This is startup
evidence only; terminal acceptance and independent prefix review remain pending.
No next slot is automatically dispatched. The frozen plan/source, selected LR
anchors, fresh/failure evidence and closed holdout remain unchanged.

Launch receipt: `campaign004-rank-09-v2-launch.json`. Observe this worker to
completion; do not relaunch. After acceptance and exit, review rank prefix before
any slot10 dispatch.


### Campaign004 rank slot9 independently reviewed

Hybrid D/8 at LR0.000625, seed3501 completed1,200 updates and passed
parent acceptance (1,055.16s including startup/review). Final H200 validation CE
was4.9540804930 and raw CE4.9830292708; the D/4 LR anchor H200 was4.9636715290,
a0.0095910361 screening improvement. This is one seed and one of six rank probes;
no rank selection, architecture adoption or fresh-confirmation claim is made.

After both processes exited, the independent prefix audit passed under shared
locks in30.28s with231,768,064-byte peak RSS. It verified LR cleanup history,
rank scientific and transport evidence, and preserved379 file identities plus
exact bytes for371 non-checkpoint files. Eight live screen weight files remain.
The remaining-capacity gate requires25,667,182,740 bytes including8GiB reserve;
observed free space was28,387,106,816 bytes. No cleanup or training occurred in
this review. Test holdout remains closed.

Evidence: `campaign004-rank-09-v2-terminal.json`,
`campaign004-rank-prefix-10-v2-review.json` and its finished guard. Commit/push
these before serial slot10 (gated MLA D/8 at selected LR0.00125, seed3501).


Rank9 review evidence was committed/pushed as `bdc9b29` before serial slot10.
The next controller36655 passed preparation/readiness and started sole child36681:
gated MLA D/8, width512, LR0.00125, seed3501, 1,200 updates. Running process
control reports no error, about28.38GB free against25.67GB required including
8GiB reserve. Launch receipt: `campaign004-rank-10-v2-launch.json`. No later
slot launches automatically; terminal acceptance and independent prefix11 review
are required before slot11.


### Campaign004 rank slot10 independently reviewed

Gated MLA D/8 at LR0.00125, seed3501 completed1,200 updates and passed
parent acceptance (781.25s including preparation/review). Final H200 validation
CE4.8286763122 and raw CE4.8307202561 compare with D/4 anchor H2004.8214491355:
D/8 is0.0072271767 worse on this screening seed. This does not complete rank
selection or establish an architecture conclusion. All raw/H200 pairs remain
protected; no fresh confirmations or holdout evaluations have run in this round.

After both processes exited, independent prefix11 review passed in43.85s under
shared locks and the unchanged CPU guard, peak RSS231,964,672 bytes. It verified
LR cleanup history, both rank trials and their transport receipts, archived408
file identities and398 exact non-checkpoint files. Ten live screen weights remain.
Observed free space27,178,233,856 bytes exceeds24,458,051,804 required including
8GiB reserve. No cleanup occurred.

Evidence: `campaign004-rank-10-v2-terminal.json`,
`campaign004-rank-prefix-11-v2-review.json` and its finished guard. Commit/push
before serial slot11: KDA D/8, LR0.000625, seed3501, 1,200 updates.


Rank10 review evidence was committed/pushed as `65ebf7f` before serial slot11.
Controller37495 passed production preparation/readiness and started sole
child37537: KDA D/8, width512, LR0.000625, seed3501, 1,200 updates. Running
control reports no error and27.17GB free against24.46GB required including
8GiB reserve. Launch receipt: `campaign004-rank-11-v2-launch.json`. No later
slot dispatches automatically; terminal acceptance and independent prefix12
review are required before KDA D/2 slot12.


### Campaign004 rank slot11 independently reviewed

KDA D/8 at LR0.000625, seed3501 completed1,200 updates and passed parent
acceptance (1,257.16s including preparation/review). Final H200 validation CE
was4.9418159157 and raw CE4.9664300471. Against D/4 anchor H2004.9479304630,
this is a0.0061145473 screening improvement. Across the three D/8 probes,
hybrid and KDA improved slightly while gated MLA worsened slightly on this one
seed. All three D/2 probes remain; no rank selection, adoption or fresh-seed
conclusion is made. Test holdout remains closed.

After both processes exited, independent prefix12 review passed under shared
locks in55.29s, peak RSS233,013,248 bytes. It verified the cleanup history, all
three rank results and transport receipts, and archived437 file identities plus
425 exact non-checkpoint files. All12 live screen weights remain protected.
Free space25,897,988,096 bytes exceeds23,207,123,588 required including8GiB
reserve. No cleanup occurred.

Evidence: `campaign004-rank-11-v2-terminal.json`,
`campaign004-rank-prefix-12-v2-review.json` and its finished guard. Commit/push
before serial slot12: KDA D/2, LR0.000625, seed3501, 1,200 updates.


Rank11 review evidence was committed/pushed as `fd60c7a` before serial slot12.
Controller38788 passed production preparation/readiness and started sole
child38846: KDA D/2, width512, LR0.000625, seed3501, 1,200 updates. Running
control reports no error and25.90GB free against23.21GB required including
8GiB reserve. Launch receipt: `campaign004-rank-12-v2-launch.json`. No later
slot dispatches automatically; terminal acceptance and independent prefix13
review are required before gated MLA D/2 slot13.


### Campaign004 rank slot12 independently reviewed

KDA D/2 at LR0.000625, seed3501 completed1,200 updates and passed parent
acceptance (1,323.78s including preparation/review). Final H200 validation CE
was4.9264940236 and raw CE4.9551758129. KDA H200 results across D/8,D/4,D/2
are4.9418159157,4.9479304630,4.9264940236: D/2 improves0.0214364394 versus
D/4 and0.0153218921 versus D/8 on this screening seed. This is a screening
result, not a fresh-seed or adoption claim. Two remaining rank probes must
finish before the frozen selection is saved. Test holdout remains closed.

After both processes exited, independent prefix13 review passed under shared
locks in64.83s, peak RSS233,521,152 bytes. It verified cleanup history, four
rank results and transport receipts, and archived466 file identities plus452
exact non-checkpoint files. All14 live screen weights remain protected. Free
space24,620,216,320 bytes exceeds21,943,612,396 required including8GiB reserve.
No cleanup occurred.

Evidence: `campaign004-rank-12-v2-terminal.json`,
`campaign004-rank-prefix-13-v2-review.json` and its finished guard. Commit/push
before serial slot13: gated MLA D/2, LR0.00125, seed3501, 1,200 updates.


Rank12 review evidence was committed/pushed as `5932053` before serial slot13.
Controller39858 passed production preparation/readiness and started sole
child39926: gated MLA D/2, width512, LR0.00125, seed3501, 1,200 updates.
Running control reports no error and24.61GB free against21.94GB required
including8GiB reserve. Launch receipt: `campaign004-rank-13-v2-launch.json`.
No later slot dispatches automatically; terminal acceptance and independent
prefix14 review are required before the final rank probe, hybrid D/2 slot14.


### Campaign004 rank slot13 independently reviewed

Gated MLA D/2 at LR0.00125, seed3501 completed1,200 updates and passed parent
acceptance (966.81s including preparation/review). H200 validation CE4.8302985459
and raw CE4.8338051302 compare with D/4 H2004.8214491355 and D/8 H2004.8286763122.
D/4 remains the lowest-loss gated MLA rank on this screening seed; D/2 is
0.0088494104 worse. The final hybrid D/2 probe remains before the complete
frozen rank selection. No adoption or fresh-confirmation claim; holdout closed.

After both processes exited, independent prefix14 review passed under shared
locks in72.48s, peak RSS233,652,224 bytes. It verified cleanup history, all five
rank results and transport receipts, and archived495 file identities plus479
exact non-checkpoint files. All16 live screen weights remain protected. Free
space23,342,075,904 bytes exceeds20,696,701,620 required including8GiB reserve.
No cleanup occurred.

Evidence: `campaign004-rank-13-v2-terminal.json`,
`campaign004-rank-prefix-14-v2-review.json` and its finished guard. Commit/push
before serial slot14: hybrid D/2, LR0.000625, seed3501, 1,200 updates.
After slot14 use full rank selection/boundary APIs, not prefix count15.


Rank13 review evidence was committed/pushed as `7cd787a` before serial slot14.
Controller41010 passed production preparation/readiness and started sole
child41149: hybrid D/2, width512, LR0.000625, seed3501, 1,200 updates. Running
control reports no error and23.33GB free against20.70GB required including
8GiB reserve. Launch receipt: `campaign004-rank-14-v2-launch.json`. This is the
final rank probe. After acceptance and worker exit, execute the separately
prepared rank-selection wrapper, inspect/commit/push selection before cleanup,
and independently verify/commit/push cleanup before any fresh confirmation.


### Campaign004 complete rank selection independently verified

The final hybrid D/2 trial passed parent acceptance (1,310.29s including
preparation/review), H200 CE4.9527891402 and raw CE4.9848553028. All15 screening
trials are accepted. The frozen minimum-H200-CE rule (larger divisor for exact
ties) saved these configurations before any fresh-seed result:

| Family | D/8 H200 CE | D/4 H200 CE | D/2 H200 CE | Selected divisor | LR |
|---|---:|---:|---:|---:|---:|
| Hybrid | 4.9540804930 | 4.9636715290 | 4.9527891402 | 2 | 0.000625 |
| Gated MLA | 4.8286763122 | 4.8214491355 | 4.8302985459 | 4 | 0.00125 |
| KDA | 4.9418159157 | 4.9479304630 | 4.9264940236 | 2 | 0.000625 |

Hybrid D/2 beats D/8 by only0.0012913528 on this seed; this narrow screening
choice is not a stability claim. These are selected single-seed results, not
fresh confirmation or architecture adoption. Nine fresh trials and three
inference evaluations remain under the frozen plan. Test holdout stays closed.

The full production selection and independent verification finished in348.48s
under both locks and the unchanged CPU guard, peak RSS243,957,760 bytes.
All original file identities and non-checkpoint bytes remained unchanged.
The saved external archive has529 files and511 exact non-checkpoint files.
The manifest retains selected slots5,12,14 (six raw/H200 weights,3,744,596,480
bytes) and marks12 losing weights totaling7,449,344,384 bytes eligible. No
weights were removed during selection. Commit/push selection evidence before
cleanup, then independently verify/push cleanup before fresh slot15.

Evidence: `campaign004-rank-select-v2-review.json`, before archive and guard,
plus `campaign004-rank-14-v2-terminal.json`. Selection completion SHA256:
`9ad355475f0c762fbf4da56184e611902d9fc943160812f823f7b9972ff58ccc`.

Prepared `start_campaign004_fresh_v2.py` by adapting the reviewed first-rank
wrapper to unchanged `run_first_confirmation_once`. It only accepts slot15,
requires verified rank cleanup and a finished guard, retains plan/cache/MPS
checks and both shared locks, and stops after one child without retry. Syntax
and exact adaptation diff were inspected; no fresh dispatch yet.


### Campaign004 rank cleanup independently verified

Selection evidence was committed/pushed as `bd1c487` before cleanup. The frozen
production cleanup re-audited the complete rank stage and exact manifest under
shared locks, then removed exactly12 losing weights totaling7,449,344,384 bytes.
Selected raw/H200 pairs at slots5,12,14 remain byte-identical,3,744,596,480 bytes
in total. Every one of511 original non-checkpoint files retains its exact
archived bytes; every other preexisting file identity is unchanged. The final
external archive has527 files, including521 non-checkpoint files and six weights.
Original metrics, validation block sums, logs, hashes, selection decisions and
all cleanup receipts remain. Deleted losing weights cannot be reloaded. No
closed-round, fresh-confirmation, failure evidence or protected cold archive
was touched.

Independent verification accepted the full three-boundary cleanup chain. The
outer guard finished in324.58s, peak RSS260,964,352 bytes. Free space at guard
completion was29,572,743,168 bytes; remaining-campaign required free space is
26,712,927,696 bytes including the unchanged8GiB reserve. The external archive
comparison additionally confirmed the exact12-path deletion set and unchanged
non-checkpoint bytes. No training launched during selection/cleanup.

Evidence: `campaign004-rank-cleanup-v2-review.json`, before archive and guard.
Completion SHA256:
`191ec52b3fff8be397eddf5f8a33c02aa86c3f03c4be213d5ed0f37bab200814`.
Commit/push this verified boundary before `start_campaign004_fresh_v2.py --slot 15`.
The first fresh trial is selected hybrid D/2, LR0.000625, seed3511,1,200 updates;
all fresh raw/H200 pairs must be retained, including negative results.


Verified rank cleanup was committed/pushed as `6394277` before fresh slot15.
Controller45487 passed production preparation/readiness and started sole
child45508: selected hybrid D/2, width512, LR0.000625, fresh seed3511,
1,200 updates. The immutable request confirms the fresh seed and selected
configuration. Running control reports no error and29.57GB free against26.71GB
required including8GiB reserve. Launch receipt: `campaign004-fresh-15-v2-launch.json`.
No later slot dispatches automatically. Retain every fresh raw/H200 pair.
After acceptance and exit, independently review compact first-fresh history
and adapt the external slot16 wrapper to the existing frozen later-confirmation
startup/readiness/dispatch APIs before gated MLA D/4 seed3511.


### Campaign004 first fresh confirmation independently verified

Slot15 (selected hybrid D/2, LR0.000625, fresh seed3511) completed1,200
updates and parent acceptance. Full-validation H200 CE is4.999246600352049;
raw CE is5.029029321452403, over34,278 targets in17 blocks. The H200
checkpoint improves this fresh run by0.029782721100354. This is one
fresh run; no paired architecture comparison or adoption claim is available.
The closed test holdout remains unused and the selected configurations stay fixed.

After controller/child exit, the read-only first-confirmation history verifier
reconstructed the full scientific, transport, storage and compact-parent audit
under both shared locks. The outer guard finished in36.89s
with peak RSS245202944 bytes. The external archive preserves
556 file identities and exact bytes for548 non-checkpoint files,
including all selected pairs and the first fresh raw/H200 pair (eight weights
in total). Free space was28270907392 bytes against
25453566624 required, including the unchanged8GiB reserve.
No cleanup, selection change or later request occurred during this audit.

Evidence: `campaign004-fresh-15-v2-terminal.json`,
`campaign004-fresh-prefix-16-v2-review.json` and its finished guard.
Parent receipt SHA256:
`eede2435620a7f21f805a854df4e35b8ddfab31cad6702d1a8ad251cca00b330`.

Prepared `continue_campaign004_fresh16_v2.py`, restricted to slot16. It uses
unchanged `confirmation_startup_v2`, `quality_readiness_v2` and
`confirmation_dispatch_v2` APIs with one same-parent startup/seal/dispatch
sequence. An outer1,800s CPU guard covers startup and readiness, including
final replay checks; original prepare/child/review guards and both locks remain.
It requires this independent review and a finished guard, pins plan/cache/MPS,
and forbids retries or automatic later slots. Syntax and exact adaptation
diffs were checked; frozen production source is unchanged.
Commit/push this review before slot16: gated MLA D/4, LR0.00125, seed3511.
Retain every fresh pair, including negative results.


### Campaign004 closed at slot16 startup; no second fresh child launched

The slot15 review and bounded slot16 wrapper were committed/pushed as`bdc4a92`.
Slot16 controller46637 stopped after68.40s during startup with
`ValueError('Frozen slot16 request changed before receipt')`. It produced
no startup success receipt, readiness seal, child process, result or checkpoint.
The controller exited and process inspection confirmed no research workers.
Do not retry or reopen this failed campaign; its16 accepted quality results
(15 screening plus one fresh) remain evidence, not a completed fresh panel.

Read-only diagnosis under both locks and an outer CPU guard verified all556
preexisting file identities and all548 non-checkpoint byte sequences unchanged.
The only appends are the slot16 startup guard, attempt, failure receipt and
pending request. The pending request matches the independently reconstructed
frozen choice; frozen files and context still match the slot15 readiness seal.
All eight selected/fresh weight files remain intact. The failure archive
preserves all560 file identities and552 exact non-checkpoint byte sequences.
The diagnostic guard finished in5.30s, peak RSS
252280832 bytes. No deletion, training or holdout evaluation occurred.

Static diagnosis: `confirmation_startup_v2` compares the whole `bind_campaign`
return before/after writing the pending request and finishing its startup guard.
That return includes mutable `metadata_bytes`, so expected namespace growth is
compared as though it were frozen identity. Current metadata is5,841,923 bytes.
The original two intermediate bound dictionaries were not recorded, so the
precise failing field is inferred from the code and unchanged request/file
checks, pending a disposable regression reproduction. The same comparison
appears in slot17 and remaining-slot startup. Existing readiness already has
a helper separating metadata census from frozen identity.

Evidence: `campaign004-fresh-16-v2-launch.json`, terminal and startup guard;
`campaign004-fresh16-failure-review-v2.json`, diagnostic script and finished guard.
Next: a bounded disposable regression exercising real metadata growth, then
repair the affected startup identity comparison while retaining byte/file
caps, exact append audits, source/config/request checks and resource guards.
Do not replay any closed numerical fixture or reopen campaigns001–004.
Reassess the finite future plan only after regression evidence; no new training
is authorized by these failure receipts. Every fresh pair remains protected.


### Fresh-startup metadata census regression fixed

A new disposable CPU regression reproduced the closed004 startup failure in
all three affected paths: slot16, slot17 and the shared remaining-slot startup
at slot18. Each test supplies actual namespace `metadata_bytes` on every
frozen binding, rather than omitting that mutable observation as the earlier
synthetic fixtures did. All three failed at the expected frozen-request
comparison before the fix (276.72s); exact red-test source and logs are preserved.
This confirms the metadata-growth mechanism without retrying campaign004 or
replaying a closed numerical fixture.

The three startup modules now use the existing readiness `_same_frozen_identity`
helper at the comparison that spans request publication/guard completion. It
permits changes only to a valid integer metadata count with matching presence
and the existing256MiB ceiling. Every other returned binding field stays exact,
and the explicit request comparison is unchanged. Actual `bind_campaign` census
validation, per-group/file/total caps, exact evidence/control append checks,
remaining-storage checks, shared locks and all resource/no-retry guards remain.
No model, optimizer, EMA, RNG, data, recipe or quality decision code changed.

Verification:25 tests passed in561.11s, including all three live-growth startup
and receipt-replay cases, malformed census values, changes to every other
binding field, and late unexpected control appends across all three paths.
The latter still block receipt publication. Peak test RSS was244,645,888 bytes;
the4GiB RSS/8GiB disk/1,800s limits held throughout. These are disposable
history tests, not new model training or an end-to-end MPS reliability claim.
Slots19–23 share the corrected remaining-slot comparison; this regression
exercised that shared path at18 rather than rerunning every full chain.

An independent locked preservation review matched all560 closed004 file
identities, all552 non-checkpoint byte sequences and all eight selected/fresh
checkpoint files. Runtime is unchanged, and only the three startup source
files differ from the004 frozen source. New source SHA256:
`73325667bc47f62f43d322b3351c3fe65f94ddace930c333be3fb80e4fd1bcfa`.
Campaigns001–004 remain closed; no cleanup, training or test-holdout use occurred.

Evidence: `confirmation-census-v2-review.json`, before manifest, exact red-test
source, red/green logs and finished guards; review script and regression test
are retained. Next: refreshed prerequisite applicability with archive-aware
original-evidence verification, prospective metadata/capacity audit and a
finite future plan before any new training. Preserve selected/fresh/failure
weights and use audited eligible cleanup proactively if capacity is insufficient.


### Census-fix prerequisite applicability and additional archival plan

Archive-aware source applicability passed for source
`73325667bc47f62f43d322b3351c3fe65f94ddace930c333be3fb80e4fd1bcfa`.
Only the three reviewed fresh-startup comparisons changed since004; model,
optimizer, EMA, RNG, data/tokenizer and request semantics remain unchanged.
Original numerical/input reports and artifacts were verified without execution,
including121 original numerical file identities and35 archived members through
the unchanged fully verified archives. Closed002/003/004 evidence and all162
remaining tokenizer checkpoint files were reconciled. The initial read-only
audit referenced a nonexistent handoff review filename; its source and failed
guard are preserved, and the corrected run used the actual completed review.
No campaign or numerical fixture was retried.

The unchanged full-panel requirement is30,631,412,484 bytes including8GiB
reserve, versus28,386,181,120 bytes free at audit:2,245,231,364 short.
Three completed successful synthetic diagnostic checkpoints are eligible for
lossless archival under the retention policy: baseline001, handoff001 and
trace-v3-001 (all `fixture.pt`, total1,124,140,727 bytes). They are not model
selections, fresh pairs or failed checkpoints. Previous compaction excluded
them for that operation; this explicit manifest preserves their logical
identities through verified cold storage rather than discarding evidence.

`checkpoint-compaction-002/plan.json` pins exact paths/sizes/SHA256,159 excluded
selected/fresh/failed loose weights, three existing protected archives and4,291
non-checkpoint file identities. The adapted single-worker archiver retains the
original stream verification, disk/RSS/time limits, exclusive worker lock and
per-file removal journal; it removes loose copies only after every archive
member passes complete decompression/hash verification. Syntax and adaptation
diff reviewed. Commit/push the plan before applying it.
Even a zero-byte archive cannot fully close the current capacity gap. Do not
reduce reserve or launch a partial panel; selected/fresh/failure weights remain
protected. Prospective metadata and complete capacity checks still precede
any future finite-plan freeze.


### Additional diagnostic archival completed; prospective metadata passes

The archival plan was committed/pushed as`2a7bdba` before execution. Three
completed successful diagnostic weights totaling1,124,140,727 bytes were
losslessly archived into44,015 bytes; net logical space reclaimed is
1,124,096,712 bytes. All members passed complete streamed decompression and
size/SHA256 verification before loose-copy removal. Observed free space rose
from28,383,399,936 to29,507,551,232 bytes immediately after archival.
All159 protected loose selected/fresh/failed weights, three prior archives
and4,291 non-checkpoint files passed post-operation identity checks. No
checkpoint bytes were discarded and no unrelated files were removed.
Protected local archive (never push):
`research_archives/checkpoint-compaction-002/successful-diagnostics.tar.zst`,
SHA256`fce1eb1fd01157246bfe7d5d7b51aeaa96b8966eba5456885c412f310bce772a`.

Independent logical-evidence review rehashed all four archives and reconciled
all38 archived members with original file identities. It verified closed002
(250 files),003(337),004(560), all three successful diagnostic histories and
121 original numerical-prerequisite files. Use both compaction manifests for
future historical audits; original-path-only readers must restore exact bytes
or explicitly resolve archival identity. Originals in the completed diagnostic
histories are now logical archive members; do not rerun those old controllers.

Prepared a non-runnable prospective005 byte-layout preview outside
`research_runs`, carrying original numerical/input reports and archive-aware
source applicability losslessly. No005 campaign directory was created and no
plan was frozen. Independent transport review bound the numerical artifacts,
verified the unchanged96 training-token IDs and exact embedded original bytes;
no numerical execution or holdout access occurred. The unchanged full-panel
shape remains3pilots,24quality trajectories and3inference panels solely as a
capacity/layout proposal, not new execution authority or an independent result.
Its metadata envelope passes:2,092,559/2,097,152 bytes in`evidence/meta`
(4,593 bytes remaining), with855 conservative files below the2,048 limit.

Capacity remains blocked:29,516,120,064 observed free versus30,631,412,484
required, including the unchanged8GiB reserve; shortfall1,115,292,420 bytes.
No further eligible loose tokenizer checkpoints remain after this manifest;
the remaining159 are protected selected/fresh/failed evidence. Do not lower
guards, rerun a closed campaign or launch a partial panel. Additional capacity
is needed before a fresh full-panel freeze. Do not repeatedly notify or poll
an unchanged shortfall. Any scientifically different smaller future study
would need its own prospective design, validation and storage contract.

Evidence: `checkpoint-compaction-002/completed.json`, archive receipt, journal,
finished guards and `logical-evidence-review.json`;
`census-campaign005-preview-review.json`, preview payloads, preview script and
`census-preview-transport-review-v2.json`. No training/test worker remains.


### Transparent filesystem compression clears the remaining capacity gap

The renewed human request prompted an audit beyond tokenizer checkpoint weights.
Historical Markdown knowledge snapshots occupied2,037,526,528 allocated bytes.
The explicit `filesystem-compression-001` manifest compressed174 snapshots with
APFS transparent compression, preserving every original path and readable byte;
one4,011-byte snapshot remained unchanged because compression saved no blocks.
The initial storage operation stopped safely on that no-savings copy; its guard
and journal were retained, and a reviewed continuation completed the remaining
entries. No research campaign or numerical fixture was retried.

Net allocated space recovered:1,594,634,240 bytes. All175 snapshot SHA256/size,
mode, mtime and ownership checks passed. All33,352 other inventoried research
files passed unchanged checks, including159 selected/fresh/failed tokenizer
weights, four archives, historical weights, non-checkpoint evidence and corpus.
Observed free space30,967,259,136 now exceeds the unchanged30,631,412,484-byte
full-panel requirement including8GiB reserve. No model checkpoint was removed,
archived or recompressed. The old campaigns001–004 remain closed.

Evidence: `filesystem-compression-001/plan.json`, both journals, original partial
guard, finished continuation guard and `completed.json`. This changes physical
storage only; ordinary readers and old hashes continue to see identical bytes.


Campaign005 is now frozen from the exact previously reviewed prospective bytes
(plan SHA256 `7572d9dd09bb4b9c6b064179ed3439ffc90b772e53b9addec6066cd58be06d94`).
Fresh publication rechecked current source/runtime, all bound files, original
prerequisite transport, four archive identities, compression receipts and actual
full capacity. `bind_campaign` accepted numerical and training-input bindings.
The finite plan remains3pilots,24quality trajectories and3inference panels with
unchanged grid/seeds, budgets, retention and8GiB reserve. Old campaigns remain
closed, and their results are not pooled. The serial starter is identical to004
apart from campaign/receipt paths: three pilots, audited pilot cleanup, then
quality slot0 only; later dispatch requires separate accepted-history review.
Evidence: `campaign005-freeze-v2-review.json`, finished guard and frozen bytes.


### Campaign005 pilots and audited boundary verified

All three20-update resource pilots completed and passed cache/resource checks:
hybrid97.14s, gatedMLA98.90s, KDA119.58s. Sampled peak RSS was at most
3,232,137,216 bytes and MPS-driver allocation at most4,112,416,768 bytes;
no resource violation was recorded. These short checks do not establish full
training stability or quality superiority.

The production pilot-boundary review authorized removal of six successful pilot
weights totaling3,769,780,224 bytes. A separate read-only review verified the
recorded deletion manifest, all76 archived non-checkpoint byte streams and
immutable evidence hashes. Selected/fresh/failure checkpoints were not involved.
Quality slot0 is running under controller88781/child89256; no quality result or
adoption claim yet. Later slots are not automatically dispatched by this starter.
Evidence: `campaign005-pilots-reviewed-v2.json`, pilot archive, boundary receipt
and `campaign005-quality0-start-v2.json`.


### Campaign005 first quality cell accepted; serial LR screen continues

Slot0 (hybridD/4, LR0.000625, seed3501, width512,1200updates) completed
in859.33s and received parent acceptance. Independent prefix01 review verified
its scientific/transport history, raw/H200 checkpoint hashes and all preserved
non-checkpoint bytes. H200 full-validation CE4.965645637327922 versus
raw4.996666329241553, over34,278targets/17blocks. This is one screening cell;
no architecture comparison, selection or adoption follows from it. Holdout closed.

The review guard finished without violations. Remaining campaign storage fit:
29,690,634,240 bytes free versus29,384,508,436 required including the8GiB reserve.
Both original worker processes exited; process inspection found no other research
worker. The005 LR helper adapts only campaign/receipt bindings from004 and
dispatches exactly slot1 through unchanged production gates. Controller89974,
child89987, session9207 are observation hints, not durable proof of liveness.
Review accepted prefix02 after slot1 succeeds and exits before considering slot2.

Evidence: `campaign005-terminal-v2.json`, `campaign005-lr-prefix-01-v2-review.json`,
finished prefix guard,005 review/continuation helpers and slot1 launch receipt.
See `TOKENIZER_RESUME.md` and `TOKENIZER_RESUME_STATE.json` for current resumption.


### Campaign005 second quality cell accepted

Slot1 (gatedMLAD/4, LR0.000625, seed3501, width512,1200updates) completed
in542.68s and received parent acceptance. Independent prefix02 review verified
both accepted cells, checkpoint identities and non-checkpoint evidence. H200
full-validation CE4.953336852752508 versus raw4.9200928696776876: raw was
0.0332439830748204 nats/token better in this cell. Retain this EMA reversal;
do not change the prospective H200 selection rule. No architecture winner or
adoption is established by these one-seed screens. Holdout remains closed.

The review guard finished without violations. Remaining campaign storage fit:
28,480,778,240 bytes free versus28,162,784,220 required including8GiB reserve.
After verifying worker exit and no other research workers, dispatched only slot2
through the unchanged production gates (controller90595, child90610, session97599).
Review prefix03 after successful slot2 acceptance and exit before slot3.

Evidence: `campaign005-lr-01-v2-terminal.json`, `campaign005-lr-prefix-02-v2-review.json`,
finished prefix guard and `campaign005-lr-02-v2-launch.json`. Current instructions
are in `TOKENIZER_RESUME.md` and its SHA-pinned state index.


### Campaign005 first LR setting completed across three families

Slot2 (KDAD/4, LR0.000625, seed3501, width512,1200updates) completed
in991.54s and received parent acceptance. Independent prefix03 review verified
all three accepted cells and preserved checkpoint/non-checkpoint identities.
KDA H200 full-validation CE4.944542140258151 versus raw4.969660905926372.
At this first LR setting, H200 CE was hybrid4.965645637327922,
gatedMLA4.953336852752508 and KDA4.944542140258151. These reused-seed
screens establish neither tuned family superiority nor adoption. Finish the
frozen LR/rank stages and fresh confirmations; retain H200 as the selection metric.

The review guard finished without violations. Remaining campaign storage fit:
27,204,509,696 bytes free versus26,907,661,636 required including8GiB reserve.
After verifying worker exit and no other research workers, dispatched only slot3
through unchanged production gates (controller91415, child91431, session24853).
Review prefix04 after successful slot3 acceptance and exit before slot4.
No cleanup or holdout use occurred; selected/fresh/failure weights remain protected.

Evidence: `campaign005-lr-02-v2-terminal.json`, `campaign005-lr-prefix-03-v2-review.json`,
finished prefix guard and `campaign005-lr-03-v2-launch.json`. Current instructions
are in `TOKENIZER_RESUME.md` and its SHA-pinned state index.


### Campaign005 second KDA learning-rate screen accepted

Slot3 (KDAD/4, LR0.00125, seed3501, width512,1200updates) completed
in996.61s and received parent acceptance. Independent prefix04 review verified
all four accepted cells and preserved checkpoint/non-checkpoint identities.
H200 full-validation CE4.976398347271712 versus raw5.079611199806855.
Compared with LR0.000625 on the same screening seed, KDA H200 CE is
0.031856207013561 worse. This partial-grid observation is not LR selection,
fresh confirmation or adoption; the frozen grid and H200 metric remain unchanged.

The review guard finished without violations. Remaining storage fit:
25,929,609,216 bytes free versus25,652,539,052 required including8GiB reserve.
After verifying worker exit and no other research workers, dispatched only slot4
through unchanged gates (controller92236, child92267, session30754).
Review prefix05 after successful slot4 acceptance and exit before slot5.
No cleanup or holdout use occurred; selected/fresh/failure weights remain protected.

Evidence: `campaign005-lr-03-v2-terminal.json`, `campaign005-lr-prefix-04-v2-review.json`,
finished prefix guard and `campaign005-lr-04-v2-launch.json`. See the resume guide
and SHA-pinned state index for the current next action.


### Campaign005 second hybrid learning-rate screen accepted

Slot4 (hybridD/4, LR0.00125, seed3501, width512,1200updates) completed
in901.91s and received parent acceptance. Independent prefix05 review verified
all five accepted cells and preserved checkpoint/non-checkpoint identities.
H200 full-validation CE4.976714106020634 versus raw5.077557087636783.
Compared with LR0.000625 on the same screening seed, hybrid H200 CE is
0.011068468692712 worse. This partial-grid observation is not LR selection,
fresh confirmation or adoption; the frozen grid and H200 metric remain unchanged.

The review guard finished without violations. Remaining storage fit:
24,646,782,976 bytes free versus24,405,766,076 required including8GiB reserve.
After verifying worker exit and no other research workers, dispatched only slot5
through unchanged gates (controller93091, child93130, session66438).
Review prefix06 after successful slot5 acceptance and exit before slot6.
No cleanup or holdout use occurred; selected/fresh/failure weights remain protected.

Evidence: `campaign005-lr-04-v2-terminal.json`, `campaign005-lr-prefix-05-v2-review.json`,
finished prefix guard and `campaign005-lr-05-v2-launch.json`. See the resume guide
and SHA-pinned state index for the current next action.


### Campaign005 gated MLA benefits from the second learning rate

Slot5 (gatedMLAD/4, LR0.00125, seed3501, width512,1200updates) completed
in594.50s and received parent acceptance. Independent prefix06 review verified
all six accepted cells and preserved checkpoint/non-checkpoint identities.
H200 full-validation CE4.82145746133221 versus raw4.8244493101960275.
Compared with LR0.000625 on the same screening seed, gated MLA H200 CE
improved0.131879391420298. Hybrid and KDA regressed at this LR increase,
supporting equal-budget family-specific LR tuning rather than a shared LR.
This is a partial-grid observation, not selection or fresh-seed superiority;
the frozen grid and H200 metric remain unchanged.

The review guard finished without violations. Remaining storage fit:
23,404,400,640 bytes free versus23,184,041,860 required including8GiB reserve.
After verifying worker exit and no other research workers, dispatched only slot6
through unchanged gates (controller93890, child93924, session58522).
Review prefix07 after successful slot6 acceptance and exit before slot7.
No cleanup or holdout use occurred; selected/fresh/failure weights remain protected.

Evidence: `campaign005-lr-05-v2-terminal.json`, `campaign005-lr-prefix-06-v2-review.json`,
finished prefix guard and `campaign005-lr-06-v2-launch.json`. See the resume guide
and SHA-pinned state index for the current next action.


### Campaign005 gated MLA learning-rate grid screened

Slot6 (gatedMLAD/4, LR0.0025, seed3501, width512,1200updates) completed
in603.72s and received parent acceptance. Independent prefix07 review verified
all seven accepted cells and preserved checkpoint/non-checkpoint identities.
H200 full-validation CE4.857494233625267 versus raw4.919870187125091.
All three gated MLA rates are screened: H200 CE at LR0.000625/0.00125/0.0025
is4.953336852752508/4.82145746133221/4.857494233625267. LR0.00125 is
lowest in this family on this screen seed. Formal selection waits for the full
nine-cell LR stage, with rank screening and fresh confirmation still required.
No adoption or seed-population conclusion; frozen grid and metric unchanged.

The review guard finished without violations. Remaining storage fit:
22,157,549,568 bytes free versus21,962,317,644 required including8GiB reserve.
After verifying worker exit and no other research workers, dispatched only slot7
through unchanged gates (controller94679, child94703, session72867).
Review prefix08 after successful slot7 acceptance and exit before slot8.
After slot8, adapt the004 LR-boundary helper to005 for selection and audited
cleanup; the prefix helper accepts only1–8 completed cells. No cleanup or holdout
use occurred; selected/fresh/failure weights remain protected.

Evidence: `campaign005-lr-06-v2-terminal.json`, `campaign005-lr-prefix-07-v2-review.json`,
finished prefix guard and `campaign005-lr-07-v2-launch.json`. See the resume guide
and SHA-pinned state index for the current next action.


### Campaign005 KDA learning-rate grid screened

Slot7 (KDAD/4, LR0.0025, seed3501, width512,1200updates) completed
in1036.27s and received parent acceptance. Independent prefix08 review verified
all eight accepted cells and preserved checkpoint/non-checkpoint identities.
H200 full-validation CE5.118220312101873 versus raw5.274697344090305.
KDA H200 CE at LR0.000625/0.00125/0.0025 is
4.944542140258151/4.976398347271712/5.118220312101873;0.000625 is lowest
on this screening seed. Formal selection waits for the last hybrid screen.
No fresh-seed or adoption conclusion; the frozen grid and metric are unchanged.

Review guard finished without violations. Remaining storage fit:
20,878,508,032 bytes free versus20,707,195,060 required including8GiB reserve.
After verifying worker exit and no other research workers, dispatched only slot8
(controller95618, child95644, session28061). The005 LR-boundary helper was
adapted only in campaign bindings from004 and syntax/identity checked, not run.
After slot8 acceptance/exit, run selection and review it before audited cleanup;
do not call the prefix helper with9. Selected/fresh/failure weights remain
protected; no cleanup or holdout use occurred in this continuation.

Evidence: `campaign005-lr-07-v2-terminal.json`, `campaign005-lr-prefix-08-v2-review.json`,
finished guard, `campaign005-lr-08-v2-launch.json`, `campaign005_lr_boundary_v2.py`.
See the resume guide and SHA-pinned state index for the next action.


### Campaign005 LR selection and audited cleanup complete

All nine width512, D/4,1200-update LR screens passed. Final hybrid slot8
(LR0.0025, seed3501) H200CE5.110104452076268, raw5.277339023920939.
The full production selection audit and independent minimum-CE calculation agree:

| Family | LR0.000625 H200CE | LR0.00125 | LR0.0025 | Selected LR |
| --- | ---: | ---: | ---: | ---: |
| Hybrid | 4.9656456373 | 4.9767141060 | 5.1101044521 | 0.000625 |
| Gated MLA | 4.9533368528 | 4.8214574613 | 4.8574942336 | 0.00125 |
| KDA | 4.9445421403 | 4.9763983473 | 5.1182203121 | 0.000625 |

Family-specific tuning matters in this reused-seed screen. These choices feed
the frozen rank comparison; they do not establish fresh-seed superiority or adoption.
Selection guard finished177.20s, cleanup guard188.55s, without violations.
The exact manifest was committed/pushed before completion of cleanup. Removed
only12 losing-screen weights at slots1/3/4/6/7/8, totaling7,447,239,552 logical
bytes. Retained all six selected weights at slots0/2/5 (3,723,619,776 bytes).
Independent before/after comparison verified exact deletions, unchanged retained
bindings and all preserved non-checkpoint bytes;344 non-checkpoint files are
archived in the cleanup review. Deleted weights cannot be re-evaluated.
Fresh pairs, old selected/failure weights, archives and holdout remain protected.

Post-cleanup review observed27,126,628,352 free bytes versus26,907,661,636
required for the remaining campaign including8GiB reserve. Read-only process
inspection initially hit an automatic approval-review timeout; the one permitted
retry succeeded. No human authorization was missing, and no guard was bypassed.
After cleanup exit and a new process check, rank slot9 was dispatched through
the original gates. Prepared005 rank starter/review/continuation helpers differ
from004 only in campaign bindings; no production source changed.

Evidence: `campaign005-lr-summary-v2.json`, selection/cleanup before archives,
reviews and finished guards, and `campaign005-rank-09-v2-launch.json`. The
resume guide/index record the live controller and exact next accepted-prefix review.


### Campaign005 first rank screen accepted

Slot9 (hybridD/8, LR0.000625, seed3501, width512,1200updates) completed
in947.10s and received parent acceptance. Independent rank-prefix10 review
verified prior LR selection/cleanup, this accepted rank cell, checkpoint identities
and preserved metadata. H200 full-validation CE4.956084595234285 versus
raw4.9859793480260155. H200 CE is0.009561042093637 below the selectedD/4
anchor on the same screening seed. Rank selection, fresh confirmation and
inference remain; no adoption or seed-population claim.

Review guard finished30.48s without violations. Remaining storage fit:
25,847,701,504 bytes free versus25,667,182,740 required including8GiB reserve.
After confirming worker exit/no other research workers, dispatched only slot10
through unchanged gates: controller99377, session61462, observed phasestarting,
childNone. Startup history verification may precede child creation; never duplicate
the launch. After slot10 acceptance/exit, review prefix11 before slot11.
No cleanup or holdout use occurred; all selected/fresh/failure evidence preserved.

Evidence: `campaign005-rank-09-v2-terminal.json`, `campaign005-rank-prefix-10-v2-review.json`,
finished guard and `campaign005-rank-10-v2-launch.json`. Current resumption is
recorded in the guide and SHA-pinned state index.


### Campaign005 gated MLA D/8 rank screen accepted

Slot10 (gatedMLAD/8, LR0.00125, seed3501, width512,1200updates) completed
in648.80s and received parent acceptance. Independent rank-prefix11 review
verified prior LR selection/cleanup, both accepted rank cells, checkpoint identities
and preserved metadata. H200 full-validation CE4.828688418517511 versus
raw4.830738001895907. H200 CE is0.007230957185301 above the selectedD/4
anchor on the same screening seed. Rank selection, fresh confirmation and
inference remain; no adoption or seed-population claim.

Review guard finished46.22s without violations. Remaining storage fit:
24,627,044,352 bytes free versus24,458,051,804 required including8GiB reserve.
After confirming worker exit/no other research workers, dispatched only slot11
through unchanged gates: controller318, session48019, training child484.
Actual process inspection confirmed one training child; never duplicate the launch.
After slot11 acceptance/exit, review prefix12 before slot12. No cleanup or
holdout use occurred; all selected/fresh/failure evidence preserved.

Evidence: `campaign005-rank-10-v2-terminal.json`, `campaign005-rank-prefix-11-v2-review.json`,
finished guard and `campaign005-rank-11-v2-launch.json`. Current resumption is
recorded in the guide and SHA-pinned state index.


### Campaign005 KDA D/8 rank screen accepted

Slot11 (KDA D/8, LR0.000625, seed3501, width512,1200updates) completed
in1078.64s and received parent acceptance. Independent rank-prefix12 review
verified prior LR selection/cleanup, all three accepted D/8 rank cells, checkpoint
identities and preserved metadata. H200 full-validation CE4.942311713761669
versus raw4.969815671035888. H200 CE is0.002230426496483 below the selected
D/4 anchor on the same screening seed. This small difference is a screening
observation; D/2 comparisons, fresh confirmation and inference remain.
No rank selection, adoption or seed-population claim.

Review guard finished55.13s without violations. Remaining storage fit:
23,381,426,176 bytes free versus23,207,123,588 required including8GiB reserve.
Actual process inspection confirmed prior worker exit and no other research
workers. Dispatched only slot12 through unchanged gates: controller2103,
session29865, observed phasestarting, childNone. Never duplicate this launch.
After slot12 acceptance/exit, review prefix13 before slot13. No cleanup or
holdout use occurred; all selected/fresh/failure evidence preserved.

Evidence: `campaign005-rank-11-v2-terminal.json`, `campaign005-rank-prefix-12-v2-review.json`,
finished guard and `campaign005-rank-12-v2-launch.json`. Current resumption is
recorded in the guide and SHA-pinned state index.


### Campaign005 KDA D/2 rank screen accepted

Slot12 (KDA D/2, LR0.000625, seed3501, width512,1200updates) completed
in1091.09s and received parent acceptance. Independent rank-prefix13 review
verified prior LR selection/cleanup, all four accepted rank cells, checkpoint
identities and preserved metadata. H200 full-validation CE4.930133935108931
versus raw4.9562451051463. H200 CE is0.014408205149220 below the selected
D/4 anchor and0.012177778652737 below D/8 on the same screening seed.
KDA D/2 leads its three screened ranks, but remaining family rank cells, formal
rank selection, fresh confirmation and inference remain; no adoption claim.

Review guard finished65.08s without violations. Remaining storage fit:
22,086,991,872 bytes free versus21,943,612,396 required including8GiB reserve.
Actual process inspection confirmed prior worker exit and no other research
workers. Dispatched only slot13 through unchanged gates: controller3448,
session23454, observed phasestarting, childNone. Never duplicate this launch.
After slot13 acceptance/exit, review prefix14 before slot14. After slot14 use
the full rank boundary; do not call the prefix helper with15. No cleanup or
holdout use occurred; all selected/fresh/failure evidence preserved.

Evidence: `campaign005-rank-12-v2-terminal.json`, `campaign005-rank-prefix-13-v2-review.json`,
finished guard and `campaign005-rank-13-v2-launch.json`. Current resumption is
recorded in the guide and SHA-pinned state index.


### Campaign005 gated MLA D/2 rank screen accepted

Slot13 (gated MLA D/2, LR0.00125, seed3501, width512,1200updates) completed
in694.54s and received parent acceptance. Independent rank-prefix14 review
verified prior LR selection/cleanup, all five accepted rank cells, checkpoint
identities and preserved metadata. H200 full-validation CE4.830300243418004
versus raw4.833802662880457. D/4 (4.82145746133221) remains better than
D/8 (4.828688418517511) and D/2 on this screening seed. Final hybrid D/2,
formal rank selection, fresh confirmation and inference remain; no adoption.

Review guard finished72.39s without violations. Remaining storage fit:
20,823,105,536 bytes free versus20,696,701,620 required including8GiB reserve.
Actual process inspection confirmed prior worker exit and no other research
workers. Dispatched only slot14 through unchanged gates: controller4628,
session93760, observed phasestarting, childNone. Never duplicate this launch.
No cleanup or holdout use occurred; all selected/fresh/failure evidence preserved.

Prepared `campaign005_rank_boundary_v2.py` from004 using campaign/path-only
substitutions; AST parsing and exact reverse comparison passed. Neither boundary
phase has run. After slot14 acceptance/exit, select and independently verify all
rank choices and the exact manifest, commit/push evidence, then perform audited
cleanup and verification before fresh slot15. Never use rank-prefix completed15.

Evidence: `campaign005-rank-13-v2-terminal.json`, `campaign005-rank-prefix-14-v2-review.json`,
finished guard and `campaign005-rank-14-v2-launch.json`. Current resumption is
recorded in the guide and SHA-pinned state index.


### Campaign005 screening completed; rank choices verified

Final slot14 hybrid D/2 reached H200 CE4.951268515577171, raw4.981226434801457;
its child completed998.90s and parent accepted. All15 screens are now accepted.
Full rank selection and replay finished347.31s without guard violations.
An independent minimum-H200-CE calculation confirmed hybrid D/2 LR0.000625,
gated MLA D/4 LR0.00125 and KDA D/2 LR0.000625. Hybrid D/2 improves over
D/8 by0.004816079657114 CE on the reused screening seed; fresh confirmation
and inference remain mandatory. No adoption or holdout claim.

Independent manifest checks matched original result paths, sizes, hashes and
archived bindings. Keep selected raw/H200 pairs5/12/14 (3,744,596,480 bytes).
Only12 losing weights in slots0/2/9/10/11/13 are eligible (7,449,344,384 bytes).
No deletion has happened in this phase. Commit/push selection evidence before
cleanup; deleted weights cannot subsequently be re-evaluated. Preserve all
noncheckpoint metadata, older selected models, fresh pairs, failures and archives.

Evidence: campaign005-rank-select-v2-{before,review,guard}.json and the compact
campaign005-rank-selection-summary-v2.json. First-fresh helper prepared by
campaign/path-only substitutions, AST/reverse comparison passed; not executed.


### Campaign005 rank cleanup verified; first fresh confirmation dispatched

Guarded cleanup completed323.31s. Independent before/after comparison confirmed
exactly12 manifest-listed removals in slots0/2/9/10/11/13, reclaiming
7,449,344,384 logical bytes. The six selected raw/H200 weights in slots5/12/14
(3,744,596,480 bytes) and all511 prior noncheckpoint streams are unchanged.
Original hashes, results and deletion receipts remain; removed losing weights
cannot be re-evaluated. Other selected models, fresh pairs, failures, archives
and the closed test holdout were untouched.

Remaining confirmation storage was27,031,441,408 bytes free versus26,712,927,696
required, including the unchanged8GiB reserve. Selection/manifest were pushed
as3a138bc before cleanup; completed cleanup/independent summary pushed as0efb952.
After checking actual process exit, dispatched only fresh slot15 (hybrid D/2,
LR0.000625, seed3511): controller9355, session23723, observed phasestarting,
childNone. No fresh result or adoption yet. Never duplicate this launch.

The path-only005 first-fresh reviewer passed AST/reverse comparison and is not
yet executed. After slot15 acceptance/exit, review completed16, then inspect/adapt
the slot16 helper and use the current tested census-fixed gates. Later slots and
inference retain their own prerequisites. Resume guide/index contain exact paths.


### Campaign005 first fresh pair accepted and reviewed

Slot15 hybrid D/2, LR0.000625, fresh seed3511 completed1000.74s and received
parent acceptance. H200 full-validation CE4.999353112775229, raw5.029627986077138.
The independent first-fresh history review finished36.81s without guard violations,
checking frozen provenance, full history, both checkpoint identities and storage.
Both weights are protected; no cleanup occurred. Paired controls and seeds3512/3513
remain pending, so no comparative or adoption conclusion is supported yet.

Remaining storage verified25,736,900,608 bytes free versus25,453,566,624 required,
including8GiB reserve. Current source fingerprint exactly matches frozen005 and
the census-fix review; every recorded source/test pin matches. Slot16 controller
was adapted from004 by campaign/path substitutions only, AST/reverse comparison
passed. Existing25-test green evidence is applicable; no numerical fixture rerun.
After prior workers exited, dispatched only slot16: controller10478, session87554,
observed phasestarting, childNone. No live slot16 success claim yet.

Evidence: campaign005-fresh-15-v2-terminal.json, campaign005-fresh-prefix-16-v2
review/guard, campaign005-fresh16-controller-review-v2.json and slot16 launch.
After slot16 acceptance, use its dedicated confirmation-history verifier before
the slot17 later-startup path; do not reuse the first-fresh-only verifier.
All selected/fresh/failure evidence, protected archives and closed holdout preserved.


### Campaign005 fresh gated MLA accepted; slot16 compact history replay passed

Slot16 gated MLA D/4, LR0.00125, seed3511 completed696.13s and received parent
acceptance. H200 CE4.845517761609039, raw4.834619455145382. The frozen H200
metric favors gated MLA over hybrid (4.999353112775229) by0.153835351166190
on this seed; do not switch to raw because its score is lower. KDA and the two
remaining fresh seeds are pending. Both fresh raw/H200 pairs remain protected.

Dedicated slot16 history replay reconstructed the compact acceptance receipt,
checkpoint identities and storage; guard finished57.23s with no violations.
Remaining space24,580,538,368 bytes versus24,231,842,408 required, including8GiB
reserve. Wrapper diffs preserve locks, guards, immutable metadata and all original
production gates; reviewer calls the slot16 history API, controller calls the
separate slot17 later-startup and later-dispatch APIs. No production edits.

The slot17 dispatch/history/test hashes match the recorded parent tests. Shared
acceptance has a later hash pinned in inference-dispatch evidence and frozen005;
the old slot17 tests are not represented as covering that newer hash. Full live
slot16 replay passed on the current frozen source; census-fix source/test pins
also match. Details: campaign005-fresh17-controller-review-v2.json.

After prior worker exit, dispatched only slot17: controller12309, session44132,
observed phasestarting, childNone. No cleanup/holdout use or live slot17 success
claim. Next review uses the slot17 later-history verifier before slot18's remaining
startup/dispatch path. All selected/fresh/failure evidence and archives preserved.


### Campaign005 first fresh seed complete; independent slot17 replay passed

All three seed3511 cells received parent acceptance. H200 validation CE is
4.845517761609039 for gated MLA D/4, 4.9743094953134435 for KDA D/2, and
4.999353112775229 for hybrid D/2. Hybrid is worse by0.1538353511661903 versus
gated MLA and0.02504361746178585 versus KDA on this seed. Slot17 raw CE is
4.998094061861975; its child completed1117.89s and parent accepted1405.94s.
These are one-seed comparisons, not adoption. Seeds3512/3513 and frozen
statistical/inference gates remain. The H200 decision metric is unchanged.

Dedicated slot17 compact-history reconstruction finished73.19s without guard
violations, binding18 completed records and six live selected/fresh checkpoint
pairs. Space verified23,286,218,752 bytes free versus22,968,331,216 required,
including the unchanged8GiB reserve. Every fresh raw/H200 pair remains protected;
no cleanup or holdout use occurred. The full review preserves metadata bytes,
checkpoint identities and prior boundary evidence.

Prepared exact-slot18 startup/dispatch and slot17-review wrappers with AST/diff
and signature checks. Current full production source matches frozen005; current
census-fix source/test pins match. Historical slot18-only test hashes precede
remaining-slot extensions and are not misrepresented as current coverage.
Later inference evidence binds current dispatch/acceptance, and the census review
binds current startup. No production edits or closed numerical fixture rerun.

After all prior workers/reviewer exited, dispatched only slot18 (KDA D/2,
LR0.000625, fresh seed3512): controller14449, session32349. Startup may precede
child creation. No slot18 success claim. Next independent review must use
remaining_confirmation_history_v2 explicitly slot18 before preparing slot19.
Evidence: campaign005-fresh-seed3511-accepted-v2.json,
campaign005-fresh-prefix-18-v2-review/guard.json, campaign005-fresh18-wrapper-review-v2.json,
campaign005-fresh18-census-applicability-v2.json and slot18 launch receipt.


### Campaign005 second-seed KDA accepted and independently reviewed

Slot18 KDA D/2, LR0.000625, seed3512 completed1125.81s; parent accepted1493.48s.
H200 validation CE4.967920234487431, raw4.994988798030418. Its seed3512
paired controls remain pending; no adoption or metric change. Both weights stay
protected alongside all prior fresh pairs and selected/failure evidence.

The dedicated slot18 history reconstruction passed93.02s, checking19 completed
records and7 retained selected/fresh pairs, compact transport, immutable
metadata and all three cleanup boundaries. Remaining capacity21,985,591,296 bytes
versus21,704,820,024 required includes the unchanged8GiB reserve. No cleanup.
Exact-slot19 wrapper changes retain all original production gates; AST/diff/API
checks passed. Frozen source, applicable inference manifest and census-fix
source/test pins match. No production edit or numerical fixture rerun.

After worker/reviewer exit, dispatched only slot19 (hybrid D/2, LR0.000625,
seed3512): controller17187, session61533, guarded startup. No slot19 success claim.
Its eventual idle review must use slot19_history_v2, not slot18-only history.
Evidence: campaign005-fresh-18-v2-terminal.json, fresh-prefix-19 review/guard,
campaign005-fresh19-wrapper-review-v2.json and slot19 launch. Resume guide/index
record exact next actions; local checkpoints/archives remain outside Git.


### Campaign005 second-seed hybrid accepted and independently reviewed

Slot19 hybrid D/2, LR0.000625, seed3512 completed1033.10s; parent accepted1493.51s.
H200 validation CE4.979351837490887, raw5.008163495363069. KDA on the same
seed has H200 CE4.967920234487431, so hybrid is worse by0.011431603003456.
Gated MLA seed3512 remains pending. This partial fresh panel does not establish
adoption; retain the frozen H200 metric, all seeds and inference requirements.

The exact-slot19 independent compact-history replay passed113.63s, checking20
completed records and8 retained selected/fresh pairs, immutable metadata and
all three cleanup boundaries. Remaining capacity20,697,972,736 bytes versus
20,445,458,952 required includes the unchanged8GiB reserve. Every fresh pair,
selected/failure checkpoint and protected archive remains retained; no cleanup.

AST/diff/API checks verified the slot19 reviewer and exact-slot20 wrapper;
current full source matches frozen005 and applicable inference/census pins.
No production edits or numerical fixture rerun. After all workers/reviewer
exited, dispatched only slot20 (gated MLA D/4, LR0.00125, seed3512):
controller20347, session27655, guarded startup. No slot20 success claim.
Its idle review must use slot20_history_v2 before slot21; later scopes differ.
Evidence: campaign005-fresh-19-v2-terminal.json, fresh-prefix-20 review/guard,
campaign005-fresh20-wrapper-review-v2.json and slot20 launch.


### Campaign005 second fresh seed complete; gated MLA leads both observed seeds

Slot20 gated MLA D/4, LR0.00125, seed3512 completed725.50s; parent accepted1291.34s.
H200 CE4.84407892743844, raw4.836186593180043. Seed3512 H200 comparisons:
gated MLA4.84407892743844, KDA4.967920234487431, hybrid4.979351837490887.
Together with seed3511, gated MLA has the lowest H200 CE on both observed fresh
seeds. This is an interim descriptive result; complete seed3513 and all frozen
statistical/inference rules before adoption. Do not switch to raw or pool004.

Dedicated slot20 compact-history replay passed137.14s, binding21 completed
records and9 retained selected/fresh pairs, immutable metadata and all three
cleanup boundaries. Remaining capacity19,433,660,416 bytes versus19,223,734,736
required includes the unchanged8GiB reserve. All fresh pairs and selected/failure
evidence remain protected; no cleanup or holdout use.

Exact-slot21 wrapper/reviewer AST, diffs and API scopes checked; production source
matches frozen005 and applicable inference/census pins. No production edits or
numerical fixture rerun. After all prior workers/reviewer exited, dispatched only
slot21 (gated MLA D/4, LR0.00125, seed3513): controller23736, session83274,
guarded startup. No slot21 success claim. Its idle review must use
final_confirmation_history_v2 explicitly21; this API supports21..23 but final23
cannot append another quality slot. Inference requires completed23.
Evidence: campaign005-fresh-seeds3511-3512-accepted-v2.json, slot20 terminal,
fresh-prefix-21 review/guard, campaign005-fresh21-wrapper-review-v2.json and
slot21 launch. Resume guide/index preserve the exact continuation sequence.


### Campaign005 third-seed gated MLA accepted; final-history replay passed

Slot21 gated MLA D/4, LR0.00125, seed3513 completed737.94s; parent accepted1420.76s.
H200 CE4.839211356994297, raw4.824693534822825. All three gated MLA fresh
cells are now accepted, but KDA/hybrid seed3513 and inference remain pending.
No adoption or switch from the frozen H200 decision metric.

Dedicated final_confirmation_history_v2 replay explicitly for21 passed162.14s,
checking22 completed records and10 retained selected/fresh pairs, immutable
metadata and all three cleanup boundaries. Remaining capacity18,182,131,712 bytes
versus18,002,010,520 required includes the unchanged8GiB reserve. All protected
models, every fresh pair, failures and archives remain intact; no cleanup.

Exact-slot22 wrapper AST/diff/API checks passed; production source matches005
and applicable inference/census pins. After prior workers/reviewer exited,
dispatched only slot22 (KDA D/2, LR0.000625, seed3513): controller27897,
session5372, guarded startup. No slot22 success claim. Review22 before final23.

A read-only inference resumption map records prepare_inference under an outer
guard and both locks, run_panel for the three fixed serial seeds, then the frozen
review_final_campaign_v2 procedure with final inference/3513/review.json. This
map grants no skipped prerequisites and no inference was prepared. The final
procedure must be pinned in the frozen files, verify all three controllers, and
publish outside the sealed campaign. See campaign005-inference-resumption-map-v2.json.
Evidence: slot21 terminal, fresh-prefix-22 review/guard,
campaign005-fresh22-wrapper-review-v2.json and slot22 launch.


### Campaign005 third-seed KDA accepted; final confirmation dispatched

Slot22 KDA D/2, LR0.000625, seed3513 completed1166.35s; parent accepted1971.89s.
H200 CE4.947641727241584, raw4.975336175863581. Gated MLA on seed3513 has
H200 CE4.839211356994297. KDA and gated MLA now have all three fresh cells;
hybrid seed3513 and inference remain pending. No adoption or metric change.

Exact-slot22 final-history replay passed188.16s, checking23 completed records
and11 retained selected/fresh pairs, immutable metadata and all three cleanup
boundaries. Remaining capacity16,881,770,496 bytes versus16,738,499,328 required
includes the unchanged8GiB reserve. Every fresh pair, selected/failure checkpoint
and archive remains protected; no cleanup.

Final-slot23 wrapper AST/diff/API checks passed; current production source matches
frozen005 and applicable inference/census pins. After all prior workers/reviewer
exited, dispatched only slot23 (hybrid D/2, LR0.000625, seed3513):
controller32901, session62286, guarded startup. No slot23 success claim.
After acceptance/exit, independently replay23/completed24; no quality slot24.
Use the pinned inference resumption map and inspect all frozen prerequisites
before prepare_inference/run_panel. Final quality/cost review remains gated on
accepted inference/3513/review.json and all controller/history receipts.
Evidence: slot22 terminal, fresh-prefix-23 review/guard,
campaign005-fresh23-wrapper-review-v2.json and final slot23 launch.


### Campaign005 all nine fresh quality cells accepted

Final slot23 hybrid D/2, LR0.000625, seed3513 has H200 CE4.976864544264859
and raw CE5.002648414981093; parent acceptance took2079.58s. All nine fresh
confirmation cells are now parent accepted. Mean H200 CE over the three seeds
is gated MLA4.8429360153472585, KDA4.96329048568082 and hybrid4.985189831510325.
Gated MLA has the lowest H200 CE on each fresh seed. This is descriptive local
Shakespeare validation evidence, conditional on the adaptive finite choices;
no test-holdout, seed-population or larger-scale claim. No adoption yet.

The final slot23 history reviewer explicitly requests23/completed24 and archives
full non-checkpoint bytes with checkpoint hashes. No quality slot24 is allowed.
The inference wrapper requires the finished independent review, fixed005 source,
plan/cache/MPS identities and both locks. Its outer CPU guard covers only
prepare_inference; the fixed serial run_panel owns individual parent/child
guards. All21 relevant inference source/test pins match the159-test consolidated
manifest; those tests are synthetic and do not replace real prerequisite replay.
The final quality/cost procedure is present in the original frozen plan.
Every fresh raw/H200 pair, selected model, failure and protected archive remains
retained. No cleanup, production-source edit or closed fixture retry occurred.
Evidence: campaign005-fresh-panel-accepted-v2.json, final slot23 terminal,
review_campaign005_fresh23_v2.py and campaign005-inference-wrapper-review-v2.json.

Final independent history replay passed219.31s, confirming24 records and12
retained selected/fresh pairs,8,192,035 metadata bytes and all three cleanup
boundaries. End-of-review free15,577,501,696 bytes exceeds8,858,370,048 required.
Reviewer and prior workers exited. Inference controller38659/session36174 is
now in guarded preparation for the fixed serial3511/3512/3513 panel; inference
is not yet accepted. Sandbox MPS preflight failed before any attempt/publication;
host MPS was confirmed and the unchanged wrapper launched there. This is an
environment preflight record, not a retried inference seed or numerical fixture.
Resume guide/index and the status tool now expose the exact inference continuation.


### Campaign005 inference preparation completed

Shared inference preparation passed under the outer504.62s guard; internal
preparation/replay finished265.37s, both with no resource violation. The exact
panel request SHA256 is688225dd78304f6bdf57029e1106fcb80dc033e08ef0dcb8085dec3b27c629af.
Controller38659 remains active in first-seed readiness replay; no inference
measurement or seed acceptance is claimed. The fixed serial panel still owns
3511/3512/3513; do not duplicate preparation or dispatch.
Evidence: campaign005-inference-preparation-reviewed-v2.json contains the
immutable request and completed guards. All protected pairs remain retained.


### Campaign005 first inference seed accepted

Seed3511 completed its child in141.94s; guarded parent preparation303.37s and
review545.92s finished without violation. All six cache-correctness checks passed.
Actual state bytes for batch1 are hybrid399,360, gated MLA327,680 and KDA335,872;
batch8 scales to3,194,880 /2,621,440 /2,686,976. Hybrid state is larger than
both homogeneous controls in this short-context workload. This single seed is
not the final campaign cost decision; retain the frozen three-seed aggregation.

Controller38659/session36174 remains active, preparing seed3512 serially.
No new launch or locked replay was performed. Immutable seed3511 metadata,
measurements and completed process/guard receipts are preserved byte-for-byte
in campaign005-inference-seed3511-accepted-v2.json. All selected/fresh pairs,
failures and protected archives remain retained; no adoption or holdout use.


### Campaign005 second inference seed accepted

Seed3512 child completed143.33s; guarded parent preparation629.41s and
review810.92s finished without violation. All six cache-correctness checks
passed. Measured state bytes match3511 for each family/batch: hybrid399,360,
gated MLA327,680 and KDA335,872 at batch1, scaling eightfold at batch8.
No final decision: seed3513 and frozen quality/cost review remain required.

Controller38659/session36174 is preparing final seed3513. Immutable seed3512
metadata, measurements and completed guard/process receipts are preserved in
campaign005-inference-seed3512-accepted-v2.json. No duplicate launch, cleanup,
production change, closed retry, holdout use or adoption. All fresh pairs remain
protected. Parent replay costs grow with accepted history but remain within
the existing1800s phase guards; no limits were relaxed.


### Campaign005 inference panel complete; independent final review started

Final seed3513 child completed147.01s; parent preparation930.57s and
review1090.28s finished within unchanged guards. All three seed reviews and
controller receipts are accepted. Serial inference terminal completed5248.11s;
actual controller38659 and all children exited. Full inference metadata,
measurements, controls and receipts are losslessly saved in
campaign005-inference-panel-accepted-v2.json. No model/checkpoint cleanup.

Independent reviewer44715/session5954 now owns both locks under an outer1800s
CPU guard. It invokes the original frozen review_final_campaign_v2 procedure,
checks full quality/inference provenance and publishes exclusively outside the
sealed campaign within128KiB. Source/procedure pins and wrapper AST/API checks
passed. Require review success, finished guard and reviewer exit before a final
decision. No adoption claim, rule change, holdout use or new campaign launch.


### Final audit environment failure preserved and corrected

External reviewer44715 stopped18.98s while reconstructing tokenizer identity:
its wrapper omitted TIKTOKEN_CACHE_DIR and attempted a GPT2 vocab download;
DNS failed. No final decision was published. The wrapper, failed guard and
failure receipt remain immutable. This is a read-only audit environment failure,
not a training/inference failure or the closed failed numerical fixture.

Both existing local cache SHA256 pins verified, and all archived accepted
inference bytes rehashed unchanged. A new external wrapper differs only by
explicit pinned cache setup and a separate output namespace. Frozen source,
procedure, statistical rules and guards are unchanged. Reviewer44945/session83295
now runs that separately recorded read-only audit. No experimental trial was
repeated. Require its finished guard and successful final report before decision.
Evidence: campaign005-final-cache-environment-diagnosis-v2.json.


### Campaign005 closed: hybrid fails both adoption routes

Independent final review completed985.06s, guard finished with no reason and
reviewer exited. All24 quality trajectories, three inference seeds, full history
and protected artifacts were verified. Hybrid is not adopted; existing adopted
recipes remain unchanged. No posthoc promotion of005 gated NoPE MLA versus the
prior adopted RoPE MLA, since that comparison was not this prospective study.

Fresh mean H200 CE: hybrid4.985189831510325, gated MLA4.8429360153472585,
KDA4.96329048568082. Hybrid-minus-control means are+0.142253816163066 and
+0.021899345829505; every seed is worse than each control. Simultaneous upper
bounds at blocks2048/4096/8192 are0.176936/0.190790/0.205227 versus gated MLA
and0.029720/0.032640/0.036977 versus KDA. Strict and neutral quality gates fail.
Raw diagnostic means are5.013479965473767 /4.831833194382750 /4.989473011918658
(hybrid/gated MLA/KDA); the H200 decision metric remains unchanged.

All seeds measure batch1 state399,360 /327,680 /335,872 bytes, scaling eightfold
at batch8. Hybrid uses21.875% more state than gated MLA and18.9024% more than KDA;
memory gate fails. Median seed-paired hybrid/control latency ratios (prefill,
decode) at batch1/batch8: gated MLA(1.815,1.012)/(1.559,1.025);
KDA(1.026,1.793)/(0.881,1.327). Each seed uses eight measured repeats; warmups
excluded. This is local context128/prefill32/decode64 MPS evidence, not general
production, long-context or population-seed superiority. Validation remains
adaptively reused; holdout closed. Preserve every fresh pair and failure.

Decision: campaign005-final-decision-v2.json. Original audit cache-environment
failure remains immutable; the separately recorded cached read-only review
passed without repeating experimental trajectories. Campaign005 is now closed.

Next architecture direction: tokenizer sparse versus dense FFNs, with the
historical stored-capacity wider-dense control. Read-only feasibility found
historical MoE restricted to MHA and tokenizer contracts restricted to dense.
TOKENIZER_MOE_PREPARATION.md specifies proposed width256 adopted RoPE MLA
backbone, exact FFN arithmetic, bounded equal LR tuning, prospective strict
comparison, diagnostics/tests and mandatory capacity audit. No new source edits
or training; proposal is not frozen and fit is not asserted. Activation/norm/
residual families remain queued; further hybrid tuning and separate EMA/horizon
work are deferred with rationale. No blanket storage cleanup.


### Next FFN direction: capacity audit and activation contract

The proposed width256 three-arm MoE panel does not fit: 18,782,075,912 free
bytes required versus 15,515,467,776 observed, a 3,266,608,136-byte gap. The
projection retains all three selected screens and nine fresh pairs, optimizer/EMA/
export allowances, 256 MiB metadata and the unchanged 8 GiB reserve. Even its
final tensor payload plus reserve exceeds observed free space before headers
and extra overlap. MoE is deferred; no controls or fresh seeds were dropped and
no protected evidence was removed. This is a feasibility result, not a failed run.

Select the prospective same-shape SiLU versus exact-GELU gate fallback on the
adopted width256 RoPE MLA backbone. Its complete two-arm projection requires
15,169,995,168 bytes, with only 345,472,608 bytes of observed margin: recheck
before freeze/launch. Historical character GEGLU did not transfer at width256;
this is a new subword/MLA hypothesis without pooling or a positive forecast.

A separate tokenizer_mla_gate_v1 contract is implemented. 154 targeted tests
passed: exact SiLU incumbent weights/logits/optimizer parity, parameter coverage,
GELU formula/gradients, full/cached causality, same-shape checkpoint rejection
and historical contract checks. Guard finished in 1.32s, peak RSS 558,956,544
bytes, reserve intact. No training, quality scoring or model cleanup. Remaining
checkpoint resume/H200, tokenizer/cache/update tests, finite controller, storage
reconciliation and complete freeze are required before pilots. Details and
next actions: TOKENIZER_MLA_ACTIVATION_PREPARATION.md; immutable direction decision:
tokenizer-evidence/moe-preparation-001/direction-decision.json. Earlier MoE draft
and original source references remain preserved as historical evidence.


### Activation serialized-state prerequisite passed

Four new bounded synthetic CPU tests passed for both SiLU and exact GELU.
Resumed online weights, optimizer, EMA and global/loader RNG matched uninterrupted
four-step execution exactly. Production H200 export/load, tokenizer identity
rejection, reset/interleaved caches and CE-only semantics passed. Real torch
serialization stayed in RAM; no checkpoint file or corpus training/evaluation
was performed. Filesystem publication remains a separate integration gate.

The successful guard finished17.48s with peak RSS3,352,215,552 bytes below4GiB
and the8GiB disk reserve intact. Preserve two harness failures: v1 incorrectly
requested logit-z; v2 used /tmp rather than canonical /private/tmp as a RAM key.
No production source change or closed experimental/numerical retry occurred.
Runtime and shared checkpoint/EMA/optimizer/schedule/loader/cache/writer hashes
match completed005; only three activation contract files differ. This does not
relabel the old hybrid tests as activation evidence or authorize training.

Review: tokenizer-evidence/activation-preparation-001/state-prerequisite-review.json.
Next: finite two-arm rules/seed registry/statistics, retention-aware serial
controller and filesystem-producer integration, then exact capacity/source/input/
runtime freeze before pilots. Existing protected models/pairs/failures remain.


### Activation finite design and seed registry tested

22 pure tests passed for exact two-arm configurations, ordered stage/selection/
retention barriers and conditional-text statistics. An independent loop oracle
confirmed weighted-tail bootstrap arithmetic. Seed audit checked3,024 recorded
plans/requests: pilot3591, screen3601 and fresh3611/3612/3613 were unused.
Draft design:2pilots+6LR screens+6fresh runs; all9 possible selected-LR branches
recorded. Both arms get0.000625/0.00125/0.0025; minimum H200 selects, lower-LR
ties. Single GELU-minus-SiLU contrast uses PCG64 seed20261031,2,000 draws and
linear quantile1-0.05/3 across three block sizes. Strict mean improvement>0.02,
every seed better and all upper bounds negative; no neutral-cost route.

Source applicability confirms only two pure research modules were added since
model/state checks. No quality training, cleanup, holdout scoring or executable
freeze. Production serial controller/filesystem integration, actual audit/cleanup
receipts and complete resource/source/runtime/input binding remain required.
Evidence: tokenizer-evidence/activation-preparation-001/finite-design-v1.json.

### October31 activation session/transport prerequisite

A new activation receipt-ledger core passed49 tests, including the complete
14-trajectory synthetic sequence, independent LR selection, exact retention
boundaries and preservation of both selected screens plus all six fresh pairs.
Real bounded filesystem serialization/publication and torch.load used tiny
synthetic payloads; no corpus or research training occurred. Complete producer
inventories are pinned at acceptance. Tampered/incomplete evidence, STOP,
failures, orphan requests, symlinks and output overflow block progression.
Guard54.21s, peakRSS241,418,240bytes, 180s/4GiB/8GiB-reserve limits intact.
The first44-test run passed but its post-test reporting wrapper failed; both
versions and that diagnostic are preserved. No closed fixture was retried.

This is controller-core/transport evidence only. The ledger has no launch/delete
API; test audit verdicts are synthetic. Executable worker, independent scientific
review, complete audited cleanup and final source/runtime/input/resource freeze
remain necessary. No pilots launched and no new model result or adoption.
[Evidence](tokenizer-evidence/activation-preparation-001/session-prerequisite-review.json).

### October31 activation worker/acceptance prerequisite

76 worker/audit/regression tests and six independent-parent integration tests
passed. Actual activation shapes underwent20 synthetic CPU updates per gate to
check real optimizer/EMA payloads, exact export values and corruption rejection.
Guard100.70s, peakRSS1,149,943,808bytes,180s/4GiB/8GiB-reserve limits intact.
Worker orchestration/resource traces and parent integration use explicit test
doubles; they do not establish MPS pilot readiness or model quality. CPU cache
allocation matches196,608bytes atbatch1 and1,572,864bytes atbatch8 for both gates.

The bounded worker body and independent acceptance adapters are implemented.
Executable freeze/dispatch, guarded parent execution, complete retention cleanup
and inference/resource binding remain required before pilots. No corpus training,
holdout scoring, protected-checkpoint change or closed-fixture retry occurred.
[Evidence](tokenizer-evidence/activation-preparation-001/worker-prerequisite-review.json).


### October31 activation dispatch/retention prerequisite

24 tests passed in34.99s under the180s/4GiB/8GiB-reserve guard; peak RSS
239,960,064bytes. The new entry binds a complete frozen plan and finite design,
source/runtime/input/prerequisite identities and remaining capacity; it connects
bounded serial dispatch, guarded child entry and independent parent acceptance.
The retention executor uses complete stage reviews, exact allowlists/hashes,
intent/deleted journals and preserved inventories; partial failure forbids retry.

Real harmless subprocesses test inherited locks and exit handling. Connected
entry/parent tests use explicit process/numerical doubles. Both actual cleanup
boundaries were exercised only on tiny disposable fixture files, retaining every
selected/fresh pair. The first attempt's two preflight test-cap failures and
source snapshots are preserved; production limits were not reduced.

Inference protocol/runner, final applicability/capacity review and executable
freeze remain pending. No research training, checkpoint cleanup, holdout scoring
or closed-fixture retry occurred. No new model result or adoption.
[Evidence](tokenizer-evidence/activation-preparation-001/dispatch-retention-prerequisite-review.json).


### October31 activation inference prerequisite

65 tests passed, including the new finite timing/order/aggregation contract,
CPU export-value/lineage checks, connected three-seed worker/parent receipt chain,
and dispatch/retention regressions. Guard37.77s, peakRSS241,385,472bytes;
180s/4GiB/8GiB-reserve limits intact. GPU, process, loading and completed-quality
operations are explicit doubles in integration tests; no MPS benchmark or model
quality claim follows. No research training, checkpoint cleanup or holdout scoring.

Inference fixes both fresh H200 arms at all three seeds, first96 training IDs,
batches1/8, correctness before two warmups/eight alternating measurements,
32-token prefill and64-token decode. Report seed-paired latency ratios and actual
cache/allocator observations separately from the unchanged strict quality rule.
Final campaign decision wrapper and exact capacity/source/runtime/input freeze
remain pending before pilots. [Evidence](tokenizer-evidence/activation-preparation-001/inference-prerequisite-review.json).


### October31 activation001 frozen before pilots

Final independent decision logic passed34 tests. Exact implemented meta-device
accounting confirms28,492,416 parameters and788,953,140-byte trajectory allowance
per arm; all nine LR branches require15,169,995,168 free bytes including eight
retained pairs,256MiB metadata and8GiB reserve. Cached tokenizer/input hashes and
source/runtime/prerequisite applicability passed. No model forward/update,
checkpoint access, holdout score or scientific result in this audit.

The finite two-pilot/six-screen/six-fresh plan is frozen. Initial publication and
executable binding succeeded, followed by an external review-writer Path/string
error. Both failure and original procedure are preserved; independent read-only
verification passed on the unchanged plan and empty campaign. First guarded pilot
is next after commit/push. [Freeze review](tokenizer-evidence/activation-preparation-001/campaign001-freeze-review.json).


### October31 activation001 pilots and audited retention completed

Both20-update MPS pilots passed independent tensor/EMA, full-validation, cache
and resource acceptance. SiLU/GELU worker times were46.97/57.55seconds; peak RSS
1,839,251,456/1,837,727,744bytes and driver memory2,501,705,728bytes for both.
Raw/H200 cached parity passed at batches1/8; actual state196,608/1,572,864bytes.
These are feasibility checks, not comparative quality evidence or adoption.

The frozen retention boundary deleted exactly four audited pilot weights,
1,118,318,800 logical bytes, preserving all59 other campaign files. Manifest,
journal, inventory and terminal guard were independently verified; complete
portable text evidence is saved. Deleted pilot weights cannot be re-audited.
Observed free15,573,012,480bytes exceeded remaining requirement15,169,847,947.
No selected/fresh/failure checkpoint was deleted; test holdout remains closed.
Next is the first of six1200-update screens under the unchanged frozen source.
[Review](tokenizer-evidence/activation-preparation-001/campaign001-pilot-boundary-reviewed.json).


### October31 activation001 first quality screen accepted

SiLU at LR0.000625, seed3601, completed1200updates and independent acceptance.
Full-validation rawCE4.863816353, H200CE4.900551548;211.99s training and275.87s
worker duration. PeakRSS1,720,582,144bytes and MPS driver2,501,705,728bytes stayed
within frozen limits. This is one adaptive screening cell, not an activation
comparison or adoption. All checkpoints retained; test holdout remains closed.
The matched exact-GELU screen is now running serially.
[Accepted-prefix review](tokenizer-evidence/activation-preparation-001/campaign001-quality-prefix-01-review.json).


### October31 activation001 first matched screening pair

Both LR0.000625 cells (seed3601,1200updates) passed independent acceptance.
H200 CE: SiLU4.900551548, exact GELU4.898298625; GELU-minus-SiLU=-0.002252923.
Raw CE: SiLU4.863816353, GELU4.862704081. H200 was worse than raw by0.036735195
and0.035594544 respectively. Track the averaging pattern without changing the
frozen selection endpoint or introducing a posthoc EMA study into this panel.
This small single-screen-seed contrast is not adoption or fresh confirmation.
GELU training214.05s/worker278.14s; peakRSS1,721,024,512bytes, driver2,501,705,728bytes.
All guards passed; no cleanup and holdout stays closed. Slot02 GELU LR0.00125 running.
[Review](tokenizer-evidence/activation-preparation-001/campaign001-quality-prefix-02-review.json).


### October31 activation001 middle-rate GELU screen accepted

Exact GELU LR0.00125, seed3601, completed1200updates and independent acceptance.
Raw CE4.727150151; H200 CE4.726793996. Raising LR from0.000625 improved H200
by0.171504628 on this screening seed, much larger than the first-rate gate
contrast0.002252923. H200 now slightly improves raw by0.000356154; the earlier
averaging disadvantage therefore does not persist across these two GELU rates.
No general EMA conclusion, changed endpoint or adoption follows.
Training213.41s/worker297.45s; peakRSS1,718,665,216bytes, driver2,501,705,728bytes.
All guards passed, checkpoints retained, holdout closed. Matched SiLU LR0.00125
is running as slot03. [Review](tokenizer-evidence/activation-preparation-001/campaign001-quality-prefix-03-review.json).


### October31 activation001 middle-rate paired screen accepted

SiLU LR0.00125, seed3601, completed1200updates and independent acceptance.
RawCE4.723106871/H200CE4.722533288; GELU-minus-SiLU H200 delta=+0.004260708
at this rate, reversing the lower-rate delta=-0.002252923. Increasing LR from
0.000625 improves H200 by0.178018260 for SiLU and0.171504628 for GELU. Thus
within this one screening seed, LR effects are much larger than the gate
contrast; neither screening difference establishes an activation winner.
H200 slightly improves SiLU raw by0.000573584 at this rate. No endpoint change.
Training212.66s/worker297.91s, peakRSS1,650,491,392bytes, driver2,501,705,728bytes.
All guards passed, no cleanup, holdout closed. Slot04 SiLU LR0.0025 is running;
the final GELU rate follows before any frozen selection.
[Review](tokenizer-evidence/activation-preparation-001/campaign001-quality-prefix-04-review.json).


### October31 activation001 SiLU three-rate screen completed

SiLU LR0.0025, seed3601, completed1200updates and independent acceptance.
RawCE4.908889975/H200CE4.805960225. Its three H200 endpoints at ascending
LR0.000625/0.00125/0.0025 are4.900551548/4.722533288/4.805960225. The middle
rate is lowest on this screening seed; the highest rate is worse by0.083426937
despite H200 improving its own raw endpoint by0.102929750. No fresh-seed or
general-optimum claim. Persisted selection waits for the final GELU screen.
Training210.84s/worker325.27s, peakRSS1,706,131,456bytes, driver2,501,705,728bytes.
All guards passed; no cleanup, all current pairs retained, holdout closed.
Slot05 GELU LR0.0025 is running; then save selection and audit losing-screen
cleanup before fresh confirmation.
[Review](tokenizer-evidence/activation-preparation-001/campaign001-quality-prefix-05-review.json).


### October31 activation001 screen selection and cleanup complete

All six1200-update screens passed independent acceptance. H200 full-validation CE:

| LR | SiLU | Exact GELU | GELU minus SiLU |
| --- | --- | --- | --- |
|0.000625|4.900551548|4.898298625|-0.002252923|
|0.00125|4.722533288|4.726793996|+0.004260708|
|0.0025|4.805960225|4.807614079|+0.001653854|

The frozen minimum-H200 rule saved LR0.00125 for both arms. This screening seed
favors SiLU by0.004260708 at the selected rate; no fresh-seed/adoption conclusion.
Final GELU rawCE4.906256948/H200CE4.807614079;207.35s training/332.93s worker;
peakRSS1,697,267,712bytes, driver2,501,705,728bytes. All guards passed.

Independent selection review confirmed both saved minima and exact losing slots
00/01/04/05. Frozen cleanup removed their eight weights (2,236,641,184 logical
bytes), preserving all240 other campaign files including the four selected
weights. Independent post-cleanup replay verified selection, journals, terminal
guards, retained hashes and remaining capacity:14,436,069,376free bytes versus
14,050,805,346 required including the8GiB reserve and all future fresh pairs.
Deleted losing weights cannot be re-audited; complete portable text remains.

Next: fresh seeds3611/3612/3613 at the saved rates, alternating arm order. No
source/endpoint/grid change, holdout scoring or protected checkpoint deletion.
[Selection review](tokenizer-evidence/activation-preparation-001/campaign001-screen-selected-review.json),
[retention review](tokenizer-evidence/activation-preparation-001/campaign001-screen-released-review.json).


### October31 activation001 first fresh SiLU accepted

Fresh SiLU seed3611 at saved LR0.00125 completed1200updates and independent
acceptance. RawCE4.707916448/H200CE4.722778111; H200 is worse by0.014861663
in this cell. This unpaired endpoint is not an activation conclusion, and the
frozen H200 endpoint remains unchanged. Training206.79s/worker338.83s; peakRSS
1,697,906,688bytes and driver2,501,705,728bytes; all guards passed.
Matched GELU seed3611 is running as slot07. Every fresh raw/H200 pair remains
protected; no cleanup or holdout scoring.
[Review](tokenizer-evidence/activation-preparation-001/campaign001-quality-prefix-07-review.json).


### October31 activation001 first fresh pair accepted

Both seed3611 recipes at saved LR0.00125 passed independent acceptance.
SiLU rawCE4.707916448/H200CE4.722778111; GELU rawCE4.707687207/H200CE4.722655893.
GELU-minus-SiLU H200 delta=-0.000122218 (raw delta=-0.000229241): nearly identical
on this seed, far smaller than the strict mean0.02 adoption improvement. No
adoption or equivalence conclusion from one pair; all three pairs and frozen
conditional-text/inference/final reviews remain required. H200 worsened both
raw endpoints by about0.0149 on this seed; preserve the frozen endpoint.
GELU training206.89s/worker346.67s, peakRSS1,698,004,992bytes, driver2,501,705,728bytes.
All guards passed, every fresh pair protected, holdout closed. Slot08 GELU
seed3612 now runs first in the predeclared alternating arm order.
[Review](tokenizer-evidence/activation-preparation-001/campaign001-quality-prefix-08-review.json).


### October31 activation001 fresh GELU seed3612 accepted

GELU seed3612 at saved LR0.00125 completed1200updates and independent acceptance.
RawCE4.702213629/H200CE4.698578620; averaging improves this endpoint by0.003635009.
The matched SiLU seed is still pending, so this is not a paired conclusion.
Training205.52s/worker353.74s; peakRSS1,695,776,768bytes, driver2,501,705,728bytes.
All guards passed, no cleanup or holdout scoring. Matched SiLU seed3612 runs as
slot09; every fresh raw/H200 pair and selected screen remains protected.
[Review](tokenizer-evidence/activation-preparation-001/campaign001-quality-prefix-09-review.json).


### October31 activation001 second fresh pair accepted

Both seed3612 recipes at saved LR0.00125 passed independent acceptance. SiLU
rawCE4.699689255/H200CE4.695734123; GELU rawCE4.702213629/H200CE4.698578620.
GELU-minus-SiLU H200 delta=+0.002844497, reversing seed3611's-0.000122218.
The frozen every-fresh-seed-improves condition therefore fails; GELU cannot
qualify under this panel's adoption rule regardless of seed3613. This does not
establish population superiority or equivalence. Complete the final pair and
inference/final review as planned; no premature terminal receipt or changed rule.
SiLU training205.09s/worker378.39s; peakRSS1,695,023,104bytes, driver2,501,705,728bytes.
All guards passed; no cleanup/holdout scoring. Slot10 SiLU seed3613 running;
all fresh raw/H200 pairs and selected models remain protected.
[Review](tokenizer-evidence/activation-preparation-001/campaign001-quality-prefix-10-review.json).


### October31 activation001 final-seed SiLU accepted

SiLU seed3613 at saved LR0.00125 completed1200updates and independent acceptance.
RawCE4.701461123/H200CE4.712309397; averaging worsens this cell by0.010848274.
The matched GELU endpoint is pending. Training205.24s/worker393.60s; peakRSS
1,514,586,112bytes, driver2,501,705,728bytes. All guards passed; no cleanup or
holdout scoring. Final GELU seed3613 runs as slot11. Then the finite inference
panel and independent final review remain required; the every-seed quality
condition already failed on3612. All fresh and selected pairs stay protected.
[Review](tokenizer-evidence/activation-preparation-001/campaign001-quality-prefix-11-review.json).


### October31 activation001 complete fresh quality panel

All12 quality trajectories and both pilots passed independent acceptance.
Both families use their independently selected LR0.00125. Fresh H200 CE:

| Seed | SiLU | Exact GELU | GELU minus SiLU |
| --- | --- | --- | --- |
|3611|4.722778111|4.722655893|-0.000122218|
|3612|4.695734123|4.698578620|+0.002844497|
|3613|4.712309397|4.714239386|+0.001929989|

Mean GELU-minus-SiLU=+0.001550756: GELU is slightly worse on average and loses
two of three seeds. Both strict mean improvement and every-seed improvement
conditions fail. No population-equivalence claim; conditional-text intervals,
finite inference and independent terminal review still must be recorded.
Final GELU rawCE4.703934880/H200CE4.714239386;205.06s training/401.30s worker;
peakRSS1,692,663,808bytes, driver2,501,705,728bytes. All guards passed.

Training is complete; do not dispatch another training step. First inference
pair seed3611 is running in physical storage slot3511, followed serially by
3612/3613 and summary. These writer bucket names do not reuse old scientific
seeds or artifacts. Preserve all16 selected/fresh checkpoint files, all other
evidence and the closed test. No cleanup or model adoption.
[Full accepted-prefix review](tokenizer-evidence/activation-preparation-001/campaign001-quality-prefix-12-review.json).


### October31 activation001 inference failed; no retry

First inference seed3611 (storage slot3511) exited1 after39.24s. Its output
census enumerated control/process-inference-3511.json.tmp; the parent atomically
replaced that file before Path.stat, raising FileNotFoundError inside worker
budget(). The exact traceback, failed worker guard, process receipt, request,
attempt and failure latches are preserved. Worker guard failed at34.43s with
RSS821,805,056bytes; this was a file-census race, not a memory-limit crossing.
No complete timing result or accepted inference receipt was published.

Independent read-only audit confirmed no research worker/process group remains,
all12 quality trajectories and both boundaries still verify, and all16 retained
selected/fresh checkpoint hashes match. Original source/plan stayed unchanged.
GELU's completed quality failure remains supported; inference costs and the
frozen final wrapper are incomplete. Do not report a formal terminal receipt.

Do not retry or patch activation001, rerun training, or regenerate weights.
The next engineering step is a separate versioned inference-only continuation:
reproduce the race using disposable filesystem metadata; test bounded census
synchronization/revalidation without dropping transient bytes or weakening
unknown/symlink/hardlink/cap checks; bind all original accepted training,
selection, retention and checkpoint evidence; preserve frozen source and use
explicit applicability for any external adapter. Freeze a new dedicated root
and exact unchanged timing/quality/resource contract before measurements.
This is planned recovery work, not an approved result or completed fix.
[Failure review](tokenizer-evidence/activation-preparation-001/campaign001-inference-failure-review.json).


### October31 activation inference recovery: metadata prerequisite passed

The original control temporary-file/stat race is reproduced deterministically on
disposable metadata. A separate reader outside frozen source serializes census
under the existing metadata lock, revalidates inventory and enforces a two-second
elapsed allowance without ignoring missing files or temporary bytes. Seventeen
tests passed, including actual atomic publication, two waiting readers and a
bounded parent/child process. File/group caps, alias rejection, old-plus-temporary
write accounting and deadlines remain tested. Guard finished in0.583s with
233,783,296-byte peakRSS; no model or corpus loaded. Original source fingerprint,
campaign inventory and all422 non-checkpoint hashes remain unchanged.

Lock audit correction: the original process supervisor already locks its scans;
the uncovered call is the worker inference-budget census. Do not nest a new
lock around already-serialized supervisor/writer calls. See the
[recovery design](activation_inference_recovery_v1/README.md) and
[guarded review](tokenizer-evidence/activation-preparation-001/inference-recovery-census-prerequisite-001-review.json).
This is a preparation result, not an accepted timing measurement. The external
binder, new-root controller, independent acceptance/final report and exact source
freeze remain prerequisites. Activation001 is never retried; its failed evidence
and protected checkpoints stay intact. SiLU remains the incumbent; no new
scientific quality conclusion or formal terminal receipt is claimed here.


### October31 activation001 imported quality statistics and evidence binding

The external read-only binder passed28 contract tests and actual integration:
all438 original files,14 accepted dispatches,both retention boundaries,and all
16protected checkpoints (4,473,282,368bytes) verified. Original inference failure
is pinned explicitly and grants no timing credit. No original source or campaign
file changed. [Binding review](tokenizer-evidence/activation-preparation-001/inference-recovery-binding-prerequisite-001-review.json).

The imported-quality adapter passed9 tests and ran the unchanged predeclared
statistical engine on the original six accepted fresh endpoints, with complete
source/evidence verification before and after. No weights loaded or new
validation scoring. Mean H200 CE:SiLU4.710273877112203,GELU4.711824633037997;
GELU-minus-SiLU+0.001550755925793319,losing2/3seeds.

| Block length | Gate upper bound | Descriptive95% interval, GELU minus SiLU |
| --- | --- | --- |
|2048|+0.002470789135|[+0.000692766861,+0.002409247687]|
|4096|+0.002569505209|[+0.000385030443,+0.002518861133]|
|8192|+0.002788169461|[+0.000105328239,+0.002716096356]|

All three negative-upper-bound conditions fail, alongside the mean and every-seed
criteria. The descriptive intervals are above zero, supporting a small GELU
disadvantage conditional on these trained seeds and adaptively reused validation
text. They are not seed-population or independent-holdout intervals. Original
bootstrap draws/seed/quantile/block rules are unchanged; no posthoc quality or
cost route. [Analysis](tokenizer-evidence/activation-preparation-001/inference-recovery-quality-prerequisite-001-analysis.json),
[guarded review](tokenizer-evidence/activation-preparation-001/inference-recovery-quality-prerequisite-001-review.json).

The incumbent remains unchanged. This is a quality-only review, explicitly
`quality_reviewed_inference_pending`, not a terminal campaign receipt. No new
inference ran. Next complete the separate new-root worker/controller, independent
acceptance/final assembly and exact external freeze before finite inference;
see [recovery design](activation_inference_recovery_v1/README.md). The failed
activation001 root is never retried; selected/fresh pairs,failures,and closed
holdout remain protected. Both new evidence-producing wrappers executed once.


### October31 separate inference recovery001 frozen

External recovery worker/controller/final implementation passed41 integration
tests (34execution/7final). The suite uses real disposable bounded metadata and
guards, explicit model/GPU/process/source-audit doubles, and connects all three
serial pairs to independent acceptance and a final receipt. Earlier37-test
suite001 and its source snapshot remain preserved; review then found and fixed
the failed-final-to-controller continuation gap. Applicable002 adds a root-level
failure/partial latch and four regressions; no failed phase can advance.

A real-source applicability/freeze audit passed: all original438files and
16protected checkpoints bind to unchanged source/runtime; original loader,
cache checks, numerical timing/statistical engines and bounded supervisor are
reused under exact source pins. Outputs and acceptance live in a separate root.
No original inference entry or frozen producer is patched/retried. Capacity:
12,993,671,168bytes free versus8,858,224,705bytes required remaining; zero new
checkpoint allocation, conservative256MiBmetadata envelope and8GiBreserve.

Plan SHA256 `29b8265276d6bedb933bbb3f522b179ffa1b8403b18ac7f7b62b5b6638ee5dde` at
`research_runs/tokenizer-activation-inference-recovery-001/evidence/meta/recovery-plan.json`.
[Freeze review](tokenizer-evidence/activation-preparation-001/inference-recovery001-freeze-review.json),
[execution002](tokenizer-evidence/activation-preparation-001/inference-recovery-execution-prerequisite-002-review.json),
[final002](tokenizer-evidence/activation-preparation-001/inference-recovery-final-prerequisite-002-review.json).
No inference launched during freeze. Commit/push first, then dispatch only
seed3611; require independent acceptance before3612/3613, saved summary before
separate terminal review. Quality failure and incumbent remain unchanged.
All selected/fresh pairs, failures, archives and closed holdout stay protected.


### October31 recovery001: valid first pair, second entry failed permanently

Seed3611 completed33.995s and passed independent parent acceptance plus portable
read-only review; worker28.687s,peakRSS784,973,824bytes. The synchronized census
survived the actual first inference workload. Seed3612 then failed9.751s after
launch during child entry, before model loading/timing. No3612 result or3613
attempt exists; no worker remains. Original activation001 and recovery001 must
never be retried, and their frozen runtime source must remain unchanged.

The trace ends at `worker.review_outputs`'s whole-root before/after census equality
while replaying prior3611 acceptance. The parent concurrently updates the active
3512 monitor. Individually serialized scans do not guarantee whole-directory
stability across a longer original-source audit. Exact differing entries were
not logged; attribution to a particular mutable monitor field is inferred.
The doubled process tests did not cover actual parent updates during later-child
prior-prefix review. A real concurrent subprocess regression is now required.

[Failure audit](tokenizer-evidence/activation-preparation-001/inference-recovery001-second-entry-failure-review.json)
verified all first-pair artifacts/guards/process/model bindings/timing arithmetic
against the [accepted prefix](tokenizer-evidence/activation-preparation-001/inference-recovery001-prefix-01-review.json),
and rehashed all16original protected checkpoints. All remain intact; no cleanup,
training, weight regeneration or holdout scoring occurred.

Next prepare a separate versioned continuation for remaining3612/3613, importing
valid3611 timing without repeating it. Compare completed-slot immutable evidence
separately from active control state while retaining global synchronized quota,
STOP/failure/path/file-count/alias/transient-byte/resource checks. Test actual
parent/child monitor contention and immutable-file tampering before a new finite
freeze. Quality failure and incumbent are unchanged; only one inference pair is
accepted, so no aggregate cost conclusion or formal terminal closure is available.


### October31 recovery v2 scoped verification and carried-pair prerequisites

The separate v2 inventory helper passed27 tests, including10 sequential real
bounded pipe children. Actual parent monitor updates reproduce rejection by
whole-root before/after equality while the prior completed slot remains byte
identical. Scoped file/control hashes accept that valid overlap and reject
same-size prior-result changes, prior-control changes, extra prior files and
unknown controls. Global quotas, path/alias/file-count, STOP/failure, transient
and resource checks remain enforced. Historical exact differing monitor fields
were not logged; this test demonstrates the mechanism, not that missing detail.

The first-pair importer passed13 provenance tests and an actual read-only audit
of accepted seed3611. All27 files of the failed recovery001 root are pinned,
including the second-entry failure; original training and all16 protected
checkpoint hashes were reverified. No inference was repeated. Both failed roots
and frozen source sets are unchanged; no matching workers remain.

[Scoped inventory review](tokenizer-evidence/activation-preparation-001/inference-recovery-v2-scoped-inventory-prerequisite-001-review.json),
[carried-pair review](tokenizer-evidence/activation-preparation-001/inference-recovery-v2-carried-pair-prerequisite-001-review.json),
[v2 integration gates](activation_inference_recovery_v2/README.md).
These are component prerequisites, not a completed controller or launch freeze.
Next integrate carried3611 plus only new3612/3613, test actual later-child entry
concurrency and final assembly, then freeze/commit/push before dispatch. GELU
still fails the quality gates; no complete timing aggregate or final receipt yet.


### October31 inference recovery002 frozen for two remaining pairs

V2 execution40 and final7 tests passed. Four real bounded child subprocesses
exercise production later-child entry, prior-prefix replay and output review
while the actual parent monitor changes. Source/model/GPU portions are explicit
doubles in these tests. Valid monitor changes pass; prior-result/guard tampering
and unknown controls fail. Serial tests preserve carried3611 credit, dispatch
only3612/3613, enforce failure/partial/final latches and assemble all three pairs.

The separate real-source freeze audit reverified original training, all16
protected checkpoints, all27 closed recovery001 files and the accepted3611
result without repeating it. Both failed roots and frozen source sets are intact.
New root: `research_runs/tokenizer-activation-inference-recovery-002`.
Plan SHA256 `f28fe8194baf1fe4b05a1c480b4868e41a1b7b66fe7410e163e52f883edbb1b3`.
Free12,988,067,840bytes versus8,858,181,766required remaining; no new weights.

[Freeze review](tokenizer-evidence/activation-preparation-001/inference-recovery002-freeze-review.json),
[execution tests](tokenizer-evidence/activation-preparation-001/inference-recovery-v2-execution-prerequisite-001-review.json),
[final tests](tokenizer-evidence/activation-preparation-001/inference-recovery-v2-final-prerequisite-001-review.json).
Commit/push this freeze before dispatch3612, then independently accept before3613;
summary precedes a separate guarded final review. No inference launched during
freeze. No training, checkpoint regeneration, cleanup or holdout scoring.


### October31 activation comparison closed: retain SiLU

Recovery002 completed the two remaining inference pairs,3612/3613, and carried
accepted3611 unchanged from recovery001. Both new pairs passed parent acceptance
and separate read-only prefix reviews. The guarded final decision finished37.038s;
independent closure reconstruction finished55.992s, peakRSS231,161,856bytes.
No worker remains. Original activation001 and recovery001 failures stay preserved
and closed; no training or accepted timing pair was repeated.

Both arms selected LR0.00125. Fresh H200 means are SiLU4.7102738771 and exact
GELU4.7118246330: GELU minus SiLU **+0.0015507559 nat/token**, worse on2/3seeds.
All three conditional gate upper bounds are positive (+0.002471/+0.002570/
+0.002788 at2048/4096/8192 blocks). Mean, every-seed and conditional-bound gates
all fail. **GELU is not adopted; the existing SiLU MLA recipe remains unchanged.**
These are reused-validation, fixed-trained-seed findings, not population-seed,
independent-test or larger-model transfer claims.

| Batch | GELU/SiLU prefill latency | GELU/SiLU decode latency | Cache bytes, each arm |
| --- | --- | --- | --- |
|1|1.002140|1.001036|196,608|
|8|1.002410|1.000247|1,572,864|

Ratios are medians of three seed-paired ratios of eight-repeat median latency,
excluding warmups. Local MPS workload: context128, prefill32,64 teacher-forced
decode calls. Measurements are near parity; these tiny differences do not
establish a general speed advantage or statistical equivalence. There is no
predeclared cost-only adoption route. Cache sizes are identical. New worker
peakRSS814,202,880/823,885,824bytes stayed below4GiB; all guards finished cleanly.

[Independent closure audit](tokenizer-evidence/activation-preparation-001/inference-recovery002-closed-review.json),
[terminal receipt](tokenizer-evidence/activation-preparation-001/inference-recovery002-closed-portable/evidence/meta/recovery-final-complete.json),
[final decision](tokenizer-evidence/activation-preparation-001/inference-recovery002-closed-portable/evidence/meta/recovery-final-decision.json),
[timing summary](tokenizer-evidence/activation-preparation-001/inference-recovery002-closed-portable/evidence/meta/recovery-inference-summary.json).
All16 selected/fresh checkpoint files, both failure roots and archives remain
protected; no cleanup or holdout scoring. The complete experiment is closed.
Next review normalization/residual architecture and full storage feasibility
before proposing another finite panel. MoE remains deferred on its recorded
capacity requirement; do not trim controls/seeds or delete protected pairs to fit.


### October31 next normalization design: matched-epsilon block LayerNorm

Selected prospective two-arm block RMSNorm versus affine LayerNorm on the adopted
D256 RoPE MLA/SiLU backbone; only eight pre-branch normalizers change. The local
constructor audit found an epsilon confound: default RMSNorm uses FP32 epsilon
2^-23, default LayerNorm1e-5. Proposed LayerNorm explicitly uses2^-23 while the
control preserves its exact incumbent path. Centering and a learned bias remain
joint changes; no isolated-centering causal claim. Candidate adds2,048 parameters.

The guarded meta-shape/optimizer audit found15,170,221,264bytes required against
12,984,197,120free, a2,186,024,144byte gap. Even final tensor payload plus reserve
exceeds free space by77,657,600bytes before headers/logs/overlap. Reducing safety
allowances alone cannot make this full retained panel fit. No pilot or training
started; no protected evidence changed. Bounded CPU prerequisites can proceed.
See [prospective design](TOKENIZER_BLOCK_NORM_PREPARATION.md) and
[audit](tokenizer-evidence/block-norm-preparation-001/shape-storage-review.json).


### Standalone FP32 normalizer prerequisite completed October31

The separate [specification](block_norm_v1/README.md) passed66 bounded CPU tests:
independent FP64 formulas and analytic input/gain/bias derivatives; six input
regimes; bit-exact native RMS control forward/backward; eight independent
normalizers with RNG-preserving constant initialization; precise parameter counts;
signature round trips and rejection of wrong epsilon/dtype/shape/affine/bias,
autocast, mixed/shared/aliased modules and unsupported mutation.

Near-constant shifted LayerNorm input has greater rounding sensitivity: observed
maximum output/input-gradient/gain-gradient discrepancies versus FP64 were
0.0002103/0.01536/0.0009631. Ordinary random/shifted output discrepancies were
below6.26e-7. All passed tolerances declared before execution; no tolerance was
relaxed. This does not establish full-model stability, MPS agreement or quality.

Attempt001's66 tests passed, but its enclosing final metadata audit failed on a
string-versus-Path error. That failure, log and exact source snapshot are retained.
Attempt002 fixed only that wrapper conversion; its complete audit passed in0.388s
at285,900,800bytes peakRSS. Original source, all13 frozen recovery modules, three
closed-root inventories and noncheckpoint hashes were unchanged. No models,
corpus/checkpoint data, training, MPS, cleanup or holdout were accessed.

[Applicable review002](tokenizer-evidence/block-norm-preparation-001/standalone-spec-prerequisite-002-review.json),
[preserved failure001](tokenizer-evidence/block-norm-preparation-001/standalone-spec-prerequisite-001-review.json).
Next resolve full model/optimizer/checkpoint/cache and source applicability as a
separate versioned integration contract. Do not patch frozen old code or bypass
its RMS-only validators. Full-model initialization, cache, resume/EMA, serialization
and MPS remain prerequisites. The recorded training-capacity blocker remains.


### Separate runtime baseline completed October31

The [integration contract](block_norm_v1/INTEGRATION_CONTRACT.md) maps the model,
optimizer, checkpoint/H200 reconstruction and cache dependencies into a separate
`research.block_norm_v1.runtime` namespace. Its43-module unmodified copy passed
five guarded CPU tests: exact original-control initialization/RNG/state, logits
and loss, old-new cached outputs/state, optimizer group equivalence and preserved
RMS-only rejection. This is baseline transport evidence, not a candidate quality,
full-versus-cached numerical, gradient/update, serialization or MPS result.

Attempt001 stopped before tests on a sandbox safety-entry PermissionError. The
unchanged sources passed attempt002 with authorized process access in0.851s,
peakRSS543,424,512bytes. Original source,13 frozen recovery modules and three
closed-root inventories/nonweight hashes remained unchanged. No training or
checkpoint/corpus load, cleanup or holdout scoring occurred.

[Review002](tokenizer-evidence/block-norm-preparation-001/runtime-baseline-prerequisite-002-review.json)
and [copy manifest](tokenizer-evidence/block-norm-preparation-001/runtime-baseline-copy-manifest.json).
Next implement the explicit full-model normalization policy only in the new
namespace, preserving old-policy behavior and binding candidate configuration,
actual modules, optimizer, strict checkpoint/EMA and cache contracts. Frozen
source sets remain untouched. The finite plan and capacity gate remain pending.


## Named full-model policy construction completed October31

The isolated runtime now implements `tokenizer_mla_block_norm_v1` with explicit
RMS-incumbent and matched-epsilon LayerNorm signatures. It validates the exact
config field set and eight actual modules, placement, FP32 parameters, final
affine LayerNorm, untied tables, serial unit residuals and SiLU/MLA backbone.
Non-FP32 construction/execution and autocast are rejected. Only three copied
model files changed plus one new policy module; optimizer, checkpoint, EMA,
training and cache implementations remain unchanged. Frozen original source and
all13 recovery runtime files were preserved.

Forty checks passed in attempt001. Attempt002 added the direct new RMS-policy
versus original-source comparison, passing41 tests without changing runtime code.
The named RMS control is bit-exact for initial state/RNG, synthetic full logits/CE
and original-versus-new cached outputs/state. Candidate shared initial weights
and RNG match the control; only eight zero biases are added (2,048 parameters).
Optimizer checks confirm unique complete parameter coverage, unchanged shared
groups and all new bias vectors in AdamW at base LR with independent decay.
Invalid config/signatures, wrong epsilon/placement/dtype, mixed/shared normalizers,
changed residual/gate/final norm and mutable model signatures were rejected.

Guard002:1.280s, peakRSS658,538,496bytes; four CPU threads,120s cap,4GiB RSS cap,
8GiB reserve. Both attempts and exact source snapshots are retained. Three closed
activation-root inventories and nonweight hashes were unchanged. No model
checkpoint/corpus load, training, cleanup, MPS or holdout scoring occurred.

This is construction/contract evidence. No gradients, optimizer updates, candidate
full-versus-cached numerical equivalence, real checkpoint serialization, resume
or H200 transport has been demonstrated. Current checkpoint save/export code
does not yet persist a dedicated constructed normalization signature: implement
and test that transport before any candidate artifact is accepted. In-memory
architecture-config comparison alone is insufficient. Capacity and finite-plan
freeze remain pending.

[Review002](tokenizer-evidence/block-norm-preparation-001/model-policy-prerequisite-002-review.json),
[reviewed source delta](tokenizer-evidence/block-norm-preparation-001/model-policy-delta-001.json).


## Raw and H200 normalization-signature transport completed October31

The copied checkpoint writer/loader and H200 exporter now persist and validate a
dedicated constructed normalization signature, including exact typed version,
backbone/normalizer semantics, placements and parameter names. Saved config must
agree with that signature and the actual receiving model; all saved tensors must
have the complete expected inventory, shapes, dtypes and finite values. Old
policies omit the new signature and reject artifacts carrying it.

Twenty-six guarded CPU tests passed for both arms: actual raw/H200 serialization
and strict loads, H200 optimizer-resume rejection, malformed/missing/cross-arm/
epsilon/signature/dtype/nonfinite rejection before mutation, and final EMA-state
restoration. A synthetic two-snapshot H200 oracle explicitly changes all norm
gains/biases, confirming their inclusion without folding. Maximum gain error
versus independent FP64 arithmetic was4.8312e-8 and LayerNorm bias error8.223e-13.
All unchanged parameters stayed bit-exact; export preserved the raw parent and
CPU RNG. No actual optimizer update or training trajectory was run.

Guard:3.412s, peakRSS1,323,057,152bytes, four threads,120s cap,4GiB RSS cap and
8GiB reserve. Four fresh synthetic artifacts total684,354,160bytes and remain
retained with SHA256/size manifest; no binary checkpoints enter Git. No trained
checkpoint or corpus was accessed, no cleanup/MPS/holdout scoring occurred, and
frozen source plus closed campaign inventories/nonweight hashes were unchanged.

[Review](tokenizer-evidence/block-norm-preparation-001/checkpoint-transport-prerequisite-001-review.json),
[retained fixture manifest](tokenizer-evidence/block-norm-preparation-001/checkpoint-transport-fixture-manifest.json).
These tests restore an empty optimizer's groups and final EMA accumulator only.
They do not establish populated optimizer-state or interrupted trajectory/loader
RNG replay. That is next, alongside strict raw-resume state requirements; candidate
cache/gradient numerical and MPS checks remain. No training readiness or quality
gain follows. Guard-observed free space was12,084,506,624bytes after fixtures;
against the recorded15,170,221,264 full-panel requirement, the capacity gap is now
3,085,714,640bytes. This observation is not a new audited launch approval.


## Populated optimizer and interrupted CPU replay completed October31

The new policy now rejects incomplete raw-resume metadata before loading model
weights. Preflight checks model/device/PyTorch/progress/RNG contracts, exact
optimizer group order/names/rates and scheduled step, complete finite FP32
Muon/AdamW slots, Adam counters and EMA state. This policy-specific helper lives
in the copied runtime; old-policy behavior is unchanged. H200 export restores
raw weights/EMA without pretending to resume the optimizer; existing raw-parent
and layout checks remain mandatory.

Thirty CPU tests passed using fresh synthetic batch1/context2 data: three updates
per arm, saving after update1 and replaying updates2/3. Final model weights,
populated optimizer slots, EMA state, batches, losses, global RNG and loader RNG
were bit-exact between uninterrupted and restored execution. Tests also rejected
missing optimizer/training/RNG state, changed model contracts, group order/rates,
missing/wrong-dtype slots, nonfinite state, wrong Adam counts and invalid EMA.
No original corpus or trained research checkpoint was used. This is ten synthetic
optimizer updates in an implementation fixture, not a quality trajectory.

A separate three-test read-only regression rederived both retained synthetic H200
exports bit-exactly, rejected their deliberately incomplete full-resume state and
preserved legacy low-level resume semantics. Original/recovery source and three
closed campaign inventories/nonweight hashes stayed unchanged. All evidence and
synthetic artifacts remain retained; no cleanup, MPS or holdout scoring occurred.

Replay guard:12.266s, peakRSS2,186,706,944bytes. Export regression:1.900s,
peakRSS958,185,472bytes. Both used four threads,120s cap,4GiB RSS cap and8GiB
reserve. Two new raw files plus synthetic ID files total890,211,626bytes under
the declared1,200MiB quota; their hashes are recorded without pushing binaries.

[Replay review](tokenizer-evidence/block-norm-preparation-001/resume-replay-prerequisite-001-review.json),
[export regression](tokenizer-evidence/block-norm-preparation-001/resume-export-regression-001-review.json),
[fixture manifest](tokenizer-evidence/block-norm-preparation-001/resume-replay-fixture-manifest.json).
Next complete candidate full/cached causality/reset/interleaving and numerical
gradient checks, then strict MPS and tokenizer-bound trainer/controller integration.
This small CPU replay does not establish those or full-budget training stability.
Latest guard free space11,481,591,808bytes leaves a3,688,629,456byte gap against
the recorded15,170,221,264 full-panel requirement. Training remains gated; do not
poll unchanged capacity, narrow the panel or delete protected evidence to fit.


## CPU cache and normalization-gradient integration completed October31

33 guarded CPU tests passed for both block-normalization arms. Full versus cached
execution covers batch1/2/8 and context17, plus batch1/context128, mixed chunk
sizes and nonzero continuation offsets. Largest absolute logit discrepancy was
8.3447e-7. Two interleaved requests remained independent (maximum7.7487e-7 versus
full execution); resetting one cache left the other unchanged. Reset/fresh cached
outputs and same-length causality checks were bit-exact. Cache storage remains
196,608bytes per batch item (1,572,864 at batch8), identical for both arms.

The explicit reference independently computes RMS/Layer normalization in FP64
and casts at FP32 branch boundaries, sharing attention/FFN/final readout with
the tested model. Maximum logit discrepancy was1.0729e-6, CE discrepancy0, all
parameter-gradient discrepancy1.8627e-8 and norm-input-gradient discrepancy
1.8627e-9. This checks normalization math/placement through the model, not an
independent oracle for the shared attention/FFN implementation. The new RMS
control's complete parameter gradients were bit-exact to original source. All
tolerances were declared before attempt001 and stayed unchanged.

The inherited plain MLA cache lacked normalization-policy and position checks.
The isolated new policy now binds cache signature/layer/geometry/dtype/device,
requires distinct storage and contiguous positions, and resets its position.
Validation covers all four caches before any layer writes. Wrong arm, batch,
dtype, layer order, shared/aliased caches, stale/nonzero-empty positions and
overflow were rejected without partial cache mutation. Old-policy cache behavior
is preserved. This narrow cache-file extension is recorded in the source delta.

Guard:1.418s, peakRSS1,201,012,736bytes, four CPU threads,120s cap,4GiB RSS cap
and8GiB reserve. No checkpoints/corpus were loaded, no binary artifacts or
optimizer updates were produced, and no cleanup/MPS/holdout scoring occurred.
Original/recovery source and closed campaign inventories/nonweight hashes stayed
unchanged. [Review](tokenizer-evidence/block-norm-preparation-001/cache-numerical-prerequisite-001-review.json),
[predeclared source delta and tolerances](tokenizer-evidence/block-norm-preparation-001/cache-numerical-delta-001.json).

Next strict MPS numerical/resource validation with no fallback, then actual
tokenizer-bound trainer/controller integration and finite-plan freeze. These
CPU results do not establish MPS agreement, long-context behavior, training
stability or quality improvement. The full-panel capacity gate remains unmet.


## Strict MPS numerical prerequisite completed October31

11 tests passed for both block-normalization arms with MPS fallback and fast math
explicitly disabled. Model parameters and tested gradients were MPS FP32. Cases
cover batch1/8/context17 and batch1/context128 full/cached execution, nonzero
continuation offsets, batch2/context9 interleaving/causality/reset, CPU-cache
device-mismatch rejection and batch1/context5 gradient comparison to the CPU
normalization reference. Original MPS RMS control logits/CE were bit-exact; its
gradients passed the tighter predeclared control tolerance.

Maximum absolute errors: CPU–MPS logits8.9407e-7; MPS full/cached1.0729e-6;
interleaved requests9.5368e-7; reset0; CPU-reference logits9.5368e-7; CE0; all
parameter gradients1.9325e-7. Tolerances were fixed before attempt001 and not
relaxed. The CPU reference independently changes normalization arithmetic only,
sharing attention/FFN/readout; this remains a scoped numerical check.

Guard:3.340s, peakRSS1,154,957,312bytes, peak sampled MPS driver1,159,659,520bytes.
Limits remained RSS4GiB/driver6GiB, reserve8GiB,600s and four CPU threads. Driver
observations were sampled every0.25s, not an instantaneous peak guarantee. Runtime
code was unchanged from the CPU cache prerequisite. No checkpoints/corpus were
loaded, no binary artifacts, optimizer updates, cleanup or holdout scoring were
performed, and frozen source/closed campaign inventories remained unchanged.

[Review](tokenizer-evidence/block-norm-preparation-001/mps-numerical-prerequisite-001-review.json),
[prospective tolerances and resource plan](tokenizer-evidence/block-norm-preparation-001/mps-numerical-plan-001.json).
This is not a throughput benchmark, MPS optimizer/resume run, resource pilot or
quality result. Next validate actual tokenizer/data binding and trainer entry on
a bounded synthetic fixture with its own correctly derived binding, then strict
MPS optimizer/resume and controller/retention/statistical preparation. Do not use
these small numerical checks to bypass capacity or freeze an incomplete panel.


## Generated-text tokenizer and trainer integration completed October31

The actual copied `run_training` entry and GPT-2 raw-text loader passed12 CPU
checks on generated Unicode/special-marker text, with a correctly computed
`tokenizer_no_bias_prior`. Both normalization arms completed two updates at
batch1/context128, using Muon/AdamW, cooldown and H200. Each generated file has
6,408 train tokens and1,599 validation tokens under80/20 raw-text splits; no test
split exists in the fixture, and the real corpus/closed holdout were not read.

The raw training contract, complete resume preflight, tokenizer-bound raw/H200
loading, exact H200 equality to saved EMA tensors, metrics/summary and parent
hashes passed. Self-consistent but wrong corpus, train-ID and tokenizer priors,
saved tokenizer mismatches and changed resume evaluation seed were rejected.
The campaign's paired512 batch8/context128 sampler separately matched independently
sampled starts; the trainer itself used independent batch1. Runtime source was
unchanged. This is integration evidence, not model-quality or production resource
readiness, full-trainer interrupted replay, MPS optimizer/resume or a controller.

Attempt001 failed during setup before optimizer updates: its two-step fixture
set cooldown start1, violating start<total-1. Attempt002 corrected start to0
without changing runtime validators. Both plans, source snapshots, logs, guards
and all generated files are preserved. Attempt002 guard12.437s, peakRSS
1,419,116,544bytes, four threads,120s cap and8GiB reserve. Successful outputs total
1,118,435,545bytes (four checkpoints plus metadata/text); the failed attempt retains
40,902bytes of generated text. Both attempts left frozen source and closed roots
unchanged. No cleanup, historical checkpoint loading or quality training occurred.

[Passing review](tokenizer-evidence/block-norm-preparation-001/trainer-binding-prerequisite-002-review.json),
[corrected prospective plan](tokenizer-evidence/block-norm-preparation-001/trainer-binding-plan-002.json),
[retained fixture manifest](tokenizer-evidence/block-norm-preparation-001/trainer-binding-fixture-manifest.json).
Next predeclare strict MPS optimizer/resume checks with storage accounted before
writes; prefer bounded in-memory replay and existing evidence where applicable.
Controller/retention/selection/statistical readiness and exact finite seed/order
freeze remain pending. Full-panel capacity remains gated; no partial quality
pilot or thinning controls/seeds/protected evidence to fit.


## Strict MPS optimizer/resume prerequisite failed October31

Two immutable in-memory transport attempts used generated GPT-2 text, batch1/
context128, strict MPS FP32, fallback=0 and fast_math=0. No new binary files were
written. Attempt001 completed three uninterrupted RMS updates but rejected resume
because the saved contract says `mps` while actual parameters report `mps:0`.
The isolated new-policy preflight now accepts these device0 aliases, preserving
other-device and PyTorch-version rejection. It also checks MPS RNG vector length
against the current backend before loading any model weights. CPU/legacy branches
and all mathematical operations remain unchanged; explicit plan deltas bind this.

Attempt002 passed all five malformed-MPS-RNG and three device/version rejection
checks without changing model weights or RNG, accepted both device aliases, and
completed the two resumed updates. Its RMS model comparison FAILED the original
atol2e-6/rtol2e-4: one element out of the first failing172,032-element tensor had
absolute error3.050663508e-6 (relative0.002040846). This is a failed numerical gate,
not an accepted approximate replay. No tolerance was widened. Fail-fast stopped
before the LayerNorm arm. Full optimizer/EMA comparisons and the final exact RNG/
batch/draw assertions were not reached, so their agreement is not established.

Both guards completed with frozen original/recovery sources, closed roots and
retained generated fixtures unchanged. Attempt001:3.229s, RSS2,064,138,240bytes,
peak sampled driver1,190,920,192bytes. Attempt002:4.325s, RSS2,183,725,056bytes,
peak sampled driver1,242,398,720bytes. Limits stayed4GiB RSS/6GiB driver/8GiB
reserve/600s/4CPU. Across attempts eight synthetic RMS optimizer updates ran;
no candidate updates, real-corpus quality training, holdout scoring or cleanup.

[Failed numerical review](tokenizer-evidence/block-norm-preparation-001/mps-resume-prerequisite-002-review.json),
[device/RNG delta and unchanged tolerances](tokenizer-evidence/block-norm-preparation-001/mps-resume-plan-002.json).
The generic wrapper mentions archive hashes, but these failed tests stopped before
emitting their final observation properties: no in-memory archive digest or binary
payload was retained. Exact source, plans, failure logs and guard evidence remain.
Future diagnostics must emit intermediate observations before any assertion.

Close both attempts; do not overwrite or rerun them, loosen tolerance or infer a
root cause. Next prepare a separately bounded diagnostic that first verifies the
loaded model/optimizer/EMA and RNG states, then records exact batch/CPU/MPS draws
and per-step losses/gradients/updates with parameter names before final comparison.
Separate a transport/state defect from backend numerical variation using a
prospectively specified control, with no quality training. MPS resume and all
later controller/finite-plan gates remain incomplete; capacity remains gated.


## RMS MPS diagnostic: exact transport, variation without resume

The separately planned trace001 completed eight synthetic updates: a three-step
RMS reference, a separately initialized three-step uninterrupted repeat and a
resume for updates2/3. It did not invoke either closed failed prerequisite. All
original tolerances remain unchanged, and diagnostic completion is not acceptance.

The fresh repeat started with identical state. The checkpoint round trip restored
all recorded model, optimizer and EMA tensor values/layouts plus group, schedule,
EMA-counter and CPU/MPS/loader RNG metadata exactly. Serialization did not change
RNG. Every compared batch, CPU/MPS random draw, pre/post-draw RNG state and scalar
loss matched exactly. First observed numerical differences were confined to the
embedding gradient: fresh-repeat step1 max4.6566e-10; resumed step2 max2.3283e-10.

The fresh uninterrupted repeat subsequently diverged more than the resumed run.
After step3, fresh-repeat max parameter difference4.902482033e-5 with5,472elements
outside the unchanged2e-6+2e-4*abs(reference) threshold; resumed max3.956258297e-6
with3elements outside. Named intermediate gradients, model/optimizer/EMA hashes,
layouts and all comparisons are retained in the event trace. The observed pattern
supports backend numerical variation independent of checkpoint transport. It does
not prove the precise kernel cause, justify relaxed tolerance or establish MPS
resume readiness. Evidence is one RMS seed, one fresh-repeat control, batch1/
context128; no LayerNorm or full trainer interruption was tested.

Read-only inspection of installed torch.optim._muon confirms that its Newton-
Schulz orthogonalizer casts updates to bfloat16. FP32 parameters, gradients and
saved optimizer slots therefore do not imply FP32 optimizer intermediates. BF16
quantization/nonlinear orthogonalization is a plausible amplification mechanism,
not a causal result from this trace. Installed module/version/function hashes and
line locations are recorded; no optimizer math or backend configuration changed.

Guard7.896s, peakRSS2,297,741,312bytes, peak sampled MPS driver1,191,608,320bytes;
strict fallback0/fast_math0,4CPU,4GiB RSS/6GiB driver/8GiB reserve/600s. No new binary
files or cleanup. The445,082,087-byte in-memory archive's SHA was emitted before
comparisons and verified unchanged; payload bytes were not persisted. Original
source,13 recovery modules, closed roots and generated fixtures stayed unchanged.

[Trace analysis](tokenizer-evidence/block-norm-preparation-001/mps-resume-trace-001-analysis.json),
[full events](tokenizer-evidence/block-norm-preparation-001/mps-resume-trace-001-events.jsonl),
[installed Muon precision](tokenizer-evidence/block-norm-preparation-001/installed-muon-precision-review.json).
Next prospectively isolate repeated embedding backward under default and explicit
strict deterministic controls, then identical-gradient Muon repeat/perturbation
sensitivity as warranted. Use separate bounded diagnostics and disclose backend
support/errors; do not silently change precision, fallback or accept a new threshold.
MPS resume, controller and finite quality-plan readiness remain gated.


## Isolated MPS embedding and Muon diagnostics completed October31

A fresh embedding-only diagnostic used the real generated-text first batch,
FP32 upstream gradients, eight identical backward passes per mode and a unique-
ID control. Repeated token IDs produced8distinct gradient hashes in8passes both
with deterministic algorithms off and with strict deterministic mode on
(warn_only=False). No warning/error or fallback occurred. Maximum repeat delta
was1.862645149e-9; maximum error versus sequential FP64 analytic sums1.513399184e-9.
Unique-ID controls were bit-exact to the oracle and across passes in both modes.
This isolates observed variation to repeated-index embedding backward in this
local case; the strict flag alone does not solve it. It does not identify the
specific kernel implementation or establish universal backend behavior.

A separately frozen Muon sensitivity probe used fixed synthetic FP32 weights,
gradients and populated momentum at shapes256x64 and672x256, with four repeats
per input and deterministic mode off/on. Identical inputs gave bit-exact updated
parameters in every case. Seeded gradient perturbations near1.86e-9 produced
max parameter differences9.765848517e-6 and1.265760511e-5, respectively; results
were the same across deterministic modes. BF16 orthogonalizer inputs reconstructed
from the installed formula changed51/416elements. These reconstructed intermediates
were not instrumented inside the optimizer, and the in-place/out-of-place momentum
paths were not separately compared. Actual parameter outputs were measured.

Together with the earlier exact-restoration trace, this supports small embedding-
backward variation followed by optimizer sensitivity as a plausible explanation.
The synthetic perturbations are not the full-model trace's actual gradient pairs;
there is no complete causal reconstruction or accepted numerical resume result.
No tolerance changed, failed attempt reopened, runtime/math edited, precision
changed or persistent deterministic setting promoted. Global settings were restored.

Embedding guard2.776s/RSS599,015,424bytes/peak sampled MPS driver1,082,621,952bytes;
Muon guard1.972s/RSS454,410,240bytes/driver8,880,128bytes.32embedding backwards and
32synthetic single-matrix optimizer updates, no GPT/model training. Strict MPS
fallback0/fast_math0 and all4CPU/4GiB/6GiB/8GiB/600s guards remained. No new binary
files, cleanup, quality/holdout scoring or changes to frozen source/closed roots.

[Analysis](tokenizer-evidence/block-norm-preparation-001/mps-kernel-diagnostics-analysis.json),
[embedding events](tokenizer-evidence/block-norm-preparation-001/mps-embedding-diagnostic-001-events.jsonl),
[Muon events](tokenizer-evidence/block-norm-preparation-001/mps-muon-diagnostic-001-events.jsonl).
Next design and prospectively test a separate deterministic MPS embedding-gradient
prototype before any runtime adoption. One candidate is unique-token grouping plus
a FP32 membership-matrix product and unique-index output copy, entirely on MPS;
first verify backend support, analytic gradients, repeated-call determinism and
bounded cost/memory without changing production. Preserve native forward behavior,
no CPU fallback, no new quality claim and no tolerance widening. Any eventual
runtime change needs explicit source delta, both-arm equivalence/replay/resource
checks and a fresh finite-plan binding. Quality/controller readiness stays gated.


## Standalone grouped MPS embedding-gradient prototype passed

The prototype preserves native functional.embedding forward and computes backward
entirely on MPS: sorted unique IDs, an FP32 membership matrix times upstream rows,
and a copy to distinct rows of the dense gradient. It introduces no CPU fallback,
Muon precision change, padding, sparse/frequency-scaled semantics or runtime edits.
Its scoped contract is first-order FP32, vocabulary50,257/width256, rank2int64 IDs,
and1..1,024tokens. Higher derivatives and other embedding options are unsupported.

11 tests passed: one numerical/repeat test spanning11cases and ten unsupported-
contract rejections.88backwards covered generated batch1 and paired batch8/context128,
unique IDs, one repeated ID, alternating IDs, zero/constant upstream, noncontiguous
IDs and noncontiguous upstream. Native forward outputs were bit-exact; all gradient
hashes matched within and across deterministic-off/strict-on modes. Unused rows
were exact zero. Maximum absolute gradient error versus independent sequential
FP64 sums was1.002636907e-8, below prospective atol1e-7/rtol2e-5. Unique/zero/constant
cases matched the oracle exactly. These are standalone bounds, not a changed
full-model resume tolerance or a general determinism guarantee.

Guard5.613s, peakRSS624,197,632bytes and peak sampled MPS driver1,082,884,096bytes;
fallback0/fast_math0,4CPU and all4GiB/6GiB/8GiB/600s limits preserved. Native forward
and backward observations took roughly3–7ms after initial cases, but first-use
outliers reached121ms and timings include checks/synchronization; this is not a
native-versus-prototype benchmark or campaign resource pilot. No optimizer steps,
model/quality training, new binary artifacts, cleanup or holdout use occurred.
Global deterministic settings were restored; frozen sources/closed roots and
prior generated fixtures stayed unchanged. All source/evidence snapshots retained.

[Analysis](tokenizer-evidence/block-norm-preparation-001/grouped-embedding-prototype-analysis.json),
[prospective contract and tolerances](tokenizer-evidence/block-norm-preparation-001/grouped-embedding-prototype-plan-001.json),
[review](tokenizer-evidence/block-norm-preparation-001/grouped-embedding-prototype-001-review.json).
The prototype remains outside runtime imports. Next design explicit versioned
embedding-backward config/model/checkpoint/source bindings before integration;
never silently reinterpret prior checkpoints or old model-policy signatures.
Bind all-MPS reduction semantics, CPU inference/reference behavior, token limits
and first-order scope. Both prospective normalization arms must share that policy,
with fresh controls and declared FP32 reduction-order change. Verify initialization/
forward/cache equivalence, both-arm gradients, serialized transport and full-model
MPS repeat/resume under the unchanged original gate before resource/controller
readiness or finite quality-plan freeze. Earlier failed attempts remain closed.


## October 31 versioned grouped embedding integration: 33 checks passed

The isolated runtime now has an explicit `tokenizer_mla_block_norm_grouped_embedding_v2`
architecture policy and exact `embedding_backward` specification. Its MPS backward
core is AST-identical to the tested standalone prototype: unique-token FP32
membership matmul and distinct-row copy, first order, vocabulary50,257/width256,
maximum1,024 gradient tokens. CPU gradients use the native reference path; no-grad
inference uses native embedding on either device. Padding, max-norm, sparse or
frequency-scaled gradients, autocast and altered shapes/dtypes remain unsupported.
This is a declared FP32 reduction-order change for both future normalization arms,
not a reinterpretation of the old native-embedding policy or prior checkpoints.
Original source/requirements and the13 frozen recovery modules are unchanged.

33 checks passed under the frozen integration plan. Both RMS and matched-epsilon
LayerNorm preserved old-policy initialization, RNG, parameter inventories, CPU
forward and CPU gradients exactly. Native MPS full/cached outputs were also
bit-exact between old/new policies. Full-versus-cached maximum error was8.940697e-7
under inherited atol3e-4/rtol3e-4. Four full-model gradient repeats per arm at shapes
1x128 and8x17 were bit-exact for every parameter (16 MPS backwards total). Maximum
error against native CPU embedding/shared MLA/FFN with independent FP64 block
normalization was5.564652e-8 under inherited gradient atol3e-5/rtol3e-3. This is
scoped repeatability evidence; full8x128 gradients and optimizer replay are pending.

Four untrained raw payloads serialized in RAM and reloaded exactly. Old/new policy
checkpoint and cache interchange was rejected in both directions before mutation;
missing/malformed specifications and actual module changes were rejected. These
raw payloads have no optimizer/training state, and strict-resume loading rejects
them. H200 export/reconstruction and populated optimizer resume are not established
by this transport check. No binary files or optimizer updates were produced.

Guard8.165s; peakRSS1,806,532,608bytes and sampled MPS driver1,244,495,872bytes.
Strict fallback0/fast_math0,4CPU and4GiB/6GiB/8GiB/600s guards held. Frozen sources,
closed campaigns and retained generated fixtures passed unchanged audits. No
quality training, deletion or holdout scoring occurred. SiLU remains retained;
no normalization quality conclusion is available.

Next freeze a new v2 both-arm MPS repeat/resume prerequisite with populated optimizer,
EMA and RNG state, using the original failed gate (atol2e-6/rtol2e-4), then finish
v2 H200/inference transport and full-workload/resource/controller/statistical gates.
Do not rerun historical failed fixtures, widen tolerances or change Muon precision.
Full-panel capacity remains unresolved; no partial panel, seed/control/width trimming
or protected-weight deletion. Freeze exact finite seeds/order before quality launch.


## October 31 v2 MPS repeat/resume: both arms exactly reproduced

The new, separately frozen `grouped-embedding-resume-plan-001.json` exercised the
explicit grouped-embedding v2 policy with new prerequisite seeds373101/373102/373103.
No historical failed v1 fixture or closed campaign was rerun. Runtime files,
original source/requirements and13 recovery modules were unchanged. Installed
Muon precision, including its BF16 orthogonalizer, was not altered.

Both RMS and matched-epsilon LayerNorm passed the original model/optimizer/EMA
acceptance tolerance (atol2e-6/rtol2e-4), with observed maximum errors **zero**.
Each arm ran three reference updates, three fresh-repeat updates and two updates
after restoring the step-one RAM checkpoint:16 synthetic updates total on real
GPT-2 loading of generated text, batch1/context128. Every intermediate gradient
and post-update model/optimizer/EMA tensor hash matched. Initialization, restored
state, schedules/groups/counters, batches, CPU/MPS random draws and all final RNG
states matched exactly. This short prerequisite supports the scoped grouped
embedding workaround; it does not establish global determinism or model quality.

Each arm also rejected five malformed MPS RNG states and three invalid device/
PyTorch-version bindings before any model/optimizer/EMA/RNG mutation. Both `mps`
and `mps:0` aliases passed preflight. Complete one-update raw states serialized to
RAM archives of445,082,791bytes (RMS) and445,128,471bytes (LayerNorm); their hashes
and restored state hashes are retained. Payload bytes were ephemeral, no binary
files were written, and no historical/selected quality checkpoint was loaded.
H200 serialization/export and production controller resume remain separate gates.

The2 parameterized tests passed in15.70s; guard elapsed16.741s, peakRSS2,469,740,544
bytes and sampled MPS driver1,242,398,720bytes. Four CPU threads, strict fallback0/
fast_math0 and4GiB/6GiB/8GiB/600s limits held. Source, closed-root and existing
fixture audits passed unchanged. No quality training, cleanup or holdout scoring.

Next prepare v2 H200 file export/reconstruction and full8x128 paired generated-loader
MPS readiness with a bounded storage projection before writing any new fixture.
Verify exact EMA replacement, parent hashes, tokenizer identity, inference/cache
parity and refusal to resume an online optimizer from H200. Keep raw/H200 fixtures
and failure evidence under the retention policy. Then finish controller, retention,
selection, statistics and inference/resource gates and freeze the finite quality
plan. Full-panel capacity remains unresolved; do not launch a partial panel,
trim controls/seeds/width, delete protected pairs or repeatedly poll unchanged space.
SiLU remains retained; no normalization quality result is available.


## October 31 v2 full-batch trainer and H200 transport passed

The separately frozen `grouped-embedding-h200-plan-001.json` passed9 checks.
Both normalization arms used the actual isolated `run_training` with GPT-2 on
generated Unicode/special-marker text, paired512 sampling, batch8/context128,
strict MPS, two updates and2,048 target tokens each. This closes the prior
full-batch generated-loader gap; it is not an actual-corpus quality/resource pilot.
The paired sampler was checked against independently drawn starts before training.

Both raw/H200 pairs were serialized to exclusive capped files. H200 matched all
saved EMA parameters exactly (88 RMS/96 LayerNorm), copied nonlearned buffers
exactly, and preserved parent SHA, tokenizer/config and versioned model signatures.
Export preserved CPU/MPS RNG. Raw complete resume preflight passed; H200 optimizer
resume was rejected before model mutation. Malformed tokenizer metadata and
re-export to an existing path were rejected. Reconstructed CPU/MPS and cached
H200 inference at1x17 and8x17 passed inherited atol3e-4/rtol3e-4, with maximum
CPU/MPS error7.748604e-7 and full/cached error1.072884e-6.

A prospective storage projection reserved1,216,348,160bytes above8GiB, including
two448MiB raw caps, two112MiB H200 caps, metadata and source/log allowances; no
temporary checkpoint copies. The capped writer rejects overflow before writing.
The completed fixture retains14 files totaling1,118,443,110bytes, including four
binary checkpoints under `research_state/block-norm-v2-h200-fixture-001`.
All files are SHA-bound in `grouped-embedding-h200-fixture-manifest.json`; retain
these pairs for dependent tests. No old checkpoint was deleted or modified.

Guard elapsed11.985s; peakRSS1,628,127,232bytes and sampled MPS driver2,501,722,112
bytes. Preflight free space10,347,212,800bytes; last sampled free space9,034,428,416
bytes, above the8GiB reserve. These are observed filesystem values, not inferred
space savings. Four CPU threads, strict fallback0/fast_math0 and existing time/
memory/disk guards held. Original/copied runtime sources,13 frozen recovery modules,
closed campaigns and prior fixtures passed unchanged audits. No holdout scoring.

Next implement the new isolated controller/config/source bindings and bounded
selection/statistical/retention/serial-dispatch tests, with full-validation and
inference evidence paths. Reuse retained v2 fixtures read-only where possible;
new binary production needs a new reserve projection. Test failure/stop handling,
finite stage/order and exact selection rules before freezing any quality launch.
Actual-corpus pilot cost, full-horizon behavior and full retained-panel capacity
remain unestablished. Do not launch a partial panel, trim controls/seeds/width,
remove protected pairs or poll unchanged capacity. SiLU remains retained; no
normalization quality conclusion follows from this generated-text prerequisite.


## October 31 finite-stage, selection and statistics rules passed

The new pure `campaign_rules.py` passed67 CPU tests under prospective
`campaign-rules-plan-001.json`. It suggests exactly two20-step pilots, six1200-step
screens and six fresh confirmations on the explicit v2 backbone. Both arms have
LRs0.000625/0.00125/0.0025; each independently selects minimum full-validation H200
CE, breaking exact ties toward lower LR. Screens alternate arm order by ascending
LR; confirmations alternate by declared fresh-seed order. All18 suggested actions,
including pilot/screen audit boundaries, selection publication and final review,
were checked. Premature, incomplete, duplicate, reordered, failed or malformed
records are rejected. Persisted selection cannot change at retention/confirmation.

The seed contract requires distinct typed pilot/screen/three-fresh/evaluation/
bootstrap values. Test values373301/373302/373311–313/373303/373304 are synthetic
inputs only, not scientific seed reservations. Final seed freshness/census and
exact launch configuration/source/order remain to be frozen after prerequisites.
Pilot cooldown/EMA and the complete worker configuration still need their explicit
controller binding; these pure row specifications are not a complete training plan.

Statistics use aligned2048-target base sums, grouping to2048/4096/8192 and2000
PCG64 common paired resamples per size. RNG resets to the declared bootstrap seed
for each size. Sampled target counts weight short tails; the same text indices
apply to all three trained pairs. Linear quantile1-0.05/3 supplies each gate upper
bound. An independent Python scalar/per-seed calculation with separate NumPy
linear quantiles agrees to3.469447e-18 under predeclared absolute5e-14. Repeated
arithmetic is exact. Strict gates reject equality at mean-0.02, any seed0 or upper0.
Malformed inputs, nonfinite/overflow arithmetic and inconsistent panels are rejected.

Numeric eligibility always returns `adopted=False`. These functions do not prove
checkpoint/source/config/tokenizer/target identity or full validation coverage.
Boundary flags must later resolve immutable audited receipts; they do not authorize
cleanup. Retention only identifies four losing-screen indices while protecting both
selected screens and all fresh pairs; no paths are removed. The executable ledger
must handle pending requests, locks, STOP/failure, source drift and serial acceptance.

Guard1.793s, peakRSS236,732,416bytes,4CPU/120s/4GiB/8GiB limits. No model, corpus or
checkpoint payload was loaded, no MPS or training ran, and no binary file/cleanup/
holdout access occurred. Prior fixture bytes were hash-audited read-only, including
both retained v2 raw/H200 pairs. Original/runtime/recovery sources and closed-root
inventories remained unchanged. Full-panel capacity remains gated.

Next implement and test the new isolated manifest/config binding and append-only
session ledger before any worker launch: bind exact source inventories/environment,
corpus/tokenizer/split/validation coverage and v2 trial configurations; enforce one
pending immutable request, exact accepted prefixes, persistent terminal failure/STOP,
selection and audited retention receipts. Then connect full-validation/inference
and worker/parent acceptance with retained fixtures. Do not treat this pure rule
pass as controller, cleanup, quality adoption or launch readiness.

## Read-only campaign contracts passed, October31,2026

The isolated preparation contract passed66 guarded CPU checks. It builds and
validates exact v2 configurations for both20-step pilots, six1200-step screens
and all54 fresh-slot/LR-selection combinations (62 checks of configurations).
Pilot cooldown and H200 begin at13of20 with EMA interval1; quality keeps800of1200
and interval10. No output directory or model is created by the builder. Changed
fields, wrong slots, path aliases and source inventory drift are rejected.

The preparation manifest binds369 source files, the unchanged original source
and13 recovery modules, strict four-thread environment, installed Muon source,
base recipe, corpus bytes, tokenizer and train/validation identities. The actual
validation has34,278 targets/111,536 bytes:267 full128-target windows plus102,
and16 full2048-target statistical blocks plus1510. Only train and validation
raw-text slices were encoded; test was neither tokenized nor scored. This binds
intended coverage, not evaluator loss provenance.

Guard:4.02235s, peak RSS272,367,616B, free disk9,226,452,992B;120s/4GiB/8GiB
guards passed. No MPS/model/optimizer work, checkpoint writes, cleanup or new
quality result. Earlier fixtures, closed campaigns and runtime source stayed
unchanged. The manifest always has launch_permitted=false; fixture seed values
are unreserved. A future immutable ledger must bind the selected LR and exact
request/config/source, one pending worker, accepted prefix, STOP/failure and
retention receipts. Full validation/inference and worker/parent prerequisites,
actual-corpus pilots, scientific seed census and full-panel capacity remain.

Evidence: `campaign-contract-plan-001.json`, `campaign-contract-001-review.json`,
`campaign-contract-analysis.json`, `campaign-contract-preparation-manifest-001.json`
and `campaign-contract-decision.json` in the block-norm preparation evidence directory.

## Synthetic append-only ledger passed, October31,2026

The isolated ledger passed42 bounded CPU checks. Its32-event replay accepts
exactly two pilots, six screens and six fresh confirmations, with one pending
request, per-arm selected LR, exact accepted prefix and boundary indices.
Content-addressed configuration/request/receipt objects and sequenced event
hashes reject edits, missing files, aliases and stale callers. A nonblocking
file lock serializes writes. Durable STOP/failure retains the pending identity
and blocks later acceptance; source drift does not prevent recording STOP.

A caller must retain the expected head outside the ledger: a hash chain alone
cannot detect removal of a valid suffix. That check passed, but external head
persistence/recovery is still a controller integration task. Interrupted or
invalid operations may leave valid unreferenced objects; these are preserved
and cannot advance accepted state. Partial event writes fail closed.

This version only accepts explicitly synthetic parent and retention receipts.
It cannot launch workers, validate real checkpoints or authorize cleanup, and
always reports launch_permitted=false. The passing replay is not real-artifact
acceptance. Guard:13.82545s, peak RSS276,971,520B, free disk9,223,426,048B,
within120s/4GiB/8GiB constraints. No model/optimizer/checkpoint work or quality
result; prior fixtures, runtime and closed campaigns remain unchanged.

Ledger mechanics are complete for this prerequisite. Next prioritize scored
validation/provenance and inference checks against retained v2 fixtures, then
bind real worker/parent artifacts and externally persisted heads. Do not extend
metadata-only scaffolding absent a concrete failure. Actual-corpus pilots, full
retention-aware capacity, scientific seeds and finite launch freeze remain gated.
Evidence: `campaign-ledger-plan-001.json`, `campaign-ledger-001-review.json`,
`campaign-ledger-analysis.json` and `campaign-ledger-decision.json` in the
block-norm preparation evidence directory.

## Cache capacity restored and retained-fixture validation passed, October31,2026

The human authorized unused shared dataset-cache deletion. The exact manifested
cleanup removed9.32GiB allocated, with17.87GiB free afterward. Research weights,
archives, model caches and immutable evidence were verified unchanged. The
remaining checkpoint audit found no additional verified deletion allowlist; historical
numerical payloads remain pending any specific archival/dependency audit.
This clears the current15,170,221,264B full-panel projection with about3.74GiB
headroom; recheck before dispatch. Capacity is no longer the current blocker.
See `tokenizer-evidence/dataset-cache-cleanup-001/completed.json` and checkpoint review.

Read-only validation attempt001 failed: the new scorer's combined MPS-to-CPU
float64 conversion returned zeros. A separate tiny-tensor diagnostic reproduced
it for literal values and crossentropy. The corrected path transfers FP32 to CPU
first and converts there. Failed001 and its exact source are retained. Original
`src/research/coverage_eval.py:54` already uses this safe MPS path; historical
quality results and source remain unchanged. No tolerance was relaxed.

New attempt002 passed34 checks using both retained generated-text v2 H200
checkpoints: all1598 validation targets,13 context windows including62-target
tail, independent CPU FP64 logsumexp oracle, exact MPS replay, RNG preservation,
artifact/hash and malformed-report rejection. Maximum per-window mean NLL
errors were3.786403e-7 RMS and3.410218e-7 LayerNorm (limit3e-5). A separate
8327-target uniform-logit analytic oracle checked66 windows and5 blocks, with
short tails7/135 and mean error2.758057e-7 (limit2e-6). No test split was scored.
Guard7.41663s, RSS808,304,640B, peak sampled MPS driver1,103,872,000B.

This is evaluation correctness on two-step fixtures, not a quality comparison.
No optimizer update, new binary checkpoint, cleanup of research evidence or
model/runtime change occurred. Next bind real worker/parent artifact acceptance,
external ledger heads and retention receipts; complete inference measurement
acceptance, seed census, resource pilots and final finite freeze. Avoid further
metadata-only expansion absent a concrete failure.

## Actual retained artifact acceptance passed, October31,2026

Parent-side acceptance passed52 guarded checks on both retained generated-text
v2 output bundles. Each binds actual config/raw/H200/metrics/summary/export files
and379 source files. Raw optimizer/RNG/EMA resume preflight passes without a step;
all88 RMS or96 LayerNorm averaged parameters exactly match parent EMA, and
nonlearned buffers match the raw parent. Summary, log and checkpoint agree on
updates, tokens, LR schedule, evaluation settings, timing and averaging lineage.

The parent independently recomputes full validation. Its scores exactly match
the earlier CPU-oracle-checked reports. A fabricated score with internally
consistent sums/CE/byte arithmetic is rejected by recomputation: structural
validation alone cannot authenticate a worker's loss. Corrupt configurations,
metadata, signatures, tensors, lineage and each of six file hashes are rejected.
No saved artifact is edited, and CPU/MPS RNG state is preserved.

Guard11.85870s, RSS1,820,884,992B, peak sampled MPS driver1,163,706,368B;
no training, optimizer update, new binary or holdout evaluation. This proves
acceptance of actual existing two-step fixtures, not a new worker dispatch or
quality claim. Receipts still prohibit launch and adoption.

Next connect a thin serial worker/parent path, external head persistence and
audited retention boundaries to these tested primitives, complete inference
measurement acceptance, reserve fresh scientific seeds and freeze the finite
pilot/quality plans. Capacity currently fits after authorized cache cleanup;
recheck at dispatch. Avoid additional metadata-only expansion absent a concrete
failure. Evidence: artifact-acceptance-plan-001/review/analysis/decision and source
snapshot in the block-norm preparation evidence directory.

## Real readonly worker dispatch passed, October31,2026

Two actual child processes sequentially loaded the retained v2 RMS/LayerNorm
fixtures, scored all validation targets, exited and were reaped. Only then did
the parent audit each six-file bundle and independently recompute its score.
Both exactly match the previous CPU-oracle-checked reports. All22 checks passed,
including duplicate-attempt and STOP rejection,17 corrupt process/report/guard
variants, and rejection of parent impersonation of a child. Sources (382 files),
configuration, requests, environment, argv, parent PID and inherited locks bind
the handoff. The frozen bounded subprocess supervisor is reused unchanged.

Child wall times were4.132s/2.620s; peak child RSS653,885,440B and sampled
MPS driver1,125,711,872B. Parent peak RSS1,222,098,944B and driver1,155,317,760B.
A live-child parent uses synchronous guard checks so the supervisor owns cleanup;
the parent's strict MPS audit guard begins after child exit. No training, new
binary, deletion or holdout scoring occurred. Attempt001 failed test collection
before starting a worker; its source/log/plan are preserved. Attempt002 changes
only the test's relative import to an absolute namespace.

This closes the real readonly worker-to-parent transport prerequisite. It does
not freeze or launch the scientific campaign. Next finish inference measurement
acceptance and connect durable external heads and audited retention receipts to
the actual campaign path, then census seeds, freeze the finite pilots and run
the actual-corpus resource pilots. Ledger mechanics need no further expansion.
Evidence: fixture-dispatch-plan-001/002, original source snapshots,002-campaign,
reviews, analysis and decision in the block-norm preparation evidence directory.

## Cached inference passed; scientific seeds reserved, October31,2026

The retained v2 pair passed21 guarded inference checks. Before any warmup, both
arms at batch1/8 passed full96 versus cached32+64 teacher-forced calls and cache
reset/replay; maximum absolute logit difference1.43052e-6. Both cache sizes are
196,608B at batch1 and1,572,864B at batch8. LayerNorm adds8,192B of learned
parameter storage. Two warmups and eight alternating paired repeats per batch
produce eight warmup/32 measured rows. Loading/cache allocation stay outside
synchronized timing intervals; saved-value/provenance review and independent
median/throughput arithmetic pass.19 corrupted reports and a swapped-arm request
are rejected. CPU/MPS RNG and all retained artifacts remain unchanged.

The22.577s guard observed peak RSS869,646,336B and MPS driver1,154,236,416B.
These generated two-step fixture timings are implementation checks only: no
quality or speed winner, deployment peak or cost adoption conclusion. Real
campaign inference still needs its worker/process provenance and fresh trained
pairs. No optimizer updates, new binary files, deletion or holdout scoring.

A separate120s-guarded freshness census scanned40,314 recorded research/config/
source/log text files (3,701,570,300B), finding no whole-word occurrence of the
seven proposed seeds. Reserve pilot374001, screen374002, fresh374011/374012/374013,
evaluation374003 and bootstrap374004. Both arms share paired training seeds.
The census excludes only its own new output directory, bytecode/binary artifacts,
raw corpus and external files; it claims freshness in recorded repository work.
The exact file/hash inventory and audit source are retained.

Next build/freeze the minimal finite pilot entry using these reserved seeds and
tested config/worker/acceptance primitives, then run the two actual-corpus20-step
resource pilots. Keep their weights until the required audits and dependent
prerequisites finish. Before cleanup or quality dispatch, connect durable external
heads and audited retention receipts; do not expand already-passed ledger mechanics.
Final quality freeze remains separate. Evidence: inference-measurement-plan001/
review/result/analysis/decision and scientific-seed-census-001.

## Two actual-corpus resource pilots frozen, October31,2026

The finite pilot plan is now frozen for RMS then LayerNorm: seed374001,20updates,
LR0.0025, batch8/context128, cooldown/EMA start13 and H200 interval1. It binds
388 source files, exact configs/data/environment, completed prerequisites and
the15,170,221,264B complete-panel projection; observed free space19,149,811,712B.
Quality dispatch and checkpoint cleanup remain prohibited by this pilot freeze.

A concrete fixture-to-production audit gap was fixed before freeze: the trainer
logs at evaluation indices, not each update. The20step pilot must have rows0/19;
1200step quality runs must have0/300/600/900/1199.52 existing artifact checks and
12 sparse-metric checks pass. The original runtime and logging schedule are
unchanged, and the previous acceptance source remains preserved. The finite
trial acceptance scope additionally binds exact configuration/spec/seeds.

Run the two bounded children serially, independently audit raw/H200, resume state,
full validation and raw/H200 cache/reset at batches1/8, and anchor each accepted
receipt outside the campaign directory only after parent guards close. A partial
attempt or absent external anchor blocks progression; no retry or cleanup.
Keep both pilot pairs until complete audits and dependent prerequisites finish.
Evidence: pilot-acceptance-plan001/review/decision and pilot001-plan/freeze/build.

## Pilot001 closed after validation-adapter failure, October31,2026

The actual-corpus RMS pilot completed20updates/20,480targets in4.073884 measured
training seconds and wrote its raw and H200 checkpoints (eight EMA snapshots).
The child then failed during post-training validation: `validation.load` required
exactly train/val split keys, while the inherited80/10/10 checkpoint loader
exposes train/val/test. The earlier generated80/20 fixtures did not exercise
this boundary. The failure occurred before full-validation scoring or cache
audits. No parent acceptance exists, and the LayerNorm pilot did not start.

The child exited and was reaped; the root is now closed with STOP, never retried.
Both checkpoint files, frozen source, configuration, logs and failure evidence
are protected. The failed32.737s child guard observed peak RSS1,480,212,480B
and MPS driver2,510,094,336B, below limits; this is a postprocessing error rather
than a resource-limit breach. It is not a passed pilot or a quality result.

The inherited loader does encode/materialize the held-out split, but the frozen
trainer uses only training batches and sampled validation. No test loss, test
training batch or test-based decision was computed. Do not claim the holdout
was never tokenized. Next correct the read-only scorer/inference interface with
an explicit train/val-only view and actual80/10/10 boundary tests, keeping model
math and raw boundaries fixed. Audit the saved RMS outputs in separate recovery
evidence without retraining or relabeling the failed process. A separately frozen
continuation may carry verified RMS evidence and run the unstarted LayerNorm
pilot. Rebuild remaining capacity without cleanup credit for these failed weights.
Evidence: pilot001-failure.json, failure-evidence, close-failure.py and original
pilot001 freeze/source. Free space after closure:18,575,335,424B.

## Actual-corpus validation recovery passed, October 31, 2026

The concrete split-interface failure is fixed with an explicit read-only
train/validation view. It reuses the frozen raw split boundaries, encodes only
train and validation, and verifies tokenizer and training-prior identity.
The training runtime and model math are unchanged. The actual 80/10/10 boundary
test observes exactly two encoding calls and no exposed test array or sampler.

All 62 guarded checks passed: 34 validation checks, 21 inference checks and
seven actual-corpus recovery checks. The saved RMS pilot raw/H200 pair passed
independent parent artifact acceptance, optimizer/RNG/EMA preflight, full
validation and raw/H200 cached-versus-full/reset checks at batches 1 and 8.
All 34,278 validation targets are accounted for. H200 CE is 9.611318003763374;
the independent CPU FP64 oracle is 9.611318312664661, with maximum window-mean
error 4.108e-7. This 20-update pilot score is diagnostic, not a quality comparison.
The audit confirmed 20 updates, 20,480 tokens and eight EMA snapshots; it made
zero optimizer updates and preserved checkpoint bytes and RNG state.

The recovery guard finished in 46.615 seconds, peak RSS 1,380,466,688 bytes and
MPS driver 1,162,608,640 bytes. The failed pilot001 process remains failed and
closed, with its source, STOP and all artifacts unchanged. This separate
read-only acceptance does not relabel that process. Next freeze a new finite
continuation carrying these audited RMS components and run only the unstarted
LayerNorm resource pilot. Recheck remaining capacity with the failed RMS pair
protected. Quality training remains unfrozen; SiLU remains adopted.
Evidence: validation-recovery-plan-001.json, validation-recovery-001-review.json,
validation-recovery-001-rms-result.json and validation-recovery-decision.json.

## LayerNorm resource pilot accepted, October 31, 2026

Pilot002 slot1 completed20 updates/20,480 tokens with eight EMA snapshots, and
passed independent parent artifact acceptance, full-validation replay and all
four raw/H200 batch1/8 cache/reset checks. The child exited cleanly and was reaped;
its46.375-second guard observed peak RSS1,544,667,136B and MPS driver2,501,705,728B.
The parent audit completed in10.132 seconds. Measured training time was3.509446s.
Full-validation H200 CE9.567695740230638 is a short-pilot diagnostic only, not
evidence for adoption; no quality ranking or test loss was computed.

RMS resource observations and separately audited checkpoint components are carried
explicitly; the original RMS process remains failed and closed. Both pairs are
retained, and the failed RMS pair is protected from cleanup. About17.95GB remains
free. Next measure the actual trained pair's descriptive inference, then connect
the finite quality sequence to real parent acceptance and audited retention
boundaries before final quality freeze. The original15.17GB projection assumes
stage-boundary retention; do not reinterpret it as a retain-all projection.
No additional ledger-only scaffolding is needed. Evidence: pilot002-result.json,
pilot002-accepted-01.json and pilot002-completed-evidence/.

## Trained pilot-pair inference completed, October 31, 2026

The actual20-update RMS and LayerNorm H200 checkpoints passed all four full/cache/
reset comparisons before timing. With two warmups and eight alternating repeats,
median32-token prefill/64-token decode times were8.477/291.908ms for RMS and
8.952/278.811ms for LayerNorm at batch1; at batch8 they were13.713/297.073ms
and14.222/305.529ms respectively. LayerNorm decode time was4.49% lower at batch1
and2.85% higher at batch8, with identical cache allocations196,608/1,572,864B.
This mixed single-pilot result supports no consistent speed advantage. It is
descriptive only; two models were resident, cache allocation/loading excluded,
and timing values are not independently reproducible hashes or adoption gates.

The24.224-second guarded run peaked at RSS881,606,656B and driver1,155,842,048B.
Both original pilot directories, weights and failed-process status remained
unchanged. The new RMS measurement request explicitly rebinds only the additive
continuation source; original requests and recovery evidence remain immutable.
No training, test scoring or deletion occurred.

The remaining quality projection keeps both existing pilot pairs without cleanup
credit, retains all six screens through saved LR selection, then permits only
exactly audited losing-screen weight removal. Two selected screen pairs plus six
fresh pairs define the later peak. Original per-trajectory allowances,256MiB
metadata and8GiB reserve require15,170,221,264 additional free bytes, versus
approximately17.95GB available. This projection fits, but is not launch authority:
real quality dispatch, external acceptance anchors and the exact retention
boundary must be connected and checked before final finite quality freeze.
Evidence: pilot-pair-inference-001-plan/result/review/guard.json and
quality-remaining-capacity-001.json. SiLU remains adopted; no norm quality result.

## Finite normalization quality plan frozen, October 31, 2026

The real quality dispatcher now connects the accepted pilot worker/parent path
to1,200-update trials, a7,200-second total worker/parent deadline, independent
full-validation/cache replay, and external acceptance anchors. Requests and
partial attempts cannot be retried. Six screens span both arms and three LRs;
per-arm minimum H200 CE (lower-LR tie-break) is saved before three fresh pairs.
The original quality thresholds and simultaneous conditional-text bootstrap
remain unchanged; adoption still needs a separate final review.

The exact retention boundary re-audits all six screens, persists selection, then
allows only four losing raw/H200 pairs with a size/SHA/path manifest, fsynced
intent/deletion journal and complete protected-evidence verification. Two chosen
screen pairs, all six fresh pairs, both pilots and all non-weight evidence remain.
The failed RMS pilot remains permanently protected; no cleanup credit is assumed.

Integration001 failed during collection due to a relative test import; no test,
training or deletion ran. Its source and log are preserved. Corrected002 passed
nine checks including disposable-file cleanup/corruption/re-entry handling and
actual parent acceptance of both trained pilot models. No scientific checkpoint
was deleted. The final frozen entry bound successfully with395 source inventory
entries and17,937,055,744B free versus the
15,170,221,264B initial requirement. All CPU/thread/memory/disk/holdout guards remain.
Evidence: quality-integration-001/002-plan/review/source, quality001-plan/freeze/
preflight and quality-retention-disposable-002. The first quality screen is next.

Quality001 slot00 accepted (2026-10-31T22:54:47.339243+00:00): rms_incumbent, seed374002, LR0.000625, 1,200 updates/1,228,800 targets, H200 full-validation CE4.883680153820. Independent parent tensor/optimizer/EMA/validation/cache audits passed and the child exited cleanly. This is one endpoint; no adoption or partial-screen selection. Evidence: quality001-slot00-result.json and accepted-00.json.

Quality001 slot01 accepted (2026-10-31T23:00:01.867456+00:00): layer_matched_epsilon, seed374002, LR0.000625, 1,200 updates/1,228,800 targets, H200 full-validation CE4.880529115287. Independent parent tensor/optimizer/EMA/validation/cache audits passed and the child exited cleanly. This is one endpoint; no adoption or partial-screen selection. Evidence: quality001-slot01-result.json and accepted-01.json.

Quality001 slot02 accepted (2026-10-31T23:05:48.222426+00:00): layer_matched_epsilon, seed374002, LR0.00125, 1,200 updates/1,228,800 targets, H200 full-validation CE4.690080241450. Independent parent tensor/optimizer/EMA/validation/cache audits passed and the child exited cleanly. This is one endpoint; no adoption or partial-screen selection. Evidence: quality001-slot02-result.json and accepted-02.json.

Quality001 slot03 accepted (2026-10-31T23:11:15.403533+00:00): rms_incumbent, seed374002, LR0.00125, 1,200 updates/1,228,800 targets, H200 full-validation CE4.694150685727. Independent parent tensor/optimizer/EMA/validation/cache audits passed and the child exited cleanly. This is one endpoint; no adoption or partial-screen selection. Evidence: quality001-slot03-result.json and accepted-03.json.

Quality001 slot04 accepted (2026-10-31T23:16:46.542625+00:00): rms_incumbent, seed374002, LR0.0025, 1,200 updates/1,228,800 targets, H200 full-validation CE4.772951029132. Independent parent tensor/optimizer/EMA/validation/cache audits passed and the child exited cleanly. This is one endpoint; no adoption or partial-screen selection. Evidence: quality001-slot04-result.json and accepted-04.json.

Quality001 slot05 accepted (2026-10-31T23:22:21.601112+00:00): layer_matched_epsilon, seed374002, LR0.0025, 1,200 updates/1,228,800 targets, H200 full-validation CE4.784594486091. Independent parent tensor/optimizer/EMA/validation/cache audits passed and the child exited cleanly. This is one endpoint; no adoption or partial-screen selection. Evidence: quality001-slot05-result.json and accepted-05.json.

## Six-screen selection and audited retention complete

All six1,200-update endpoints passed independent parent acceptance. Full-validation
H200 CE for RMS/LayerNorm was4.883680/4.880529 at LR0.000625,
4.694151/4.690080 at0.00125, and4.772951/4.784594 at0.0025. The frozen per-arm
minimum rule selects0.00125 for both. LayerNorm's small selected-rate screening
improvement (-0.004070nat/token) is not a fresh confirmation or adoption result.

The boundary independently re-audited all six checkpoint pairs, saved selection,
then removed exactly eight manifested losing-screen files (slots0,1,4,5), totaling
2,236,771,928 logical bytes. The two selected pairs, all non-weight evidence and
both original pilot directories verified unchanged; the failed RMS pair remains
protected. Exact manifest, intent/deletion journal, guard and receipt are retained.
The deleted weights cannot be reloaded; all prior audit evidence remains.

Post-cleanup source/receipt/retained-file verification passed; approximately16.77GB
is free and the remaining requirement is14.05GB. Next run the six frozen fresh
endpoints at LR0.00125, seeds374011/374012/374013, with every raw/H200 pair protected.
All three descriptive inference pairs and final quality review remain required.
Evidence: quality001-selection.json, quality001-screen-retention/, and
quality001-boundary-result/next/guard.json. No test scoring or adoption occurred.

Quality001 slot06 accepted (2026-10-31T23:30:51.552785+00:00): rms_incumbent, seed374011, LR0.00125, 1,200 updates/1,228,800 targets, H200 full-validation CE4.709659289211. Independent parent tensor/optimizer/EMA/validation/cache audits passed and the child exited cleanly. This is one endpoint; no adoption or partial-screen selection. Evidence: quality001-slot06-result.json and accepted-06.json.

Quality001 slot07 accepted (2026-10-31T23:36:36.371349+00:00): layer_matched_epsilon, seed374011, LR0.00125, 1,200 updates/1,228,800 targets, H200 full-validation CE4.706465472198. Independent parent tensor/optimizer/EMA/validation/cache audits passed and the child exited cleanly. This is one endpoint; no adoption or partial-screen selection. Evidence: quality001-slot07-result.json and accepted-07.json.

Quality001 slot08 accepted (2026-10-31T23:42:26.336186+00:00): layer_matched_epsilon, seed374012, LR0.00125, 1,200 updates/1,228,800 targets, H200 full-validation CE4.720327997809. Independent parent tensor/optimizer/EMA/validation/cache audits passed and the child exited cleanly. This is one endpoint; no adoption or partial-screen selection. Evidence: quality001-slot08-result.json and accepted-08.json.

Quality001 slot09 accepted (2026-10-31T23:50:51.285184+00:00): rms_incumbent, seed374012, LR0.00125, 1,200 updates/1,228,800 targets, H200 full-validation CE4.728084822601. Independent parent tensor/optimizer/EMA/validation/cache audits passed and the child exited cleanly. This is one endpoint; no adoption or partial-screen selection. Evidence: quality001-slot09-result.json and accepted-09.json.

Quality001 slot10 accepted (2026-10-31T23:57:00.233968+00:00): rms_incumbent, seed374013, LR0.00125, 1,200 updates/1,228,800 targets, H200 full-validation CE4.715869467650. Independent parent tensor/optimizer/EMA/validation/cache audits passed and the child exited cleanly. This is one endpoint; no adoption or partial-screen selection. Evidence: quality001-slot10-result.json and accepted-10.json.

Quality001 slot11 accepted (2026-11-01T00:06:08.450927+00:00): layer_matched_epsilon, seed374013, LR0.00125, 1,200 updates/1,228,800 targets, H200 full-validation CE4.710493693825. Independent parent tensor/optimizer/EMA/validation/cache audits passed and the child exited cleanly. This is one endpoint; no adoption or partial-screen selection. Evidence: quality001-slot11-result.json and accepted-11.json.

## Block-normalization comparison closed, November 1 UTC / October 31 Los Angeles

All six LR screens, six fresh confirmations and three fresh inference pairs
completed with independent acceptance. Both arms selected LR0.00125.

| Fresh seed | RMSNorm H200 CE | LayerNorm H200 CE | LayerNorm minus RMSNorm |
|---|---:|---:|---:|
| 374011 | 4.709659 | 4.706465 | -0.003194 |
| 374012 | 4.728085 | 4.720328 | -0.007757 |
| 374013 | 4.715869 | 4.710494 | -0.005376 |

Mean delta is **-0.00544214 nat/token**. Every seed improves and simultaneous
conditional-text upper bounds are negative at all declared block sizes:
2048: -0.00382454; 4096: -0.00420806; 8192: -0.00391838. The required mean
delta below -0.02 is not met. **Do not adopt LayerNorm; retain RMSNorm and SiLU.**
This is a small consistent validation improvement that fails the prospective
practical-effect threshold, not a claim of no effect. Independent grouping,
PCG64 resampling and linear quantiles agreed with the frozen arithmetic within1e-12.
Intervals condition on these trained seeds and reused text; they do not establish
population-seed or holdout confidence. Centering and learned bias changed jointly.

All three H200 inference pairs passed full/cache/reset checks for batches1/8.
Median across the three paired timing medians (milliseconds,32-token prefill /
64-token decode) was:

| Batch | RMSNorm | LayerNorm |
|---|---:|---:|
| 1 | 8.720 / 275.931 | 8.664 / 284.162 |
| 8 | 13.942 / 290.713 | 14.104 / 292.302 |

LayerNorm batch1 decode was slower in all three pairs; batch8 was mixed.
Cache allocations are identical (196608B/item). Timings are descriptive, with
two models resident and allocation/loading excluded; there is no cost-only
adoption route. All guards passed; approximately13.35GB remained free.
Every fresh raw/H200 pair, both selected screens and original failed pilot
are preserved. No additional checkpoint deletion or test scoring occurred.
A STOP marker closes quality001; do not rerun its training, inference or review.
Evidence: quality001-final-review.json, quality001-fresh-inference-* and
quality001-closure.json in tokenizer-evidence/block-norm-preparation-001/.


## EMA quality001: first fresh trajectory accepted

Seed187610011 completed1200updates and41dual-EMA snapshots. H100 full-validation
CE is4.715769963489097 versus H2004.726289522424742, delta-0.010519558935645
nat/token. This is one of three fixed fresh trajectories and is insufficient
for adoption; the required final mean improvement exceeds0.02, with everyseed
and all simultaneous conditional-text bounds also negative. Continue the frozen
panel without interim selection, early stopping or replacement seeds.

Independent raw-resume/export audits and both full-validation recomputations
passed. Worker elapsed292seconds, peak RSS1.827GB and MPS driver2.535GB;
parent acceptance15.24seconds. All guards passed. Paired inference full/cache/reset
checks passed at batches1/8. Median prefill/decode milliseconds were H200
8.212/285.844 versus H1008.839/276.767 at batch1; H20013.918/288.110 versus
H10013.781/291.969 at batch8. Timing is descriptive; cache196608B/item remains
identical. All raw/H100/H200 files are retained, no test scoring or deletion.
Evidence: ema-horizon-preparation-001/quality001-slot00-{review,inference-result,
review-guard}.json and portable slot00-evidence/. Next is frozen quality slot1.


## EMA quality001: second fresh trajectory accepted

Seed187610012 completed1200updates and41EMA snapshots. H100 full-validation CE
is4.6866778909108255 versus H2004.691822250918006, delta-0.005144360007180
nat/token. Both completed seeds favor H100; their descriptive mean delta is
-0.007831959471412. This partial panel does not satisfy or replace the final
three-seed adoption decision. Continue fixed seed187610013 without early stopping.

Independent parent validation, raw/export acceptance and fresh inference checks
passed. Worker peak RSS1.806GB, driver2.527GB; all guards finished. Median
prefill/decode(ms) at batch1: H2008.250/280.984, H1008.924/283.620; batch8:
H20013.882/306.252, H10013.992/287.625. Timing remains descriptive and mixed;
cache196608B/item is unchanged. All three new checkpoint files remain protected.
No test scoring, deletion, architecture or hyperparameter changes occurred.
Evidence: ema-horizon-preparation-001/quality001-slot01-review.json,
slot01-inference-result.json, slot01-review-guard.json and slot01-evidence/.


## EMA horizon comparison closed: retain H200

The finite H100-versus-H200 panel completed all three fresh1200-update online
trajectories with41snapshots per EMA. Both averages used each same online model;
only averaging half-life differed. All raw resume/export audits, independent
parent full-validation recomputations and fresh inference/cache/reset checks
passed. No source, LR, schedule, seed or grid change occurred after freeze.

| Fresh seed | H200 validation CE | H100 validation CE | H100 minus H200 |
|---|---:|---:|---:|
| 187610011 | 4.726289522 | 4.715769963 | -0.010519559 |
| 187610012 | 4.691822251 | 4.686677891 | -0.005144360 |
| 187610013 | 4.700717490 | 4.689580086 | -0.011137404 |

Mean delta is **-0.0089337743 nat/token**. Every fresh seed improves; simultaneous
conditional-text upper bounds are negative at2048(-0.0064582934),
4096(-0.0058233907) and8192(-0.0049000548) tokens. The declared mean delta
below-0.02 is not met. **Do not adopt H100; retain H200, RMSNorm and SiLU.**
This is a consistent small validation gain below the prospective practical-effect
threshold, not evidence that the two averages are identical. Independent
contiguous grouping, PCG64 resampling and NumPy linear quantiles agree with
frozen arithmetic within1e-12. The intervals condition on three trained seeds
and reused validation text; they are not population-seed or holdout confidence.
The experiment is consistent with a smaller averaging lag helping this400-update
tail, but it does not isolate lag as the cause or establish a new optimal half-life.

All three inference pairs passed. Median across paired medians, milliseconds
for32-token prefill/64-token teacher-forced decode:

| Batch | H200 | H100 |
|---|---:|---:|
| 1 | 8.212 / 280.984 | 8.839 / 276.767 |
| 8 | 13.918 / 293.076 | 13.861 / 288.526 |

Per-seed decode timing was mixed; cache196608B/item is identical. This is
descriptive timing of two resident models, excluding loading/cache construction;
there is no cost-only adoption route. All12 raw/H100/H200 checkpoint files
(pilot plus three fresh triples) remain verified and protected. About10.19GB
was free at closure, above the8GiB reserve. No deletion or test scoring occurred.
The inherited trainer materialized test tokens; validation used only train/val.
STOP closes tokenizer-ema-horizon-v1-001; do not rerun training, inference or
this final review. No new control checkpoint is promoted from these results.

Evidence: tokenizer-evidence/ema-horizon-preparation-001/quality001-final-review.json,
its executable source/guard, all three slot reviews/inference reports, and
quality001-closure.json. Final quality source inventory remains399 files,
including395 byte-identical inherited files.

Next boundary: review a distinct structural FFN question (gated versus ungated
at a declared comparable matrix budget), residual scaling at fixed depth, and
MoE feasibility against the recorded evidence. Do not repeat the closed gate-
activation, block-normalization, wider-hybrid, tying or EMA campaigns. Choose
one concrete architecture or HP hypothesis with a full retained-output capacity
projection before implementation/training; no next grid or launch is frozen yet.


## FFN scientific resource pilots and paired inference completed

Both frozen20-update pilots passed real child training and independent parent
acceptance. Each processed20,480 targets and saved8 H200 snapshots. Raw resume,
optimizer/EMA/export/metric/schedule/tokenizer checks passed; independent parent
scoring reproduced the full34,278-target validation report exactly. These short
pilots are excluded from LR selection and all quality/adoption conclusions.

| Arm | Training seconds | Worker seconds | Parent-inclusive seconds | Worker peak RSS | Worker peak MPS driver |
|---|---:|---:|---:|---:|---:|
| SwiGLU672 | 3.614 | 10.710 | 16.492 | 1.437GB | 2.502GB |
| Ungated SiLU1008 | 3.499 | 10.285 | 16.234 | 1.484GB | 2.502GB |

Both strict resource guards and clean process reaping passed. Linear scaling
puts training alone around210–217seconds per1200-update run; this is only a
rough estimate and does not replace the7,200-second full-trial guard.

Trained H200 full/cache/reset checks passed at batches1/8 before alternating
2warmup/8repeat timing. Median32-token prefill /64-token decode milliseconds:
SwiGLU8.058/279.251 versus ungated8.427/278.982 at batch1;
SwiGLU13.919/294.506 versus ungated13.768/284.885 at batch8. Cache is identical
196,608B/item. This single short-pilot pair is descriptive, with two models
resident and loading/cache construction excluded; no consistent deployment or
quality advantage is established and there is no cost-only adoption route.

All four pilot weight files remain retained, totaling1,118,322,228B. There was
no deletion or test scoring. Actual free space16,045,850,624B exceeds the
remaining conservative full-panel requirement15,629,411,630B by416,438,994B.
This accounts for both retained pilots, full serialization limits and metadata
allowances; future dispatch must recheck capacity. No fixture deletion credit.

Next connect/test only full-duration finite quality acceptance and the exact
losing-screen retention boundary, then freeze the complete3LR-per-arm screen
and3freshpaired-seed panel. Keep all pilot sources immutable; additive quality
work must use a new namespace. Neither accepted pilot is rerun.
Evidence: ffn-structure-preparation-001/pilot001-review.json, both external
pilot001-accepted anchors, pilot001-completed-evidence/ and pilot001-inference-*.


## Finite FFN quality plan frozen

The additive research/ffn_quality_v1 path reuses the established bounded serial
controller, independent full-validation/cache acceptance, paired-statistics and
exact screen-retention logic. It binds FFN configurations and88/84 EMA parameter
inventories without editing any frozen pilot or historical source. Seventy-six
checks passed, including finite orders/ties/adoption arithmetic, all12config
slots, disposable-only cleanup/corruption/re-entry rejection, and real read-only
parent acceptance of both retained scientific pilots. No scientific weights
were deleted.

Integration001 passed all76tests but its external report serialization rejected
pytest tuples. Original source/guard/plan/disposable receipts are preserved.
Distinct002 normalizes report properties to native JSON lists, records its log,
and passed all76checks with the protected pilot roots unchanged. No model or
scientific training retry occurred.

Frozen quality001 contains six1200-update screens at0.000625/0.00125/0.0025,
shared screening seed187620002, alternating arm order. Each arm selects minimum
full-validation H200 CE with lower-LR tie-break. Save selection after all six
accepted endpoints; independently re-audit all six and retire only four losing
raw/H200 pairs through exact manifests/journals before any fresh run. Keep both
selected pairs, both resource pilots and every fresh raw/H200 pair. Then run
three fresh paired seeds187620011/12/13 at the fixed selected rates.

All quality runs retain cooldown800, H200 from800 every10updates, batch8/context128
and1,228,800 targets. Adoption requires mean ungated-minus-gated delta<-0.02,
every fresh delta<0 and all simultaneous conditional-text uppers<0 at the three
frozen block sizes. All three paired inference measurements and a separate final
review remain required; no interim or cost-only adoption.

Full remaining capacity requirement is15,629,411,630B versus16,034,213,888B free.
The bounded checkpoint writer streams to a temporary inode then publishes a hard
link and unlinks the temporary name under the census lock; it makes no second
physical payload copy. Aggregate raw/H200 cap728MiB plus bounded metadata fits
the788,953,140B per-slot cap;256MiB round metadata covers control overlap. The
conservative earlier pilot headroom remains reserved. No speculative deletion
credit or relaxed guard. First quality screen is next.

Evidence: quality-integration001-failure.json, quality-integration002-review.json,
quality-retention-disposable001/002/, quality001-plan/freeze/preflight/capacity, and quality001-protected-pilot-*.


FFN quality001 slot00 accepted (2026-11-01T04:13:05.206410+00:00): SwiGLU672, seed187620002,
LR0.000625,1,200 updates/1,228,800 targets and41H200 snapshots. Full-validation
H200 CE is **4.920534463529** over34,278 targets. Independent parent
raw/optimizer/EMA/export/full-validation/cache audits passed; the child exited
cleanly and all guards finished. Training took225.11seconds; worker265.68seconds
and parent acceptance10.78seconds. Worker peak RSS1.574GB/MPS driver2.510GB.

This first fixed screen supports no selection or comparative quality conclusion.
Next is frozen slot01, ungatedSiLU1008 at the same seed and LR. No checkpoint
deletion or test scoring occurred. About15.50GB is free versus15.07GB remaining
conservative requirement. Both pilot pairs and the accepted screen pair remain
retained. Evidence: quality001-slot00-result.json/evidence/ and
quality001-accepted-00.json in ffn-structure-preparation-001/.


FFN quality001 slot01 accepted (2026-11-01T04:19:11.075544+00:00): ungatedSiLU1008, screening
seed187620002, LR0.000625,1,200 updates/1,228,800 targets and41H200 snapshots.
Full-validation H200 CE is **4.946103441823**, versus SwiGLU672
**4.920534463529** at the same seed/rate: candidate minus control
**+0.025568978294 nat/token**. This first screening pair favors SwiGLU.
It does not settle the independently tuned, fresh-seed comparison or authorize
interim selection, stopping, new rates or a grid extension. Hidden width and
factorization change jointly; this is not an isolated gating-mechanism result.

Independent parent raw/optimizer/EMA/export/full-validation/cache audits passed,
as did clean child exit and all guards. Worker268.54seconds, parent12.04seconds;
worker peak RSS1.579GB and MPS driver2.544GB. Both screen pairs and both pilots
remain retained. No deletion or test scoring occurred. About14.95GB free exceeds
14.51GB remaining conservative requirement. Next frozen slot02 is ungatedSiLU
at LR0.00125, followed by SwiGLU at that rate. Evidence: quality001-slot01-result.json, quality001-slot01-evidence/ and quality001-accepted-01.json in ffn-structure-preparation-001/.


FFN quality001 slot02 accepted (2026-11-01T04:25:21.000319+00:00): ungatedSiLU1008 at
LR0.00125, screening seed187620002,1,200 updates/1,228,800 targets and41H200
snapshots. Full-validation H200 CE **4.722775215414** improves over this
arm's LR0.000625 screen (**4.946103441823**). The paired SwiGLU
endpoint at0.00125 is next; no rate selection or architecture conclusion is
available from this incomplete screen. The remaining frozen rates are unchanged.

Independent parent raw/optimizer/EMA/export/full-validation/cache checks and
clean child exit passed. Training221.93seconds, worker279.39seconds, parent
11.35seconds; peak worker RSS1.558GB/MPS driver2.510GB. All guards passed.
All accepted screen/pilot pairs remain retained; no deletion or test scoring.
Free14.37GB exceeds13.95GB remaining conservative requirement. Evidence:
quality001-slot02-result.json, quality001-slot02-evidence/ and
quality001-accepted-02.json in ffn-structure-preparation-001/.


FFN quality001 slot03 accepted (2026-11-01T04:31:25.581805+00:00): SwiGLU672 at LR0.00125,
screening seed187620002,1,200 updates/1,228,800 targets and41H200 snapshots.
Full-validation H200 CE **4.712840888916** versus ungatedSiLU1008
**4.722775215414** gives candidate-minus-control
**+0.009934326498 nat/token**. Both completed rate pairs favor SwiGLU
on this screening seed. The0.0025 pair and three fresh confirmations remain;
there is no interim rate selection, adoption, early stopping or grid change.

Independent parent raw/optimizer/EMA/export/full-validation/cache audits, clean
child exit and all resource guards passed. Training220.39seconds; worker286.09
seconds and parent10.04seconds. Worker peak RSS1.572GB/MPS driver2.510GB.
All pilot/screen checkpoint pairs remain retained; no deletion or test scoring.
Free13.80GB exceeds13.39GB remaining conservative requirement. Next is frozen
slot04, SwiGLU at LR0.0025. Evidence: quality001-slot03-result.json,
quality001-slot03-evidence/ and quality001-accepted-03.json in
ffn-structure-preparation-001/.


FFN quality001 slot04 accepted (2026-11-01T04:39:27.180350+00:00): SwiGLU672 at LR0.0025,
screening seed187620002,1,200 updates/1,228,800 targets and41H200 snapshots.
Full-validation H200 CE **4.803416296113** over34,278 targets is worse
than this arm's LR0.00125 result **4.712840888916**. All three SwiGLU rates
are now measured; the ungated LR0.0025 endpoint remains before the frozen
six-screen selection boundary. No interim selection or architecture adoption.

Independent parent raw/optimizer/EMA/export/full-validation/cache audits, clean
child exit and all guards passed. Training217.31seconds; worker289.46seconds,
parent10.08seconds. Peak worker RSS1.584GB/MPS driver2.510GB. All pilot and
screen pairs remain retained; no deletion or test scoring occurred. Observed
free13.05GB exceeds the12.83GB remaining conservative requirement; recheck
before dispatch. Next is frozen slot05, ungatedSiLU1008 at LR0.0025. After all
six endpoints pass, save per-arm rate selection and execute the exact audited
losing-screen retention boundary before fresh confirmations. Evidence:
quality001-slot04-result.json, quality001-slot04-evidence/ and
quality001-accepted-04.json in ffn-structure-preparation-001/.


## FFN six-screen selection and retention complete

Slot05 accepted (2026-11-01T04:45:54.354657+00:00): ungatedSiLU1008 at LR0.0025,
screening seed187620002,1,200 updates/1,228,800 targets and41H200 snapshots.
Full-validation H200 CE is **4.749980863365**, versus SwiGLU's4.803416296113
at that rate (candidate minus control **-0.053435432748 nat/token**).
Independent parent raw/optimizer/EMA/export/full-validation/cache checks, clean
child exit and all guards passed. Training213.73seconds; worker291.48seconds,
parent10.09seconds; peak worker RSS1.583GB/MPS driver2.552GB.

All six accepted full-validation screens:

| Peak LR | SwiGLU672 | UngatedSiLU1008 | Candidate minus control |
|---|---:|---:|---:|
| 0.000625 | 4.920534463529 | 4.946103441823 | +0.025568978294 |
| 0.00125 | 4.712840888916 | 4.722775215414 | +0.009934326498 |
| 0.0025 | 4.803416296113 | 4.749980863365 | -0.053435432748 |

The frozen minimum-H200-CE rule selects **LR0.00125 for both arms**, persisted
before any fresh run. The ranking reversal at0.0025 shows why the independently
tuned comparison matters: the selected-rate screen favors SwiGLU by0.009934,
although ungated SiLU wins at the highest fixed rate. This is screening evidence
from one seed, not a fresh confirmation or adoption conclusion. Width and
factorization change jointly, so it does not isolate the gating mechanism.

The boundary independently re-audited all six pairs, saved selection, then removed
exactly eight allowlisted raw/H200 files from losing slots0,1,4,5, reclaiming
**2,236,646,504 logical bytes**. The selected slots2/3, all71 non-deleted files in
the quality evidence inventory and both pilot inventories verified unchanged.
Manifest, fsynced intent/deletion journal, guard and receipt are preserved in
quality001-screen-retention/. Deleted weights cannot be reloaded; original
hashes, complete audits and all non-weight evidence remain. No test scoring.

Post-boundary verification passed: free14,753,619,968B exceeds the remaining
14,511,088,378B conservative requirement. Next is slot06 SwiGLU at LR0.00125,
fresh seed187620011, followed by the paired ungated endpoint. Complete all six
fresh endpoints at frozen rates, all three paired inference reviews and final
independent statistical review. Keep every fresh raw/H200 pair. Evidence:
quality001-slot05-result.json/evidence/, quality001-accepted-05.json,
quality001-selection.json and quality001-boundary-result.json/guards in
ffn-structure-preparation-001/.


FFN quality001 slot06 accepted (2026-11-01T04:53:57.716984+00:00): first fresh SwiGLU672
endpoint, seed187620011, frozen selected LR0.00125. It completed1,200 updates,
1,228,800 targets and41H200 snapshots. Full-validation H200 CE is
**4.700707486073** over34,278 targets. The paired ungated endpoint is pending;
no paired fresh delta, adoption, rate reselection or interim stopping follows.

Independent parent raw/optimizer/EMA/export/full-validation/cache audits, clean
child exit and all guards passed. Training214.88seconds; worker296.82seconds,
parent10.10seconds; peak worker RSS1.623GB/MPS driver2.510GB. This fresh raw/H200
pair is permanently protected under the campaign retention policy. Both selected
screen pairs, both pilots and all existing evidence remain intact. No deletion
or test scoring occurred. Free14,178,803,712B exceeds the remaining13,951,923,794B
projection. Next is slot07 ungatedSiLU1008, same fresh seed and LR. Complete the
three frozen fresh pairs, all paired inference checks and final review. Evidence:
quality001-slot06-result.json/evidence/ and quality001-accepted-06.json in
ffn-structure-preparation-001/.


FFN quality001 slot07 accepted (2026-11-01T05:00:27.705613+00:00), completing the first fresh
pair at seed187620011 and frozen LR0.00125. UngatedSiLU1008 H200 validation CE
is **4.714475188678**, versus SwiGLU672 **4.700707486073**; candidate minus
control is **+0.013767702605 nat/token**, favoring SwiGLU. This first fresh
pair agrees in direction with the selected-rate screen. It is not a completed
three-seed comparison; no interim stopping, reselection or grid extension.

The ungated endpoint completed1,200 updates/1,228,800 targets and41H200 snapshots.
Independent parent raw/optimizer/EMA/export/full-validation/cache checks, clean
child exit and all guards passed. Training212.26seconds; worker298.93seconds,
parent10.02seconds; peak worker RSS1.636GB/MPS driver2.510GB. Both first-fresh
raw/H200 pairs remain protected, as do both selected screens and pilot pairs.
No cleanup or test scoring occurred. Free13,604,524,032B exceeds13,392,765,126B
remaining conservative requirement. Next is slot08 ungatedSiLU1008 at
seed187620012/LR0.00125, then its SwiGLU counterpart in the frozen alternating
order. Finish all three fresh pairs, all paired inference reviews and final
statistical review. Evidence: quality001-slot07-result.json/evidence/ and
quality001-accepted-07.json in ffn-structure-preparation-001/.


FFN quality001 slot08 accepted (2026-11-01T05:07:04.838084+00:00): ungatedSiLU1008 at
second fresh seed187620012 and frozen LR0.00125. Full-validation H200 CE is
**4.722783352829** over34,278 targets. The same-seed SwiGLU endpoint remains
pending; no second paired delta or final architecture conclusion is available.
The first fresh pair still favors SwiGLU by0.013767702605nat/token.

This endpoint completed1,200 updates/1,228,800 targets and41H200 snapshots.
Independent parent raw/optimizer/EMA/export/full-validation/cache audits, clean
child exit and all guards passed. Training211.54seconds; worker301.42seconds,
parent11.11seconds; peak worker RSS1.583GB/MPS driver2.510GB. Every fresh raw/H200
pair remains protected, along with selected screens and both pilots. No cleanup
or test scoring. Free13,024,657,408B exceeds the remaining12,833,606,458B
projection. Next is slot09 SwiGLU672 at the same seed/rate, then the final fresh
pair, all paired inference reviews and final statistical review. Evidence:
quality001-slot08-result.json/evidence/ and quality001-accepted-08.json in
ffn-structure-preparation-001/.


FFN quality001 slot09 accepted (2026-11-01T05:13:45.691626+00:00), completing the second
fresh pair at seed187620012/LR0.00125. SwiGLU672 H200 validation CE is
**4.727638883432**, versus ungatedSiLU1008 **4.722783352829**. Candidate minus
control is **-0.004855530602 nat/token**, a small ungated advantage. The first
fresh pair favored SwiGLU by0.013767702605. Thus the two completed fresh pairs
have opposite signs; their descriptive mean candidate-control delta is
**+0.004456086001 nat/token**, favoring SwiGLU slightly. This is incomplete
confirmation evidence. Run the final frozen pair and all inference/final reviews;
no interim stopping, reselection, grid extension or architecture adoption.

Slot09 completed1,200 updates/1,228,800 targets and41H200 snapshots. Independent
parent raw/optimizer/EMA/export/full-validation/cache checks, clean child exit
and all guards passed. Training214.41seconds; worker307.46seconds, parent10.86
seconds; peak worker RSS1.608GB/MPS driver2.518GB. Every fresh raw/H200 pair,
selected screen and pilot remains retained. This run deleted no checkpoints and
did not score test. Actual free12,744,900,608B exceeds the remaining12,274,441,874B
projection. Next is slot10 SwiGLU672 at seed187620013/LR0.00125, followed by
slot11 ungated at that seed/rate. Evidence: quality001-slot09-result.json/evidence/
and quality001-accepted-09.json in ffn-structure-preparation-001/.


FFN quality001 slot10 accepted (2026-11-01T05:20:50.655072+00:00): final-seed SwiGLU672
control, seed187620013 at frozen LR0.00125. Full-validation H200 CE is
**4.711847316257** over34,278 targets. The matching ungated endpoint is the
last remaining training run. The first two fresh pairs remain mixed in direction;
no third paired delta or final architecture conclusion is available yet.

Slot10 completed1,200 updates/1,228,800 targets and41H200 snapshots. Independent
parent raw/optimizer/EMA/export/full-validation/cache checks, clean child exit
and all guards passed. Training217.00seconds; worker310.34seconds, parent11.06
seconds; peak worker RSS1.610GB/MPS driver2.510GB. Every fresh raw/H200 pair,
selected screen and pilot remains retained. No checkpoint deletion or test scoring
by this run. Actual free12,215,128,064B exceeds the remaining11,715,277,290B
projection. Next is slot11 ungatedSiLU1008 at the same seed/rate; then run all
three frozen paired inference measurements and final statistical review. Evidence:
quality001-slot10-result.json/evidence/ and quality001-accepted-10.json in
ffn-structure-preparation-001/.


## FFN structure campaign closed: retain SwiGLU672

Final slot11 accepted (2026-11-01T05:27:30.110749+00:00): ungatedSiLU1008 at seed187620013,
LR0.00125, H200 CE **4.725101035355**. It completed1,200 updates/1,228,800
targets and41H200 snapshots. Independent parent raw/optimizer/EMA/export/full-
validation/cache checks, clean child exit and all guards passed. Training214.28s,
worker328.03s, parent10.12s; peak worker RSS1.534GB/MPS driver2.510GB.

Both arms used their independently screened LR0.00125. Complete fresh results:

| Fresh seed | SwiGLU672 CE | UngatedSiLU1008 CE | Ungated minus SwiGLU |
|---|---:|---:|---:|
| 187620011 | 4.700707486073 | 4.714475188678 | +0.013767702605 |
| 187620012 | 4.727638883432 | 4.722783352829 | -0.004855530602 |
| 187620013 | 4.711847316257 | 4.725101035355 | +0.013253719098 |
| Mean | 4.713397895254 | 4.720786525621 | **+0.007388630367** |

The candidate is worse on two of three fresh seeds and on average. All three
predeclared gates fail: mean improvement beyond0.02, improvement on every seed,
and negative simultaneous conditional-text upper bounds. Independent arithmetic
recomputed the paired deltas and all2,000 common PCG64 bootstrap draws using
explicit grouping and NumPy linear quantiles, agreeing within1e-12:

| Text block size | Conditional descriptive95% interval | Simultaneous gate upper |
|---|---|---:|
| 2048 | [0.003193095, 0.011832723] | +0.012057621 |
| 4096 | [0.002329997, 0.012221908] | +0.012518783 |
| 8192 | [0.000923522, 0.013122211] | +0.013122211 |

Retain the adopted **SwiGLU672/SiLU/RMSNorm/H200** D256 MLA recipe; do not adopt
ungatedSiLU1008. The fixed-high-rate ordering reversal from screening did not
translate into an independently tuned fresh quality win. This is a joint change
to FFN hidden width and multiplicative factorization at matched matrix budget,
not proof about gating alone. The intervals condition on these three trained
seeds and reused validation text; they do not provide population-seed or test
holdout confidence. No test scoring occurred.

All three frozen paired inference measurements passed full/cache/reset correctness
at batches1/8. Both arms use196,608cache bytes per item (1,572,864 at batch8).
Ungated median decode time was0.34-1.70% lower at batch1 across pairs; at batch8
it ranged from2.07% lower to2.77% higher. These small local timing differences
are descriptive: two models resident, loading/cache construction excluded,
warmups excluded, and no cost-only adoption route. They do not rescue the failed
quality gate. All424 frozen sources, accepted artifacts and protected pilot
inventories were rechecked unchanged. The two additive review scripts adapt the
already exercised block-norm review wrappers, project exact config/raw/H200
bindings to the frozen FFN inference API, and leave frozen training code intact.

The final independent review and guarded closure completed; the quality root now
has a STOP marker. All three fresh raw/H200 pairs, both selected screen pairs and
both resource-pilot pairs remain retained. Earlier8-file losing-screen retirement
remains explained by its immutable receipt; no further checkpoint deletion.
Closure observed12,062,703,616B free. Continue the broader authorized research
loop by reviewing untested internal-normalization/position/local-mixing questions
alongside bounded attention-score/optimizer HP questions, rejecting already closed
comparisons and selecting one distinct justified next hypothesis. Forecast its
complete retained-output requirement before freezing any new training panel.

Evidence in ffn-structure-preparation-001/: quality001-slot11-result.json/evidence/,
quality001-accepted-11.json, quality001-fresh-inference-plan.json and all three
pair result/review/guard files, quality001-final-review.py/json/guard and
quality001-closure.json/guard. The continuation prompt now follows the closed
FFN result and next-direction review; the old research heartbeat remains paused.

## Next direction: MLA attention-score transfer; first arithmetic gate passed

At the closed FFN boundary, reviewed the historical QK/FFN internal-normalization,
position and readout results alongside attention-score and optimizer HP questions.
Selected a distinct two-arm **0.5 versus1.0 attention multiplier** comparison on
the adopted D256 tokenizer MLA backbone. The half-strength choice was learned in
character MHA; its current MLA transfer has not been independently compared. The
current implementation correctly applies0.5 in full/cached attention. This is
not a repair of the roadmap's historical pre-integration issue. Internal norms
and position alternatives had weak/negative historical evidence; short convolution
and MoE remain separate architectural candidates. No closed campaign is reopened.

Both multipliers passed the actual attention component's independent complex-RoPE
forward/gradient oracle and CPU/MPS uneven-cache/reset/causality checks. Maximum
FP64 forward/gradient errors7.78e-16/8.89e-15; CPU/MPS FP32 errors4.48e-7/4.77e-6.
This used private scalar overrides on copies of an incumbent module; the frozen
full-model policy still correctly rejects1.0. No corpus, checkpoint IO, scientific
training or quality comparison occurred. Full-model policy, optimizer/cache and
checkpoint/resume integration remain next, outside424 unchanged frozen sources.
See TOKENIZER_MLA_SCORE_PREPARATION.md and mla-score-preparation-001/component001.*.

Proactively retired the now-completed FFN resource-pilot weights after exact saved
hash verification: four files, **1,118,322,228B**. All266 other checkpoint/archive
files and26,231 metadata files verified unchanged. Free space after cleanup was
**13,223,596,032B**. Historical pilot receipts and closure remain immutable and
now require the later retirement receipt to explain absent weights. Every selected
and fresh pair and every failed artifact remains protected. The completed EMA
pilot triple (787,232,443B) is an additional audit candidate, not deleted or credited.

Existing full-quality SwiGLU raw/H200 pairs each actually occupy559,164,584B.
The previous conservative envelope with ten retained trajectories would require
16,747,901,448B, exceeding post-cleanup space by3,524,305,416B. This is a planning
envelope, not a validated new serialization projection or a reason to block small
integration work. Next measure the new serializer/metadata/transient limits and
prospectively test any early successful-pilot retirement boundary. Do not relax
resource guards, reduce confirmations or launch a partial panel. No training is
currently running; next executable work is the narrow full-model integration.
Evidence: mla-score-preparation-001/direction-decision.json, capacity-inventory001.json,
and ffn-pilot-retention/manifest/journal/completed plus bounded inventory shards.

## MLA score full-model/resume prerequisites passed; capacity recovery completed

The isolated score-transfer runtime now supports exactly0.5/1.0 while preserving
all incumbent settings. Both arms have28,492,416 parameters, identical initial
weights and identical optimizer groups; the0.5 control matches old logits bitwise.
Cross-arm checkpoints/caches and mutated scalar configurations are rejected.
Maximum CPU/MPS full-model logit/gradient errors were8.94e-7/1.10e-7; cached
error1.02e-6, with exact reset and causality checks. A harness-only namespace
failure before training was preserved as model001; corrected model002 passed
without altering frozen sources or weakening strict class validation.

Generated-text serialized tests passed both arms with bitwise repeat/resume
model, optimizer, EMA and RNG state, plus raw/H200 parent/export/load checks and
batch1/8 inference. The16 total generated-data updates are implementation tests,
not scientific training or quality evidence. No research corpus/test was read.
Raw files measured445,082,407B and H200114,074,435B under enforced448/112MiB caps.
Next required work is real worker/output-sink, independent parent validation and
inference acceptance; no pilot or quality plan is frozen yet.

Exact cleanup retired the completed EMA pilot raw/H100/H200 triple, reclaiming
787,232,443B and verifying269 other checkpoint/archive files plus26,234 metadata
files unchanged. Lossless cold archival then preserved all twelve successful
FFN/score resume-fixture weights:4,016,947,008B of original members became two
archives totaling1,817,863,749B, net savings2,199,083,259B. Complete decompression
verified every member before any loose copy was removed;26,491 other retained
files were rechecked. All selected/fresh/failed artifacts and older archives stayed
unchanged. Original fixture receipts remain immutable; later archive manifests
resolve their logical identities. A pre-apply static reference-type correction
was recorded separately; no failed archive/deletion attempt preceded it.

Free space afterward13,986,324,480B. The measured prospective requirement is
13,690,208,256B using eight576MiB trajectory allowances,256MiB metadata and8GiB
reserve, with successful-pilot retirement before screening and exactly audited
losing-screen retirement before fresh confirmations. This is conditional on the
real worker enforcing448/112MiB raw/export caps,16MiB per-trajectory metadata,
no duplicate whole-payload publication and tested retention boundaries. Recheck
all prerequisite consumption and actual free space before freezes; do not treat
this narrow296MB observed margin as unconditional launch authority.

Evidence: mla-score-preparation-001/model001-failure.json, model002.py/json/guard,
checkpoint-resume001.py/json/guard/trace, serialization-capacity002.json,
ema-pilot-retention/ and synthetic-archive001/, plus research/mla_score_v1/.

## Real worker, independent parent and trained inference prerequisites passed

The score half/unit arms each completed two actual generated-text run_training
updates through the bounded serial child process and output sink. Independent
parent acceptance verified optimizer/RNG/resume semantics, both raw/H200 identities,
all88 EMA tensors and two snapshots,2,048 processed training targets, and all3,014
validation targets. CPU/MPS validation CE differed by at most2.775e-7. Both raw
checkpoints measured445,084,199B and both H200 exports114,074,819B. Existing streaming
publication and new individual448/112MiB limits passed; no duplicate payload or
temporary checkpoint remained. These are generated-data engineering checks only.

The resulting trained pair passed full/cache/reset checks at batches1/8 before
warmups and eight alternating timing repeats. Cache bytes were196,608 per item.
Timing is descriptive for this two-update fixture; it establishes no score-arm
quality or deployment-performance advantage. The research corpus and test stayed
closed. All frozen and worker-bound sources were rechecked unchanged.

Six successful worker checkpoints (both new score pairs and the prior successful
ungated FFN worker003 pair) are now losslessly archived in
`research_archives/mla-score-worker-001/successful-workers.tar.zst`. Full member
verification preceded loose-copy removal, and26,524 other files were rehashed.
Original1,677,471,074B became849,909,661B, recovering827,561,413B net. The failed FFN
worker002 control and every selected/fresh/failure artifact remain unchanged.
Use worker-archive001's exact manifest to resolve or restore absent loose paths.

Capacity003 corrects duplicate metadata accounting: the enforced256MiB complete
campaign budget already includes8.25MiB per trajectory, below its16MiB allowance,
including temporary metadata publication. Thus the complete proposed high-water
is eight560MiB weight allowances +256MiB metadata +8GiB reserve =13,555,990,528B.
The active serializer is included in its weight allowance and publishes by
same-inode link/unlink. No guard or confirmation requirement was reduced. Observed
free13,654,347,776B gives98,357,248B headroom at that observation only. The earlier
13,690,208,256B estimate double-counted128MiB of trajectory metadata.

Next execute the narrow prospective pilot/losing-screen retention-boundary checks,
then fresh seed census, finite pilot freeze, guarded resource pilots and final
quality freeze. Recheck actual free space and charge all new output before dispatch.
No scientific pilot/quality plan is frozen and no scientific worker is running.
Evidence: mla-score-preparation-001/trainer-worker001, trained-inference001,
worker-archive001, capacity003 and worker003-review.json; implementation in
research/mla_score_v1. No new ledger framework was introduced.

## Finite resource pilots completed; quality freeze is next

The narrow pilot and losing-screen retention test passed using the bounded,
locked executor and tiny disposable byte files. Incomplete or failed stages,
wrong saved LR selection, aliasing, fresh-confirmation paths, missing evidence,
and retry attempts were rejected. Protected sentinels and all remaining evidence
were unchanged. This tests eligibility/file handling; actual retirement still
requires independent complete payload/inference/selection audits.

A read-only census reserved pilot198630001, screen198630002, fresh198630011/12/13,
evaluation198630003 and bootstrap198630004 after scanning42,354 recorded files
(3,789,456,073B) with no collisions. No seed was chosen using a quality result.
The first frozen pilot wrapper (pilot001) stopped before config publication,
model construction or updates because its path variable shadowed the imported
data-contract module. Its source, plan, log and STOP remain preserved and closed.
The separate pilot002 wrapper uses an explicit data_contract alias; static
regression and real execution confirm the correction. Its new frozen root carries
forward the explicitly unconsumed reserved seed panel. No historical fixture was
reopened and no guard or validator was weakened.

Both pilot002 arms completed20updates/20,480 training targets and eight H200
snapshots at LR0.0025. Independent parent acceptance and full-validation coverage
passed, followed by complete paired cache/reset/inference barriers and timing.
Worker process times were40.57s/47.34s; parent acceptance4.81s/4.39s, below600s
per trajectory. Peak worker RSS was1,490,141,184B and MPS driver2,510,094,336B,
below4GiB/6GiB. Pilot losses are excluded from scientific LR selection and quality
conclusions. The test holdout was never scored.

All pilot-bound source bytes remain unchanged. research/mla_score_v1 is now a
frozen pilot namespace; final quality code must be additive outside it. The planned
quality scope remains six1200-step screens (three rates per arm), then three fresh
paired seeds, with identical prospective adoption gates and no extension.
Screen and fresh seeds are reserved and still unused.

Actual free space at final review: 12513693696B. The two accepted pilot raw/H200
pairs occupy1118328272B and remain protected until dependent preparation closes.
After exact successful-pilot retirement, prospective free would be13632021968B
against the complete13,555,990,528B requirement, margin76031440B.
This is conditional recovery credit, not completed deletion or launch authority.
Use the tested exact boundary after independent audits, measure actual space,
and freeze the full quality plan before training. Do not treat retained eligible
pilots as a permanent storage blocker or shrink the confirmation panel.

Evidence: mla-score-preparation-001/retention-boundary001, scientific-seed-census-001,
pilot001-failure-review, pilot002-plan/review/final-review and preserved failed/
completed metadata copies. No scientific quality endpoint has been launched.

## Full finite score quality comparison frozen and started

Additive code in research/mla_score_quality_v1 reuses the frozen score runtime,
bounded writer, full-validation/cache arithmetic and tested retention executor.
All74 finite-rule/config/selection/retention checks passed. Independent additive
acceptance reloaded both real pilot raw/H200 pairs and checked optimizer/EMA,
full-validation and cached inference before closing their dependencies. Neither
frozen pilot source nor the earlier failed attempt was modified or rerun.

The exact successful-pilot retirement removed four audited files totaling
1,118,328,272B under global/campaign locks. Independent payload/inference audits,
path/size/SHA256 allowlist, fsynced intent/deletion journal and post-verification
confirmed26,448 other research files unchanged. The failed pilot001 evidence,
selected models, all fresh pairs and archives remain protected. These four pilot
weights cannot be restored; original receipts and copied deletion evidence remain.

The final quality freeze measured14,089,822,208B free against13,555,990,528B:
eight560MiB weight allowances +256MiB complete metadata +8GiB reserve. Each active
trajectory has separate448MiB raw/112MiB H200 caps and a576MiB process-output cap.
Declared active campaign metadata groups total128,188,416B; bounded external
anchors/reviews/retirement shards add at most32MiB, together below the256MiB
allowance. Temporary publication is charged; checkpoint publication adds names,
not another payload. No speculative pilot cleanup or partial panel is needed.

quality001 freezes all twelve1200-update endpoints: six equal-budget screens,
three rates per multiplier (0.000625/0.00125/0.0025), followed by three reserved
fresh paired seeds. Save each arm's minimum full-validation H200 CE, lower-LR tie
break, only after all six screens independently pass. Then audit and retire only
four losing screen pairs; retain both selected screen pairs and all six fresh
raw/H200 trajectories. No extension, replacement seeds, retry or cost-only route.
All three fresh pairs require paired inference and independent final statistics.
The gate remains mean(unit-half)<-0.02, every fresh delta<0, and all simultaneous
conditional-text upper bounds<0 at2048/4096/8192 tokens (2000 common PCG64 draws).

First screen slot00 (multiplier0.5, LR0.000625) has started under the frozen hash.
No quality result or adoption claim is made here. Both score namespaces are now
frozen; continue the exact serial prefix and actual process/anchor state, keeping
the test holdout closed. Evidence: quality-integration001, quality001-pilot-retirement,
quality001-plan/freeze/preflight/capacity/start-review in mla-score-preparation-001.

### First accepted quality endpoint

quality001 slot00 (multiplier0.5, LR0.000625, screening seed198630002) completed
all1200updates/1,228,800 targets and41H200 snapshots. Independent parent checkpoint,
optimizer/EMA, full-validation and cache checks passed. H200 validation CE is
4.896530022nat/token across34278 targets. This is one
screening endpoint, not a candidate comparison, selected rate or adoption result.
Remaining capacity fits: observed13514715136B versus required
12996825432B after charging the accepted retained pair.
The next frozen slot01 (multiplier1.0 at the same LR) has started serially. Keep
all six screens through selection and every fresh pair through final review.
Evidence: quality001-slot00-result.json and its preserved metadata, plus the
external quality001-accepted-00 anchor. No test scoring or interim selection.


### First matched MLA score screening comparison

quality001 slot01 accepted at 2026-11-01T06:31:57.562317+00:00: multiplier1.0 at
LR0.000625 and screening seed198630002 completed1200updates/1,228,800targets,
with41H200 snapshots and88 EMA parameters. Independent parent checkpoint,
optimizer/EMA, full-validation and cache audits passed. H200 CE across34278
validation targets is 4.863321954179, versus 4.896530022241 for
multiplier0.5 at the same rate: unit-minus-half -0.033208068063nat/token.
This is the first screening comparison only; it does not select a rate or
establish adoption. Finish all six equal-budget screens, then all three fresh
pairs at the independently selected rates under the original frozen gate.

Worker peak RSS1,640,792,064B and MPS driver2,510,094,336B remained within
4GiB/6GiB limits; worker and parent guards finished without violations.
Observed free 12,939,902,976B exceeds the remaining complete-panel
requirement 12,437,660,336B after charging both retained
raw/H200 pairs. Slot02 (multiplier1.0, LR0.00125) has started serially, with
capacity rechecked by the frozen runner. Keep all six screen pairs until the
saved-selection retention boundary; the holdout remains closed. Evidence:
mla-score-preparation-001/quality001-slot01-result.json, preserved slot01
metadata and quality001-accepted-01.json.


### Third score screening endpoint accepted

quality001 slot02 accepted at 2026-11-01T06:37:05.177388+00:00: multiplier1.0,
LR0.00125, screening seed198630002. All1200updates/1,228,800targets and
41H200 snapshots completed; independent parent checkpoint/optimizer/EMA,
full-validation and cache checks passed. H200 CE across34278 validation targets
is 4.692032518055, compared with 4.863321954179 for the same
multiplier at LR0.000625. This single-seed rate comparison makes no rate selection
or adoption decision. The matched multiplier0.5/LR0.00125 endpoint is next,
followed by both frozen LR0.0025 screens. No grid change or interim stopping.

Worker peak RSS1,642,840,064B and MPS driver2,510,094,336B remained below
4GiB/6GiB; worker and parent guards finished without violations. After retaining
all three raw/H200 pairs, observed free 12,364,083,200B exceeds the
remaining full-panel requirement 11,878,495,240B. Slot03
has started serially with the same frozen source and all guards. Preserve all
screens until the complete six-screen selection/retirement boundary and every
fresh pair through final review; no test access. Evidence: quality001-slot02-result.json,
quality001-slot02-evidence/ and quality001-accepted-02.json in
mla-score-preparation-001/.


### Second matched score comparison accepted

quality001 slot03 accepted at 2026-11-01T06:42:22.271513+00:00: multiplier0.5,
LR0.00125 and screening seed198630002 completed1200updates/1,228,800targets,
with41H200 snapshots. Independent checkpoint/optimizer/EMA, full-validation
and cache checks passed. H200 CE across34278 targets is 4.703659447036,
versus 4.692032518055 for multiplier1.0 at the same rate. Unit-minus-half
is -0.011626928981nat/token. Both completed matched-rate comparisons favor1.0
on this screening seed, but the gap is smaller at0.00125 than at0.000625.
This is no rate selection or adoption result; finish the two LR0.0025 screens
and all three reserved fresh pairs under the original gate.

Worker peak RSS1,643,544,576B and MPS driver2,510,094,336B remained below
4GiB/6GiB; both guards finished without violations. Observed free
11,865,415,680B exceeds remaining complete-panel requirement
11,319,330,144B with four raw/H200 pairs retained. Slot04
(multiplier0.5, LR0.0025) has started serially. All screen pairs remain protected
until the complete six-screen saved-selection boundary; no test access, grid
extension or changed source. Evidence: quality001-slot03-result.json,
quality001-slot03-evidence/ and quality001-accepted-03.json in
mla-score-preparation-001/.


### Fifth score screening endpoint accepted

quality001 slot04 accepted at 2026-11-01T06:47:42.882435+00:00: multiplier0.5,
LR0.0025 and screening seed198630002 completed1200updates/1,228,800targets
and41H200 snapshots. Independent checkpoint/optimizer/EMA, full-validation
and cache audits passed. H200 CE across34278 targets is 4.793428589630,
worse by 0.089769142594nat/token than this arm at LR0.00125
(4.703659447036). The upper tested rate therefore did not improve the
incumbent on this screening seed. This does not select a rate or establish
adoption: complete the final unit/LR0.0025 endpoint, then apply the frozen
per-arm minimum-CE/lower-LR tie rule to all six accepted screens.

Worker peak RSS1,611,890,688B and MPS driver2,510,094,336B stayed within
4GiB/6GiB limits; both guards finished without violations. Observed free
11,260,493,824B exceeds the remaining complete-panel requirement
10,760,165,048B with five raw/H200 pairs retained.
Slot05 (multiplier1.0, LR0.0025) has started serially. After it passes,
independently re-audit all six and retire only the four losing screen pairs
through the frozen boundary, then run all three fresh confirmation pairs.
No test scoring, grid extension or checkpoint removal occurred in this iteration.
Evidence: quality001-slot04-result.json, quality001-slot04-evidence/ and
quality001-accepted-04.json in mla-score-preparation-001/.


## MLA score six-screen selection and retention complete

All six frozen1200-update screens passed independent checkpoint/optimizer/EMA,
full-validation and cache acceptance. Each completed1,228,800training targets
and41H200 snapshots; validation covers34278 aligned targets. Final slot05
(multiplier1.0/LR0.0025) reached CE4.832040597159. No grid extension, changed
source, pilot-loss selection or replacement seed occurred.

| Learning rate | Multiplier0.5 H200 CE | Multiplier1.0 H200 CE | Unit minus half |
| --- | ---: | ---: | ---: |
| 0.000625 | 4.896530022 | 4.863321954 | -0.033208068 |
| 0.00125 | 4.703659447 | 4.692032518 | -0.011626929 |
| 0.0025 | 4.793428590 | 4.832040597 | +0.038612008 |

The persisted per-arm minimum-full-validation-CE/lower-LR tie rule selects
**LR0.00125 for both arms**. On this screening seed, multiplier1.0 improves
CE at the two lower rates but worsens it at0.0025. At the selected common rate
the difference is -0.011626928981nat/token. This is screening evidence only;
the three reserved fresh pairs and original strict gate decide adoption.
The result makes an unqualified claim that multiplier1.0 always improves this
backbone inappropriate.

The frozen boundary independently reloaded and re-audited all six raw/H200
pairs before persisting selection and retiring slots00/01/04/05 only. Eight
files totaling2,236,660,384B were removed under the shared worker/campaign locks,
exact path/size/hash manifest, durable intent/deletion journal and bounded
guards. All72 retained campaign evidence files were reverified unchanged.
Selected slots02/03 remain; no fresh, failure, archive or unrelated path was
removed. Retired weights cannot be restored; original accepted receipts and
all nonweight evidence remain immutable, with the later retention manifest
explaining their absent checkpoint paths. A read-only recorder independently
verified the boundary and preserved its selection, inventory, manifest, journal,
receipt and guards in quality001-screen-retention/.

Post-retirement free12,979,437,568B exceeds the full remaining
requirement12,437,660,336B, including the unchanged8GiB
reserve. Fresh slot06 (multiplier0.5/LR0.00125/seed198630011) has started serially;
slot07 is its matched1.0 partner. Complete all six fresh trajectories, all three
paired inference reviews and independent final statistics. Keep every fresh
raw/H200 pair and the test holdout closed. Evidence: quality001-slot05-result.json
and preserved metadata, quality001-accepted-05.json, quality001-boundary-result.json,
quality001-boundary-record-guard.json and quality001-screen-retention/ in
mla-score-preparation-001/.


### First fresh MLA score endpoint accepted

quality001 slot06 accepted at 2026-11-01T06:59:36.996505+00:00: multiplier0.5,
selected LR0.00125, first reserved fresh seed198630011. All1200updates,
1,228,800training targets and41H200 snapshots completed. Independent parent
checkpoint/optimizer/EMA, full-validation and cache audits passed; H200 CE is
4.712354865011 across34278 validation targets. This is one fresh endpoint,
not a paired difference or adoption result. Slot07 multiplier1.0 is now running
at the same seed and rate. No selection change, interim stop or grid extension.

Worker peak RSS1,653,833,728B and MPS driver2,510,094,336B remained below
4GiB/6GiB; worker and parent guards finished without violations. Retained
selected screen pairs and the fresh raw/H200 pair were verified. Observed free
12,407,627,776B exceeds remaining complete-panel requirement
11,878,495,240B. All fresh confirmations remain protected.

The existing read-only three-pair inference driver was adapted outside frozen
namespaces, changing only score-module imports and the exact quality freeze SHA.
Syntax validation passed; execution is deferred until all twelve endpoints are
accepted, when its exact requests/anchors and runner hash will be frozen. No
inference result or new numerical claim is made for this unexecuted driver.
Evidence: quality001-slot06-result.json, quality001-slot06-evidence/ and
quality001-accepted-06.json; driver quality001-fresh-inference.py in
mla-score-preparation-001/. Keep the test holdout closed.


### First fresh MLA score pair completed

quality001 slot07 accepted at 2026-11-01T07:05:12.665572+00:00, completing the first
fresh pair at seed198630011 and selected LR0.00125 for both arms. Multiplier1.0
H200 CE is 4.699208080398, versus 4.712354865011 for0.5:
unit-minus-half -0.013146784613nat/token across34278 aligned validation targets.
Both endpoints completed1200updates/1,228,800training targets and41H200 snapshots
and passed independent checkpoint/optimizer/EMA, full-validation and cache audits.
This one-seed improvement does not establish adoption or justify interim stopping.
Complete the other two frozen pairs and all three inference reviews before the
independent strict-gate calculation; retain both raw/H200 pairs.

Worker peak RSS1,648,738,304B and MPS driver2,518,482,944B remained within
4GiB/6GiB limits; both guards finished without violations. Observed free
11,848,216,576B exceeds remaining full-panel requirement
11,319,330,144B. Slot08 (multiplier1.0, seed198630012,
LR0.00125) has started in the frozen unit-first order for the second pair.

The independent final-statistics driver was adapted from the completed FFN
review outside frozen namespaces, with score identities and neutral handling
of either gate outcome. It rechecks all12 accepted endpoints, all three
inference reviews, aligned validation blocks, explicit independent bootstrap
quantiles and the frozen gate. A passing gate requires a separate adoption
review; the script cannot silently adopt. Syntax validation passed; it has
not been executed and provides no new statistical conclusion yet. Evidence:
quality001-slot07-result.json, quality001-slot07-evidence/, quality001-accepted-07.json
and quality001-final-review.py in mla-score-preparation-001/. Holdout stays closed.


### Second-seed score candidate accepted

Slot08 accepted at 2026-11-01T07:10:56.377832+00:00: multiplier1.0, selected LR0.00125,
fresh seed198630012; H200 CE4.705125006811 across34278 validation targets.
All1200updates/1,228,800training targets and41H200 snapshots completed; independent
checkpoint/optimizer/EMA, full-validation and cache checks passed. Its matched
control slot09 is now running. There is no second paired difference or adoption
conclusion yet; retain every fresh pair and complete the frozen panel.

Peak worker RSS1,558,102,016B and MPS driver2,518,482,944B stayed within4GiB/6GiB;
both guards finished without violations. Free11,279,716,352B exceeds
remaining full-panel requirement10,760,165,048B. No cleanup,
grid/source change or holdout access. Evidence: mla-score-preparation-001/
quality001-slot08-result.json, quality001-slot08-evidence/ and quality001-accepted-08.json.


### Second fresh MLA score pair completed

Slot09 accepted at 2026-11-01T07:16:43.884398+00:00, completing seed198630012 at
selected LR0.00125 for both arms. Multiplier0.5 H200 CE4.722457821258
versus multiplier1.0 CE4.705125006811 gives unit-minus-half
-0.017332814447nat/token across34278 aligned validation targets. Both fresh pairs
so far favor1.0, but there is no adoption or interim stopping decision. Complete
the final pair, all three paired inference reviews and independent frozen-gate
statistics. Each endpoint passed1200updates/1,228,800targets/41H200 snapshots
and independent checkpoint/optimizer/EMA, validation and cache acceptance.

Worker peak RSS1,583,513,600B and driver2,518,482,944B were below4GiB/6GiB;
both guards finished without violations. Free10,710,462,464B exceeds
remaining full-panel requirement10,200,999,952B. Slot10
(multiplier0.5/LR0.00125/seed198630013) has started serially. All fresh raw/H200
pairs remain protected; no cleanup, source/grid change or test access. Evidence:
mla-score-preparation-001/quality001-slot09-result.json, quality001-slot09-evidence/
and quality001-accepted-09.json.


### Final-seed MLA score control accepted

Slot10 accepted at 2026-11-01T07:22:42.657530+00:00: multiplier0.5, selected LR0.00125,
fresh seed198630013; H200 CE4.707110792553 across34278 validation targets.
All1200updates/1,228,800training targets and41H200 snapshots completed, with
independent checkpoint/optimizer/EMA, full-validation and cache audits passed.
The final matched candidate slot11 is now running. There is no third-pair
difference or adoption decision yet. Complete it, then all three paired
inference reviews and independent statistics under the frozen gate.

Worker peak RSS1,551,777,792B and MPS driver2,518,482,944B remained below
4GiB/6GiB; both guards finished without violations. Free10,151,968,768B
exceeds remaining complete-panel requirement9,641,834,856B.
All selected and fresh raw/H200 pairs remain protected; no cleanup, grid/source
change or test access. Evidence: mla-score-preparation-001/quality001-slot10-result.json,
quality001-slot10-evidence/ and quality001-accepted-10.json.


## MLA attention multiplier comparison closed: retain0.5

All twelve1200-update endpoints, three fresh paired inference reviews and the
independent final statistics review passed. Slot11 multiplier1.0 completed the
final fresh pair with H200 CE4.692675884339. Both arms selected LR0.00125 from
the original equal three-rate screens; no grid extension, pilot-loss selection,
seed replacement or holdout evaluation occurred. All449 frozen source entries
were reverified unchanged at closure.

| Fresh seed | Multiplier0.5 CE | Multiplier1.0 CE | Unit minus half |
| --- | ---: | ---: | ---: |
| 198630011 | 4.712354865 | 4.699208080 | -0.013146785 |
| 198630012 | 4.722457821 | 4.705125007 | -0.017332814 |
| 198630013 | 4.707110793 | 4.692675884 | -0.014434908 |

Mean unit-minus-half is **-0.014971502424nat/token**.
Every fresh seed improved and all simultaneous conditional-text upper bounds
are negative. However, the frozen requirement was mean delta strictly below
-0.02; this comparison fails that minimum-improvement gate. Therefore **retain
multiplier0.5 with D256 MLA/SwiGLU672/SiLU/block RMSNorm/H200**. The data support
a small consistent validation improvement from1.0 in these three runs, not
absence of an effect; the prospective practical threshold prevents adoption.
No cost-only route or retrospective threshold change is allowed.

| Text block size | Descriptive95% interval | Simultaneous gate upper bound |
| --- | --- | ---: |
| 2048 | [-0.017749179, -0.011981252] | -0.011724943 |
| 4096 | [-0.016979215, -0.012739207] | -0.012548857 |
| 8192 | [-0.016721147, -0.012374552] | -0.012035767 |

The separate implementation reproduced paired loss arithmetic and2000 common
PCG64 bootstrap draws at2048/4096/8192-token groupings, with linear quantiles
and simultaneous upper quantile1-0.05/3. These are intervals conditional on
reused validation text and these three trained seeds, not population-seed,
independent holdout or universal architecture guarantees.

All three full/cache/reset barriers passed at batches1/8 before timings. Two
warmups and eight alternating repeats per cell were completed under unchanged
guards. Timings are descriptive medians; both H200 models were resident and
model loading/cache allocation were excluded. Mixed timing differences and
identical cache sizes establish no consistent deployment-cost advantage.

| Pair | Batch | Prefill0.5 /1.0 ms | Decode0.5 /1.0 ms | Cache bytes per arm |
| --- | ---: | --- | --- | ---: |
| 1 | 1 | 8.320 / 8.715 | 289.938 / 283.768 | 196,608 |
| 1 | 8 | 14.114 / 13.748 | 298.384 / 298.548 | 1,572,864 |
| 2 | 1 | 8.847 / 9.019 | 291.754 / 292.610 | 196,608 |
| 2 | 8 | 13.667 / 13.826 | 314.810 / 300.039 | 1,572,864 |
| 3 | 1 | 9.067 / 8.519 | 286.192 / 294.305 | 196,608 |
| 3 | 8 | 14.188 / 13.940 | 300.852 / 293.040 | 1,572,864 |

Inference peak RSS was824,164,352B and driver1,155,842,048B. All three guard
reviews finished without violations. Closure preserved both selected screen
pairs and all six fresh raw/H200 trajectories:4,473,320,768 retained weight bytes.
The prior exact four-loser retirement remains final; resolve absent screen
weights through its manifest. Closure observed9,603,719,168B free;
this is not a capacity guarantee for another campaign. No research worker
remains active. All earlier campaigns and failed attempts remain closed.

Evidence in mla-score-preparation-001/: quality001-slot11-result.json/evidence/,
quality001-accepted-11.json, quality001-fresh-inference-plan.json, all three pair
result/review/guard files, quality001-final-review.json/guard and
quality001-closure.json/guard.

At this boundary, both architecture and bounded HP hypotheses were reviewed.
Next prepare per-token affine RMS normalization of the SwiGLU product before
down projection versus none on the retained backbone. This is distinct from
closed block-normalization and FFN-factorization studies. Character results
are a weak prior only.100-update warmup with bounded peak-LR tuning was reviewed
and deferred given weak mixed character evidence and no current stability
mechanism. See TOKENIZER_FFN_NORM_PREPARATION.md and its direction-decision.
No new scientific seeds or training panel are frozen.

The new component001 harness failed before numerical work by parsing Python
source as JSON during hash verification; its source/plan/failure remain closed
and immutable. Separately recorded component002 corrected only this harness
check and passed12 generated CPU FP64/FP32 cases, all forward/gradient comparisons
and per-token causality, with identical shared matrices and672 additional gain
parameters per block. No full-model/MPS/optimizer/resume/quality claim follows.
Next perform narrow explicit model/checkpoint/cache integration outside frozen
namespaces; measure actual retained-output high-water and audit eligible capacity
recovery before any new finite pilot/quality freeze. Small numerical work is
not blocked by a future training-capacity constraint.

## SwiGLU-product RMS implementation prerequisites passed

The explicit additive D256 MLA product-normalization runtime passed both arms'
CPU/MPS full-model, all-gradient, optimizer and request-cache checks. The control
retains bitwise identical initialization/logits; the candidate adds2,688 AdamW
gains with shared weights/groups unchanged. Serialized repeated/resumed model,
optimizer, EMA and RNG states matched exactly, and H200 CPU/MPS/cache inference
passed at batches1/8. These are generated-text implementation checks, not a
scientific quality result. Actual worker/parent validation and inference precede
fresh census and finite pilot/quality plans.

Proactive lossless archival of14 completed successful normalization-prerequisite
weights recovered2,266,198,096B net, with27,046 retained files verified unchanged.
Exact original members remain protected in ffn-norm-prerequisites-001 archives;
use the later manifests, not absent loose paths. Selected/fresh/failure artifacts
were excluded. The capacity recovery enabled serialized tests immediately; a
complete future retained-output projection remains required.

## FFN product-normalization real execution prerequisites complete

Both serial two-update generated workers passed independent parent raw/H200/
optimizer/EMA/RNG audits and exact full-validation recomputation. Trained paired
full/cache/reset inference passed at batches1/8. No scientific quality claim is
available. The finite198740 seed panel is reserved after a43,153-file collision
census; all scientific seeds remain unused and no new pilot/quality plan is frozen.

Audited lossless archival of six completed resume files recovered957,460,848B
net with27,031 survivors unchanged, enabling both workers without reducing guards.
Those checkpoint identities resolve via ffn-norm-preparation-001/resume-archive001.
The prospective full retained panel requires13,555,990,528B free; capacity001's
9,661,427,712B observation leaves3,894,562,816B deficit. Continue finite pilot
source and exact eligible recovery audit before any scientific dispatch; do not
repeat stale measurements as progress or consume a partial panel.


Finite FFN product-normalization pilot orchestration and the complete six-screen/
three-fresh-pair design now pass config, real checkpoint/parent full-validation,
CLI and exact retention/capacity integration checks. No scientific seed consumed
or pilot/quality plan frozen. Lossless archival of224 successful non-selected
worker/numerical weights recovered644,140,089B net; all retained evidence verified.
capacity002 records10,279,190,528B free versus13,555,990,528B required. Source/design
and known eligible recovery are complete; no worker is active. Wait for actual
full-panel capacity or new storage evidence, checking hourly and staying quiet
while unchanged. See TOKENIZER_FFN_NORM_PREPARATION.md and exact later archive
manifests; do not rerun closed prerequisites against intentionally absent paths.


FFN product-normalization resource pilots are complete: both20-step endpoints
passed independent parent checkpoint/full-validation acceptance and paired
full/cache/reset inference. This is resource readiness, not a scientific quality
result. Authorized obsolete numerical archive retirement reclaimed5,443,835,089B;
34,575 retained files verified unchanged. Actual post-pilot free14,676,287,488B
exceeds the complete13,555,990,528B requirement. Next additive finite quality
orchestration and real pilot integration, exact pilot retirement after dependencies
close, then freeze six equal-LR screens and all three fresh confirmation pairs.
See TOKENIZER_FFN_NORM_PREPARATION.md; the storage wait is resolved.


FFN-product normalization quality001 is now frozen and scientific screening has
started.74 integration checks and real pilot payload/full-validation/cache audits
passed; exact successful pilot retirement reclaimed1,118,391,708B with all retained
evidence verified. The finite design remains six equal-LR1,200-step screens then
three fresh pairs, all paired inference and independent final statistics. See
TOKENIZER_FFN_NORM_PREPARATION.md for the exact freeze and capacity accounting.
No adoption or new scientific quality conclusion is claimed at launch.


## FFN product-normalization screening complete

All six1,200-step endpoints passed independent checkpoint, full-validation and
cache acceptance. Both arms selected LR0.00125 using the frozen minimum-CE rule.
Screen H200 CE (none/RMS) was4.9243453/4.8871934 at0.000625,
4.7064604/4.7193965 at0.00125, and4.7702461/4.7424677 at0.0025.
These reused-seed screens select rates, not architecture adoption. The selected
RMS recipe trails none by0.0129361 on this screen; all three fresh pairs remain
required regardless of that direction.

The frozen boundary independently re-audited all six endpoints and retired only
eight weights from the four losing screen pairs, reclaiming2,236,787,128B.
All72 retained files, including both selected raw/H200 pairs (slots02/03),
verified unchanged. Original receipts stay immutable; absent loser paths resolve through
ffn-norm-preparation-001/quality001-screen-retention/. Portable exact screen
metadata is saved in quality001-screen-evidence/. The first fresh confirmation
is dispatched with all guards; all fresh raw/H200 pairs will remain protected.


## FFN product-normalization campaign closed: retain no product norm

All twelve independently accepted1,200-step endpoints, all three paired full/cache/
reset inference reviews, and independent final statistics are complete. Both arms
selected peak LR0.00125 from the same three-rate screen. Fresh H200 CE:

| Fresh seed | No product norm | Product RMSNorm | RMS minus none |
|---|---:|---:|---:|
| 198740011 | 4.714070902 | 4.713468354 | -0.000602548 |
| 198740012 | 4.726630458 | 4.734607213 | +0.007976755 |
| 198740013 | 4.732428258 | 4.726370305 | -0.006057953 |

Mean RMS-minus-none delta is+0.000438751: no quality gain. One of three seeds
regressed, and all simultaneous conditional text upper bounds are positive
(0.005718593/0.006452445/0.006978979 for2048/4096/8192-token blocks).
The mean-gain, every-seed and text-bound gates all fail. Retain no FFN-product
normalization, score0.5/SwiGLU672/SiLU/block RMSNorm/H200 on D256 MLA.

Shared-screen effects changed with LR and did not yield a confirmed improvement
after equal tuning. This does not identify an amplitude/stability mechanism.
Intervals condition on three trained seeds and reused validation text; they do
not establish population-seed uncertainty or independent holdout confidence.
All three trained cache/reset reviews passed at batches1/8. Timing is descriptive
(two resident models; allocation/loading excluded), with no cost-only adoption.

Final review and closure are in ffn-norm-preparation-001/quality001-final-review.json
and quality001-closure.json. All477 frozen sources and retained checkpoints were
rechecked. The run root has terminal STOP; do not rerun its training, boundary,
inference or final-review commands. Selected screens02/03 and all six fresh
raw/H200 pairs remain protected. Original receipts and failed attempts are intact.

At this research boundary, causal local mixing and bounded warmup/peak-LR tuning
were reviewed. Select warmup100 versus none for prerequisites only, with equal
bounded peak-LR tuning; local mixing needs explicit placement/cache-history design
and remains deferred. See TOKENIZER_WARMUP_PREPARATION.md. No new scientific seed
or panel is frozen. The small generated schedule/toy-resume prerequisite passed;
full-model Muon/AdamW/raw/H200/cache and real-worker prerequisites remain.


Warmup prerequisites now pass actual full-model Muon/AdamW group rates, a short
generated repeat/raw-resume/EMA/H200 fixture, two real bounded workers with
independent full-validation acceptance, and paired trained cache/reset inference.
The full1,200-step rate schedule was checked on real optimizer groups; actual
short fixtures used four updates and warmup2, not scientific warmup100 training.
There is no new quality result or scientific seed reservation. The final receipt
serialization failure was preserved and recovered from sealed evidence without
rerunning numerical work. See TOKENIZER_WARMUP_PREPARATION.md and its exact
model-receipt-recovery001/trainer-worker001/trained-inference001 evidence.

Authorized retirement of two obsolete FFN numerical archives reclaimed
1,051,131,606B with35,257 survivors unchanged. The next concrete action is finite
pilot/source and real-payload integration followed by exact eligible recovery,
actual capacity recheck and fresh seed census; capacity001 records a conditional
full-panel recovery path. All selected/fresh/failed weights remain protected.


## Warmup resource pilots completed

Finite pilot001 completed both120-update arms with actual100-update warmup in
the challenger. Each independently accepted endpoint has122,880 targets,
88 EMA tensors and21 H200 snapshots. Paired full/cache/reset inference passed
at batches1/8. Independent final002 resource/artifact review passed. The first
read-only final review omitted two strict environment variables and stopped
before auditing; its source/plan/failure are preserved closed. Final002 used
the required environment without repeating training or inference measurements.
No pilot loss selects a learning rate or supports an adoption claim.

The additive warmup_campaign_v1 source is frozen with487 source dependencies.
Exact retention-boundary001 and actual generated-worker payload/parent
pilot-integration001 passed before dependent worker weights were retired.
Fresh census reserved panel198850: pilot001/evaluation003 used as planned;
screen002, fresh011/012/013 and bootstrap004 remain unused. Only the resource
pilot plan is frozen; no scientific quality plan or worker is active.

Exact obsolete-numerical-cleanup-003 reclaimed4,691,175,838B from eight obsolete
numerical archives and four successful generated-worker weights, verifying
35,434 survivors unchanged. All original receipts and every selected/fresh/failed
artifact remain protected, including all model-resume001 publication-failure
outputs. Retired identities resolve through the later manifest and cannot be
restored. Capacity002 then measured14,070,431,744B versus13,555,990,528B required.
After pilots, final002 measured12,916,207,616B; the1,118,332,368B pilot weights
remain present. Their conditional exact retirement would yield14,034,539,984B.
This is conditional capacity, not permission to dispatch an unfrozen panel.

Next implement minimal additive warmup_quality_v1 orchestration outside all
frozen namespaces: six equal-LR1200-update screens, per-arm minimum full-validation
H200 CE/lower-LR tie selection, then all three fresh pairs, paired inference and
independent statistics. Validate real pilot payload/cache integration, close
pilot tensor dependencies, audit exact successful-pilot retirement, remeasure
complete capacity and verify unused scientific seeds before final quality freeze.
Keep warmup100 versus none, LR0.000625/0.00125/0.0025, cooldown/H200start800,
H200interval10/41snapshots, all frozen gates, confirmations and resource guards.


## Warmup finite quality comparison frozen

Warmup quality001 is frozen after75 integration checks and real accepted pilot
raw/H200/optimizer/RNG/full-validation/cache audits passed. Both arms have88 EMA
tensors; pilot-to-quality impersonation and mutated source bindings were rejected.
All487 predecessor sources are unchanged;497 Python sources are bound by the
additive warmup_quality_v1 contract. The only arm difference is the declared
100-update warmup schedule, with equal bounded peak-LR tuning.

Exact successful-pilot retirement reclaimed1,118,332,368B after tensor dependencies
closed. All27,049 other research files verified unchanged. Original pilot receipts
remain immutable; absent weights resolve through quality001-pilot-retirement's
manifest/journal and portable runtime evidence, and cannot be restored or rerun.
The final freeze measured14,039,228,416B free against13,555,990,528B required,
including all retained output/serialization/metadata allowances and8GiB reserve.

Freeze SHA:9f890ee1e534b966713139c9f4740488573133977b5416e43315e94df9775f94.
The first screen has been dispatched through research.warmup_quality_v1.quality_execution.
Keep exactly six1200-update screens at LR0.000625/0.00125/0.0025, persist each
arm's minimum full-validation H200 CE/lower-LR tie selection, independently
re-audit and retire only four losing screen pairs, then all six fresh trajectories
in frozen paired order, three paired inference reviews and independent statistics.
No grid extension, pilot-loss selection, partial panel, retry or adoption from
incomplete results. H200 start800/interval10 yields41 snapshots per endpoint.
Screen/fresh seeds were unused at freeze; screen198850002 is now in use.
Fresh198850011/012/013 and bootstrap198850004 remain unused at launch.

Read current process/log/acceptance-anchor state before any dispatch. Poll the
owned active worker, never duplicate it. Recheck actual remaining full-panel
capacity for each slot. Preserve selected models, every fresh pair, failures,
immutable sources/receipts and the closed test holdout. No new scientific quality
result or adopted-recipe change is claimed at launch.


Warmup quality001 screening and retention boundary completed. All six1200-update
endpoints passed independent parent acceptance and were re-audited at selection.
Both arms selected peak LR0.00125 (selected slots02/03) by the frozen minimum
full-validation H200 CE/lower-LR tie rule. Exactly four losing raw/H200 pairs
were retired:2,236,662,944B reclaimed,72 retained files unchanged. Original
acceptance receipts remain immutable; absent paths resolve via
quality001-screen-retention manifests/journal. Selected pairs remain protected.
Fresh confirmation has started in the frozen paired order using seeds198850011,
198850012,198850013. Preserve all six fresh trajectories, complete all three
paired inference reviews and independent final statistics before any decision.
No adoption or quality conclusion follows from screening alone.


## Warmup quality001 closed without adoption

All twelve1200-update endpoints passed independent checkpoint/optimizer/EMA,
full-validation and cache acceptance; all three paired inference reviews passed.
Both arms selected peak LR0.00125 with equal search budgets. Fresh H200 CE:

| Seed | No warmup | Warmup100 | Warmup minus none |
|---|---:|---:|---:|
|198850011|4.7279747502|4.7315034892|+0.0035287390|
|198850012|4.7220432808|4.7236056867|+0.0015624060|
|198850013|4.7028434423|4.7054898574|+0.0026464151|

Independent mean delta is+0.0025791866880477015 nats/token. All three seeds
slightly worsen. Simultaneous conditional text-bootstrap upper bounds are
+0.0047709126/+0.0053859092/+0.0046459678 for2048/4096/8192-token blocks.
The mean<-0.02, every-seed-improves and all-upper-bounds-negative gates all fail.
Retain no warmup with the adopted D256 MLA score0.5/SwiGLU672/SiLU/blockRMS/
no product normalization/H200 recipe. No grid extension or retry.

These estimates condition on three trained seeds and reused validation text;
they are not population-seed or test-holdout confidence. The test remains closed.
Warmup changed integrated LR, so this is a selected recipe comparison, not an
isolated causal estimate at equal integrated step size. Descriptive inference
has identical cache sizes and no cost-only adoption route.

Guarded closure wrote terminal STOP, preserving selected screen slots02/03 and
all six fresh raw/H200 pairs (4,473,325,888B total). Deleted loser paths remain
resolved by quality001-screen-retention. All497 frozen sources are unchanged;
no worker remains. Closure free9,484,255,232B is an observation, not readiness
for a new complete training panel. Evidence: quality001-final-review.json,
quality001-closure.json and all three fresh-inference pair reviews.

At the boundary, reviewed causal local mixing against bounded independent-decay
HP tuning. Selected the distinct causal three-tap normalized-FFN-input question;
decay0.00006-versus0 is deferred, not prohibited. The standalone local-mix
component passed six generated CPU FP64/FP32 and MPS FP32 forward/all-gradient,
causality, uneven-cache/reset cases. No scientific seeds, corpus training or
new pilot/quality freeze occurred. Full-model/optimizer/checkpoint/H200/cache
integration remains next; see TOKENIZER_LOCAL_MIX_PREPARATION.md.


Local causal FFN mixing remains an untested quality hypothesis. Full-model CPU/MPS
initialization, optimizer grouping, payload identity and nonzero-lag cache tests
passed. Candidate serialized repeat/resume model/optimizer/EMA/RNG was exact, but
the subsequent H200 export was correctly rejected because the harness supplied
an intermediate rather than final-horizon checkpoint. This failed fixture is
preserved closed. A separate final-horizon worker pair is source-ready, awaiting
measured capacity; no new scientific seeds or quality claims. See
TOKENIZER_LOCAL_MIX_PREPARATION.md and local-mix-preparation-001 for exact evidence.


Local causal FFN mixing final-horizon worker prerequisites now pass both arms:
actual four-update run_training, final raw/H200 serialization, independent parent
full-validation acceptance and paired full/cache/reset inference at batches1/8.
This establishes integration only; no new quality comparison or adoption.
The failed intermediate-export and initial pre-dispatch path attempts remain
closed with original evidence preserved. User-authorized obsolete failed weight
retirement recovered5.89GB while retaining selected models and all fresh pairs.
Next finite pilot design/integration, fresh seed census, actual full-output
capacity and finite guarded freezes. See TOKENIZER_LOCAL_MIX_PREPARATION.md.


Local causal FFN mixing resource pilots are complete: both20-update endpoints
passed independent checkpoint/full-validation acceptance, paired cache/reset
inference and final resource review. This establishes readiness only, with no
quality/adoption claim. Exact retirement of obsolete generated prerequisites
recovered1,118,405,952B while preserving all selected/fresh evidence. Next minimal
additive finite1200-step orchestration, real pilot integration and exact pilot
retirement before full-panel capacity/seed recheck and quality freeze. See
TOKENIZER_LOCAL_MIX_PREPARATION.md for census failure preservation, corrected
entry point,521 frozen sources and reserved198970 seed identities.


Local causal FFN mixing quality001 is now frozen and the first1200-step screen
is running.75 finite checks plus real pilot payload/full-validation/cache audits
passed in additive local_mix_quality_v3; two earlier copied config-check failures
remain preserved without changing the tested model. Exact pilot retirement
reclaimed1,118,415,676B with27,372 retained files verified. The plan binds531
sources, six equal-LR screens and all three fresh confirmation pairs, with
strict original quality gates and closed test. No quality or adoption conclusion
at launch. See TOKENIZER_LOCAL_MIX_PREPARATION.md for the exact freeze and routing.


## Local-mixing screening complete; fresh confirmations started

All six1200-step screens passed independent acceptance. Both arms selected
LR0.00125: none H200 validation CE4.706012847929603, causal3
4.698945099031134 (delta−0.007067748898469). The lower-rate gap−0.07968228
is substantially larger; this screening seed does not establish adoption.
At LR0.0025, none/causal3 CE were4.778780302320959/4.97667545638055.
The frozen three fresh pairs and independent inference/statistics remain required.

Exact boundary re-audited all six endpoints, persisted selection, retired only
four losing raw/H200 pairs (2,236,835,192B), and verified72 retained files.
Selected02/03 remain protected; original receipts are immutable. Absent losing
paths resolve through local-mix-preparation-001/quality001-screen-retention.
Actual post-boundary free13,730,594,816B; slot06 started through the frozen
remaining-full-panel capacity guard. Keep all fresh pairs, closed holdout and
all original guards. No new quality/adoption conclusion. At the next campaign
boundary consider the user concern about repeated LR searches when choosing
a prospective standard-LR comparison versus another equally tuned comparison.


## Local mixing closed without adoption

All twelve endpoints, three paired inference reviews and independent final
statistics are complete. Both arms selected LR0.00125. Fresh none/causal3 CE:

| Seed | None | Causal3 | Candidate minus control |
|---|---:|---:|---:|
|198970011|4.6991971921|4.6884827798|−0.0107144124|
|198970012|4.7075981792|4.7101195406|+0.0025213615|
|198970013|4.7005193711|4.6909489906|−0.0095703805|

Mean delta−0.0059211438 misses the frozen−0.02 threshold; one seed worsens,
and2048/4096-token simultaneous text bounds include zero. All three quality
gates fail. Retain no local mixing with D256 MLA score0.5/SwiGLU672/SiLU/
blockRMS/no product norm/no warmup/H200. The candidate also adds3072 parameters
and8192 cache bytes per item; measured cache is196608 versus204800B. Timing
is descriptive with two resident models, not a deployment or speed win.

Terminal STOP closes training/boundary/inference/final commands. All selected
02/03 and six fresh raw/H200 pairs remain protected. Losing screens resolve
through the exact later retention manifest. Original evidence and all531 frozen
sources are unchanged; holdout remains closed. See local-mix-preparation-001/
quality001-final-review.json and quality001-closure.json for the independent
bootstrap bounds, inference provenance and closure.

At this boundary both parallel-residual topology and independent weight decay
were reviewed. Select the previously deferred0.00006-versus0 decay question at
common peak LR0.00125. Recent repeated LR selection and the user's concern
about repeated screening motivate a prospective standard-recipe comparison
with equal zero LR-search budgets, not a separately optimized best-LR claim.
This does not alter any closed campaign. See TOKENIZER_DECAY_PREPARATION.md.


## Fixed-LR decay comparison closed without adoption

All six fresh endpoints, three passing paired inference reviews and independent
final statistics are complete. Both arms used common peak LR0.00125 with zero
LR-search budgets; no screening or rescue was performed. Fresh H200 CE:

| Seed | Decay0.00006 | Zero decay | Zero minus decay |
|---|---:|---:|---:|
|199080011|4.7141558584|4.7234604882|+0.0093046298|
|199080012|4.7027188086|4.7109193908|+0.0082005822|
|199080013|4.7300105276|4.7400008942|+0.0099903666|

Zero decay worsened all three seeds. Mean delta+0.0091651929 nats/token and all
three frozen gates fail. Conditional95% text intervals are wholly positive at
all declared block sizes;2048/4096/8192 simultaneous upper bounds are
+0.0128724072/+0.0141141413/+0.0158320476. These are conditional reused-text
intervals, not population-seed or holdout confidence. Retain decay0.00006 with
the adopted D256 MLA score0.5/SwiGLU672/SiLU/blockRMS/no product norm/no local
mix/no warmup/H200 recipe. This supports retaining regularization in this fixed
recipe, not a globally optimal decay or independently optimized-LR claim.

All inference correctness/reset checks passed; both arms retain196,608B cache
per item. Timing is descriptive and provides no alternate adoption route.
Terminal STOP closes decay quality training/inference/final commands; all six
fresh raw/H200 pairs and every historical selected/fresh model remain protected.
Original pilot receipts resolve through exact pilot-retirement manifests. No
holdout scoring, closed-fixture rerun or frozen-source edits occurred.

At the boundary both architecture and bounded HP were reviewed. Select parallel
separate-norm residual topology on the subword MLA/blockRMS backbone for
prerequisites only; the negative character-MHA prior remains explicit. Defer
cooldown start600-versus800 as a separate schedule question. Continue prospective
fixed-common-LR comparisons with equal zero search budgets. Read
TOKENIZER_PARALLEL_RESIDUAL_PREPARATION.md; no new seed/training plan is reserved.

## Parallel residual comparison closed without adoption

All six fresh 1,200-update endpoints were independently accepted at common
LR 0.00125 with zero LR-search budgets. Three paired full/cache/reset inference
reviews passed at batches 1/8; both arms use 196,608 bytes of cache per item.
Independent final statistics passed provenance and arithmetic checks.

| Seed | Serial H200 CE | Parallel H200 CE | Parallel minus serial |
|---|---:|---:|---:|
|199190011|4.7176393932|4.7296534829|+0.0120140896|
|199190012|4.7007159542|4.7068419199|+0.0061259657|
|199190013|4.7184464682|4.7267622133|+0.0083157451|

Parallel composition worsened all three seeds. Mean delta is
+0.008818600140528715 nats/token; all three frozen quality gates failed.
Conditional text-bootstrap simultaneous upper bounds at block sizes
2048/4096/8192 are +0.0112559128/+0.0115174044/+0.0117277727. These reused-text
intervals are conditional on the three trained seeds, not population-seed or
holdout confidence. This is a fixed standard-recipe result, not a comparison of
separately optimized optima or a general rejection of parallel architectures.

Retain serial residual composition with D256 MLA score0.5/SwiGLU672/SiLU/blockRMS,
no product norm, no local mixing, no warmup, independent decay0.00006 and H200.
Terminal STOP closes every parallel-residual quality training/boundary/inference/
final command. All six fresh raw/H200 pairs (3,354,992,304 bytes) remain protected,
as do historical selected/fresh models and their lossless RAW archives. All600
frozen sources remain unchanged. No holdout was scored. Closure free space
10,045,964,288 bytes is an observation, not next-panel readiness.

Both architecture and bounded HP were reviewed at the boundary. Select cooldown
start600 versus800 for prerequisites; defer fixed serial residual branch scaling.
The prospective comparison retains common LR0.00125, three fresh pairs, equal
zero LR-search budgets and original strict gates. No new scientific seeds or
pilot/quality plan are reserved. See TOKENIZER_COOLDOWN_PREPARATION.md and
[tokenizer evidence](tokenizer-evidence/parallel-residual-preparation-001/quality001-closure.json).

Cooldown resource preparation has advanced to two independently accepted20-update
pilots, passing paired inference and independent resource review. This is not a
scientific quality result. The next finite comparison remains exactly three fresh
pairs at common peakLR0.00125 with zero LR screening; cooldown600/800 also changes
total LR exposure. Quality001 is now frozen with623 sources; exact pilot retirement
passed and the first fresh trajectory is active. There is no quality result yet. All selected/fresh bytes remain protected, including the lossless
storage-recovery-007 RAW archives. See TOKENIZER_COOLDOWN_PREPARATION.md for the
current source, capacity, seed and next-action evidence.

## Cooldown onset comparison closed without adoption

All six fresh1,200-update endpoints were independently accepted at common peak
LR0.00125 with zero LR-search budgets. All three paired full/cache/reset inference
reviews passed at batches1/8,196,608B cache per item. Independent final statistics
verified both provenance and a separate arithmetic/bootstrap implementation.

| Seed | Cooldown800 H200 CE | Cooldown600 H200 CE | 600 minus800 |
|---|---:|---:|---:|
|199300011|4.7127084386|4.7222150878|+0.0095066492|
|199300012|4.7216895012|4.7284279587|+0.0067384574|
|199300013|4.7107724355|4.7180821944|+0.0073097588|

Earlier cooldown worsened all three seeds. Mean delta is
+0.007851621813797643 nats/token; all three frozen gates failed.
Simultaneous conditional text-bootstrap upper bounds at blocks2048/4096/8192 are
+0.0107926732/+0.0113509813/+0.0114523509. These reused-text intervals condition on
the three trained seeds, not population-seed uncertainty or an independent holdout.
No test split was scored. This is a fixed standard-recipe comparison, not separately
optimized optima. Earlier cooldown also lowers multiplier sum1020 to930; it does
not isolate schedule shape at equal LR exposure. No rescue or grid extension.

Retain cooldown800 with D256 MLA score0.5/SwiGLU672/SiLU/blockRMS/serial residual,
no product norm/local mix/warmup, independent decay0.00006 and H200. Terminal STOP
closes all cooldown quality training/inference/final commands. All six fresh
raw/H200 pairs(3,354,990,192B), all historical selected/fresh models,
and protected storage004/006/007 archives remain intact. All623 sources unchanged.
Closure free9,726,046,208B is an observation, not next-panel readiness.

Both architecture and bounded HP were reviewed. Select fixed0.5 versus unit scaling
of both complete serial attention/FFN branches; defer global gradient clipping0.5
versus incumbent1.0. Negative character0.1-gain results are an adverse prior; no
gain is assumed, old scores/seeds are not pooled and closed campaigns stay closed.
The generated standalone branch-scaling component passed12 CPU FP64/FP32 and MPS
FP32 forward/all-gradient/causality cases, exact unit reference and identical shared
initialization/RNG. This is not full-model integration or a quality result. See
TOKENIZER_BRANCH_SCALE_PREPARATION.md and cooldown quality001-final-review.json.

## Fixed branch scaling: full-model prerequisites, no quality result

Both unit/fixed_half actual MLA models passed shared-initialization/RNG/group checks,
exact unit-incumbent logits and independent complete-branch equations. Both have
28,492,416 parameters/88 EMA tensors/196,608B cache per item. Four-update generated
repeat/resume raw-model/optimizer/EMA/RNG errors were exactly zero. Final-step4 H200
and CPU/MPS batch1/8 full/cache/reset checks passed; all639 bindings unchanged.
Actual run_training and independent parent full validation/inference remain next.
No scientific seed or pilot/quality plan reserved. No adopted-recipe change.
Lossless storage008 preserved six cooldown RAWs, reclaimed653,238,031B and verified
48,501 survivors plus exact restoration. All selected/fresh bytes remain protected.
See TOKENIZER_BRANCH_SCALE_PREPARATION.md for scope and next executable action.

## Fixed branch scaling: actual generated worker prerequisite complete

Both4-update unit/fixed_half worker outputs independently accepted4,096targets,
3H200 snapshots and88EMA tensors. Parent full validation matched exactly; CPU/MPS
CE differences were below3.1e-7. Paired full/cache/reset inference passed batches1/8,
196,608B cache/item both arms. All645 bindings unchanged. No scientific quality
result or recipe change. Next finite20-update resource pilots and three-fresh-pair
fixed-LR design after actual payload integration, census and full storage forecast.
Exact obsolete generated resume retirement reclaimed1,118,314,644B with39,528
survivors unchanged and enabled the complete worker pair; selected/fresh bytes and
all original evidence remain protected. See TOKENIZER_BRANCH_SCALE_PREPARATION.md.


## Fixed branch scaling: finite resource pilots complete, no quality result

Both20-update unit/fixed_half pilots independently accepted20,480targets,8H200
snapshots and88EMA tensors. Full/cache/reset paired inference passed batches1/8,
196,608B cache/item; independent final001 resource review PASS. All651 frozen
campaign sources preserve645 predecessors. Pilot losses are excluded from selection
and adoption. Panel199410 pilot/eval used; three fresh seeds/bootstrap remain unused.
Next additive six-trajectory1200-update quality orchestration with common LR0.00125,
zero LR-search budgets and original strict gates, followed by actual pilot audits,
exact retirement/capacity check and quality freeze. No active worker or recipe change.

Exact obsolete generated-worker retirement reclaimed1,118,319,636B with39,723
survivors unchanged. Lossless storage009 preserved nine historical fresh RAWs,
reclaimed2,033,806,544B and verified48,929 other files plus exact restoration.
Selected/fresh bytes and all H200 files remain protected. Complete capacity passed
before pilot dispatch; final review free11,745,796,096B plus retainedpilots1,118,329,296B
conditionally yields12,864,125,392B, above12,381,585,408B. Fresh launch still requires
new dependency closure, exact pilot audit and actual capacity remeasurement.
See TOKENIZER_BRANCH_SCALE_PREPARATION.md for evidence and next executable action.


## Fixed branch scaling: scientific panel frozen and first trajectory launched

Quality-integration001 PASS21 finite-contract tests and both real pilot payload,
optimizer/EMA/RNG/full-validation/cache audits; pilot-to-quality forgery rejected.
All659 active sources frozen,651 predecessors unchanged. Exact pilot retirement
reclaimed1,118,329,296B with28,203 retained files verified. Freeze actualfree
12,776,476,672B exceeded complete12,381,585,408B including8GiB reserve.

Quality001 is exactly three fresh pairs at common LR0.00125, unit versus fixed-half
complete serial branches, with identical cooldown800/decay/H200 and zero LR-search
budgets. First slot00 unit trajectory dispatched and actual process/config/metric
verified. No paired quality result or adoption yet. Continue the frozen six endpoints,
three paired inference reviews and independent original strict gates; no selection,
rescue, retry or partial panel. Every fresh pair remains protected. See
TOKENIZER_BRANCH_SCALE_PREPARATION.md and quality001-freeze.json for exact identity.


## Branch-scale quality001 closed; retain unit gains

All six fresh endpoints and three paired inference reviews completed. Independent
fixed-half minus unit mean validation CE delta+0.024273102061827024 nat/token;
seed deltas+0.0287286911826099,+0.024305901472923885,+0.019784713529946707.
Every seed worsened and all original gates failed. Terminal STOP closes all branch
quality commands. Preserve all six fresh raw/H200 pairs(3,354,994,032B), historical
selected/fresh models and protected archives; all659 sources remain unchanged.
Retain the incumbent unit-gain recipe. No LR screens/rescue or old score/seed pooling.

Both architecture and bounded HP reviewed: select global norm clipping0.5-versus1.0;
defer matched-epsilon final-readout LayerNorm versus final RMSNorm with pre-branch
norms unchanged. No benefit or actual clipping frequency assumed. Frozen generated
optimizer-component001 passed four CPU FP64/FP32 real Muon/AdamW clipping/order/
serialized-resume cases; no full-model/MPS/EMA/H200/cache or quality claim. Next
narrow additive full-model clipping instrumentation and identity integration,
then real worker/validation/inference, full capacity/census/resource pilots before
any quality freeze. See TOKENIZER_GRADIENT_CLIP_PREPARATION.md.


## Gradient clipping full-model prerequisites and storage010

Clipping1.0/0.5 model002 and serialized model-resume001 passed: identical shared
initialization/logits/RNG/groups, exact direct-PyTorch clipping, CPU/MPS cache/reset/
causality, foreign-threshold rejection and exact repeated/resumed model/optimizer/
EMA/RNG/clipping history(maxerror0). Final-step4 raw/H200 has3snapshots and batch1/8
CPU/MPS inference passes; cache196608B/item. All667 sources unchanged. Model001's
coefficient-oracle tolerance failure remains immutable and closed; separate002
corrected the oracle tolerance consistently with FP32 norm error. No quality claim,
scientific seed reservation or pilot/quality freeze. See TOKENIZER_GRADIENT_CLIP_PREPARATION.md.

Storage010 losslessly archived23 closed quality001/hybrid001 RAW checkpoints,
reclaimed1,105,700,860B and verified49,467 survivors unchanged. Exact restoration
passed. Original raw bytes remain protected via storage-recovery-010/raw-completed.json
and research_archives/protected-raw-storage-010/*.tar.zst, NOT retired. Together
006/007/008/009/010 preserve98 archived RAWs; all H200 original paths and storage004
fixture archive remain intact. Never purge these archives. Four new successful
generated resume weights totaling1,118,316,500B stay retained until actual worker
integration closes dependencies and a new exact retirement audit. Source work
proceeds; recheck full capacity before the worker pair, then separately forecast
the scientific panel. No retired tensor restoration or closed-fixture reruns.


## Gradient clipping generated worker pair complete

Both clip1/cliphalf four-update workers passed actual run_training, bounded raw/H200
publication, independent parent full validation and CPU/MPS CE agreement<3.1e-7.
Each completed4096targets,3H200snapshots,88EMA tensors. Independent acceptance verifies
complete clipping histories across raw/summary/metrics and rejects five malformed
history classes. Trained paired full/cache/reset inference passes batches1/8 with
196608B cache/item. All673 runtime/acceptance sources unchanged; no scientific quality
result, scientific seed reservation or pilot/quality freeze. Next finite20-update
resource-pilot orchestration, full three-fresh-pair design, actual worker dependency
audit, fresh census and measured complete capacity before pilots/final freeze.
Fixed common LR0.00125, zero LR-search, original strict gates; no rescue/selection.

Resume-retirement001 retired exactly four obsolete successful generated weights
(1,118,316,500B) after new real payload/optimizer/EMA/RNG/history/full-validation/cache
audit closed dependencies;40,488 survivors unchanged. Original evidence stays immutable
and later manifest resolves absent weights. Actualfree10,449,551,360B exceeded complete
worker-pair9,814,671,360B including reserve/metadata. New successful worker raw/H200
weights1,118,322,004B remain retained until pilot integration closes dependencies and
a new exact audit. Preserve all selected/fresh bytes,98 protected archived RAWs and
original failure evidence. No permanently retired restoration or closed-fixture rerun.


## Gradient-clipping resource pilot001 complete

Both20-update clip1/cliphalf arms independently passed actual training, bounded
raw/H200 publication, complete clipping-history acceptance, full validation and
paired full/cache/reset inference at batches1/8. Each completed20,480targets,
8H200snapshots and88EMA tensors; cache196608B/item. Independent final001 resource
review passed. All679 campaign sources frozen,673 runtime predecessors unchanged.
No scientific quality result; pilot losses cannot select a threshold.

Exactly three prospective fresh pairs use common LR0.00125 with zero LR-search,
original strict gates and no rescue. Scientific seeds199520011/012/013 and
bootstrap199520004 remain unused; pilot199520001/evaluation199520003 were consumed.
Next new research/gradient_clip_quality_v1 must integrate/audit actual pilot
payloads, complete clipping histories and six-trajectory contracts before exact
pilot retirement, capacity/seed recheck and final quality freeze. Retain clip1.0.

Worker-retirement001 reclaimed1,118,322,004B with40,676 survivors unchanged after
tested pilot integration closed tensor dependencies. Original evidence survives.
Storage011 losslessly archived16 closed MLA-score/wide-attention selected/fresh RAWs,
reclaimed2,653,116,732B and verified49,904 other files unchanged. Exact
restoration passed. Original bytes resolve through storage-recovery-011/raw-completed.json
and protected-raw-storage-011 archives; NOT retired. Together006–011 preserve114
archived RAWs. All H200 originals and storage004 fixture archive remain intact.

Final review free11,838,726,144B plus retained pilots1,118,333,904B conditionally
yields12,957,060,048B versus complete12,381,585,408B including8GiBreserve.
Recheck actual capacity after quality-source dependencies close and an independent
exact pilot audit; no speculative deletion or partial scientific panel.


## Gradient-clipping quality001 frozen and first trajectory dispatched

New research/gradient_clip_quality_v1 passed23 finite-contract tests and actual
pilot raw/H200/optimizer/EMA/RNG/clipping-history/full-validation/cache re-audits;
pilot-to-quality forgery was rejected. All687 active sources frozen,679 predecessors
unchanged. Exactly six1200-update trajectories use three paired fresh seeds at
common LR0.00125, zero LR-search and original strict gates. No screens, selection,
rescue, retry or partial panel. Every fresh raw/H200 pair remains protected.

Complete per-update clipping histories are checked against raw/summary/metric
records and when validating the accepted prefix. Final independent statistics
also report per-pair clipping observations descriptively, never for selection.
Prepared quality001-fresh-inference001.py and quality001-final-review001.py require
the exact freeze and all six accepted endpoints. Freeze SHA:
8574e2c338d9728748492329cda236657e2e8c1f0cbf4b78c926986380302d60

Exact pilot retirement reclaimed1,118,333,904B with28,383 other files
verified unchanged. Original receipts resolve absent weights through
quality001-pilot-retirement/completed.json. Freeze free12,962,930,688B exceeded
complete12,381,585,408B including8GiBreserve. Recheck remaining full-panel capacity
each dispatch. Historical selected/fresh bytes and114 protected RAW archives
remain intact. First fresh slot00 clip1 is dispatched; no paired quality result.
Seeds199520011 now in use;012/013 and bootstrap004 unused at first launch.


## Publication failure closed; zero-update continuation quality002

Original quality001 slot00 completed1200updates,1,228,800targets,41H200snapshots
and both raw/H200 files, but worker.json publication exceeded the256KiB miscellaneous
metadata group by6,295B. Its process receipt publication also failed. STOP and all
partial files/source/log/config/weights remain immutable; no endpoint was accepted
in the original root. The earlier test bounded history alone and missed aggregate
metadata high-water. No training was repeated and no seed was replaced.

Separate publication-recovery001 independently verified the actual final raw/H200,
optimizer/EMA/RNG, all1200 clipping observations, full validation and cache. It
exercised full report publication using existing2MiB resource-audit/resource-events
groups; no guard or budget was raised. Additive v2 integration verified the same
six scientific configs (only output paths differ for remaining slots), forbids
slot00 training, rejects replacement seeds and re-audits actual retained artifacts.

Quality002 frozen SHA d417380ed2581f917936efa85c6caee4ecb24bd8ef680d63864f864342761885
imports slot00 with zero updates and zero weight copies. New691-source plan
includes all687 unchanged quality001 sources. This continues exactly the original
three fresh pairs at LR0.00125 with zero LR-search and original strict gates.
Imported original raw/H200559,261,160B are protected fresh evidence, not obsolete
failed weights. Remaining previously unrun slot01 cliphalf is active; no campaign
quality conclusion. Follow quality002 anchors and the new freeze only.

Complete12,381,585,408B allowance and8GiBreserve are unchanged; existing imported
weights are charged, leaving11,822,324,248B required versus12,381,995,008B measured.
All691 sources remain frozen. Prepared quality002-fresh-inference001.py and
quality002-final-review001.py require all6 accepted endpoints and exact new freeze.
Preserve original failure evidence and all historical protected archives.


## Gradient-clipping quality002 closed without adoption

All six fresh1200-update endpoints and three paired inference reviews passed.
Independent final arithmetic and text bootstrap gave clip0.5-minus-clip1.0 mean
**+0.004133149591824823 nats/token**. All three seeds worsened and all original
quality gates failed. Retain clipping1.0 and the adopted unit-gain serial MLA recipe.

| Seed | Clip1 H200 CE | Clip0.5 H200 CE | Delta | Clipped updates: 1 / 0.5 |
|---|---:|---:|---:|---:|
|199520011|4.6916474635|4.6962045766|+0.0045571131|30 / 966|
|199520012|4.7300379660|4.7329504182|+0.0029124522|23 / 951|
|199520013|4.6964113898|4.7013412733|+0.0049298835|34 / 962|

Counts are descriptive coefficients below one out of1200 updates per trajectory,
not selectors or causal mediation estimates. Simultaneous conditional text-bootstrap
upper bounds at blocks2048/4096/8192 are +0.0058927768/+0.0065718466/+0.0067075439.
These condition on the three trained seeds and reused validation text, not population
seed uncertainty. Closed holdout; fixed standard-recipe scope, zero LR-search, no rescue.

Terminal STOP closes all quality002 training/inference/final commands. Original
quality001 publication failure stays closed and immutable. Slot00 was imported with
zero repeat updates/copies; its original failed-root raw/H200 are protected accepted
fresh artifacts, never obsolete failed payloads. All six pairs total3,355,566,960B;
all691 frozen sources remained unchanged. Closure free9,629,437,952B is an
observation, not new-panel readiness. Preserve114 historical lossless RAW archives,
all H200 originals, selected/fresh bytes and the storage004 fixture archive.

See [independent review](tokenizer-evidence/gradient-clip-preparation-001/quality002-final-review.json)
and [closure](tokenizer-evidence/gradient-clip-preparation-001/quality002-closure.json).


## Readout centering full-model prerequisites complete

Both layer_affine/rms_affine_bias arms passed explicit policy/group/checkpoint/cache
identity, CPU/MPS gradients and full/cache/reset/causality. Shared matrices/RNG/groups
match exactly; incumbent LayerNorm logits are exact, candidate function differs.
Generated four-update repeat/resume raw-model/optimizer/EMA/RNG/clipping history
matched exactly; final-step4 H2003snapshots passed CPU/MPS batches1/8. All710
bindings unchanged. No scientific result/seed reservation or active worker.
Next additive actual worker/output/independent validation/inference with explicit
readout-policy routing, then closure of resume tensor dependencies, exact retirement
audit/capacity recheck. The4 generated weights1,118,317,844B remain retained.
Storage012 saved654,730,043B losslessly from6 closed branch RAWs,50,525 survivors
verified and exact restoration PASS;120 historical RAWs now remain recoverable
in protected archives, NOT retired. All H200 originals intact. Full scientific
panel capacity and census remain separate. Retain LayerNorm/clip1.0 and adopted
recipe; AdamW beta2 .95-versus-.999 deferred. See TOKENIZER_READOUT_CENTERING_PREPARATION.md.


## Readout-centering real worker and paired inference complete

Additive research/readout_centering_worker_v1 explicitly routes the readout policy
through actual run_training, data loading, bounded raw/H200 output and independent
parent full validation. Both generated four-update layer/rms_bias arms passed:
4,096 training targets,3 H200 snapshots and88 EMA tensors each. Parent validation
was exact; CPU/MPS CE differences were2.733889967743153e-7 and2.62410942752922e-7.
Complete clipping histories agree across raw/summary/metrics. Integration001 also
rejected missing/sequence/threshold/coefficient/nonfinite history forgeries and
foreign-readout checkpoints. Paired full/cache/reset inference passed batches1/8;
cache remains196,608B per item in both arms. All717 runtime/acceptance bindings
unchanged,710 predecessors preserved. This is generated-fixture engineering
evidence, not a scientific quality result or seed/pilot/quality reservation.
The prerequisite plan's inherited kind string says generated_gradient_clip_worker_parent_prerequisite;
its explicit readout configs, source hashes, requests and accepted artifacts bind
the actual readout comparison. This descriptive label does not change execution.

Resume-retirement001 independently audited and retired exactly four obsolete
generated resume weights,1,118,317,844B, after real payload/cache integration closed
their dependencies.41,499 retained files verified unchanged; original sources,
configs, logs and receipts remain immutable. Absent weights resolve through the
later retirement manifest; never restore or rerun them. Actual post-recovery
free10,253,819,904B exceeded complete worker-pair9,814,671,360B including8GiBreserve.
Both fresh generated workers then completed. Their new raw/H200 weights total
1,118,323,284B and remain retained until pilot integration closes dependencies
and a new exact audit. Current free space is not scientific-panel readiness.

Next: minimal additive finite20-update resource-pilot orchestration and complete
THREE-fresh-pair fixed-LR design outside717 frozen sources. Re-audit actual worker
payload/cache, test finite contracts, close dependencies, fresh scientific census,
measure complete retained-output high-water including serialization overlap,
metadata and8GiBreserve, and take exact authorized recovery if necessary before
finite pilots and final quality freeze. CommonLR0.00125/equal zero LR-search,
original strict gates/no rescue/selection/partial panel. Retain final LayerNorm
and the adopted recipe; AdamW beta2 .95-versus-.999 remains deferred.
Preserve120 recoverable historical RAW archives, all H200 originals, storage004
and every selected/fresh byte. All earlier campaigns/fixtures remain closed.

Evidence: tokenizer-evidence/readout-centering-preparation-001/trainer-worker001.json,
trained-inference001.json, worker-integration001.json and resume-retirement001/completed.json.


## Readout-centering resource pilots complete

Both20-update layer/rms_bias arms passed independent acceptance, full/cache/reset
inference batches1/8 and independent final003 resource review. All729 source
bindings unchanged; no quality result or active worker. Exactly three fresh pairs
at commonLR0.00125/equal zero LR-search/original strict gates remain prospective.
Next additive quality orchestration, actual pilot re-audit/dependency closure,
exact retirement and capacity/unused-seed recheck before freeze. AdamW beta2 remains
deferred; retain final LayerNorm and adopted recipe. Import-only integration001
failure is closed; separate additive v2/integration002 passed. Worker retirement
and lossless storage013 resolved capacity;147 historical RAWs remain recoverable,
NOT retired, with all H200 originals intact. Details and exact evidence are in
[TOKENIZER_READOUT_CENTERING_PREPARATION.md](TOKENIZER_READOUT_CENTERING_PREPARATION.md).


## Readout-centering quality001 frozen and launched

All737 sources frozen,729 predecessors unchanged;24 finite-contract/publication
tests and actual retained pilot tensor/validation/cache audits passed. Separate
exact pilot retirement reclaimed1,118,335,248B with28,586 survivors verified.
Freeze free12,691,947,520B exceeds complete12,381,585,408B including8GiBreserve.
First fresh slot00 LayerNorm is running under exact freeze`409de49f511b3d94733ff788ef0aec18a26291843b6957e0ed95e539bec9546c`.
No scientific result. Complete exactly three paired1200-update comparisons at common
LR0.00125 with zero LR-search, then all three inference reviews and independent
statistics. Candidate is affine RMS WITH bias, sameepsilon1e-5/512parameters;
eight pre-branch RMSNorms fixed. Retain incumbent LayerNorm and adopted recipe pending
the original strict gates. AdamW beta2 remains deferred. All fresh pairs and147
losslessly archived RAWs stay protected. Current live state and exact next action
are in [TOKENIZER_READOUT_CENTERING_PREPARATION.md](TOKENIZER_READOUT_CENTERING_PREPARATION.md)
and the resume index; current processes/anchors supersede launch snapshots.


Readout quality001 now has two accepted endpoints and slot02 running. A pre-dispatch
capacity shortfall was resolved with storage014: six closed clipping fresh RAWs
preserved losslessly,654,173,784B reclaimed,51,335 survivors unchanged and exact
restoration PASS. No scientific retry or guard change. All153 historical RAW archives,
H200 originals and current readout pairs remain protected. Current next action and
resolver details are in TOKENIZER_READOUT_CENTERING_PREPARATION.md and resume state.


## Readout-centering quality001 closed, not adopted

All six1200-update endpoints were independently accepted, including1,228,800targets,
41H200snapshots,88EMA tensors and complete clipping history per endpoint. All three
paired full/cache/reset inference reviews passed. Independent statistics reproduced
the frozen reduction, bootstrap and gates; all737 sources remained unchanged.

| Fresh seed | LayerNorm H200 CE | RMS with bias H200 CE | RMS minus LayerNorm |
| --- | --- | --- | --- |
|199630011|4.723623907453187|4.722338028546402|-0.0012858789067847274|
|199630012|4.732624464205026|4.732265809794470|-0.00035865441055628946|
|199630013|4.705603105991945|4.706875608092376|+0.0012725021004309565|

Mean delta is-0.00012401040563668678 nats/subword token. Two seeds improve slightly;
one worsens. Mean<-0.02, every-seed improvement and simultaneous conditional-text
upper-bound gates all fail. Upper bounds for2048/4096/8192-token blocks are
+0.0008101435587669836/+0.0008345895539876811/+0.0009522751174451204.
Intervals concern reused validation text, not population seeds or held-out test.
This fixed-common-LR/zero-search comparison does not establish separately tuned
optima. No RMS-with-bias adoption, cost-only route, rescue or partial-panel decision.

Terminal STOP closes all quality training/inference/final commands. Retain final
affine LayerNorm and the adopted recipe. Preserve all six fresh raw/H200 pairs,
3,355,571,184B, all original evidence and153 recoverable historical RAW archives.
Storage014's clipping-import resolver remains protected; permanently retired
numerical tensors cannot be restored. Closure free9,388,134,400B is an observation,
not capacity approval for another panel.

At this boundary both architecture and bounded HP were reviewed: select the
previously deferred AdamW beta2 .95-versus-.999 with beta1 .9/epsilon1e-8/Muon and
outer decay fixed. Defer final LayerNorm learned bias versus fixed zero bias.
No gain assumed or prior score/seed pooling. Read TOKENIZER_ADAMW_BETA2_PREPARATION.md.

Evidence: tokenizer-evidence/readout-centering-preparation-001/quality001-final-review.json,
quality001-closure.json and quality001-fresh-inference-pair0/1/2-review.json.

## Full-model optimizer and resume integration complete

Model003 and model-resume001 passed both beta2 .999/.95 arms. Initial model state,
logits and construction RNG match exactly, with28,492,416parameters/88EMA tensors.
All Muon/AdamW group membership, order and width-scaled rates match; only declared
AdamW beta2 differs. CPU/MPS forward/gradients and full/cache/reset/causality passed.
Foreign configuration and saved group betas are rejected before state loading.

Original model001 launch failed on an absent plan before numerical work; its config
lookup used a nonexistent portable path. Model002 then failed an absolute CPU/MPS
post-update tolerance in the incumbent beta2 .999 arm. Both failures remain closed.
The separate model003 contract uses an exact full-parameter native-child update
oracle on each backend: error0. Absolute CPU/MPS post-update discrepancy is
0.00020608305931091309 for both arms; beta2 adds only2.9802322387695312e-8,
within the preregistered differential bound. Forward/gradient tolerances and resource
guards are unchanged. This does NOT establish absolute backend-update parity.

Each four-update MPS fresh repeat and serialized resume matched raw model,
optimizer, EMA, RNG and complete clipping histories exactly (maxerror0).
Step2 was serialized in capped memory; final-step4 raw/H200 exports have3snapshots.
CPU/MPS inference batches1/8 passed full/cache/reset with196608B cache per item.
All753 source/config bindings unchanged,738 predecessors preserved. This is generated
correctness evidence, not scientific quality, a pilot, or a seed reservation.

Storage015 recovered capacity without retiring any selected/fresh bytes: transparent
lossless compression of1643 historical text files reclaimed259,952,640B and verified
49,952 protected files; archival of16 closed GQA selected/fresh RAWs reclaimed
916,750,821B and verified51,579 other files. Exact restoration passed. Original
copy-only probe stopped before compression on actual capacity; separate002 passed
after text recovery. All169 RAW archives throughstorage006–015 remain protected,
recoverable, NOT retired; all H200 originals and storage004 remain intact.

Next: minimal additive actual run_training/output/worker and independent parent
full-validation/inference adapters using these tested primitives, including beta2
identity and complete clipping histories. Re-audit actual retained resume payloads,
close their tensor dependencies, then take a NEW exact authorized retirement audit
if needed. Four successful generated resume weights total1,118,319,956B and stay
retained meanwhile. Capacity001 free9,222,590,464B versus complete worker-pair
9,814,671,360B conditionally yields10,340,910,420B after dependency closure/audit;
this is a recovery route, not dispatch readiness or a scientific-panel forecast.
Source work proceeds. Then complete scientific census/output high-water/finite
resource pilots before any three-fresh-pair quality freeze. Retain beta2 .999.

Evidence: tokenizer-evidence/adamw-beta2-preparation-001/model003.json,
model-resume001.json, capacity001.json; storage-recovery-015/raw-completed.json,
compression-completed.json and restoration-check.json.

## AdamW beta2 generated worker integration complete

Both fresh generated four-update beta999/beta95 arms passed independent parent
acceptance:4096 training targets,3 H200 snapshots and88 EMA tensors each. Actual
optimizer states contain44 AdamW groups at(.9,.999)/(.9,.95),epsilon1e-8 and zero
child decay, with44 unchanged Muon groups and outer independent decay0.00006.
Full validation recomputed exactly in the parent; CPU/MPS CE differs by less than
2.9e-7. Complete clipping histories agree across raw/summary/metrics. Independent
missing/sequence/threshold/coefficient/nonfinite history, foreign beta2 config/group
and wrong-target-coverage probes were rejected. Trained-inference001 full/cache/reset
passed batches1/8,196608B cache per item. All760 runtime/acceptance bindings unchanged,
753 predecessors preserved. No scientific quality result or seed reservation.

Worker-integration001 first re-audited both real retained resume raw/H200/optimizer,
EMA/RNG/clipping histories/full-validation/cache and closed tensor dependencies.
A separate exact resume-retirement001 audit then reclaimed1,118,319,956B from four
obsolete successful generated weights;42,544 survivors verified unchanged. Original
receipts remain immutable; absent tensors resolve through the retirement manifest,
never restoration or old-fixture reruns. Actual free10,517,405,696B exceeded complete
worker-pair9,814,671,360B including8GiB reserve, enabling both fresh worker arms.
New generated worker raw/H200 weights1,118,325,460B stay retained until pilot
integration re-audits and closes dependencies, followed by a NEW exact cleanup audit.
All169 protected historical RAW archives, H200 originals, selected/fresh bytes,
storage004 and original failures remain intact.

Next: minimal additive finite20-update resource-pilot orchestration and complete
THREE-fresh-pair fixed-LR design outside760 frozen sources. Verify actual worker
payload/cache and finite contracts, close dependencies, fresh scientific census,
measure complete retained-output high-water with serialization overlap/metadata/
8GiB reserve, take exact authorized recovery if needed, then finite pilots and final
quality freeze. CommonLR0.00125/equal zero LR-search/original strict gates/no rescue,
selection or partial panel; beta1 .9/epsilon1e-8/Muon/model/schedule/H200 fixed.
Current free space is not new-panel readiness; source work proceeds and additional
lossless recovery is inventoried if the complete forecast needs it.

Retain incumbent beta2 .999. Final LayerNorm learned bias versus fixed zero is
deferred. Prior model001 missing-plan and model002 absolute CPU/MPS post-update
failures remain closed; model003 exact native updates/differential check is no claim
of absolute backend-update parity. Generated observations do not select quality.

Evidence: tokenizer-evidence/adamw-beta2-preparation-001/trainer-worker001.json,
trained-inference001.json, worker-integration001.json, worker-capacity001.json,
and resume-retirement001/completed.json.

## November2 — AdamW beta2 finite resource pilots complete

Both20-update arms independently passed actual raw/H200/optimizer/EMA/RNG and
complete clipping-history acceptance:20,480targets,8H200snapshots,88EMA tensors.
Actual44AdamW groups differ only in declared beta2(.999/.95);44Muon groups,
beta1 .9,epsilon1e-8 and outerdecay .00006 remain fixed. Paired full/cache/reset
inference batches1/8 passed at196608B/cache item. Independent final001 resource
review passed; all766 campaign sources unchanged,760 runtime predecessors preserved.
No scientific quality result. Pilot losses were not used for selection.

The prospective design fixes exactly three fresh pairs/six1200-update trajectories,
commonLR .00125,equal zero search and original strict gates. Panel199740 pilot/eval
used;fresh011/012/013/bootstrap004 remain unused. Next prepare additive
`research/adamw_beta2_quality_v1`, re-audit retained pilot payloads/cache and close
dependencies before a new exact pilot retirement audit and final quality freeze.
Retained pilot weights:1,118,337,488B. Observed free:11,507,736,576B;
conditional post-retirement:12,626,074,064B
versus complete12,381,585,408B including8GiB reserve. Remeasure before dispatch.

Worker retirement reclaimed1,118,325,460B and verified42,735
survivors. Storage016 losslessly archived13 closed accepted readout/wide RAWs,
reclaimed2,236,324,199B and verified52,040
other files unchanged. Exact restoration passed. All182 archived RAWs remain
protected/recoverable,NOT retired;all H200 originals and storage004 remain intact.
Original worker evidence stays immutable; absent generated tensors resolve through
`adamw-beta2-preparation-001/worker-retirement001/manifest.json`,never restore/rerun.
Archived RAWs resolve through `storage-recovery-016/raw-completed.json` and
`research_archives/protected-raw-storage-016/*.tar.zst`;never purge.

Evidence: `adamw-beta2-preparation-001/pilot-integration001.json`,
`pilot001-plan.json`, `pilot001-review.json`, `pilot001-final001-review.json`,
`campaign-capacity001.json`, `scientific-seed-census-001/result.json`;
`storage-recovery-016/raw-completed.json` and `restoration-check.json`.
Retain beta2 .999 and the adopted recipe; final LayerNorm bias ablation deferred.
