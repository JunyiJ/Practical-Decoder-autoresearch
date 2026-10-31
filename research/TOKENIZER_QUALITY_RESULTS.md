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
