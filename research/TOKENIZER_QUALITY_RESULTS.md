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
