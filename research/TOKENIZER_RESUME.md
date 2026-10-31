# Resume tokenizer-quality research

Start here for the **authorized tokenizer research on `codex/tokenizer-quality`**.
The older character/null-attention handoff remains historical and those campaigns
remain stopped. This document is a checkpoint, not a claim that saved PIDs are
still live. The compact [state index](TOKENIZER_RESUME_STATE.json) pins the exact
supporting evidence; ignored local progress and live processes may be newer.

## Current execution and first action

Campaign closure observed October31,2026 at08:17 Los Angeles: **wide-hybrid005 completed and
closed; hybrid not adopted**. All24 quality trajectories and3 inference seeds
passed their parent reviews. Independent final review completed985.06s under
the unchanged guard; reason null, reviewer and all workers exited.

| Seed | Gated MLA H200 CE | KDA H200 CE | Hybrid H200 CE |
| --- | --- | --- | --- |
|3511|4.845517761609039|4.9743094953134435|4.999353112775229|
|3512|4.84407892743844|4.967920234487431|4.979351837490887|
|3513|4.839211356994297|4.947641727241584|4.976864544264859|

Hybrid mean loss is worse by0.142253816 versus gated MLA and0.021899346 versus
KDA. Strict quality and neutrality both fail against both controls. Actual
hybrid state399,360 bytes at batch1 is21.875% larger than gated MLA327,680 and
18.9024% larger than KDA335,872; all seeds/batch8 agree. Memory route also fails.
Hybrid/gated-MLA median paired prefill ratios are1.815/1.559 at batches1/8;
decode1.012/1.025. Hybrid/KDA ratios are1.026/0.881 prefill and1.793/1.327 decode.
These are local context128, prefill32/decode64 MPS measurements, not production
cost or long-context claims. Existing adopted recipes remain unchanged; no
posthoc promotion of005's gated NoPE MLA control over the RoPE incumbent.

[Final decision](tokenizer-evidence/wide-hybrid-preparation-001/campaign005-final-decision-v2.json),
[independent review](tokenizer-evidence/wide-hybrid-preparation-001/campaign005-final-cached-v2-review.json),
[completed inference metadata](tokenizer-evidence/wide-hybrid-preparation-001/campaign005-inference-panel-accepted-v2.json).
Every selected/fresh pair, failure and archive remains protected. No cleanup.
The first external final-audit wrapper omitted tokenizer cache setup and failed;
its evidence remains preserved. The separate cached read-only audit passed.
No training/inference/closed numerical fixture was repeated.

From the repository root, inspect without mutating or launching anything:

```sh
git status --short --branch
.venv/bin/python research/tokenizer_resume_status.py
ps -axo pid,ppid,etime,rss,command | rg 'python.*(src[./]research|tokenizer|metadata_contention|pytest|run_training)|pytest|zstd'
```

1. Read ignored `research_state/tokenizer-quality-progress.json`; reconcile its
   next_action with actual Git/process state. Current research workers are idle.
   No005 relaunch, slot24, seed extension or changed adoption criterion.
2. Continue [activation preparation](TOKENIZER_MLA_ACTIVATION_PREPARATION.md).
   The MoE panel is deferred after a 3.27 GB conservative capacity gap; its
   original draft remains historical. No protected evidence was removed.
3. Activation campaign001 is frozen and independently verified. All preparation
   gates are complete, including34 final-decision/freeze-entry tests. Exact meta
   model/optimizer accounting confirms15,169,995,168bytes required, including
   every future retained pair and8GiB reserve. Both20-update MPS pilots are accepted; audited pilot cleanup is complete.
4. After committing/pushing pilot evidence, inspect actual workers and execute only
   the first guarded quality step. The immutable plan fixes two20-update pilots,
   six1200-update LR screens and six fresh runs. Accept both pilots and audit
   pilot cleanup before screens; save selection and audit losing-screen cleanup
   before fresh runs. Do not edit frozen source or retry any existing attempt.
   [Freeze review](tokenizer-evidence/activation-preparation-001/campaign001-freeze-review.json),
   [frozen plan](tokenizer-evidence/activation-preparation-001/campaign001-frozen-plan.json).
   Original publication succeeded; its external report writer failed on a string/
   Path mismatch. The procedure/diagnostic is preserved, and independent read-only
   verification passed without republishing or launching anything.
5. Commit/push substantive findings and this guide/index without force. Keep only
   this chat's continuation heartbeat active; the old heartbeat stays paused.

Active campaign root: `research_runs/tokenizer-activation-001`.
Plan SHA256: `508b922253f7eef8ae6fafe05cb393aefe4bb0f58225e8573857e920933016c4`.
Source SHA256: `7a4c02946a31820a7b1ff36b8cc4be42659a4fc4a62424cbc4ea5f32d6cbc487`.

Closed historical campaign root: `research_runs/tokenizer-wide-hybrid-005`.
[Launch receipt](tokenizer-evidence/wide-hybrid-preparation-001/campaign005-launch-v2.json),
[freeze review](tokenizer-evidence/wide-hybrid-preparation-001/campaign005-freeze-v2-review.json),
[frozen plan](tokenizer-evidence/wide-hybrid-preparation-001/campaign005-frozen/meta/plan.json).
Plan SHA256: `7572d9dd09bb4b9c6b064179ed3439ffc90b772e53b9addec6066cd58be06d94`.
Source SHA256: `73325667bc47f62f43d322b3351c3fe65f94ddace930c333be3fb80e4fd1bcfa`.
The completed finite design was3pilots,24quality trajectories and3inference panels, width512,
four layers,1200updates per quality trajectory, batch8/context128. Exact requests,
seeds, conditional choices and adoption rules live in the frozen plan/source.
Prior incomplete campaigns are not pooled; reused screening seeds are not new
independent evidence.

Current continuation: activation001 pilots and audited cleanup complete; first screen next; see the indexed
[FFN direction decision](tokenizer-evidence/moe-preparation-001/direction-decision.json).

## Findings to retain

| Finding | Scope and source |
| --- | --- |
| LR0.0025 transferred better than0.01 | Adopted under paired fresh-seed/width rules at64/256. [Recipe](tokenizer-evidence/quality-001/adopted-recipe.json) |
| Shorter width256 GQA recipe adopted | Combined1200updates/cooldown800/H200-tail800 atLR0.00125; not a general horizon result. [Recipe](tokenizer-evidence/horizon-001/adopted-width256-gqa-recipe.json) |
| Width256 quarter-latent RoPE MLA adopted versus MHA | LR0.00125/1200; separate comparison, not a claim against adoptedGQA. [Recipe](tokenizer-evidence/mla-001/adopted-width256-mla-recipe.json) |
| Width512 quarter-latent MLA adopted versus GQA | Lower validationCE and68.75% smaller cache, but slower decode. [Recipe](tokenizer-evidence/wide-attention-001/adopted-width512-mla-recipe.json) |
| Earlier hybrid and table alternatives not adopted | Preserve negative results; tied-table memory savings did not meet quality neutrality. [Hybrid review](tokenizer-evidence/hybrid-001/final-review.json), [table review](tokenizer-evidence/table-001/final-review.json) |
| Incomplete004 screening preferred hybridD/2 LR0.000625, gatedMLAD/4 LR0.00125, KDAD/2 LR0.000625 | One screening seed; no adoption. HybridD/2 versusD/8 differed by only0.00129135CE. [Selection](tokenizer-evidence/wide-hybrid-preparation-001/campaign004-rank-select-v2-review.json) |
| First004 fresh hybrid cell accepted | Seed3511 H200CE4.999246600352049, rawCE5.029029321452403; one cell is not a paired architecture conclusion. [Review](tokenizer-evidence/wide-hybrid-preparation-001/campaign004-fresh-prefix-16-v2-review.json) |

All claims are local adaptive-validation results on GPT-2-tokenized Shakespeare.
The test holdout remains closed. Do not compare subwordCE to characterCE or claim
general model/seed-population superiority from conditional text intervals.

## Engineering lessons and closed failures

Campaigns001–004 are closed.002/003 had metadata-lock checkpoint-write failures;
partial weights and every diagnostic remain protected. Time-bounded checkpoint
write handoffs improved the scoped synthetic transport test, not a universal
MPS guarantee.004 stopped at slot16 **before training**: startup compared a full
binding containing a legitimately growing `metadata_bytes` census. Three startup
paths now compare stable identity while still enforcing census bounds. Red
regressions reproduced three failures; green validation passed25tests. See the
[census review](tokenizer-evidence/wide-hybrid-preparation-001/confirmation-census-v2-review.json).

Completed original numerical/input evidence was carried forward with explicit
source applicability and lossless transport; it was not rerun. The closed failed
numerical fixture must never be retried. Current source changes require a fresh
contract/review; never patch an active frozen source or resume004 on new code.

## Storage recovery and protected artifacts

The storage gap is cleared. APFS transparent compression recovered
**1,594,634,240 allocated bytes** from174 historical text snapshots; one tiny
snapshot was retained unchanged. All175 original paths and2,037,153,240 readable
bytes remain identical. Checks also preserved33,352 other research files,
including159 selected/fresh/failed tokenizer weights and all four archives.
Observed free space after compression was30,967,259,136 bytes, above the full-panel
requirement30,631,412,484 including the unchanged8GiB reserve. Those values are
historical; new training legitimately uses space. Use stage-aware production
resource checks rather than comparing live free space to the initial total again.

[Compression manifest and explanation](tokenizer-evidence/filesystem-compression-001/README.md),
[completion](tokenizer-evidence/filesystem-compression-001/completed.json).
The operation safely stopped on a no-savings4KB staging copy, preserved its
journal, then finished the remaining files with a separate audited continuation.
It did not retry any research campaign or discard evidence. Standard file reads
transparently decompress; hashes still match. Moving to another filesystem may
require the full uncompressed space.

Earlier compaction retained38 exact checkpoint members in four protected local
archives. Use both [compaction001](tokenizer-evidence/checkpoint-compaction-001/plan.json)
and [compaction002](tokenizer-evidence/checkpoint-compaction-002/plan.json) manifests
and the [logical evidence review](tokenizer-evidence/checkpoint-compaction-002/logical-evidence-review.json).
Original-path-only validators must resolve these manifests or restore exact
bytes into absent paths with adequate capacity. Never rerun a closed experiment
to recreate archived or audited-deleted checkpoints.

Keep selected models, every fresh raw/H200 confirmation pair (including negative
controls), failed/unreviewed weights, immutable evidence and all existing archives.
Proactive cleanup is authorized under the
[retention policy](TOKENIZER_CHECKPOINT_RETENTION.md): explicit allowlists,
original hashes, worker lock, full verification and durable receipts. No blanket
cleanup, corpus removal or unrelated personal-file deletion. No remaining loose
tokenizer weights were eligible at the last archival audit.

## Local evidence versus what Git contains

Git contains source, finite plans, reviewed metrics, compact provenance, portable
metadata, preservation manifests and this handoff. **Git is not a backup of the
trained models.** `research_runs/`, `research_archives/` and `research_state/` are
ignored and remain on the original machine. A fresh clone alone cannot resume
or re-audit current execution; preserve/copy those roots separately, verify their
manifests and reserve enough capacity. Never push binary weights to GitHub.
The virtual environment and cached tokenizer files are local dependencies too.

One local MPS training worker, four CPU threads, no CPU fallback, no paid compute
and no subagents. Retain all time/RSS/MPS/output/metadata/disk guards. Human stop
or pause prevents new dispatch and pauses continuation. Approval-review timeouts
are execution blockers, not missing human authorization.

Detailed reading: [protocol](TOKENIZER_QUALITY_PROTOCOL.md),
[results](TOKENIZER_QUALITY_RESULTS.md),
[wide-hybrid preparation](TOKENIZER_WIDE_HYBRID_PREPARATION.md),
[attention roadmap](TOKENIZER_ATTENTION_ROADMAP.md),
[architecture roadmap](TOKENIZER_ARCHITECTURE_ROADMAP.md).

### Latest continuation: activation session core

49 session/rule tests passed, including real bounded filesystem writers on tiny
synthetic tensors and all14 finite transitions. Complete producer-file pinning,
STOP/failure/orphan handling, saved selection and exact eligible-removal receipts
are covered. The initial44 tests also passed; a post-test reporting-wrapper error
is preserved. Model and transport primitives are unchanged. No research training,
checkpoint cleanup, holdout scoring or closed-fixture rerun occurred.

Read `research/tokenizer-evidence/activation-preparation-001/session-prerequisite-review.json`
and the latest ignored `next_action`. Implement the executable activation entry,
worker and independent scientific acceptance, then wire complete retention
process/manifest/journal checks. `Session` alone has no launch/delete API and its
synthetic test audit verdicts confer no launch authority. Source/runtime/input,
inference and full-capacity freeze remain pending before two MPS pilots. Keep
all14 trajectories, protected pairs and guards; never reopen005 or old fixtures.

### Latest continuation: activation worker and acceptance

Read `research/tokenizer-evidence/activation-preparation-001/worker-prerequisite-review.json`.
76 worker/audit/regression tests and six parent-integration tests passed. Actual
CPU activation optimizer/EMA payloads were tested; MPS/training orchestration
used explicit doubles. No research pilot launched. The worker body connects
bounded writers/guards/evaluation and the independent parent adapter validates
saved evidence before ledger acceptance. Result-envelope compatibility is fixed.

Next implement the executable freeze binder and serial child dispatch, including
source/runtime/input/full-capacity gates, inherited locks, exit/no-descendant
checks and guarded parent review. Wire the complete retention process/manifest/
journal/preservation executor, finish finite inference binding and storage review,
then freeze before two MPS pilots. Do not treat callbacks or synthetic audit
verdicts as launch authority. All protected weights, guards and closed campaigns
remain unchanged; use the latest ignored next_action for exact continuation.


## Activation001 executable commands

Run from the repository with `.venv/bin/python`, `PYTHONPATH=.`,
`TIKTOKEN_CACHE_DIR=/Users/junyi/Documents/ChatGPT/LLM-pretraining-RSI/research_state/tiktoken-cache`,
`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4` and no MPS fallback.
Use the full root and plan hash above. Each invocation holds the shared/campaign
locks; inspect actual processes first. Training entry `step` performs only one
trial or approved selection/retention transition. Later inference entry `step`
performs one accepted seed pair or the final inference summary. The final module
publishes one guarded terminal decision. None automatically retries a failure.

```sh
.venv/bin/python -m src.research.tokenizer_activation_entry step --root /Users/junyi/Documents/ChatGPT/LLM-pretraining-RSI/research_runs/tokenizer-activation-001 --plan-sha256 508b922253f7eef8ae6fafe05cb393aefe4bb0f58225e8573857e920933016c4
```

Only after all12 quality runs and both audited boundaries, use module
`src.research.tokenizer_activation_inference_entry step` with the same root/hash.
After its three accepted pairs and summary, use module
`src.research.tokenizer_activation_final` with `--root` and `--plan-sha256`.
Preserve source and every selected/fresh/failure pair and archive throughout.


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
