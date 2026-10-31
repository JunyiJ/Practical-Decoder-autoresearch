# Resume tokenizer-quality research

Start here for the **authorized tokenizer research on `codex/tokenizer-quality`**.
The older character/null-attention handoff remains historical and those campaigns
remain stopped. This document is a checkpoint, not a claim that saved PIDs are
still live. The compact [state index](TOKENIZER_RESUME_STATE.json) pins the exact
supporting evidence; ignored local progress and live processes may be newer.

## Current execution and first action

Observed October31,2026 at07:57 Los Angeles: all24 quality trajectories,
including all nine fresh confirmations, are accepted and independently reviewed.
H200 validation CE:

| Seed | Gated MLA | KDA | Hybrid |
| --- | --- | --- | --- |
|3511|4.845517761609039|4.9743094953134435|4.999353112775229|
|3512|4.84407892743844|4.967920234487431|4.979351837490887|
|3513|4.839211356994297|4.947641727241584|4.976864544264859|

Gated MLA has the lowest H200 CE on every fresh seed. No adoption yet: frozen
statistical and inference gates remain required. Final history replay passed
219.31s, verifying12 selected/fresh pairs and all three cleanup boundaries.
Free15,577,501,696 bytes exceeds remaining8,858,370,048 required, including8GiB
reserve. Every fresh raw/H200 pair remains protected. No quality slot24.

All three inference seeds are parent accepted; inference controller38659 and all
children have exited. Final seed3513 passed preparation930.57s, child147.01s and
parent review1090.28s. The serial panel finished5248.11s, with no guard violation.

Independent final reviewer **44715**, session **5954**, is running under both
locks and an outer1800s CPU guard. Never duplicate this review. Require its
`campaign005-final-v2-review.json`, finished guard with no reason, and actual
reviewer exit before interpreting the final frozen quality/cost decision.

From the repository root, inspect without mutating or launching anything:

```sh
git status --short --branch
.venv/bin/python research/tokenizer_resume_status.py
ps -axo pid,ppid,etime,rss,command | rg 'python.*(src[./]research|tokenizer|metadata_contention|pytest|run_training)|pytest|zstd'
```

The status script verifies immutable evidence, shows local progress, process
records, preparation and all inference review/controller receipts. Actual PIDs
must be inspected; saved records can be stale. Process inspection is already
human-authorized. Leave unrelated programs alone.

1. Reconcile ignored `research_state/tokenizer-quality-progress.json` with actual
   processes. While reviewer44715 runs, observe its external guard; do not acquire
   its locks or duplicate the review. Inference is complete, never relaunch it.
2. Require `campaign005-final-v2-review.json` status
   `complete_independent_quality_cost_review`, a finished
   `campaign005-final-v2-guard.json` with reason null, and actual reviewer exit.
   A review file alone does not prove its enclosing guard finished successfully.
3. Interpret the unchanged frozen rule: strict quality OR quality neutrality
   plus measured memory savings. Disclose latency, raw endpoints and conditional
   adaptive-validation limits. Save the final decision; the procedure itself
   returns eligibility with adopted=false, not automatic adoption.
4. Preserve every selected/fresh raw/H200 pair, failure and archive. Do not rerun
   closed campaigns or the failed numerical fixture. No quality slot24, cleanup,
   production changes or holdout use. A review failure must be preserved and
   diagnosed, not silently retried.
5. After final review, explicitly evaluate the architecture roadmap's next
   untested family (sparse versus dense FFN priority) alongside bounded HP
   alternatives. Freeze prerequisites and a finite plan before new training.
6. Commit/push substantive evidence and this guide/index without force. Keep
   only this chat's continuation heartbeat active; the old heartbeat is paused.

[All fresh results](tokenizer-evidence/wide-hybrid-preparation-001/campaign005-fresh-panel-accepted-v2.json),
[final history review](tokenizer-evidence/wide-hybrid-preparation-001/campaign005-fresh-prefix-24-v2-review.json),
[inference wrapper review](tokenizer-evidence/wide-hybrid-preparation-001/campaign005-inference-wrapper-review-v2.json),
[inference launch](tokenizer-evidence/wide-hybrid-preparation-001/campaign005-inference-v2-launch.json).
Selected screens remain slots14/5/12; audited rank cleanup is complete.

Campaign root: `research_runs/tokenizer-wide-hybrid-005`.
[Launch receipt](tokenizer-evidence/wide-hybrid-preparation-001/campaign005-launch-v2.json),
[freeze review](tokenizer-evidence/wide-hybrid-preparation-001/campaign005-freeze-v2-review.json),
[frozen plan](tokenizer-evidence/wide-hybrid-preparation-001/campaign005-frozen/meta/plan.json).
Plan SHA256: `7572d9dd09bb4b9c6b064179ed3439ffc90b772e53b9addec6066cd58be06d94`.
Source SHA256: `73325667bc47f62f43d322b3351c3fe65f94ddace930c333be3fb80e4fd1bcfa`.
The finite design is3pilots,24quality trajectories and3inference panels, width512,
four layers,1200updates per quality trajectory, batch8/context128. Exact requests,
seeds, conditional choices and adoption rules live in the frozen plan/source.
Prior incomplete campaigns are not pooled; reused screening seeds are not new
independent evidence.

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
