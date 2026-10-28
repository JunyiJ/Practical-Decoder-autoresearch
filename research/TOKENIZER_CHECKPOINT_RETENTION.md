# Tokenizer checkpoint retention

The human authorized proactive cleanup of non-critical checkpoints on October 28,
2026 (Los Angeles): “feel free to clean up storage with non-critical check point
moving forward! instead of silently blocked by it”. This supersedes earlier
blanket “retain all checkpoints / no further deletion / external storage only”
instructions for reviewed non-critical weights. It does not reopen stopped
campaigns or permit silent retries, changes to quality rules, or holdout use.

## Protected evidence

Keep selected final recipe checkpoints and every fresh confirmation raw/H200
pair, including comparison controls and negative results. Preserve active runs,
unreviewed or failed checkpoints, immutable requests/configs, metrics and full
validation block sums, data/tokenizer identities, logs, original hashes,
selection decisions, tensor/EMA/cache audits, resource observations and stop
markers. Never delete corpus files or unrelated personal files as research cleanup.
No automatic removal of protected weights to make a panel fit.

## Non-critical weights

Successful resource-pilot weights become eligible after the complete pilot
review and dependent quality prerequisites have been recorded. Losing screening
weights become eligible only after the stage completes, all required audits pass,
and the prospective selection rule has been applied and saved. A winning LR
anchor remains protected through the rank comparison; it may become a losing
screen after the final rank selection. Never select a different configuration
for storage convenience. Keep the final selected screen pairs.

Before deletion, hold the shared worker lock, check actual processes, create an
explicit path/size/SHA256/reason allowlist, and verify hashes against saved results
and applicable audits. Exclude symlinks, unknown files, active/pending/failure
artifacts and all fresh confirmations. Journal each removal, then verify all
protected checkpoints and non-checkpoint evidence are unchanged. Commit/push the
text manifest and report reclaimed logical bytes and observed free space. No
blanket recursive cleanup; no binary weights pushed to GitHub. Deleted weights
cannot be re-audited or re-evaluated; retain this limitation with their receipts.

## Storage-aware future experiments

For unfrozen rounds, prospectively plan audited cleanup at stage boundaries and
reserve the complete campaign's maximum simultaneous retained bytes, not the sum
of all weights ever generated. Include selected screens, all fresh confirmations,
raw/optimizer/EMA/export state, temporary export/audit overlap, metadata/logs,
allowances and the unchanged 8 GiB free reserve. Keep one worker, existing memory
and time limits, full trial budgets and closed test. Cleanup failure blocks the
next stage; no speculative reclaimed space or partial-panel launch.

This requires tested retention-aware controllers. A deleted-file receipt can
explain intentional checkpoint absence only after completed audits; it must not
bypass artifact integrity, numerical prerequisites or statistical review. Old
closed controllers remain closed. Current wide-hybrid v1 retain-all guards are
not bypassed. Before freeze, implement versioned retention contracts, test
boundary cleanup and preserved evidence, recheck current disk and audit the full
high-water projection. If protected evidence prevents a fit after eligible
cleanup, explicitly report the remaining constraint and the concrete options;
do not quietly repeat unchanged storage checks indefinitely.

The initial application is documented in
[tokenizer-evidence/checkpoint-cleanup-002/README.md](tokenizer-evidence/checkpoint-cleanup-002/README.md).
