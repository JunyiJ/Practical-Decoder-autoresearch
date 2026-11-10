# Human-requested safe storage cleanup — 2026-11-09

The user explicitly requested maximal safe cleanup of unused caches and obsolete
checkpoints, without interrupting research or deleting important models/results.

## Completed now

`cache-completed.json`: 2,570 eligible cache objects removed, 48,517,120 allocated
bytes (46.27 MiB). Free space immediately after the pass was 11,588,182,016 bytes;
normal concurrent writes mean this is a point-in-time measurement. All 27 selected
and fresh research/source hashes checked before and after remained unchanged.

Only unopened, self-identifying Chromium SimpleCache payloads and eligible npm
content-addressed objects were considered, under the independent external-cache
lock. A preflight first encountered a root-owned npm object and stopped before
any deletion. The corrected eligibility filter excludes it; there was no sudo,
ownership change, permission bypass or relaxation of the single-link safeguard.
The exact failure, manifest and durable per-file journal are retained here.

Inventory also found roughly 1.9 GiB of Codex updater cache, 0.25 GiB of iTerm
updater cache, 1.6 GiB of Codex shared runtime and 0.25 GiB of dotslash assets.
These are not interchangeable with disposable download caches and were left
alone. OS speech assets, application data, settings, credentials, sessions and
active Python Metal compiler caches were preserved. Homebrew/pip caches were
already almost empty. No application was uninstalled.

## Audited checkpoint retirement, waiting for safe execution

`checkpoint-manifest.json`: 66 obsolete LR-screen payloads, **17,519,001,600
allocated bytes (16.32 GiB)**, across four completed, non-adopted studies:

- MoE: screen slots 00–08; all fresh slots 09–17 retained.
- MLA gating: screen slots 00–08; all fresh slots 09–17 retained.
- Readout mixtures: screen slots 00–08; all fresh slots 09–17 retained.
- GPT-2 vocabulary prefix: screen slots 00–05; all fresh slots 06–11 retained.

Original accepted records verify every candidate's exact hash. Plans identify
the screen/fresh boundary, terminal STOP and closure records establish completed
studies without adoption, and active/future research plans do not consume these
study payloads. All 1,034 protected metadata/selected/fresh hashes in this audit
are preserved. Original source/config/log/metric/acceptance/statistics/closure
records are never rewritten. New live endpoints are added to protection again
at execution time.

This is a narrow, explicitly recorded application of current human authorization
over the older blanket keep-all-screen plans. It does not retire fresh
confirmations, selected models, failed numerical diagnostic payloads, remaining
selected body/dropout screens or active general-corpus checkpoints. It does not
reopen or rerun any closed experiment.

Research worker PID 15837 was active and held the shared lock during this turn.
As required by AGENTS.md, the checkpoint payloads have **not** been removed yet.
A one-time cleanup waiter is running in managed execution session **60678**;
its PID/deadline are in `checkpoint-worker.json`. It polls the nonblocking shared
research lock every 20 seconds, waits for the recorded research worker to exit
normally, rechecks consumers/dependencies/all hashes, and performs one exact
journaled pass. It never stops a process. It exits after completion or 12 hours
without a safe gap. It is not a recurring automation.

Completion will appear as `checkpoint-completed.json` and the corresponding
deletion journal in this directory. If there is no safe gap, it writes
`checkpoint-timeout.json` and removes nothing. Any exception is retained in
`research_runs/storage-cleanup-2026-11-09.log`; no success is implied by a pending
worker record. The earlier shell-background launch did not leave a live process;
the managed execution session was then started and verified instead.

After the queued pass, expected free space is approximately 27 GiB if other disk
usage remains stable. This is a forecast, not reclaimed space already measured.
The completed receipt will measure actual free space and the then-current
remaining research requirement. Research state is not overwritten while another
chat owns and advances it.
