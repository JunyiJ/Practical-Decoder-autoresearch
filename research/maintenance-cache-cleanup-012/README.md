# Unused compiler caches and final capacity reconciliation 012

Removed 55,259,136 allocated bytes from eight Python Metal compiled-function /
compiled-library cache files and two Go build-cache metadata files. No Python or
Go consumer was running, no target had an open handle, and both external and
research worker locks were acquired nonblockingly. Exact path/hash/ownership/
inode/link manifests and fsynced per-file journals bind the deletion. No research
payload, source, installed environment or app data changed. Free-space gain during
this batch was 55,181,312 bytes. Closed one-shot operation; never rerun.

Batches 010–012 removed 823,345,152 allocated bytes in total. Observed free-space
changes also include unrelated activity and APFS accounting; the whole-session
increase must not be attributed to these deletions.

The entire prepared six-trajectory panel still requires
`6 * (469762048 RAW + 117440512 H200) + 268435456 metadata + 8589934592 reserve`
= 12,381,585,408 bytes. Capped serialization and same-inode publication avoid a
second full-copy charge; existing hardlinks are not counted twice. No future
trajectory is already retained, and all pilot-retirement credits are exhausted.
The human-required extra 2 GiB makes the full cleanup target 14,529,069,056 bytes.
See `capacity-reconciliation.json` for the fresh measurement and exact deficit.

## Remaining classes and blockers

- Homebrew downloads and pip have zero allocated cache bytes; uv, npx, Go module
  downloads, Cargo archives and Gradle cache roots are absent. Go metadata and
  TypeScript cache are now cleared. Existing conda packages remain linked to
  installed environments; prior supported dry-run found no unused packages.
- npm has 77,713,408 bytes in 71 root-owned files under non-writable directories.
  Normal administrator access check returned `sudo: a password is required`.
  Keynote is an authorized unused installation but retains its administrator /
  macOS access blocker. No permissions or ownership were changed.
- Codex Sparkle updater PID 20017 remains active; its installation payload is
  protected. iTerm 3.7.6 remains a pending update of installed 3.6.11. VS Code's
  translocated installation is running; Downloads contains no new eligible
  installer. Shared runtimes, app data and personal downloads remain protected.
- Chrome's remaining Widevine/geolocation downloads do not have fully matching
  installed payload evidence. Running-browser caches were not manually purged;
  the supported browser-control path was unavailable (`chrome` browser not
  available). Other system/app caches lack concrete unused-consumer evidence.
- OrbStack still has two active containers. Previously authorized unused Docker
  images were removed in 008; remaining images/runtime dependencies and volume
  data lack new unused proof. No container or service was stopped.
- Temporary storage is dominated by the active VS Code translocation; Python's
  unused compiler cache was the newly verified removable class. No new abandoned
  installer operation was identified.
- Research state/dependencies are unchanged: selected/fresh historical evidence
  stays protected; prior concluded-pilot/failed-write retirements are complete.
  No new dependency closure justifies another payload deletion or broad rehash.

Capacity remains blocked on the complete target, not declared successful merely
because the panel's base budget fits. One concrete external option is a
human-provided APFS volume with at least 6 GiB available for a separately audited,
verified archive relocation. No destination is currently mounted or authorized.
Wake on added capacity, a new disposable candidate, dependency closure, changed
requirement, or resolved administrator/UI access. Preserve the existing schedule;
do not repeat these closed scripts, unchanged broad scans or notifications.
