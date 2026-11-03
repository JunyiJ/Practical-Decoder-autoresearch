# Immediate cleanup pass — November 2, 2026

Reclaimed 119,410,688 allocated bytes (113.88 MiB) from 51 verified disposable
files. Summed free-space increases during the two deletion batches were
118,947,840 bytes; concurrent system writes explain differences from allocation.

The first batch removed 38 unused Chrome/Codex component download archives
(101,777,408 allocated bytes). Every archive member matched the corresponding
installed file by SHA256; all 175 installed files reverified unchanged afterward.
Four archives lacking unique installed-copy proof were excluded. Download cache
metadata, profiles, sessions, credentials and installed components were untouched.
The second batch removed 13 unused CoreSimulator Metal and Xcode SDK-stat cache
files (17,633,280 allocated bytes). Xcode unsaved-document data and research Python
Metal caches were excluded. No compiler/simulator process or open consumer existed.

Both batches used the separate nonblocking external-maintenance lock, exact
owner/link/inode/size/hash manifests, symlink rejection, immediate revalidation,
live open-handle checks and fsynced deletion intent/result journals. All deletions
verified, with zero deletion errors. An initial compiler-audit helper import
failed at exclusive manifest creation without overwriting/deleting anything; the
isolated compiler audit and deletion then succeeded. These one-shot scripts are
closed and must not be rerun. No research payload or installed software changed.

Latest measured free space: 9,581,559,808 bytes (8.92 GiB). The epsilon panel has
closed and adopted epsilon1e-6; final-LayerNorm-scale integration is now in progress.
There is no measured/frozen next-panel budget yet. With no active training panel,
the unchanged 8 GiB reserve plus requested 2 GiB headroom requires 10 GiB free:
this pass remains 1,155,858,432 bytes short. A new six-trajectory panel using the
predecessor's measured allowances would need 12,381,585,408 bytes, or
14,529,069,056 bytes with the extra 2 GiB, leaving a 4,947,509,248-byte shortfall.
That is a planning scenario, not new-panel readiness: new integration/pilot costs
must still be measured. Existing closed-panel weights cannot be credited as new
outputs. The predecessor formula includes publication overlap, 256 MiB metadata
and 8 GiB reserve; no requirement was reduced.

Additional targeted classes were inspected: package/build caches, Xcode,
installer downloads, application component caches, and size-based research
payload/archives including .pt.tmp. The 197 large archives map to original
selected/fresh acceptance records; the recorded raw/H200 hashes contain no
identical groups, and no incomplete .pt.tmp remains. No research deletion was
established as eligible. Both pending newer iTerm3.7.6 (installed3.6.11) and the
live Codex updater remain protected. Installed runtimes, active VS Code caches,
user data and root-owned npm content were retained without privilege/permission
changes. Full inventory conclusions and wake conditions are in capacity-after.json.

No stale capacity blocker needed clearing; concurrent resume/progress and new
final-scale source edits were preserved. Next: continue source/small numerical
integration, then measure complete retained-output capacity before freezing a
panel. To provide the extra space, supply an external APFS volume with at least
6 GiB available for a separately verified relocation of closed archives while
preserving their bytes and recoverability. No archive was moved in this pass.
The existing active 20-minute schedule is unchanged; skip these exhausted classes
until new disposable files, dependency closure, a measured budget or an external
storage destination provides a concrete next action.
