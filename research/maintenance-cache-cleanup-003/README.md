# Verified external cache reclamation — November 2, 2026

Removed 228 unused Clang/Swift compiler module and Homebrew linkage/command
metadata cache files: 128,765,952 allocated bytes (128,287,906 logical bytes).
Observed free-space increase during deletion: 127,270,912 bytes. Concurrent RSI
writes and APFS accounting can differ from the unlinked allocation count.

The exact manifest ranks allocated bytes and records owner, mode, device/inode,
link count, size, timestamps and disposal reason. No symlink or multiply linked
file was eligible. The separate nonblocking maintenance lock, repeated compiler/
package process and open-file checks, immediate stat revalidation and fsynced
intent/result journal protected the deletion. All 228 removals verified; zero
errors. No research payload, installed runtime, active editor cache, updater,
personal file or process was modified. This one-shot script is closed: do not rerun.

The latest capacity observation is in capacity-after.json. Five raw/H200 pairs
had been published by then (four at initial inspection), each inode counted once.
Remaining full-panel requirement was 9,585,271,928 bytes; measured free space
was 10,020,184,064 bytes, leaving 434,912,136 bytes above the mandatory budget.
This retains the full 256 MiB metadata allowance and 8 GiB reserve. Serialization
publication overlap is already inside the frozen 560 MiB per-trajectory cap.
No mandatory deficit remains; preferred extra 2 GiB headroom is short by
1,712,571,512 bytes. No storage escalation is necessary to finish this panel.

Existing resume flags already say capacity resolved/zero deficit. Concurrent
worker resume/progress files were not overwritten; this scoped observation
supersedes older free-space snapshots. Research owner should accept the current
slot independently and recheck complete remaining capacity before the next slot.
No experiment was dispatched by maintenance. The inventory and wake conditions
are recorded in capacity-after.json, including protected and exhausted classes.

The existing cleanup automation was found PAUSED. Its requested interval and
standing authorization were updated to 20 minutes while preserving PAUSED;
there will be no recurring checks until the user explicitly resumes it.
