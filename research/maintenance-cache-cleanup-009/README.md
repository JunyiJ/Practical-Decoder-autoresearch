# Completed Chrome component download retirement 009

Removed one newly downloaded Crowd Deny 2026.11.5.146 CRX after all three
archive files matched the installed component byte for byte. The separate
nonblocking maintenance lock, exact ownership/inode/link/hash manifest, two
open-handle checks, immediate revalidation and fsynced deletion journal were
used. Three installed files were verified unchanged afterward. Cache metadata,
browser profiles and installed software were preserved. This operation is closed;
never rerun its deletion script.

Removed allocation: **294,912 bytes**. Observed filesystem free-space increase
during deletion: **237,568 bytes**. Concurrent writes and APFS accounting explain
why these measurements need not agree; no specific cause was established.

The full remaining panel has six retained RAW/H200 pairs, each capped at
469,762,048 + 117,440,512 bytes, plus 268,435,456 metadata and 8,589,934,592
reserve bytes. Capped streaming serialization and same-inode publication require
no second full payload copy; overlap is covered in these allowances. The full
requirement is 12,381,585,408 bytes, or **14,529,069,056 bytes** with the requested
2 GiB headroom. No fresh trajectory exists to credit, and no pilot-retirement
credit remains. Final free space was **11,639,136,256 bytes**, leaving
**2,889,932,800 bytes** short of the cleanup target. This is partial reclamation,
not completion.

Bounded change checks found no other new package/download cache candidates.
Homebrew/pip remain empty; uv/npx/Go module downloads/Cargo/Gradle cache roots
have no new eligible entries. Previously audited root-owned npm and Go cache
access blockers remain; permissions were not changed. The Codex updater is
still running and OrbStack still serves active containers. Prior installed-app,
shared-runtime, research retention and archive classifications have no new
dependency closure; no broad unchanged audit was repeated. The current research
resume record reports all prior authorized research retirement and metadata
recovery exhausted, with selected/fresh evidence preserved.

Only Macintosh HD is mounted. A human-provided external APFS volume with at
least 6 GiB available remains a concrete option for separately verified archive
relocation. Wake on a new eligible candidate, dependency closure, cleared access
blocker, changed requirement or added capacity. The existing schedule is unchanged.
