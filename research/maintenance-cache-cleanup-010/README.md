# Superseded Chrome framework retirement 010

Removed only Chrome framework `156.0.8078.79` (759,910,400 allocated bytes).
The running `.49` and installed Current `.80` versions were retained: 1,199
entries, including all regular-file SHA-256 hashes, were independently verified
unchanged. The top-level Info.plist is `.80`; `.79` had no open consumer or updater.
Browser profiles, settings, credentials and sessions were outside the allowlist.

The exact manifest records paths, size/allocation, ownership, device/inode,
hardlink counts, symlinks and hashes. Regular files had one link. The nonblocking
external maintenance lock, immediate revalidation and fsynced per-entry deletion
journal covered the operation. All 599 entries were removed.

The original script finished deletion and retained verification, then its final
summary serialization failed with a duplicate `at` keyword. Deletion was NOT
rerun. `verify.py` separately matched every removed path to the journal and
reverified every retained entry, recording `completed.json`.

Free space increased 766,017,536 bytes between deletion start and independent
verification; APFS accounting/concurrent activity prevents equating this with
file allocation. This is a closed one-shot operation; never rerun it.
