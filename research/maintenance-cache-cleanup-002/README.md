# Bounded external cache cleanup

The user's continuing authorization covers unused caches without disrupting RSI.
An initial npm cache review found all 71 files owned by root; those files were
excluded without changing ownership or permissions. Installed Gradle runtimes,
npx environments, pending app updates and research artifacts remain protected.

The Homebrew Go module cache had no live package/build consumers or open handles.
An exact path/size/SHA256/inode manifest covered 102 user-owned regular files.
The low-priority cleanup used the independent maintenance lock and fsynced each
deletion intent/result. It removed 12 cached module download files, then stopped
with PermissionError at the first expanded module source file in a read-only
directory. No permissions were changed and the failed deletion was not retried.

The receipt records the exact reclaimed bytes. All 12 removals were verified;
all 90 retained files were hash-verified unchanged afterward. The partial cleanup
is closed. Do not rerun this one-shot script or infer complete module-cache
removal. There was no research payload deletion or process interruption, and
future Go builds may redownload the discarded cache entries.
