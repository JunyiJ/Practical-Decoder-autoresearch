# Completed downloads and type-acquisition cache retirement 011

Removed 8,175,616 allocated bytes: two Chrome component download archives
(Subresource Filtering Rules 9.73.1 and Optimization Hints 773), plus the unused
TypeScript 5.9 automatic type-acquisition cache. Every archive member matched
its retained installed file byte for byte; seven installed files were verified
unchanged after deletion. Type declarations/registry downloads were identified
by their npm package lock and had no tsserver/typingsInstaller consumer.

The first validation stopped before any manifest/deletion because the geolocation
archive contained a model.crx3 absent at the corresponding installed path. That
candidate was excluded and preserved. Widevine also lacks a matching installation
and was preserved. No unresolved download was assumed disposable.

Exact hashed manifests, ownership/inode/link checks, two consumer checks,
nonblocking external lock and fsynced deletion intents covered the successful
batch. Empty directories and all browser data/settings were retained. Observed
free-space gain was 7,905,280 bytes, subject to concurrent/APFS effects.
Closed one-shot operation; never rerun.
