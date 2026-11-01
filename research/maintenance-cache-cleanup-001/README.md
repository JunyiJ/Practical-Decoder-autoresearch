# External cache cleanup, October 31, 2026 (Los Angeles)

The human explicitly requested proactive removal of unused caches and temporary
files without impacting RSI. Removed 2,110 exact allowlisted files: 280,385,232
logical bytes and 285,990,912 allocated bytes. Observed free space increased from
13,035,032,576 to 13,320,785,920 bytes (285,753,344 bytes); concurrent filesystem
activity can affect that difference.

Candidates were Homebrew Go build, Bootsnap, pip and API caches; downloaded
Supabase and portable Ruby installers whose installed versions remain present;
and an abandoned dotslash download. All 103 regular files in that archive were
stream-hash verified against the extracted DevTools runtime; archive links were
also compared. The extracted runtime was retained. Future package operations
may regenerate these caches or download installers again.

Each removed file was user-owned, regular, singly linked, reached without symlink
ancestors, absent from the open-file census, and revalidated against its exact
path, inode, device, size, modification time and SHA256. Each unlink has fsynced
intent and completion entries in deletion-journal.jsonl. The final receipt
confirms every allowlisted path is absent. No installed runtime, research
payload, selected model, fresh confirmation, failure artifact or dataset was
removed. No worker was stopped or restarted.

The initial implementation conservatively attempted the shared research lock;
it was busy and exited before any deletion. cleanup-v1.py and manifest-v1.json
preserve that attempt. The applied version uses a separate nonblocking external
cache maintenance lock and low scheduling priority: these caches have no RSI
consumers, so research training need not stop. Research-artifact deletion must
still acquire the shared research lock and follow its retention contract.

The 1.9 GB Sparkle installation cache was excluded: it contains a newer pending
app version and has a live updater. Needed Hugging Face caches, runtime caches,
and research temporary fixtures were also preserved. The scripts are historical
one-shot evidence, not recurring commands to rerun blindly.
