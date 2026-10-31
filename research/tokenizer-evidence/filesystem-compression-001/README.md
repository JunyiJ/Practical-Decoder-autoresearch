# Transparent compression of historical text

This narrowly scoped operation preserves all 175 historical Markdown knowledge
snapshots at their original paths with identical bytes. It changes only their
APFS storage representation. It never modifies checkpoint weights or archives.

A disposable-copy probe found that the sandbox could not establish the required
compressed flag: the copy was unreadable and verification rejected it before
any original replacement. Repeating the copy-only probe with filesystem metadata
access produced an identical 14,502,094-byte readable file occupying 3,321,856
bytes (previously 14,503,936 allocated). The invalid disposable copy is not used.
The operation requires this filesystem access and verifies every staged copy.

The plan pins every candidate and an inventory of other research run/state,
archive, corpus and source files. Before replacement it verifies both inventories;
after replacement it verifies all content and protected physical metadata again.
The original path, SHA256, size, permissions, mtime and ownership must match.
A durable journal records verification before atomic replacement and afterward.
The complete 8 GiB reserve and research guards remain unchanged.

Completed: 174 snapshots compressed; one 4,011-byte snapshot retained unchanged
because its staged copy saved no allocation. The initial application stopped
before replacing that file. Its diagnostic and 122 successful replacements remain
in the original guard/journal; `finish.py` verified those paths and completed the
remaining 52 replacements under a new guard and continuation journal. This was
continuation of a file-storage operation, not a retry of a research campaign.

Recovered **1,594,634,240 allocated bytes**, with all 2,037,153,240 logical bytes
unchanged and at their original paths. Final verification covered all 175
snapshots and 33,352 other research files, including 159 protected tokenizer
checkpoints, all four archives, older checkpoints, failure evidence and corpus.
Observed free space after completion: 30,967,259,136 bytes, above the unchanged
30,631,412,484-byte full-panel requirement. Both worker locks and the 8 GiB
reserve remained enforced. See `completed.json` and both immutable journals.
