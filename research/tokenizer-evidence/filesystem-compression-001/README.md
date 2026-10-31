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
