# Lossless storage recovery and experiment resumption

Authorization: the human asked to find another way to free the remaining space
without deleting selected models or fresh confirmations, and resume experiments.
This operation changes storage representation of exact historical text and
inactive temporary test fixtures. It deletes no selected/fresh model checkpoint.
The running app updater, its staged update, installed runtimes, downloaded models
and unrelated maintenance were inspected only and left untouched.

## Transparent historical-text compression

compression-plan.json binds 1,842 exact historical text files and 44,832 protected
files. compress.py stages each copy using APFS compression, verifies original
size/SHA256/mode/mtime/ownership, then atomically replaces only a verified copy
with lower allocated storage. Original readable bytes and paths are unchanged.
Every selected/fresh checkpoint is outside the replacement allowlist and was
verified physically unchanged. The original sources and prior receipts remain
valid by their logical hashes. Prior physical stamps for only the manifested text
files resolve through compression-journal.jsonl; closed operations are not rerun.

Completed allocation recovery: 1,248,325,632 bytes. Original text remains readable
normally; no special loader or archive extraction is needed. To decompress later,
reserve full space and use verified staging before replacement, preserving bytes.

## Verified archive of inactive temporary fixtures

All 144 exact /private/tmp/*-fixtures roots matched recorded historical research
test logs and the retained pytest basetemp harness. No pytest process or open file
used these roots. tmp-plan.json.gz records 289,359 exact regular files, directories
and symlinks, original paths, metadata, hashes and source/log identities. Actual
training checkpoints and selected/fresh models are outside this scope.

research_archives/research-pytest-temp-004.tar.zst retains all 798,045,756 logical
file bytes, including failed-test diagnostics. The archive is 33,004,054 bytes.
The complete archive was decompressed and every member, byte hash and link target
verified before any loose copy was removed. A durable batch intent binds the full
manifest/archive; tmp-journal.jsonl records each exact removal. Original research
logs and source files were reverified unchanged. The roots previously occupied
1,493,590,016 allocated bytes; filesystem metadata accounting is separate.

Never rerun closed fixtures to recreate their temporary paths. Use the compressed
manifest and verified archive for inspection or exact restoration, with capacity
reserved and symlink targets reviewed first. This is lossless preservation, not
retirement of historical failure evidence. The archive is local and protected;
only its manifest, source, journals and receipts belong in Git.

## Capacity and resumption

After recovery, capacity003 measured 13,340,770,304 bytes free against the complete
12,381,585,408-byte three-pair requirement, including the unchanged 8GiB reserve.
The two approaches recovered approximately2.7GB of allocated payload storage;
actual volume free-space changes also include filesystem metadata and concurrent
system activity, so they are reported separately rather than over-attributed.

All548 research source bindings verified unchanged. A fresh six-seed census
passed, and the finite decay pilot was frozen and resumed serially at LR0.00125.
No LR screening, guard relaxation, partial panel or model deletion was used.
See decay-preparation-001/capacity003.json and pilot001 evidence for subsequent
resource-pilot results. No quality-adoption claim follows from resource pilots.
