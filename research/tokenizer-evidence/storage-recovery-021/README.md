# Protected lossless RAW storage021

Six accepted fresh RAW checkpoints from the closed final LayerNorm bias quality
study are preserved in `research_archives/protected-raw-storage-021/`. They are
recoverable models,not retired payloads; never purge these archives.

`raw-completed.json` maps original paths,size and SHA to verified archive members.
`materialize.py ORIGINAL_REPO_RELATIVE_PATH DESTINATION` verifies exact bytes and
requires capacity before producing an owned copy; the archive stays protected.
`restoration-check.json` records an exact restoration test followed by removal of
only the owned temporary copy. No closed model evaluation or training was rerun.

All six H200 checkpoints remain at their original paths. Archived6 RAWs recovered
653,245,163 logical bytes;54,107 other files verified unchanged. Shared locks,
process checks,durable journals and the unchanged8GiB reserve were enforced.
