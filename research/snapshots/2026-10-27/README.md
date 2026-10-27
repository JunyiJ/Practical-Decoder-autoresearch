# Portable research snapshot — 2026-10-27 human stop

Read [the handoff](../../RESEARCH_HANDOFF.md) first. Research and its heartbeat are paused; restoration is not execution authorization.

- **3893 exact metadata files**, 1940 canonical knowledge records, all27 machine-readable conclusions.
- Archive: [research-metadata.tar.gz](research-metadata.tar.gz), 13.27 MiB compressed.
- SHA256: `27cf7f4bbdc10927d916ddb5fdf9041b40bef7d1e298a34811ddafed0e227ea6`. Every member's original path, byte count and SHA256 is in [metadata-manifest.json](metadata-manifest.json).
- Completed findings source: `2f69633da1a2c560952b1738db2c47a0927e9133`; latest completed training source: `37c3150d946c1d4884e88d48021b36801d626465`.
- Unfinished source: branch `codex/null-attention-paused-20261027`, commit `007eb93190e98371d83ab2b26e3b3dce89838e34`. Not merged into main.

## Directly readable files

| File | Purpose |
|---|---|
| [stop.json](stop.json) | Human instruction, stopped processes, paused automation/goal |
| [conclusions-index.json](conclusions-index.json) | Indexed27 conclusion records; complete original JSON is in the archive |
| [best-observed-config.json](best-observed-config.json) | Adopted recipe, selection evidence, scores and later context |
| [incumbent-request.original.json](incumbent-request.original.json) | Exact latest-study width64 control request; historical paths and seed, do not execute into old outputs |
| [incumbent-config.template.yaml](incumbent-config.template.yaml) | Same training/model recipe with relative corpus path, vocab65 and mandatory fresh seed/output placeholders; JSON syntax is valid YAML |
| [null-cpu-stop-review.json](null-cpu-stop-review.json) | Exact3,375 passed IDs and2,598 remaining/interrupted IDs; partial evidence, not a full5973-case pass |
| [verify_snapshot.py](verify_snapshot.py) | Read-only archive/member verification, with optional explicit extraction into a new directory |

## Included and omitted

The archive includes frozen run-root JSON, canonical knowledge entries, research helper source, current null preparation metadata/logs, stop records and selected historical failure reviews. It retains original absolute paths and file bytes. Use a separate extraction root and a documented path mapping when moving machines; do not rewrite frozen records in place.

It **does not include** trained weights, optimizer checkpoints, tensor capture payloads, corpus bytes, complete raw trial directories or retained scratch checkpoints. The archived review records describe completed audits; they cannot replace missing tensors for a new independent audit. Back up the original ignored evidence directories separately if full model recovery is required.

Historical helper code is preserved as evidence. Some helpers refer to artifacts outside this bundle and to the original absolute paths. Do not treat archive extraction as a ready-to-run research environment.

```bash
python3 research/snapshots/2026-10-27/verify_snapshot.py
```

No model tests, training, scoring or research were rerun to construct this snapshot.
