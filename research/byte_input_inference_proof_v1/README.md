# Closed inference consumer qualification

All six CPU/MPS native/raw-byte/nonlexical cases passed using accepted four-step
training parents. No training entries were replayed and no new training roles
were assigned. Export/load preserves every parent EMA weight exactly. Full
validation covers34278targets per case, with an independent FP64 CE oracle on
first, block-boundary and tail windows and independent full aggregation/byte
accounting. All78 rejection transactions passed. Numerical entry100.092s;
independent saved review4.530s. Scope is consumer correctness, never quality.

Evidence under `research/tokenizer-evidence/byte-composition-boundary-001/`:
`inference-math001`, `inference-review001`, `inference-acceptance001`.
Do not replay closed entry/review commands. Binary exports/oracles remain local
under the exact payload manifest. `byte_input_cache_v1.py` was separately
qualified by `cache-math001` and `cache-review001`:12 B1/B8 CPU/MPS cases,
full128 context,1584 chunk comparisons, exact reset replay and84 rejections.
