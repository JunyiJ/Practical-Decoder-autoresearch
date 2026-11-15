# Two-depth context transfer: execution results

## Status — 2026-11-15 03:24 UTC

One of12 runs passed execution review. Slot01 is active: depth4, context256,
bundle1. The frozen plan, all roles, schedules and quality criteria remain unchanged.
No paired context effect or transfer decision is available yet.

| Slot | Depth | Context | Bundle | Updates | Training seconds | RAW CE | Time-EMA CE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 00 | 4 | 128 | 1 | 41056 | 14400.181199 | 4.851455174 | 4.839095571 |

Slot00 scored all108 development documents and100416 targets. Both finite FP32
endpoints, both fixed probes, clock accounting, EMA count, paired-stream evidence,
5269 source bindings and24 first outputs passed the frozen execution review.
The121 transient restart retirements have exact deletion journals. No failed
attempt occurred. Original test contents remain unopened.

Worker elapsed time was14773.732seconds, including229.918seconds checkpoint work.
Peak observed RSS was936001536bytes and MPS driver allocation2514272256bytes.
Memory pressure reached warning level during execution and was normal
before the next launch. RSS4GiB/MPS6GiB guards remain enforced.

Slot01 admission measured22149603328 free bytes against20446356495 required for
all remaining outputs and allowances, including10GiB reserve plus2GiB headroom.
The reclaimable-page proxy was4919967744bytes, with normal host memory pressure.
This proxy is not guaranteed free RAM. Full remaining scientific time admission
also passed. Both retained slot00 endpoints total227997366bytes.

Evidence: `tokenizer-evidence/general-rsi-depth-context-quality-001/slot-00/review.json`,
`result.json`, score files and first-generation journals; slot01
`continuation-admission.json` and `launch-verification.json`.

The hypothesis remains that the context256 recipe transfers from depth4 to depth16.
This single unpaired reference endpoint does not test that hypothesis. Next finish
slot01 unchanged, review execution, then run frozen slot02 depth16/context256.
Complete all12 runs before the predeclared within-depth and interaction decisions.
