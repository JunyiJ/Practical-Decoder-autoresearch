# Two-depth context transfer: execution results

## Status — 2026-11-15 07:36 UTC

Two of12 runs passed execution review. Slot02 is active: depth16, context256,
bundle1. The frozen plan, roles, schedules and quality criteria remain unchanged.
The first depth4 pair is descriptive evidence. Full within-depth and interaction
decisions remain pending all three paired bundles at both depths.

| Slot | Depth | Context | Bundle | Updates | Training seconds | RAW CE | Time-EMA CE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 00 | 4 | 128 | 1 | 41056 | 14400.181199 | 4.851455174 | 4.839095571 |
| 01 | 4 | 256 | 1 | 41430 | 14400.102752 | 4.752460614 | 4.736637785 |

For bundle1 at depth4, native-context EMA delta256-minus128 is−0.102457786
nats/token. Both sources improve: FineWeb−0.092971985 and Wikipedia−0.140451107.
With both endpoints evaluated at context128, the context256-trained endpoint scores
4.811416146, a smaller delta of−0.027679425. This is consistent with some benefit
persisting under common evaluation context. It does not isolate the causal training
mechanism. Equal training time permits different update/token exposure, and the
fixed optimizer and schedule remain part of the tested recipe. No early bootstrap,
quality gate, generation selection or transfer claim is made from this one pair.

Each slot scored all108 development documents and100416 targets. Both finite FP32
endpoints, both fixed probes, clock accounting, EMA count, paired-stream evidence,
5269 source bindings and24 first outputs per slot passed execution review.
Each slot has121 exact transient restart retirements. No failed scientific attempt
occurred. Original test contents remain unopened. Original qualification failures
and all their first outputs remain retained.

Slot00 elapsed14773.732seconds, including229.918seconds checkpoint work.
Peak observed RSS was936001536bytes and MPS driver allocation2514272256bytes.
Memory pressure reached warning level during slot00 and was normal before the
next launch. Slot01 elapsed14781.525seconds, including227.506seconds checkpoint
work. Peak RSS was1049493504bytes and MPS driver allocation2518466560bytes.
RSS4GiB/MPS6GiB guards remain enforced.

Slot02 admission measured20432351232 free bytes against20218359129 required for
all remaining outputs and allowances, including10GiB reserve plus2GiB headroom.
The213992103byte margin is above the complete requirement. Four retained endpoints
total455994732bytes. The reclaimable-page proxy was5820317696bytes, with normal
host memory pressure. This proxy is not guaranteed free RAM. Remaining scientific
tranche time was195084.743seconds against144000 remaining training seconds.
The first depth16/context256 worker holds the shared lock and has a verified
initial checkpoint. No other training worker is active.

Evidence: `tokenizer-evidence/general-rsi-depth-context-quality-001/slot-00/review.json`
and `slot-01/review.json`, results, score files and first-generation journals;
slot02 `continuation-admission.json` and `launch-verification.json`.

The first fresh depth4 pair supports the direction of the context256 recipe's
benefit. Whether that benefit transfers to depth16 remains unmeasured. Next finish
slot02 unchanged, review execution, then run frozen slot03 depth16/context128.
Complete all12 runs before the predeclared within-depth and interaction decisions.
Two depths provide transfer evidence, not a scaling law.
