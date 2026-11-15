# Two-depth context transfer: execution results

## Status — 2026-11-15 16:00 UTC

Four of12 runs passed execution review. Slot04 is active: depth16, context128,
bundle2. The frozen plan, roles, schedules and quality criteria remain unchanged.
The first pair at each depth is descriptive evidence. Full within-depth and
interaction decisions remain pending all three paired bundles at both depths.

| Slot | Depth | Context | Bundle | Updates | Training seconds | RAW CE | Time-EMA CE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 00 | 4 | 128 | 1 | 41056 | 14400.181199 | 4.851455174 | 4.839095571 |
| 01 | 4 | 256 | 1 | 41430 | 14400.102752 | 4.752460614 | 4.736637785 |
| 02 | 16 | 256 | 1 | 22876 | 14400.167426 | 4.715277211 | 4.705582296 |
| 03 | 16 | 128 | 1 | 23710 | 14400.324782 | 4.801574994 | 4.794862826 |

For bundle1 at depth4, native-context EMA delta256-minus128 is−0.102457786
nats/token. Both sources improve: FineWeb−0.092971985 and Wikipedia−0.140451107.
With both endpoints evaluated at context128, the context256-trained endpoint scores
4.811416146, a smaller delta of−0.027679425. This is consistent with some benefit
persisting under common evaluation context. It does not isolate the causal training
mechanism. Equal training time permits different update/token exposure, and the
fixed optimizer and schedule remain part of the tested recipe. No early bootstrap,
quality gate, generation selection or transfer claim is made from this one pair.

For bundle1 at depth16, native-context EMA delta256-minus128 is−0.089280530
nats/token. Both sources improve: FineWeb−0.079173432 and Wikipedia−0.129762326.
Slot02 common128 EMA CE is4.784371225, giving a common128 delta of−0.010491601.
The first-bundle depth16-minus4 interaction is+0.013177255 nats/token. These are
descriptive contrasts only. They do not establish transfer, an interaction, or
equivalence. The three fresh paired bundles per depth and frozen multiplicity
rules remain required. Eight endpoints and96 first outputs are retained.

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
Slot02 elapsed14917.938seconds, including240.324seconds checkpoint work.
Peak RSS was1073545216bytes. Its7565 EMA snapshots and22876 updates passed
clock and checkpoint review. Slot03 elapsed14872.130seconds, including
244.654seconds checkpoint work. Peak RSS was1020903424bytes and MPS driver
allocation2744958976bytes. Its7826 EMA snapshots and23710 updates passed review.
RSS4GiB/MPS6GiB guards remain enforced.

Slot02 admission measured20432351232 free bytes against20218359129 required for
all remaining outputs and allowances, including10GiB reserve plus2GiB headroom.
The213992103byte margin is above the complete requirement. Four retained endpoints
total455994732bytes. The reclaimable-page proxy was5820317696bytes, with normal
host memory pressure. This proxy is not guaranteed free RAM. Remaining scientific
tranche time was195084.743seconds against144000 remaining training seconds.
Slot03 admission measured19978940416 free bytes against19923951787 required
for all nine remaining runs and allowances. The margin was54988629bytes above
the complete requirement. Six retained endpoints total750402074bytes. Host
pressure was normal, with a5859524608byte reclaimable-page proxy. Remaining
tranche time was180166.805seconds against129600 training seconds.
Slot04 admission measured19676602368 free bytes against19629544445 required
for all eight remaining runs and allowances, a47057923byte margin. Eight retained
endpoints total1044809416bytes. Host pressure was normal, with a6291292160byte
reclaimable-page proxy. Remaining tranche time was165294.676seconds against
115200 training seconds. The bundle2 depth16/context128 worker holds the shared
lock, with its initial checkpoint verified and no other training worker active.
No roles, criteria, schedules or source bindings changed.

Evidence: `tokenizer-evidence/general-rsi-depth-context-quality-001/slot-00/review.json`
and `slot-01/review.json`, `slot-02/review.json`, `slot-03/review.json`, results,
score files and first-generation journals; slot04 `continuation-admission.json` and
`launch-verification.json`.

The first fresh pair at each depth supports the direction of the context256
recipe's benefit. Full transfer and interaction conclusions remain unresolved.
Representation, exposure, optimizer and schedule effects remain entangled.
Next finish slot04 unchanged, review execution, then admit and run frozen slot05
depth16/context256 bundle2.
Complete all12 runs before the predeclared within-depth and interaction decisions.
Two depths provide transfer evidence, not a scaling law.

## Active-run storage recovery — 2026-11-15 12:59 UTC

A new1,216,840byte shortfall against the conservative remaining-study forecast
triggered standing-authorized external-cache cleanup. The active worker stayed
above its12GiB disk guard and was not interrupted. Under the independent cache
lock,4949 unopened Chromium cache payloads older than one hour were retired
using exact path/size/hash and consumer checks with a durable deletion journal.
Allocated bytes removed:269025280. The immediate observed free-space gain was
269680640bytes. All research payloads and frozen sources were preserved.

Reconciliation: free19786289152bytes against19524866376required, a261422776byte
margin including the full reserve and allowances. Slot03 remains active. Evidence:
`maintenance-cache-cleanup-019/manifest.json`, `deletion-journal.jsonl`,
`completed.json` and `research-reconciliation.json`. The scientific hypothesis
and next frozen run are unchanged.

## Active-run storage recovery — 2026-11-15 17:09 UTC

A guard sample showed a transient2,573,018byte shortfall against the nearby
remaining-study forecast. Free space recovered slightly before the exact audit.
Standing-authorized external-cache recovery020 retired2585 unopened Chromium
cache payloads older than one hour, with exact hashes, consumer checks and a
durable deletion journal under the independent cache lock. Allocated bytes
removed:188940288. Observed free-space gain:187764736bytes. All eligible entries
in the two audited roots were exhausted before the256MiB batch cap.

Reconciliation: free19423948800bytes against19230413658required,
a193535142byte margin including10GiB reserve plus2GiB headroom.
All eight completed endpoints remain at their recorded sizes. Slot04 continues
under its shared lock. Research payloads, frozen sources and the scientific
hypothesis are unchanged. Next review slot04, then admit frozen slot05. Evidence:
`maintenance-cache-cleanup-020/completed.json` and `research-reconciliation.json`.

## Active-run storage recovery — 2026-11-15 19:34 UTC

A new358254byte shortfall against the full remaining-study forecast triggered
external-cache recovery021. The independent cache lock, exact path/size/hash
audit, consumer checks and durable deletion journal protected active work.
504 unopened Chromium cache payloads older than one hour were retired.
Allocated bytes removed:22372352. Observed free-space gain:22085632bytes.
All eligible entries in the two audited roots were exhausted below the256MiB cap.

Reconciliation: free19101962240bytes against19080079982required, margin21882258bytes,
including10GiB reserve plus2GiB headroom and remaining-output allowances.
All eight completed endpoints retain their recorded sizes. Slot04 continues
under the shared lock. The scientific hypothesis and protocol are unchanged.
Next review slot04, then admit frozen slot05. Evidence:
`maintenance-cache-cleanup-021/completed.json` and `research-reconciliation.json`.
