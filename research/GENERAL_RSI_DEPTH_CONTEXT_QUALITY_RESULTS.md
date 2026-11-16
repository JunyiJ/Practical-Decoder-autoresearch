# Two-depth context transfer: execution results

## Status — 2026-11-16 17:00 UTC

Ten of12 runs passed execution review. Slot10 is active: depth16, context256,
bundle3. The frozen plan, roles, schedules and quality criteria remain unchanged.
All three depth4 pairs and the first two depth16 pairs are descriptive evidence. Full within-depth and
interaction decisions remain pending all three paired bundles at both depths.

| Slot | Depth | Context | Bundle | Updates | Training seconds | RAW CE | Time-EMA CE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 00 | 4 | 128 | 1 | 41056 | 14400.181199 | 4.851455174 | 4.839095571 |
| 01 | 4 | 256 | 1 | 41430 | 14400.102752 | 4.752460614 | 4.736637785 |
| 02 | 16 | 256 | 1 | 22876 | 14400.167426 | 4.715277211 | 4.705582296 |
| 03 | 16 | 128 | 1 | 23710 | 14400.324782 | 4.801574994 | 4.794862826 |
| 04 | 16 | 128 | 2 | 23531 | 14400.451095 | 4.820412777 | 4.810480072 |
| 05 | 16 | 256 | 2 | 22550 | 14400.381925 | 4.737805592 | 4.729778713 |
| 06 | 4 | 256 | 2 | 39444 | 14400.023217 | 4.777135153 | 4.764863582 |
| 07 | 4 | 128 | 2 | 41678 | 14400.006258 | 4.857361887 | 4.847266345 |
| 08 | 4 | 128 | 3 | 41876 | 14400.102661 | 4.846172457 | 4.831474207 |
| 09 | 4 | 256 | 3 | 41390 | 14400.096636 | 4.741443764 | 4.728240306 |

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
rules remain required. Twenty endpoints and240 first outputs are retained.

For bundle2 at depth16, native-context EMA delta256-minus128 is−0.080701359
nats/token. Both sources improve: FineWeb−0.079991771 and Wikipedia−0.083543460.
Slot05 common128 EMA CE is4.805153779, giving a common128 delta of−0.005326292.
Both completed depth16 pairs favor context256 descriptively. No early quality
gate or interaction inference is made; the third pair at each depth remains
required.

For bundle2 at depth4, native-context EMA delta256-minus128 is−0.082402763
nats/token. FineWeb improves by0.078471294 and Wikipedia by0.098149410.
Slot06 common128 EMA CE4.837263894 gives a common128 delta of−0.010002451.
The bundle2 depth16-minus4 interaction is+0.001701404 nats/token. Both fresh
pairs at both depths favor context256 descriptively. The third pair at each depth
and the frozen multiplicity rules remain required; these contrasts establish
neither an interaction nor equivalence.

For bundle3 at depth4, native-context EMA delta256-minus128 is-0.103233901
nats/token. FineWeb improves by0.096826646 and Wikipedia by0.128896771.
Slot09 common128 EMA CE4.805951014 gives a common128 delta of-0.025523193.
All three fresh depth4 pairs favor context256 descriptively. The third depth16
pair and frozen joint multiplicity analysis remain required. No early quality,
transfer, interaction or equivalence decision is made.

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
Slot04 elapsed14880.742seconds, including249.502seconds checkpoint work.
Peak RSS was1022640128bytes and MPS driver allocation2744958976bytes.
Its7842 EMA snapshots and23531 updates passed review. Bundle2 depth16
context128 EMA CE is4.810480072.
Slot05 elapsed14941.763seconds, including261.439seconds checkpoint work.
Peak RSS was997965824bytes and MPS driver allocation2744958976bytes.
Its7373 EMA snapshots and22550 updates passed review, with no recovery charge.
Slot06 elapsed14835.161seconds, including257.733seconds checkpoint work.
Peak RSS was872480768bytes and MPS driver allocation2510077952bytes.
Its12765 EMA snapshots and39444 updates passed review. Training overshoot
was0.023217472seconds, with no recovery charge.
Slot07 elapsed14806.925seconds, including258.206seconds checkpoint work.
Peak RSS was908869632bytes and MPS driver allocation2510077952bytes.
Its13832 EMA snapshots and41678 updates passed review. Training overshoot
was0.006258429seconds, with no recovery charge.
Slot08 elapsed14807.609seconds, including259.371seconds checkpoint work.
Peak RSS was917536768bytes and MPS driver allocation2510077952bytes.
Its13864 EMA snapshots and41876 updates passed review. Training overshoot
was0.102661408seconds, with no recovery charge. The third depth4 reference
is complete. Its paired context256 result has passed execution review.
Slot09 elapsed14833.721seconds, including258.282seconds checkpoint work.
Peak RSS was906706944bytes and MPS driver allocation2510077952bytes.
Its13721 EMA snapshots and41390 updates passed review. Training overshoot
was0.096635945seconds, with no recovery charge.
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
115200 training seconds.
Slot05 admission measured19344519168 free bytes against19335137103 required
for all seven remaining runs and allowances, a9382065byte margin. Ten retained
endpoints total1339216758bytes. Host pressure was normal, with a6020136960byte
reclaimable-page proxy. Remaining tranche time was150413.933seconds against
100800 training seconds. That run has completed and passed execution review.
Slot06 admission measured19386142720 free bytes against19040729761 required
for all six remaining runs and allowances, a345412959byte margin. Twelve retained
endpoints total1633624100bytes. Host pressure was normal, with a5092720640byte
reclaimable-page proxy. Remaining tranche time was135472.170seconds against
86400 training seconds. That run has completed and passed execution review.
Slot07 admission measured19056570368 free bytes against18812732395 required
for all five remaining runs and allowances, a243837973byte margin. Fourteen
retained endpoints total1861621466bytes. Host pressure was normal, with a
5143248896byte reclaimable-page proxy. Remaining tranche time was120637.009seconds
against72000 training seconds. That run has completed and passed execution review.
Slot08 admission measured18752978944 free bytes against18584735029 required
for all four remaining runs and allowances, a168243915byte margin. Sixteen
retained endpoints total2089618832bytes. Host pressure was normal, with a
5367300096byte reclaimable-page proxy. Remaining tranche time was105830.084seconds
against57600 training seconds. That run has completed and passed execution review.
Slot09 admission measured18499842048 free bytes against18356737663 required
for all three remaining runs and allowances, a143104385byte margin. Eighteen
retained endpoints total2317616198bytes. Host pressure was normal, with a
5340446720byte reclaimable-page proxy. Remaining tranche time was91022.474seconds
against43200 training seconds. That run has completed and passed execution review.
Slot10 admission measured18168246272 free bytes against18128740297 required
for both remaining runs and allowances, a39505975byte margin. Twenty retained
endpoints total2545613564bytes. Host pressure was normal, with a5470633984byte
reclaimable-page proxy. Remaining tranche time was76188.754seconds against28800
training seconds. The bundle3 depth16/context256 worker holds the shared lock,
with its initial checkpoint verified and no other training worker active.
No roles, criteria, schedules or source bindings changed.

Evidence: `tokenizer-evidence/general-rsi-depth-context-quality-001/slot-00/review.json`
and `slot-01/review.json`, `slot-02/review.json`, `slot-03/review.json`,
`slot-04/review.json`, `slot-05/review.json`, `slot-06/review.json`,
`slot-07/review.json`, `slot-08/review.json`, `slot-09/review.json`, results,
score files and first-generation journals; slot10 `continuation-admission.json` and
`launch-verification.json`.

All three depth4 pairs and both completed depth16 pairs support the direction of the context256
recipe's benefit. Full transfer and interaction conclusions remain unresolved.
Representation, exposure, optimizer and schedule effects remain entangled.
Next finish slot10 unchanged, review execution, then admit and run frozen slot11
depth16/context128 bundle3.
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

## Active-run storage recovery — 2026-11-15 20:06 UTC

A new11496878byte shortfall triggered recovery022. Exact audits retired377
unopened disposable cache payloads across Chrome, Code and Codex, including
validated orphan sparse Chrome payloads. Homebrew compiled-cache candidates
were exhausted. Total allocated bytes removed:34746368. Each pass used the
independent maintenance lock, hash and consumer checks, and a durable journal.

During recovery, slot04 finished training and wrote both endpoints. The final
forecast counts those new files once: free18796359680bytes against
18785021056required, margin11338624bytes, including
10GiB reserve plus2GiB headroom and remaining-output allowances. Earlier
per-pass deficits use the trigger forecast and are superseded by this measured
reconciliation. All eight prior endpoints retain their recorded sizes.
Slot04 remains active in endpoint evaluation under the shared lock.
The scientific hypothesis and protocol are unchanged. Next review slot04,
then admit frozen slot05. Evidence: `maintenance-cache-cleanup-022/` exact
manifests, deletion journals, receipts and `research-reconciliation.json`.

## Active-run capacity shortfall — 2026-11-15 20:22 UTC

A new298209324byte deficit triggered exact external-cache recovery023.
Aged dense Chromium candidates were exhausted. The package-cache audit first
stopped before manifest creation or deletion at its identity safeguard. A
separate repair excluded non-owned or linked entries before selection and
retired11 verified unopened cached wheel payloads,23826432allocated bytes.
Homebrew compiled-cache and bottle-payload candidates were exhausted.
Installed runtimes, user state and research payloads were preserved.

Final free space18661433344bytes versus18937321580required leaves
275888236bytes unmet. All ten prior endpoints retain their recorded
sizes. Slot05 remains the sole worker under its shared lock and unchanged
hard guards. This resource condition is not a scientific quality result.
No next launch until complete remaining-study admission passes. A512MiB
storage addition/freeing covers this observed deficit with churn margin.
Wake on changed capacity, new eligible cache or a closed consumer dependency;
lock release enables exact research-payload audit. Do not repeat unchanged
broad scans. Hypothesis and frozen criteria remain unchanged. Evidence:
`maintenance-cache-cleanup-023/research-reconciliation.json`, original audit
failure, separate repair, manifests and deletion journal.

## Capacity restored at run boundary — 2026-11-16 00:21 UTC

After slot05 finished, the released research lock enabled an exact obsolete-payload
audit. Two accepted pseudo-repetition engineering restart files had no remaining
consumer. Their six-run scientific panel and downstream uses are complete.
Retirement removed890056704 allocated bytes, with original receipt hashes,
open-consumer checks, a durable deletion journal and5905 retained-file hash checks.
All scientific endpoints, selected weights, frozen bindings, first outputs and
original RNG-harness failure payloads remain. This narrow retirement supersedes
the historical retain-both wording for these two engineering files only.

The measured full remaining-study storage requirement passed after recovery.
Memory pressure was normal at launch and the time allowance covers six full runs
plus30% overhead. Slot06 launched with unchanged roles, sources, criteria and
schedule. The prior capacity blocker is resolved. Evidence:
`storage-cleanup-2026-11-16-pseudo-engineering-001/manifest.json`,
`deletion-journal.jsonl`, `completed.json`, and slot06 admission/launch verification.
