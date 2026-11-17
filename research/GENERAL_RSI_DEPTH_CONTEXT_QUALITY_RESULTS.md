# Two-depth context transfer: completed results

All12 frozen runs passed execution review. Native context256 improves development
CE at both D256 depths4 and16 under the fixed14400-second recipe. The direct
context-by-depth interaction remains unresolved. This establishes two-depth
transfer of this recipe, not a scaling law or equivalence of effects.

## Frozen primary decision

Delta is CE256 minus CE128 in nats/token. Negative favors context256.
Intervals use100000 shared paired document bootstrap draws, averaging the three
fresh bundles before resampling108 documents. Bonferroni controls the three
prospective contrasts at family alpha0.01. Each marginal interval is99.6667%.
These intervals are conditional on the three bundles, not population-seed intervals.

| Contrast | Three paired deltas | Mean | Adjusted interval | Decision |
| --- | --- | --- | --- | --- |
| delta4 | -0.102458, -0.082403, -0.103234 | -0.096031 | [-0.111309, -0.081081] | context256_superior |
| delta16 | -0.089281, -0.080701, -0.099996 | -0.089993 | [-0.106336, -0.074394] | context256_superior |
| delta16_minus_delta4 | +0.013177, +0.001701, +0.003238 | +0.006039 | [-0.006058, +0.017831] | unresolved_not_equivalent |

Both depths pass all three same-sign pairs, the0.02 practical threshold,
interval exclusion of zero, and the0.05 source-regression guard. Source mean
deltas are FineWeb−0.089423/Wikipedia−0.122499 at depth4 and
FineWeb−0.084345/Wikipedia−0.112613 at depth16. The interaction mean+0.006039
is below the practical threshold and its interval crosses zero. It provides no
supported strengthening or weakening claim.

## Secondary evaluation and learning curves

All contrasts below are descriptive. They cannot rescue or change the primary gate.

| Evaluation | Depth4 mean delta | Depth16 mean delta | Depth16-minus4 |
| --- | --- | --- | --- |
| raw native | -0.094650 | -0.090435 | +0.004215 |
| time_ema common128 | -0.021068 | -0.012125 | +0.008943 |
| raw common128 | -0.021065 | -0.013247 | +0.007818 |

All six common128 EMA paired deltas favor context256. The benefit is smaller
than native-context evaluation. A training-associated difference remains at a
common evaluation context, but these comparisons do not isolate its mechanism.
Evaluation context contributes substantially to the observed native contrast.
Exposure, optimizer, clipping, representation and schedule effects remain entangled.

The retained eight-document RAW probe has8586 targets. Three-bundle mean CE:

| Depth/context | One-third | Two-thirds | Final RAW on same subset |
| --- | --- | --- | --- |
| 4/128 | 5.055398 | 4.929287 | 4.521360 |
| 4/256 | 5.018723 | 4.785194 | 4.407049 |
| 16/128 | 5.196691 | 4.923946 | 4.513580 |
| 16/256 | 5.097182 | 4.776178 | 4.372308 |

This subset favors context256 at both intermediate times on average. It cannot
establish full-development onset or isolate the final-third cooldown. The saved
120-second training-loss bins descend in every arm through the final phase.
Depth4 clips every update in the final bin; depth16 clips about2.7–5.6%.
This measured difference reinforces the fixed-recipe scope. It is not evidence
that either depth uses its optimal optimizer or clipping setting.

## Fixed first-output diagnostics

All288 first completions are retained and rated. There were no rerolls or output
selection. Each group below contains36 outputs:12 prompts across three bundles.
Greedy and temperature0.8/top-k40 each use the frozen seed42 and64-token cap.

| Depth/context | Decoder | Mean repeated-trigram fraction | Terminal loops | Relevance counts0/1/2/U | Consistency counts0/1/2/U |
| --- | --- | --- | --- | --- | --- |
| 4/128 | greedy | 0.5797 | 20/36 | 9/26/1/0 | 7/24/1/4 |
| 4/128 | sampled | 0.0309 | 1/36 | 12/24/0/0 | 5/31/0/0 |
| 4/256 | greedy | 0.6344 | 22/36 | 9/27/0/0 | 8/19/4/5 |
| 4/256 | sampled | 0.0372 | 0/36 | 8/28/0/0 | 6/30/0/0 |
| 16/128 | greedy | 0.5995 | 19/36 | 6/30/0/0 | 10/20/3/3 |
| 16/128 | sampled | 0.0197 | 0/36 | 13/21/2/0 | 5/31/0/0 |
| 16/256 | greedy | 0.6183 | 18/36 | 8/28/0/0 | 8/23/3/2 |
| 16/256 | sampled | 0.0336 | 0/36 | 12/22/2/0 | 3/32/0/1 |

Only5/288 outputs received relevance2. Greedy outputs often loop; sampled outputs
repeat less but usually drift or make ambiguous claims. One output stopped at EOS;
287 reached the64-token cap. No unexpected early stop occurred. Improved CE does
not establish usable reasoning, factual reliability or instruction following.

Ratings were manually assigned by the primary Codex agent using identity-hidden
text. The agent knew aggregate loss results from execution reviews. This is not
independent human validation. The saved quotes and reasons make judgments auditable.
Some consistency0 annotations flag obvious physical/category errors in addition
to literal prompt or internal contradictions. Treat those ordinal summaries as
exploratory broader consistency diagnostics, not a strictly literal contradiction
benchmark or exhaustive fact check. No rating was changed after joining identities.
Unassessable is separate and never a pass. Per-prompt equal-weight means, assessable
counts and per-bundle summaries are in the generation evidence. No CE composite or
new adoption rule was created from these diagnostics.

## Exposure, compute and time

| Depth/context | Parameters | Mean sampled targets/run | Mean nominal corpus passes | Mean accounted PFLOPs/run |
| --- | --- | --- | --- | --- |
| 4/128 | 28492416 | 42533546.7 | 4.253 | 4.7169 |
| 4/256 | 28492416 | 41732778.7 | 4.173 | 4.7265 |
| 16/128 | 36773376 | 24020992.0 | 2.402 | 5.0926 |
| 16/256 | 36773376 | 23115093.3 | 2.311 | 5.1187 |

At equal synchronized time, context256/context128 sampled-target ratios are
1.0091/0.9464/0.9884 at depth4 and0.9648/0.9583/0.9637 at depth16. The sampler
can revisit or miss positions; nominal passes are not distinct-data coverage.
The partial operator estimate counts dense forward/backward matrix work and
Muon Newton–Schulz matrix work. It excludes EMA, normalization, activation,
softmax/CE, embedding scatter, clipping, AdamW, copies, evaluation and generation.
It is neither total hardware telemetry nor a6NT proxy. Actual updates, tokens,
partial FLOPs and training time are retained together for every curve bin.

Totals:384968 updates,394207232 sampled targets,172802.742052 synchronized
training seconds (48.000762h),178329.579083 supervised seconds (49.535994h).
The2.742052 seconds above48h are final-update overshoot. Checkpoint work totals
3037.574780s, native endpoint scoring589.203132s and generation1037.604558s.
The remaining862.454563 supervised seconds include untimed common128 scoring,
probes, initialization, export work outside the checkpoint timer and supervision.
Do not assign that remainder to any one operation. Run sums exclude between-run
waits, cleanup and this analysis. Qualified engineering time is separately
651.198172s. There were no failed scientific attempts or replay charges.

## Execution and preservation audit

Final slot11 depth16/context128 bundle3 completed23133 updates and
14400.242093207 training seconds, RAW CE4.812361150 and EMA CE4.803556180.
It retained7660 EMA snapshots and passed execution review at01:29UTC.

All24 finite FP32 CPU endpoints passed per-run review and their hashes were
rechecked at completion. They occupy3134428248 bytes. All5269 bound sources are
unchanged. All12 reviews verified schedules, time-EMA counts, paired sampling,
100416 development targets, probes and24 first outputs each. All1452 transient
restart retirements have exact deletion journals. Original RNG-harness failure002,
its payloads and first outputs remain separate from accepted repair003. Selected
Shakespeare weights and fresh confirmations remain protected. The original test
remains unopened.

Observed scientific peak RSS was1073545216 bytes and MPS driver allocation
2753347584 bytes, below4GiB/6GiB limits. These counts overlap and must not be added.
Storage and host pressure histories, including capacity recovery, remain evidence.
The final worker exited0 and the completion audit acquired the shared lock.
No training worker is active.

Completion audit reimplemented the bootstrap using per-draw document frequencies;
all three intervals agree within1e-12 and all primary decisions agree. Generation
aggregation replayed exactly. The first audit attempt hit a sandbox denial reading
macOS swap state after writing secondary analysis. Its source, output and failure
receipt remain preserved. The separately identified retry verified identical
numerical output and completed the read-only memory snapshot. This is an execution
failure, not a negative scientific result.

At completion audit, free space was16724758528 bytes. All endpoints already exist;
remaining publication allowance is256MiB, with12GiB reserve retained and no active
optimizer or atomic checkpoint overlap. The earlier full-study forecast deficit
is historical and does not block completed-panel analysis. No additional cleanup
or training is required for this study.

## Conclusion and next action

The prediction that the context256 recipe transfers from depth4 to depth16 is
supported under the frozen native-context criterion. An interaction was not
resolved. Common128 retains smaller gains. Generation quality remains weak.
These findings apply to this finite corpus, tokenizer, native representation,
D256 architecture and fixed time/optimizer/schedule recipe. They do not establish
scaling behavior, larger-model generality, optimized depth-specific recipes,
equivalence, or a causal explanation of the context benefit.

This study is complete. Do not rerun it or pool historical confirmations into it.
The next research action is an evidence-grounded boundary review of the completed
results and primary-paper versions before selecting any new campaign. No new
training panel is selected here. Preserve all fresh endpoints and first outputs.

Evidence: `tokenizer-evidence/general-rsi-depth-context-quality-001/` contains the
frozen `plan.json`, `decision.json`, `secondary-accounting.json`,
`generation-evaluation/`, and `completion-review.json`. The analysis source is
`general_rsi_depth_context_completion_v1.py`. Source-bound first-attempt provenance
is retained in `completion-audit-attempt01.py` and its failure receipt.

## Historical execution notes (superseded by completed results above)


## Status — 2026-11-16 21:21 UTC

Eleven of12 runs passed execution review. Final slot11 is active: depth16,
context128, bundle3. The frozen plan, roles, schedules and quality criteria remain unchanged.
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
| 10 | 16 | 256 | 3 | 22294 | 14400.662006 | 4.709962556 | 4.703559781 |

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
rules remain required. Twenty-two endpoints and264 first outputs are retained.

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

## Active-run storage recovery — 2026-11-16 17:49 UTC

A new1,915,110byte shortfall against the full remaining-study forecast triggered
standing-authorized cache recovery024. Under the independent cache lock,1516
unopened Chromium SimpleCache payloads older than one hour were retired after
exact path/size/hash and consumer checks, with a durable deletion journal.
Allocated bytes removed:65818624. All eligible entries were exhausted below
the256MiB batch cap. The active worker was not interrupted.

Reconciliation measured17791811584free bytes against17730124006required,
a61687578byte margin including10GiB reserve plus2GiB headroom.
All5269 bound sources and20 retained endpoints passed hash verification.
The hypothesis, frozen protocol and next action remain unchanged: finish slot10,
review execution, then admit slot11 depth16/context128 bundle3. Evidence:
`maintenance-cache-cleanup-024/completed.json` and `research-reconciliation.json`.

## Active-run storage recovery — 2026-11-16 19:29 UTC

A new3,209,318byte shortfall against the full remaining-study forecast triggered
standing-authorized cache recovery025. Under the independent cache lock,866
unopened Chromium SimpleCache payloads older than one hour were retired after
exact path/size/hash and consumer checks, with a durable deletion journal.
Allocated bytes removed:30441472. All eligible entries were exhausted below
the256MiB batch cap. The active worker was not interrupted.

Reconciliation measured17755058176free bytes against17728051302required,
a27006874byte margin including10GiB reserve plus2GiB headroom.
All5269 bound sources and20 retained endpoints passed hash verification.
The hypothesis, frozen protocol and next action remain unchanged: finish slot10,
review execution, then admit slot11 depth16/context128 bundle3. Evidence:
`maintenance-cache-cleanup-025/completed.json` and `research-reconciliation.json`.

## Partial active-run storage recovery — 2026-11-16 20:00 UTC

A new2,332,154byte forecast deficit triggered exact cache recovery026. It removed
12 eligible unopened Chromium entries,270336allocated bytes. No eligible pip/npm
payloads remained. Targeted Homebrew/Conda inventory found only retained Homebrew
API/manifest metadata and no Conda payloads. Research cleanup cannot acquire the
shared lock while slot10 runs. No worker was interrupted.

After concurrent writes, free17575862272bytes remained4,358,778bytes below the
17580221050byte full forecast, including10GiB reserve plus2GiB headroom.
All5269 bound sources and20 retained endpoints passed hash verification.
The scientific hypothesis and protocol are unchanged. Finish and review slot10,
then audit obsolete payloads under the released lock if capacity still falls short.
Launch slot11 only after complete remaining-study admission passes. Wake on new
eligible candidates, sufficient free space or worker lock release; do not repeat
unchanged broad scans. An external option is64MiB additional usable storage.
Evidence: `maintenance-cache-cleanup-026/research-reconciliation.json`, both exact
cleanup manifests/journals, and `remaining-cache-inventory.json`.

## Swap growth and partial recovery — 2026-11-16 20:45 UTC

Host swap allocation grew from5120 to6144MiB, causing a new1,081,230,074byte
forecast deficit. Exact cache recovery027 retired45 newly eligible unopened
Chromium entries,5832704allocated bytes. All eligible entries were exhausted.
The other cache classes retain their prior exhaustion evidence. The active
research lock remains held, so research payloads were not removed.

Reconciliation measured16502829056free bytes against17579298746required,
a1,076,469,690byte shortfall. The12GiB reserve/headroom and all remaining endpoint,
serialization and failure allowances remain in the requirement. All5269 bound
sources and20 retained endpoints passed hash verification; worker48074 stays healthy.

A targeted check of existing candidate evidence found the joint-integration
restarts already absent. ReLU twenty-update pilot acceptance and STOP are valid,
but closure of their full-horizon prerequisite consumers is not established.
Those payloads remain preserved. Finish/review slot10, then audit remaining obsolete
engineering/pilot payloads under the released lock. Slot11 requires full admission.
The hypothesis and scientific protocol are unchanged. A concrete external option
is1.25GiB additional usable storage. Evidence:
`maintenance-cache-cleanup-027/research-reconciliation.json` and exact manifest/journal.

## Run11 review and final-run admission — 2026-11-16 21:21 UTC

Slot10 passed execution review with7354 EMA snapshots,24 first outputs and121
exact transient retirements. Its common128 EMA CE is4.782998467. The paired
context128 endpoint is pending. Elapsed time was14962.205seconds, including
274.994seconds checkpoint work. Peak RSS was981598208bytes and peak MPS driver
allocation2744958976bytes. There were no replay charges or scientific failures.

After the worker exited0 and released the shared lock, exact cleanup retired eight
accepted short combined-shape003 inference payloads and three accepted byte-pilot
H200 inference payloads. Their downstream consumers initialize fresh or read the
retained metadata. All source, acceptance, failure and first-output evidence stays
intact. All original002 RNG-failure payloads, pilot RAW parents, selected/fresh
models and22 scientific endpoints remain retained. The manifest, consumer closure,
exact deletion journal and44863 retained-file hash checks are in
`storage-cleanup-2026-11-16-final-run-001/`. Recovered allocation was1289719808bytes.

Final-run admission measured17908748288bytes free against17834332955 required,
including the10GiB reserve plus2GiB headroom, remaining endpoints, restart atomic
overlap, metadata and failure allowances. Host pressure was normal. The reclaimable
page proxy was5104041984bytes and the remaining science tranche61226.548seconds.
Slot11 depth16/context128 bundle3 launched with frozen roles and14400 training
seconds. Its initial checkpoint hash, sole-worker ownership and held lock passed
verification. RSS4GiB/MPS6GiB guards remain active through all phases.

The hypothesis remains unresolved: the completed pairs favor context256, with
smaller common128 differences, but exposure, representation, optimizer and schedule
uncertainty remain. Complete the final pair before the frozen multiplicity-adjusted
within-depth and interaction tests, blind generation ratings and full-panel audit.
This can establish transfer/replication across two depths, not a scaling law.

## Active-run capacity recovery — 2026-11-16 22:49 UTC

At22:49UTC, a new free-space decrease left a712166520byte deficit against the
full remaining forecast after active-checkpoint accounting. Exact external-cache
recovery028 removed918 unused entries and recovered60026880allocated bytes.
Free space was16722681856bytes, above the12GiB reserve. The final worker remains
healthy under its guards and holds the shared research lock. All5269 sources and22
retained endpoints passed hash verification. No research payload was deleted and
no scientific criterion changed. See `maintenance-cache-cleanup-028/research-reconciliation.json`.
Leave the final run active. Reassess on new eligible capacity, sufficient free space,
or completion/failure; do not repeat unchanged broad scans. About0.8GiB of additional
usable space would cover the current deficit with modest churn headroom.
