# Generated workers and resource pilots complete

All three new four-update child workers independently passed RAW/H200, complete
optimizer/EMA/routing/exposure/clipping, private samplers and deterministic CPU/
MPS RNG acceptance. Each endpoint passed full34278-target original native
validation on CPU AND MPS and H200 cache/resetB1/8. Maximum CE discrepancy was
2.8539562180185385e-7. All three subsequent20-update pilots passed the same
acceptance,20480targets,8H200 snapshots and88/88/128 learned EMA tensors. Maximum
pilot CPU/MPS CE discrepancy was2.8255780293307e-7. Validation excludes router
auxiliary loss. No holdout or scientific quality seed was used.

The finite worker/source gate verified18configs (three arms×4/20horizons×3LRs),
rejected1200before data/model work, rejected train-mode/mutated-target/legacycfg
scoring, and rejected24 partial/wrong-target/nonfinite/incorrect-normalization/
aux-included score reports. The six-file tokenizer_moe_worker_v1 namespace is
now immutable; all1025 bindings preserve1019 predecessors and1004 scientific
sources. Do not weaken its4/20-only gate for the scientific experiment.

| Arm | Pilot training seconds | Conservative1200 projection |
|---|---:|---:|
| dense672 |5.5474|758.35s|
| dense1344 |6.1122|838.56s|
| MoE top2of4/expert336 |7.4749|1037.57s|

Projection=2×(pilot training time/20×1200+measured fixed child overhead+parent
acceptance time). All pass original7200s. These are estimates, not full-run
stability or quality guarantees. All pilot updates were finite and unclipped.
Across pilots, sampled peakRSS was3130703872B and driver2686271488B, below original
4GiB/6GiB limits. Dynamic routing, later optimizer behavior, kernel-cache growth
and additional history can differ over1200updates; full guards remain mandatory.

Each MoE layer assigned40960 token/expert pairs over20updates, with no unused
expert. Cumulative hard-assignment fractions ranged.1924-.3522; assignment
entropy ranged1.3547-1.3848nats. Every learned parameter received a gradient on
each pilot update. This is not mean softmax entropy, semantic specialization,
balanced quality or a full-run exposure guarantee. The separate dormant-expert
proof remains necessary even though these ordinary pilots used every expert.

Both roots have terminal STOPs; their builders/training/acceptance commands are
closed. No numerical worker/controller/session is active. prerequisite-close001
verified every original acceptance/guard/process and all12 original RAW/H200
hashes,3487764052B. Sixty-eight metadata files were mirrored losslessly with ZERO
weight copies. All worker/pilot weights remain protected, with no retirement or
cleanup. All earlier scientific sources/evidence/STOPs/retirements are unchanged.

## Prospective scientific question and cost

The existing transfer rationale and contrary character-width256 MoE results remain
in evidence-review.md and ../../MOE_QUALITY_RESULTS.md. Correctness and resources
now permit full scientific-runtime preparation; they do not show a gain. The
design in quality-design001.json keeps dense672 as the active-FFN-matched control
and dense1344 as the stored-FFN-matched control. Only MoE can qualify against BOTH.
The exact four-expert/top-two/336-width/.01aux anchor was fixed before worker/
pilot losses. No router-loss/expert-count/placement grid or capacity rescue.

Exactly9full LR screens (3rates per arm), then9fresh endpoints (3new shared seeds
per arm),1200updates/1228800targets/41H200 snapshots each. Min complete H200 MPS
native CE within arm, lower-LR exact ties, immutable selection after all9screens.
Fresh order dense-wide-moe / wide-moe-dense / moe-dense-wide. Gates preserve the
prior proposed mean delta below-.02nat/native-token against each dense control,
every fresh paired delta negative, and negative simultaneous conditional-text
upper bounds for2contrasts×3blocks2048/4096/8192,2000PCG64draws,linearquantile
1-.05/6. These conditions predate scientific scores. No wider-dense fallback or
cost-only adoption. Intervals condition on trained seeds and adaptively reused
validation text, not holdout or population-seed confidence.

Keep ALL9screens+9fresh pairs. Complete draft floor21771120640B,target23918604288B
includes6native560MiB+6wide640MiB+6MoE640MiB caps,256MiBmetadata,64MiBsidecars,
8GiBreserve and measured external/entry headroom. Already retained worker/pilot
files consume current free space; none receive deletion credit. Actual free
92351168512B passes. This is prospective, not a freeze or verified full-horizon
serialization cap. Conservative18-trajectory projection15806.9s (~4.39h), plus
unmeasured final serving/statistics overhead; not an elapsed-time promise.

Next implement NEW additive1200training/worker/parent/controller and independent
statistics/selection/confirmation. Validate actual pilot-derived1200history
serialization inside unchanged per-arm caps, accepted-prefix/ancestry/source/
config/seed/guard/credit/selection/STOP forgeries and independent nonconstant
grouping/bootstrap/quantiles. All18accepted before all9fresh serving/cache and
separate final review. Only after source/data/runtime/prerequisite/full LIVE
budget and actual unused-scientific-seed census pass may a NEW quality plan/root
be frozen and executed. No seed panel, root or quality freeze currently exists.
Do the enabled source/numerical work under existing authorization, without
replaying closed stages or seeking redundant consent. Proposed70/20/10 remains
proposed; these prerequisite runs do not count as numerical quality exploration.
