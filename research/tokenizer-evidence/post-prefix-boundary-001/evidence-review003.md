# Real-data repeat and serialized transport passed for all three arms

The new tokenizer_moe_resume_v1 namespace explicitly binds moe_model and its
genuine backbone_template. It has no legacy cfg.model alias. Original native
267688 train IDs,34279 validation IDs, tokenizer/raw-split hashes and private
paired512 samplers are verified. Only the train/validation slices are encoded;
the holdout is not encoded or scored. The schema has distinct RAW and H200 kinds.

Actual resume001 passed the complete declared proof for dense672, dense1344 and
top2of4 expert336. Per arm: two fresh four-update real-data trajectories and a
separate serialized2+2 trajectory agree exactly in tensor dtype/shape/SHA256 bytes
and scalar state, model/optimizer/EMA/H200/logits, full clipping/routing history,
private samplers and CPU/MPS RNG. Each endpoint processes4096 targets and has
3 H200 snapshots. EMA tracks88/88/128 learned tensors and excludes integer route
counters. Complete proof training totals37888 targets, including the additional
1024-target dormant-expert fixture; this is correctness work, not quality evidence.

| Arm | Serialized two-update RAW bytes | Rejected forgeries |
|---|---:|---:|
| dense672 |445091431|20|
| dense1344 |469864039|20|
| MoE4/top2/expert336 |469984599|22|

All62 adversarial payloads were rejected before destination or final live RNG
mutation. They cover wrong kind/architecture, CPU/MPS RNG and sampler encoding,
copied config/data identity, RoPE, missing history/gradient names, per-parameter
update counts, scheduled LR, Adam step/shape/dtype/nonfinite moments, EMA names/
coverage and MoE route counters/history. Backend RNG decoding is probed
transactionally and restored, following the already proved metadata preflight.
The producer and validator audit CE, auxiliary term and total separately.

All128 MoE learned parameters happened to receive gradients on every update in
the ordinary1024-token fixture. Do not infer that this holds generally. The
additional declared real-data fixture zeroes routers before one update to force
lower-index ties. It independently verifies24 dormant expert matrices, zero
parameter update/decay counts, absent optimizer slots at globalstep1, and a valid
RAW preflight. Saved Adam step scalars must equal each parameter's actual update
count, and Muon slots exist only for parameters ever updated. Integer per-layer
assignment counts equal the complete history; clipping uses the actual active
parameter inventory rather than a constant128 assumption.

The complete serial memory/output forecast was recorded before launch. Maximum
serialized buffer512MiB; all tensors stayed in memory. CPU state was collected,
MPS training state released, then the CPU tree encoded; buffers/trees were
released before continuing. This describes lifetimes and does not establish a
comparative memory saving. Original600s/RSS4GiB/MPS6GiB/reserve8GiB guard passed:
41.37s, peakRSS2155413504B, driver2652717056B. Complete metadata-only floor
9876074496B and+2GiBtarget passed actual~95.985GBfree. No disk tensor files or
retirement occurred. All1019 source bindings, including the4new transport files
and new proof, preserve1014 predecessors and1004 original scientific bindings.

Terminal STOP-resume001 closes the proof and all its arms. Never rerun that proof command or its repeated/forged test trajectories,
model001 or earlier proofs. All new sources/results/guards/
plans are immutable. Original scientific STOPs/weights/failures remain protected.
There is no active worker, real generated-worker acceptance, resource pilot or
scientific quality result. Fixture600460/train and600461/evaluation are USED;
no future scientific seed has been assigned.

## Next executable action

Implement a new finite generated-worker namespace using the accepted model and
transport. Independently accept all3 four-update RAW/H200 pairs from real child
processes: exact config/request/source/PID/guard ancestry, private sampler replay,
deterministic CPU/MPS RNG endpoints, optimizer/EMA/exposure/clipping/routing state,
complete34278-target native validation on CPU AND MPS and cache/resetB1/8. CE
must exclude the router auxiliary term. The old prefix scorer or cfg schema must
not be passed a fake native model config. No original command or trained fixture
is replayed; these are new generated-worker prerequisite trajectories.

The prospective full worker output forecast retains all3 pairs, native560MiB+
wide640MiB+MoE640MiB=1929379840B, plus16MiBmetadata,8GiBreserve and measured
1267904512Bexternal/1458176Bentryheadroom. Floor11805454336B,target13952937984B.
This is an executor-preparation forecast; verify actual serialization and live
capacity before creating a new root. Atomic .tmp must fit the endpoint cap;
no retired/hardlink/prospective-cleanup credit. Root/plan must be new and idle,
all1019 bindings/prerequisites/runtime/data checked, strict original guards and
one shared worker lock. New fixture seeds are allowed; no scientific seed panel.

After actual worker acceptance, admit and execute finite20-update resource pilots,
then complete fair tuning/statistics/source contracts and full fresh-panel
budget/seed census before scientific freeze. Old negative results, both dense
controls and the no-cost-only-adoption rule remain. No new quality result follows
from correctness, in-memory serialization sizes or resource feasibility.
