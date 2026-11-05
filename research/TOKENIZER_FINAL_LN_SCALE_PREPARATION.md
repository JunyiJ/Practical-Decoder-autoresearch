# Final LayerNorm scale ablation preparation

Use the newly adopted AdamW epsilon1e-6 baseline.

## AdamW epsilon1e-6 adopted after original strict gates passed

All six1200-update fresh endpoints and all three paired inference reviews passed.
Independent eps6-minus-eps8 deltas:-0.022838512805131144,-0.034333479511555944,
-0.028255318629028782; mean-0.028475770315238624 nats/token. The original mean
improvement>0.02,every-seed and all conditional text-bound gates passed. Upper
bounds2048/4096/8192:-0.021995713638369736,-0.019390242512917873,
-0.020417804237483028. Separate adoption audit confirmed paired config equality
except epsilon/output paths,saved optimizer identities and all original gates.
Adopt epsilon1e-6 with betas(.9,.999),Muons,width rates,decay.00006,model/schedule
fixed. First frozen candidate is representative,not loss-selected. All six
fresh raw/H200 pairs3,355,576,176B remain protected. No test scoring; conditional
text intervals on reused validation are not seed-population/holdout confidence.

New current recipe: adamw-epsilon-preparation-001/adopted-width256-mla-epsilon-recipe.json.
Historical baseline files,frozen defaults and guards remain immutable; future
code must explicitly bind this adopted recipe in an additive namespace. Terminal
STOP closes all epsilon quality training,inference and final commands; no reruns.

Boundary review selects final LayerNorm learned scale versus fixed unit scale,
keeping learned bias,centering,epsilon1e-5 and eight pre-branch RMSNorms fixed.
AdamW beta1 .9-versus-.95 is deferred,with beta2 .999/adopted epsilon1e-6 fixed.
Final-scale component001 passed24 CPU FP64/FP32/MPS FP32 generated cases:explicit
forward/all-gradient equations,exact native parity,initial-function/shared-bias/
RNG equality,bias-gradient sum,finite differences and exact suffix/future-gradient
checks. Maxforward4.462275904337787e-7,maxgradient2.8988415756714403e-6;all863
bindings unchanged. No full-model candidate or quality evidence. Candidate removes
256 parameters,not parameter-matched. Next explicit full-model scale/group/
checkpoint/cache policy integration on adopted epsilon1e-6,without bypassing
existing learned-scale guards. Source/small numerical work proceeds.
Evidence:quality001-final-review.json,quality001-closure.json,adopted recipe,
final-ln-scale-preparation-001/direction-decision.json,component001-result.json.

The full-model next step must preserve shared initialization/logits/RNG and all
unrelated optimizer groups. Fixed-unit removes only final scale:28,492,160 versus
28,492,416 parameters,87/88EMA tensors,43/44AdamW groups,44Muon both. Keep learned
final bias. Require explicit foreign scale-policy checkpoint/cache rejection and
CPU/MPS raw optimizer/EMA/RNG/history repeat/resume/H200/full-cache-reset checks.
No new scientific seeds or pilot/quality freeze. One worker/four threads and all
original guards. Do not restore retired tensors or rerun closed fixtures.

## Full-model scale integration passed; serialized resume reserve failure closed

Model001 passed both learned/fixed-unit arms on adopted epsilon1e-6:identical
shared initialization/logits/RNG and unrelated groups,28,492,416/28,492,160
parameters,88/87parameter tensors,44/43AdamW and44Muon groups. Explicit readout
equations,CPU/MPS gradients/cache/reset/causality and foreign scale checkpoints/
caches passed. All878bindings unchanged. Only final scale removed; learned bias
retained. No serialized H200/EMA acceptance or scientific quality claim yet.

Original resume001 closed failed:disk_reserve guard exited90 at7.51s with
8,491,757,568B free below8GiB. RSS2,812,198,912B/MPSpeak2,508,668,928B stayed
within limits. Learned fresh repeat exact0 recorded; full resume/H200/both-arm
acceptance was NOT completed. Initial free10,204,749,824B minus445,087,744B
allocated payload implies1,267,904,512B extra filesystem consumption at guard.
1GiB OS swap observed afterward; cause not proved. Source/config/log/guard/trace
and failure receipt immutable; never retry closed001. New exact failed-retirement
removed only445,084,647B obsolete generated payload;54976survivors unchanged.

Separate resume002 source/plan is prepared,UNEXECUTED. It avoids retaining two
full reference tensor trees by comparing dtype/shape/SHA256 byte identities and
releases serialized bytes after decoding. No measured benefit claimed. Original
numeric/RSS/MPS/time/8GiB guards unchanged; complete capacity forecast strengthened
to11,049,021,440B with observed extra filesystem headroom. Measured
9207844864B leaves1841176576B deficit after recovery. Check free hourly,
quiet/no records when unchanged. Wake on actualfree>=11,049,021,440B or supplied
external volume for verified protected-archive relocation. Then execute separate
resume002,followed by actual worker/parent validation/inference; later full-panel
capacity/census/pilots and original three-pair gates. New independent human work
need not wait; no partial panel or relaxed guard. No worker active.

Storage028 losslessly archived all6 newly closed ADOPTED epsilon fresh RAWs,
reclaimed651099194B,54921survivors unchanged,exact restoration PASS.
Resolve originals via storage-recovery-028/raw-completed.json and protected-raw-
storage-028 archives. All203 archived RAWs are protected/recoverable,NOT retired;
all H100/H200 originals/storage004 intact. Original027 stopped before archive
creation/deletion when unrelated maintenance added a README; separate028 excludes
named unrelated maintenance and reuses unchanged-input copy proof. Do not repeat
closed recovery or numerical fixtures. Current adopted epsilon1e-6 remains.

## Inline recovery029 and compact-reference CPU audit

Applied latest immediate recovery/unused-app authorization. Under nonblocking
research lock,329 newly closed audit JSONs compressed losslessly,42,385,408
allocated bytes reclaimed;original paths/readable bytes unchanged and12 affected
archive/H200 hashes reverified. No model weights changed. External cache lock
was available;current updater/package/app classes inspected. Google download
cache only4KiB;active Chrome cache,pending Codex updater and required shared
runtimes protected. OrbStack/Keynote/Navigator usage metadata unavailable;absence
from processes alone did not authorize deleting installations. Existing cleanup
003/004 external classes and cleanup005 actual archive selection audit exhausted.
Only202 selected/fresh RAW archives remain after005's exact obsolete losing-control
retirement; original203 snapshots are historical. No retired member restoration.
Current006 copy-only native checkpoint probes report zero savings; not rerun or
modified. Separate maintenance work left untouched.

Measuredfree8897056768B;full generated-pair requirement11,049,021,440B,
requirement deficit2151964672B. Cleanup target with2GiB extra is
13,196,505,088B,target deficit4299448320B. Current verified eligible
classes exhausted. Concrete option:additional local free bytes to meet that
target,or external volume with>=6GiB for verified protected-archive relocation.
Hourly free/wake checks only while unchanged;no repeated broad scans,records,
commits or notifications. New blocker/requirement/eligible evidence triggers
coordinated recovery IN THAT TURN. Existing8GiB guard and scientific requirements
unchanged. Full resume002 remains unexecuted; no worker active.

Independent CPU state-digest-oracle001 passed24 behavioral mutations of real
finite AdamW state(width7/256,FP32/FP64):model value,shape,dtype,optimizer rate,
RNG byte and scalar changes detected against direct tensor/metadata equality.
Fresh clone identities and RNG preservation passed. This audits compact reference
semantics;no full-model repeat/resume or memory saving claimed. New concrete
independent human work need not wait. Evidence:storage-recovery-029/,
final-ln-scale-preparation-001/state-digest-oracle001.json.


## Final-scale resume003 and actual worker acceptance complete

Storage returned above the full integration target after maintenance008 removed
only the human-identified unused Docker images. Live preflight free13,607,890,944B
exceeded11,049,021,440B plus2GiB headroom. All original bindings matched; no worker
was active. Prepared resume002 stayed unexecuted: preflight found its terminal
print referenced removed max-error fields. Separate frozen003 changes only new
attempt paths and the final reporter; numerical operations/tolerances/guards
stayed identical. Both learned/fixed-unit arms passed exact dtype/shape/SHA256
state-byte and scalar/RNG fresh-repeat and serialized-resume identities, complete
clipping history, final raw/H200 export with3 snapshots and CPU/MPS batch1/8
full/cache/reset. RSSpeak2,269,659,136B and MPSpeak2,501,689,344B were observed;
no controlled comparative memory-saving claim. Original failed001 is immutable.

Seven additive final_ln_scale_worker_v1 adapters now bind888 sources, preserving
881 predecessors. Independent re-audit of the actual003 payloads passed optimizer,
EMA/RNG/history, CPU/MPS full validation and paired cache checks. History/count,
foreign-scale/group and coverage forgeries were rejected; learned bias and adopted
epsilon1e-6 enforced. Then a NEW exact resume-retirement001 audit permanently
retired four obsolete successful generated weights1,118,312,993B after dependency
closure, nonblocking shared lock, no open consumers, exact owner/inode/path/hash
checks and durable journal. All45,900 retained files verified unchanged. Missing
generated weights resolve via that manifest; never restore or rerun.

Actual serial trainer-worker001 accepted both four-update generated-text arms:
4,096 targets,3H200 snapshots,88/87EMA tensors,44/43AdamW and44Muon groups,
complete clipping histories. Parent full validation recomputation exact; CPU/MPS
CE discrepancies2.4222548411501066e-7/2.6099441896576536e-7. Trained-inference001
full/cache/reset passed batches1/8,196,608B cache/item; per-file overflow rejected.
Metadata mirrored exactly; new worker RAW/H200 weights1,118,318,689B remain
retained until pilot dependencies close and a NEW exact retirement audit.

No scientific seed reservation,pilot/quality freeze or scale-quality claim. No
active worker. Next minimal finite20-update pilot orchestration and complete
three-fresh-pair design in NEW final_ln_scale_campaign_v1 outside888 frozen
sources; re-audit actual worker tensors/cache/history, close dependencies and
measure complete output high-water including serialization,metadata,8GiB reserve
and2GiB cleanup target before census/pilots. Recover immediately on a new blocker;
source integration proceeds. Learned/fixed-unit scale comparison retains learned
bias, centering,epsilon1e-5,eight RMSNorms and adopted AdamW epsilon1e-6. Fixed
unit removes256 parameters. CommonLR.00125,zero LR-search,original strict gates,
exactly3fresh pairs,no selection/rescue/partial panel. Beta1 ablation deferred.
All202 selected/fresh RAW archives and actual H100/H200/storage004 preserved.
Evidence:final-ln-scale-preparation-001/worker-terminal001-review.json.


## Resource pilots completed through separate compact-receipt continuation

Six final_ln_scale_campaign_v1 modules passed prospective20-update/two-arm and
complete three-fresh-pair fixed-LR contract tests plus actual worker full-validation/
optimizer/EMA/history/cache re-audits. Pilot-only retention rejects fresh targets.
All894 bindings preserved888 predecessors. Worker-retirement001 then permanently
retired only four dependency-closed generated weights1,118,318,689B;46,100 retained
files verified unchanged. Original evidence immutable,never restore or rerun.
Storage030 reclaimed24,190,976 allocatedB through184 newly closed metadata files.
Full budget12,381,585,408B passed live before census/dispatch;extra2GiB target was
not met after eligible recovery. Panel200070 census found no matches in54,914
recorded files/5,100,919,969 readable bytes. No old score/seed pooling.

Original pilot001 stopped on acceptance-receipt metadata cap after the learned
worker completed20 updates and parent validation passed. Unit slot01 had no
launch or updates. Original STOP/source/log/config/guard/accepted.json.tmp remain
immutable. Never rerun its prepare/run/review commands. Separate v2 references
the immutable worker and request SHA instead of duplicating its full source
inventory. Real publication fixture150,155B fit unchanged262,144B group cap;
foreign references/source bridges rejected. All898 sources preserve894 previous
bindings. Separate frozen pilot002 imported the learned endpoint with ZERO
repeat updates or weight copies,then ran only previously unstarted fixed-unit.
Accepted learned RAW/H200 remain in original failed001 root,unit in002. Classify
by actual acceptance,not failed-root name. Both are retained successful pilots.

Both intended endpoints now have20 updates/20,480 targets/8H200 snapshots,
88/87EMA,44/43AdamW,44Muon,adopted epsilon1e-6 and complete clipping histories.
Parent validation and paired full/cache/reset batches1/8 passed. Independent
final003 resource checks passed;1200-step conservative2x projections5800.67/
5755.12seconds under7200,all RSS/MPS/time/reserve guards unchanged. Final001
review stayed unexecuted after checking capacity; final002 review launch omitted
strict environment flags and failed during startup before its guard/resource
checks. Both sources/logs preserved;separate final003 used the full strict env.

The full-capacity gate remains FALSE. Around1GiB extra filesystem consumption
beyond new unit tensors was observed;cause unproved. Inline recovery031/032
compressed367/2,568 previously uncompressed closed metadata/text files losslessly,
reclaimed76,763,136/117,678,080 allocatedB,original paths/readable bytes unchanged.
Twelve affected protected model/archive hashes verified after each operation.
Closed updater/build/app/research classes rechecked;current verified options
exhausted. Active updaters/shared runtimes/current and live Chrome versions,
root-owned npm,containers/volumes/user data preserved;no apps/services stopped.
Future exact pilot retirement would reclaim1,118,330,653B only after quality
integration closes dependencies. Current precise deficits/options are recorded
in campaign-capacity002.json; no partial panel or relaxed capacity gate.

Next minimal NEW final_ln_scale_quality_v1 finite1200-step orchestration outside
898 frozen sources,with compact receipts and cross-root pilot acceptance. Re-audit
real pilot RAW/H200/optimizer/EMA/RNG/history/cache,six-trajectory and metadata
contracts,close dependencies,then NEW exact pilot retirement. Recheck complete
live capacity and fresh/bootstrap nonuse before quality freeze. Source work
proceeds despite capacity; no idle wait ban. Fresh011/012/013 and bootstrap004
remain reserved UNUSED,pilot001/eval003 used. Exactly3 fresh pairs/commonLR.00125,
zeroLRsearch/original strict gates/no selection/rescue/retry/partial panel.
No scale scientific quality result or active worker. Learned scale/bias and
adopted epsilon1e-6 recipe retained;beta1 ablation deferred. All202 selected/fresh
RAW archives and actual H100/H200/storage004 protected.
Evidence:pilot002-terminal-review.json,pilot002-final003-review.json,
pilot001-publication-failure-review.json,worker-retirement001/,storage030–032.


## Finite final-scale quality source ready; exact pilot retirement verified

New final_ln_scale_quality_v1 passed27 contracts and independent real pilot
RAW/H200/optimizer/EMA/RNG/clipping/full-validation/cache re-audits. All906 sources
preserve898 predecessors. Exactly3 fresh pairs at commonLR.00125,zero LRsearch,
1200updates/41H200 snapshots per trajectory and original strict gates remain.
Compact acceptance receipts resolve immutable worker/request SHA; full-horizon
metadata publication stays within unchanged group caps. No quality training.

After actual downstream closure, exact retirement removed only four accepted
cross-root pilot tensors1,118,330,653B;46,372 survivors verified unchanged. The
cleanup's final verifier then incorrectly parsed the intentionally truncated
original accepted.json.tmp. Original source/log/guard/completed receipt and parser
exception remain immutable,closed;never rerun cleanup or restore tensors. New
additive final_ln_scale_quality_v2 verifies arbitrary evidence by exact bytes,
parsing only complete declared JSON. Separate verification002 rehashed all46,372
survivors and passed. All912 bindings preserve906; acceptance implementation is
byte-identical and all6 configs/time/RSS/MPS/reserve guards unchanged.

Prepared freeze/inference/final-review sources passed AST,compact-receipt mapping,
independent nonconstant synthetic bootstrap and complete budget review. The
bootstrap test uses separate synthetic seeds; fresh200070011/012/013 and
bootstrap200070004 remain unused. All226 saved run config files checked. No
quality freeze or scientific result; learned scale/bias/epsilon1e-6 recipe retained.

Storage033 reclaimed22,908,928 allocatedB from124 newly closed metadata files,
losslessly preserving paths/readable bytes;12 protected payload hashes verified.
All current eligible classes exhausted. Actual post-retirement capacity is in
quality-source-review002.json: no further conditional pilot credit. Mandatory
12,381,585,408B gate and14,529,069,056B cleanup target remain. Prepared freeze
builder UNEXECUTED. Next unchanged heartbeat: one silent hourly free-space check
only. Wake at actualfree>=12,381,585,408B or external APFS>=6GiB for verified
protected-archive relocation;new eligible evidence triggers inline recovery.
Then recheck912 bindings/prepared source refs,strict environment,fresh/bootstrap
nonuse and complete live capacity before the new one-shot freeze. No partial
panel,relaxed guard,closed retry or retired restoration. All202 selected/fresh
RAW archives and actual H100/H200/storage004 preserved.


## Quality001 frozen and first fresh learned-scale trajectory launched

Coordinated external maintenance010–012 reclaimed823,345,152 allocatedB and
exhausted verified remaining classes. Mandatory complete budget now fits; extra
2GiB cleanup target remains unmet. No duplicate closed cleanup or relaxed guard.
Wake001 rechecked912 sources,prepared source refs,strict environment,retirement
proof and current process state. One-shot freeze builder then rechecked all226
saved run configs for unused fresh seeds and froze exactly3fresh pairs at
commonLR.00125/equal zero LRsearch/original strict gates. Freeze SHA
d00b8e7960322b7a139b4481c7ea238661831f440c46abaafb16c64401e6c1ce;
free12,441,579,520B versus required12,381,585,408B. All six future RAW/H200 pairs
remain protected;full serialization/metadata/reserve accounting unchanged.

Slot00 learned-scale(seed200070011) launched under one local MPS worker/four
threads,7200sec/RSS4GiB/MPS6GiB/reserve8GiB/output guards. First completed update
and1024targets observed. Parent/worker PIDs and mutable guard/process snapshot
are recorded in quality001-slot00-launch-observation001.json. This is active
training,not endpoint acceptance or a scientific quality result. No holdout
scoring/adoption. Seed200070011 now used;200070012/013 and bootstrap200070004
remain unused. Follow actual processes/logs/STOP/anchors and never rerun slot00
or the freeze builder. On clean parent acceptance proceed serially to slot01
fixed_unit,same first-pair seed;check remaining complete budget before each
new slot. No second worker,rescue,replacement seed,partial-budget dispatch or
retired restoration. All912 bindings and original failure/retirement evidence
preserved;learned-scale/epsilon1e-6 incumbent stays adopted pending full gates.


Post-launch forward capacity pressure was detected after initialization/concurrent
filesystem changes,cause unproved. New inline external-cache recovery001 acquired
its independent nonblocking lock while the numerical research lock was busy.
Package/build roots had no new eligible files;71root-owned npm files77,713,408
allocatedB retain their normal administrator blocker. No app/worker stopped or
research payload changed;other classes remain exhausted per maintenance012.
Observed free12,229,713,920B is151,871,488B below the full forward projection before
accepted-weight credit. The active slot00 guard remains healthy above8GiB and
continues. Recompute actual remaining full budget after independent acceptance
and before slot01;no below-budget next launch or stale freeze-time assumption.
See quality001-active-capacity-recovery001/completed.json.
