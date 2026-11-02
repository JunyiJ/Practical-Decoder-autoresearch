# Tokenizer global gradient-clipping preparation

Current status: quality002 CLOSED_NOT_ADOPTED. All six fresh endpoints and three
paired inference reviews passed; clip0.5 worsened all seeds (mean +0.0041331496).
Retain clipping1.0. All prior commands are terminally closed; no active worker.

Question: does global L2 gradient clipping0.5 versus incumbent1.0 improve the retained
1200-update subword MLA recipe at common peak LR0.00125? Apply clipping after backward
and before schedule-rate application, independent decay, Muon/AdamW updates and EMA.
Use error_if_nonfinite=True. PyTorch's epsilon1e-6 affects the exact coefficient even
at the threshold. No learned parameters, RNG draws or cache change is proposed.

The incumbent remains unit gains/D256 MLA score0.5/SwiGLU672/SiLU/blockRMS/serial/no
product norm/no local mix/no warmup/independent decay0.00006/cooldown800/H200start800.
Lower clipping changes gradients only where the coefficient changes; Muon normalization
and Adam moments mean raw gradient scaling need not scale updates proportionally.
No benefit or clipping frequency is assumed. Actual pre/post norm and coefficient
measurement is required before scientific training. Do not choose thresholds from
pilot loss or expand the finite comparison.

The original deferred direction incorrectly described the final readout as RMSNorm.
Correction from accepted H200/raw inspection: it is affine LayerNorm epsilon1e-5,
with256 learned scale and256 bias parameters; all eight pre-branch norms are RMS.
The original direction-decision.json is preserved as historical evidence. The new
boundary question isolates final centering with scale, bias and epsilon matched.
See TOKENIZER_READOUT_CENTERING_PREPARATION.md for the current next action.

Generated optimizer-component001 tested prescribed norms0,.25,.5,.75,1,2 against an
independent float64 global-norm/coefficient oracle. Maximum gradient error was
2.2522872067387567e-8. Audited clipped gradients followed by manual shrinkage and
independent child steps exactly matched the actual HybridOptimizer. Serialized
in-memory resume exactly matched parameters and optimizer state; foreign threshold
identity, NaN/Inf rejection before update, gradient-absent preservation and unchanged
RNG passed. Serialized payload stayed below1MiB. CPU-only guarded evidence, no corpus
or scientific seed, no full-model/MPS or quality claim. Do not rerun this closed check.

Full-model model001 failed its coefficient oracle assertion: CPU FP32 reduction
error yielded coefficient0.3771889806 versus independent FP64 oracle0.3771862250,
a2.76e-6 difference above the2e-6 absolute bound despite the allowed2e-5 relative
norm tolerance. Original source/plan/log/failure remain closed. Separate model002
keeps norm tolerance and exact direct-PyTorch gradient parity, using consistent
relative2e-5 coefficient tolerance. Both CPU/MPS arms passed. Model math and the
actual clipping helper were unchanged; no scientific endpoint was retried.

The generated four-update fixture measured pre-norms1.0693–1.1661 with coefficient
ranges0.85760–0.93511 at1.0 and0.42881–0.46760 at0.5. Every one of these generated
updates clipped; this is not a scientific training frequency estimate. Step2 was
serialized only in capped memory; only final-step4 raw/H200 files were written.
Final H200 retained88EMA tensors and196608B cache per item, unchanged by threshold.
The additive trainer records complete per-update pre/post norms and coefficients
in raw resume state and final summary, plus evaluated metric rows. That actual
run_training/output/worker path now passes the bounded generated test below.

The actual workers retained all four measured per-update clipping observations in
raw resume state and summary; evaluated metric rows match their corresponding
updates exactly. Independent acceptance rejects missing, misordered, foreign-threshold,
nonfinite and inconsistent coefficient histories. Generated worker pre-norms were
1.2937–1.3033; coefficient ranges0.76733–0.77297 and0.38372–0.38649 respectively.
These four generated updates per arm all clipped, not a scientific frequency or
benefit estimate. Both parent full-validation reports recomputed exactly; CPU/MPS
CE differences3.0137e-7/2.6949e-7. Paired H200 cache/reset checks passed batches1/8.
Timing is descriptive generated-fixture evidence, not a production advantage.

Historical worker-stage next action (completed below): finite20-update resource-pilot orchestration
and complete THREE-fresh-pair fixed-LR design outside frozen673 sources, reusing tested
serial/acceptance/inference/retention primitives. Integrate/re-audit actual successful
worker raw/H200/optimizer/EMA/RNG/clipping history/cache and finite contracts, close
tensor dependencies, perform a fresh scientific seed census and measure complete
retained-output high-water including serialization overlap/metadata/8GiB reserve.
Use exact authorized recovery if needed, then finite pilots and final quality freeze.
Common LR0.00125, zero LR-search budgets, original strict gates; no screens/selection/
rescue/retry/partial panel. Standard-recipe scope, closed holdout, no old score pooling.

The new adapter integration closed successful resume tensor dependencies before
resume-retirement001 independently audited and retired exactly four generated
weights totaling1,118,316,500B;40,488 survivors remained unchanged. Original sources,
config/log/receipts remain immutable and later manifest resolves absent tensors.
Never restore or rerun that fixture. Actualfree10,449,551,360B exceeded complete
worker-pair9,814,671,360B including reserve and metadata, enabling both workers.
New successful worker raw/H200 weights1,118,322,004B remain retained until pilot
integration closes dependencies and a new exact audit. Current free space is not
scientific-panel readiness; source work proceeds and additional authorized lossless
recovery can be inventoried when the measured full high-water needs it.

Storage010 recovered1,105,700,860B by losslessly archiving23 closed quality001/hybrid001
RAWs, verified49,467 survivors and exact restoration. All selected/fresh raw bytes
remain recoverable, NOT retired; all H200 files remain original-path. Manifests and
protected archives resolve original identities. Current generated worker weights
remain retained. This resolved the resume-pair capacity requirement before execution;
new scientific-panel capacity must be measured separately.

All earlier campaigns remain closed. Branch half-gain mean+0.024273102061827024 and
all three seeds worse; unit gains retained. Preserve every selected/fresh model,
original failure sources/logs/configs/receipts and protected archives. Ninety-eight
historical RAWs remain losslessly recoverable via storage006/007/008/009/010 manifests
and protected archives, NOT retired; all H200 original paths and storage004 fixture
archive remain intact. Never restore permanently retired generated payloads or rerun
closed fixtures. Current free space is not a new-panel guarantee. Small source and
numerical work proceeds; exact authorized recovery needs dependency closure, path/
size/hash allowlists, shared locks/process checks, durable journals and survivor
verification. One local worker/four threads, all resource guards, no paid compute
or subagents. Approval-review timeouts are execution blockers, not missing consent.


## Gradient-clipping resource pilot001 complete

Both20-update clip1/cliphalf arms independently passed actual training, bounded
raw/H200 publication, complete clipping-history acceptance, full validation and
paired full/cache/reset inference at batches1/8. Each completed20,480targets,
8H200snapshots and88EMA tensors; cache196608B/item. Independent final001 resource
review passed. All679 campaign sources frozen,673 runtime predecessors unchanged.
No scientific quality result; pilot losses cannot select a threshold.

Exactly three prospective fresh pairs use common LR0.00125 with zero LR-search,
original strict gates and no rescue. Scientific seeds199520011/012/013 and
bootstrap199520004 remain unused; pilot199520001/evaluation199520003 were consumed.
Next new research/gradient_clip_quality_v1 must integrate/audit actual pilot
payloads, complete clipping histories and six-trajectory contracts before exact
pilot retirement, capacity/seed recheck and final quality freeze. Retain clip1.0.

Worker-retirement001 reclaimed1,118,322,004B with40,676 survivors unchanged after
tested pilot integration closed tensor dependencies. Original evidence survives.
Storage011 losslessly archived16 closed MLA-score/wide-attention selected/fresh RAWs,
reclaimed2,653,116,732B and verified49,904 other files unchanged. Exact
restoration passed. Original bytes resolve through storage-recovery-011/raw-completed.json
and protected-raw-storage-011 archives; NOT retired. Together006–011 preserve114
archived RAWs. All H200 originals and storage004 fixture archive remain intact.

Final review free11,838,726,144B plus retained pilots1,118,333,904B conditionally
yields12,957,060,048B versus complete12,381,585,408B including8GiBreserve.
Recheck actual capacity after quality-source dependencies close and an independent
exact pilot audit; no speculative deletion or partial scientific panel.

Observed pilot clipping: clip1 clipped0/20 updates, pre-norm0.40428–0.82294;
cliphalf clipped13/20, pre-norm0.40428–0.81842. These20-update observations
are not full-horizon clipping-frequency estimates or threshold-selection evidence.


## Gradient-clipping quality001 frozen and first trajectory dispatched

New research/gradient_clip_quality_v1 passed23 finite-contract tests and actual
pilot raw/H200/optimizer/EMA/RNG/clipping-history/full-validation/cache re-audits;
pilot-to-quality forgery was rejected. All687 active sources frozen,679 predecessors
unchanged. Exactly six1200-update trajectories use three paired fresh seeds at
common LR0.00125, zero LR-search and original strict gates. No screens, selection,
rescue, retry or partial panel. Every fresh raw/H200 pair remains protected.

Complete per-update clipping histories are checked against raw/summary/metric
records and when validating the accepted prefix. Final independent statistics
also report per-pair clipping observations descriptively, never for selection.
Prepared quality001-fresh-inference001.py and quality001-final-review001.py require
the exact freeze and all six accepted endpoints. Freeze SHA:
8574e2c338d9728748492329cda236657e2e8c1f0cbf4b78c926986380302d60

Exact pilot retirement reclaimed1,118,333,904B with28,383 other files
verified unchanged. Original receipts resolve absent weights through
quality001-pilot-retirement/completed.json. Freeze free12,962,930,688B exceeded
complete12,381,585,408B including8GiBreserve. Recheck remaining full-panel capacity
each dispatch. Historical selected/fresh bytes and114 protected RAW archives
remain intact. First fresh slot00 clip1 is dispatched; no paired quality result.
Seeds199520011 now in use;012/013 and bootstrap004 unused at first launch.


## Publication failure closed; zero-update continuation quality002

Original quality001 slot00 completed1200updates,1,228,800targets,41H200snapshots
and both raw/H200 files, but worker.json publication exceeded the256KiB miscellaneous
metadata group by6,295B. Its process receipt publication also failed. STOP and all
partial files/source/log/config/weights remain immutable; no endpoint was accepted
in the original root. The earlier test bounded history alone and missed aggregate
metadata high-water. No training was repeated and no seed was replaced.

Separate publication-recovery001 independently verified the actual final raw/H200,
optimizer/EMA/RNG, all1200 clipping observations, full validation and cache. It
exercised full report publication using existing2MiB resource-audit/resource-events
groups; no guard or budget was raised. Additive v2 integration verified the same
six scientific configs (only output paths differ for remaining slots), forbids
slot00 training, rejects replacement seeds and re-audits actual retained artifacts.

Quality002 frozen SHA d417380ed2581f917936efa85c6caee4ecb24bd8ef680d63864f864342761885
imports slot00 with zero updates and zero weight copies. New691-source plan
includes all687 unchanged quality001 sources. This continues exactly the original
three fresh pairs at LR0.00125 with zero LR-search and original strict gates.
Imported original raw/H200559,261,160B are protected fresh evidence, not obsolete
failed weights. Remaining previously unrun slot01 cliphalf is active; no campaign
quality conclusion. Follow quality002 anchors and the new freeze only.

Complete12,381,585,408B allowance and8GiBreserve are unchanged; existing imported
weights are charged, leaving11,822,324,248B required versus12,381,995,008B measured.
All691 sources remain frozen. Prepared quality002-fresh-inference001.py and
quality002-final-review001.py require all6 accepted endpoints and exact new freeze.
Preserve original failure evidence and all historical protected archives.


## Gradient-clipping quality002 closed without adoption

All six fresh1200-update endpoints and three paired inference reviews passed.
Independent final arithmetic and text bootstrap gave clip0.5-minus-clip1.0 mean
**+0.004133149591824823 nats/token**. All three seeds worsened and all original
quality gates failed. Retain clipping1.0 and the adopted unit-gain serial MLA recipe.

| Seed | Clip1 H200 CE | Clip0.5 H200 CE | Delta | Clipped updates: 1 / 0.5 |
|---|---:|---:|---:|---:|
|199520011|4.6916474635|4.6962045766|+0.0045571131|30 / 966|
|199520012|4.7300379660|4.7329504182|+0.0029124522|23 / 951|
|199520013|4.6964113898|4.7013412733|+0.0049298835|34 / 962|

Counts are descriptive coefficients below one out of1200 updates per trajectory,
not selectors or causal mediation estimates. Simultaneous conditional text-bootstrap
upper bounds at blocks2048/4096/8192 are +0.0058927768/+0.0065718466/+0.0067075439.
These condition on the three trained seeds and reused validation text, not population
seed uncertainty. Closed holdout; fixed standard-recipe scope, zero LR-search, no rescue.

Terminal STOP closes all quality002 training/inference/final commands. Original
quality001 publication failure stays closed and immutable. Slot00 was imported with
zero repeat updates/copies; its original failed-root raw/H200 are protected accepted
fresh artifacts, never obsolete failed payloads. All six pairs total3,355,566,960B;
all691 frozen sources remained unchanged. Closure free9,629,437,952B is an
observation, not new-panel readiness. Preserve114 historical lossless RAW archives,
all H200 originals, selected/fresh bytes and the storage004 fixture archive.

See [independent review](tokenizer-evidence/gradient-clip-preparation-001/quality002-final-review.json)
and [closure](tokenizer-evidence/gradient-clip-preparation-001/quality002-closure.json).


Storage014 COMPLETE: six closed gradient-clip quality002 fresh RAWs losslessly
archived,654,173,784B reclaimed,51,335 other files verified unchanged and exact
restoration PASS. Original RAWs resolve through storage-recovery-014/raw-completed.json
and protected research_archives/protected-raw-storage-014/*.tar.zst. This includes
accepted imported clip slot00 in the original publication-failed quality001 root;
it is accepted FRESH, never obsolete failed payload. No bytes retired.153 RAWs total
remain recoverable in storage006–014; all H200 originals and current readout fresh
endpoints stay loose/protected. Slot02 capacity shortfall was before process/request
creation; no quality retry. Actual free11,577,294,848B exceeded remaining11,263,061,680B
and slot02 resumed under the unchanged freeze/guards.
