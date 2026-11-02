# Tokenizer parallel-residual preparation

The fixed-LR decay comparison is fully closed: zero decay worsened every seed
(mean+0.0091651929 nats/token), so independent decay0.00006 remains adopted.
Both architecture and bounded HP questions were reviewed at this boundary.
Select the previously queued separate-norm residual topology question. Defer
cooldown start600 versus800 as a separate prospective schedule comparison.

For each block, control uses u=x+A(N1(x)), y=u+F(N2(u)). Candidate uses the
same u but y=u+F(N2(x)). N1 and N2 remain distinct affine RMSNorms, epsilon1e-5,
gain initialized1, no bias. Keep MLA score0.5, SwiGLU672/SiLU, no product norm,
no local mixing, no warmup, independent decay0.00006 and H200. Branch gains stay
one. This changes dependency structure; it does not implement concurrent kernels.

The historical character MHA/LayerNorm parallel study worsened every fresh seed
at both widths. That is a negative prior. The distinct question here concerns the
adopted subword MLA/blockRMS backbone and short1200-update recipe; no gain is
assumed, and no historical CE or seed is pooled. Parameter shapes/construction
RNG must be identical across arms; initial functions and gradients should differ.
Expected28,492,416 parameters,88EMA tensors and196,608FP32 cache bytes per item
must be verified in actual full-model integration, not asserted from this plan.

Prospective quality design: exactly three fresh pairs at common LR0.00125,
equal zero LR-search budgets,1200updates/cooldown800/H200800, original strict
quality gates and all three paired inference reviews. This is a standard-recipe
comparison, not separately tuned optima. No LR rescue, partial panels, relaxed
guards or interim stopping. The current resource-pilot status and remaining quality-freeze gate are below.

Standalone component001 passed all12 generated cases: serial and parallel,
batches1/3, CPU FP64/FP32 and MPS FP32. Independent tokenwise equations agree
with maximum forward error5.960464477539062e-7 and all-gradient error
1.9073486328125e-6. CPU FP64 finite differences passed. Suffix perturbations
leave prefix outputs unchanged and future-input gradients exactly zero.
Shared matrices and construction RNG match exactly; the topology functions
intentionally differ. Separate normalization and unknown-kind rejection passed.
All556 predecessor sources remain unchanged. The frozen source/plan/result and
guard are in parallel-residual-preparation-001/component001*. No corpus,
scientific seed, checkpoint or training panel was used. This generic one-head
composition check is not full-model MLA, optimizer or cache evidence.

Full-model model001 passed both arms with identical shared initialization/RNG,
28,492,416 parameters,88EMA tensors and identical Muon/AdamW group membership.
The serial arm exactly matches adopted-model initial logits; parallel logits
intentionally differ. CPU/MPS forward/gradient, uneven cached chunks/reset and
causality passed. Both arms use196,608B cache per item. Foreign topology
checkpoints/caches and a mutated block were rejected before use. The new named
MLA separate-RMS policy is isolated under parallel_residual_v1/runtime; historical
serial-only contracts remain unchanged.

Serialized model-resume001 passed both four-update arms: fresh repeat and
interrupted step2 replay matched raw model/optimizer/EMA/RNG exactly (maxerror0).
Step2 serialization stayed in capped memory; final-step4 raw/H200 exports have
three H200 snapshots over88 tensors. CPU/MPS batch1/8 full/cache/reset inference
passed. All572 bound sources remained unchanged, including559 predecessors.
Original plans/logs/guards/results remain retained. The generated resume weights
were later retired by exact resume-retirement002; resolve absent paths there.
These generated fixtures remain closed and do not establish scientific quality.

Actual generated worker pair and trained-inference003 are complete. Both arms
passed bounded run_training publication and independent parent full validation:
4updates/4,096targets/3H200 snapshots/88EMA tensors each. CPU/MPS validation CE
agreement is within2.8e-7. Paired batch1/8 full/cache/reset checks passed; each
arm retains196,608B cache per item. Timings are descriptive fixture measurements,
not an adoption route. No scientific quality claim or seed reservation.

First worker-integration001 failed before training/deletion because the reused
validation-data reader resolved the historical serial-only model policy.
Original source/plan/log/failure stay closed. Additive parallel_residual_worker_v2
routes validation to the tested isolated runtime. Integration002 re-audited both
real resume payloads, optimizer/EMA/RNG, full validation and paired inference;
585 runtime/acceptance sources remained unchanged. Exact resume-retirement002
then retired four obsolete successful generated weights (1,118,314,132B), with
37,999 survivors verified. Original receipts resolve through that manifest.

Trainer-worker002 accepted serial slot00, then the remaining-space guard stopped
slot01 before process creation:9,196,064,768B available versus9,210,691,584B required.
Its failure and accepted prefix remain closed and immutable. Lossless
storage-recovery-005 compressed82 exact historical text files, reclaimed
76,353,536 allocated bytes and verified47,103 protected files unchanged; original
paths/readable bytes and all weights survived. Complete remaining capacity was
rechecked at9,435,860,992B. Separately bound trainer-worker003 independently
revalidated/imported serial slot00, executed only the unlaunched parallel slot01,
and accepted it. No repeated serial updates or relaxed/partial-panel guards.
Resolve accepted pair paths through trainer-worker003.json bindings: serial is
in prerequisite002 and parallel in prerequisite003; do not assume one root.

Finite20-update resource-pilot orchestration and the complete three-fresh-pair
fixed-LR design are now tested in additive parallel_residual_campaign_v1.
Pilot-integration001 failed before payload audit/training on a harness field-name
mismatch (residual versus topology); source/plan/log/failure remain closed.
Separate pilot002 entry and integration002 passed finite contracts and real
worker raw/H200/optimizer/EMA/RNG/full-validation/cache re-audits. All591 campaign
sources are frozen, including585 unchanged runtime/acceptance predecessors.
Retention-boundary001 passed exact pilot retirement and rejected selection,
fresh cleanup, incomplete/foreign plans and repeated execution. After dependency
closure, worker-retirement001 retired exactly four obsolete generated weights
(1,118,319,188B), verifying38,174 survivors; original002/003 receipts resolve
through that later manifest. No selected/fresh scientific payload was retired.

Storage-recovery-006 preserved32 older selected/fresh raw optimizer checkpoints
in protected lossless one-member Zstandard archives and recovered3,474,600,510B.
All original raw bytes are preserved, and every H200 model remains at its original
path. Full streamed member verification preceded each duplicate loose-copy
removal;47,303 other files were verified unchanged. An exact raw restoration to
an owned temporary copy passed size/SHA verification, then only that temporary
copy was removed. These raw checkpoints are NOT retired. Resolve their original
paths through storage-recovery-006/raw-completed.json and its protected archives
under research_archives/protected-raw-storage-006; materialize.py provides exact
restoration without any training. Never purge these archives as obsolete payloads.
The earlier storage-recovery-004 fixture archive remains protected.

Capacity001 measured13,405,265,920B free against the complete12,381,585,408B
requirement: six retained raw/H200 pairs, serialization accounting,256MiB metadata
and8GiB reserve. Fresh census001 scanned46,955 files/4,165,631,841B with no seed
collisions. Panel199190 is reserved: pilot001/eval003 are used; fresh011/012/013
and bootstrap004 remain unused. No LR-screen seed exists.

Both scientific resource pilots are COMPLETE: each20updates/20,480targets,
8H200 snapshots and88EMA tensors, independently accepted. Paired full/cache/reset
inference passed batches1/8,196,608B per item for both arms. Independent final001
resource review passed, including original guards, output bounds and conservative
1200-update timing projections. Pilot losses are not selection or quality evidence.
Final review retained1,118,328,784B pilot weights and observed12,055,719,936B free;
exact retirement after quality-source dependency closure would conditionally yield
13,174,048,720B. Remeasure actual capacity before dispatch; current pilots stay
retained until that new exact audit. No active worker or scientific quality result.

Next minimal additive finite1200-update quality orchestration in NEW
research/parallel_residual_quality_v1 outside all591 frozen campaign sources.
Reuse tested fixed-LR three-pair serial/acceptance/inference/retention primitives;
adapt serial/parallel_separate_norms,88EMA and196,608B cache identities. Verify
actual retained pilot payload/cache and exactly six fresh-trajectory contracts.
Close dependencies, independently audit exact successful-pilot retirement,
recheck full capacity and unused fresh/bootstrap seeds, then final quality freeze,
six fresh trajectories, three paired inference reviews and independent statistics.
Common LR0.00125, equal zero search budgets, no screens/rescue/retry or pilot-loss
selection; original strict gates and standard-recipe scope. All earlier campaigns
and failed fixtures remain closed. Architecture/HP direction remains parallel
residual topology with cooldown600-versus800 deferred; no boundary-quality result
has appeared to revise that decision.

Closure free space is only an observation; source and small numerical work can
proceed. Recheck actual full retained-output high-water with8GiB reserve before
any training dispatch. Preserve every selected/fresh model, original evidence
and exact archive identities. See parallel-residual-preparation-001/direction-decision.json.
