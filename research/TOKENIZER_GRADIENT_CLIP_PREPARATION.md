# Tokenizer global gradient-clipping preparation

Current status: model002 and model-resume001 PASS both thresholds1.0/0.5.
Identical initialization/logits/RNG/groups,28,492,416parameters/88EMA tensors;
CPU/MPS gradient, full/cache/reset and causality checks pass. Exact direct-PyTorch
clipped gradients, foreign-threshold checkpoint rejection and unchanged RNG verified.
Four-update repeated/resumed raw model/optimizer/EMA/RNG/clipping history agree exactly
(maxerror0); final-step4 raw/H200 with3snapshots and CPU/MPS batches1/8 pass.
All667 bound sources unchanged,660 predecessors preserved. No active worker,
scientific quality result, seed reservation or pilot/quality freeze.

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

At the boundary both architecture and bounded HP were reviewed. Defer a distinct
matched-epsilon final-readout LayerNorm versus incumbent final RMSNorm question,
keeping all pre-branch RMSNorms unchanged. Earlier block LayerNorm failed; this is
an adverse prior, not evidence for the different placement. Centering and affine
bias remain joint changes. That direction needs explicit policy/group/initialization/
checkpoint/cache checks before use. See gradient-clip-preparation-001/direction-decision.json.

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
in raw resume state and final summary, plus evaluated metric rows. Its actual
run_training/output/worker path still needs the next bounded test.

Next executable action: minimal additive worker/output and independent parent
full-validation/inference adapters outside frozen667 bindings, including explicit
clipping-history acceptance. Re-audit the actual retained resume raw/H200/optimizer/
EMA/RNG/history/cache payloads and close tensor dependencies. Then exact independent
retirement audit for obsolete generated weights if needed, and actual complete
worker-pair capacity recheck. capacity001 records the observed free space and the
1,118,316,500B retained generated-weight recovery route; no deletion before dependency
closure/audit. Source work proceeds. After worker/inference: fresh scientific census,
complete measured retained-output high-water including metadata/serialization overlap/
8GiB reserve, finite resource pilots, then exactly three fresh pairs at fixed common
LR0.00125 with zero LR-search budgets and original strict gates. No selection/rescue/
retry/partial panel; standard-recipe scope, no old score/seed pooling, closed holdout.

Storage010 recovered1,105,700,860B by losslessly archiving23 closed quality001/hybrid001
RAWs, verified49,467 survivors and exact restoration. All selected/fresh raw bytes
remain recoverable, NOT retired; all H200 files remain original-path. Manifests and
protected archives resolve original identities. Current generated resume weights
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
