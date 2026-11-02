# Tokenizer global gradient-clipping preparation

Current status: frozen optimizer-component001 PASS four generated CPU FP64/FP32
cases using real Muon/AdamW, thresholds1.0 and0.5. All660 source bindings unchanged,
including659 predecessors. No actual full-model clipping-frequency, MPS, EMA/H200,
cache or quality claim. No scientific seeds, pilot/quality freeze or active worker.

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

Next executable action: narrow additive full-model integration outside frozen660
bindings. Enforce explicit1.0/0.5 threshold identity and unit-gain adopted model;
verify identical shared initialization/logits/RNG and optimizer groups. Measure real
per-update pre/post gradient norms and coefficients, retaining the existing clipping
order and nonfinite failure behavior. Test CPU/MPS raw-model/optimizer/EMA/RNG repeat/
resume, foreign-threshold checkpoint rejection, final-horizon H200 and full/cache/reset
inference. Then actual bounded run_training worker and independent parent validation/
inference, fresh scientific census, complete measured retained-output high-water
including serialization overlap/metadata/8GiB reserve, finite resource pilots and
final quality freeze. Exactly three fresh pairs at common LR0.00125 with equal zero
LR-search budgets, original strict gates, no screens/rescue/retry/partial panel.
Standard-recipe scope, no old score/seed pooling, closed holdout.

All earlier campaigns remain closed. Branch half-gain mean+0.024273102061827024 and
all three seeds worse; unit gains retained. Preserve every selected/fresh model,
original failure sources/logs/configs/receipts and protected archives. Seventy-five
historical RAWs remain losslessly recoverable via storage006/007/008/009 manifests
and protected archives, NOT retired; all H200 original paths and storage004 fixture
archive remain intact. Never restore permanently retired generated payloads or rerun
closed fixtures. Current free space is not a new-panel guarantee. Small source and
numerical work proceeds; exact authorized recovery needs dependency closure, path/
size/hash allowlists, shared locks/process checks, durable journals and survivor
verification. One local worker/four threads, all resource guards, no paid compute
or subagents. Approval-review timeouts are execution blockers, not missing consent.
