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
guards or interim stopping. No scientific seeds, pilots or quality plan frozen.

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
Original plans/logs/guards/results and final raw/H200 pairs remain retained.
These generated fixtures are closed; no scientific seeds, pilot/quality plan,
quality result or active worker exists.

Next minimal additive actual run_training/output/worker and independent parent
full-validation/inference adapters, reusing tested primitives. Integrate and
re-audit the successful resume payloads, then close their tensor dependencies.
Before the complete worker pair, independently audit exact retirement of the
four successful generated weights (1,118,314,132B) if needed, preserving source,
configuration, logs and receipts. Observed free8,761,184,256B versus prospective
worker-pair9,814,671,360B (8GiB reserve, per-arm metadata and16MiB shared allowance)
has a concrete conditional recovery route: after eligible exact retirement,
9,879,498,388B. This is not deletion or dispatch readiness; source integration
proceeds and actual free space must be remeasured. Scientific-panel output
high-water remains separate. Then worker validation/inference, fresh scientific
census, finite resource pilots and final three-pair fixed-LR quality freeze.

Closure free space is only an observation; source and small numerical work can
proceed. Recheck actual full retained-output high-water with8GiB reserve before
any training dispatch. Preserve every selected/fresh model, original evidence
and exact archive identities. See parallel-residual-preparation-001/direction-decision.json.
