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

Next implement narrow explicit full-model policy/optimizer/checkpoint identity
and CPU/MPS raw-EMA-RNG resume/final-H200/full-cache-reset integration outside
the559 bound sources. Do not bypass existing serial-only model/checkpoint guards.
Then actual worker/parent validation/inference, fresh census, complete measured
retained-output high-water plus8GiB reserve and finite pilots before a final
three-fresh-pair quality freeze. No numerical worker is active.

Closure free space is only an observation; source and small numerical work can
proceed. Recheck actual full retained-output high-water with8GiB reserve before
any training dispatch. Preserve every selected/fresh model, original evidence
and exact archive identities. See parallel-residual-preparation-001/direction-decision.json.
