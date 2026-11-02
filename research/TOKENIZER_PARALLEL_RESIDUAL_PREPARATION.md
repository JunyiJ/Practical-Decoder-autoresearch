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

Next prepare a small standalone composition source and freeze its generated
CPU FP64/FP32 and MPS FP32 forward/all-gradient/causality cases. Compare each arm
against its own explicit equation using identical shared matrices. Then implement
narrow explicit full-model/optimizer/checkpoint/raw-EMA-RNG resume/final-H200/
full-cache-reset contracts outside the556 frozen sources, actual worker/parent
validation/inference, fresh census, measured full-output capacity and pilots.
Do not bypass existing serial-only model/checkpoint guards or reopen old fixtures.

Closure free space is only an observation; source and small numerical work can
proceed. Recheck actual full retained-output high-water with8GiB reserve before
any training dispatch. Preserve every selected/fresh model, original evidence
and exact archive identities. See parallel-residual-preparation-001/direction-decision.json.
