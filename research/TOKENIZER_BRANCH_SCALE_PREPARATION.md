# Tokenizer fixed residual branch-scaling preparation

Current status: standalone component001 PASS all12 generated cases. No full-model
integration, scientific seeds, resource pilot/quality plan or active worker.
Cooldown quality001 is CLOSED_NOT_ADOPTED; cooldown800 remains retained.

Compare unit gain with fixed0.5 scaling of both complete post-projection residual
branches at unchanged depth: u=x+g*A(N1(x)); y=u+g*F(N2(u)), g=1 or0.5. Keep separate
RMSNorms and serial composition. Gains are constants, add no learned parameters
or RNG draws, and must remain explicit rather than folded into H200 projections.
The unit arm must exactly reproduce the incumbent. Retain D256 MLA score0.5,
SwiGLU672/SiLU/blockRMS/no product norm/no local mixing/no warmup/independent decay
0.00006/cooldown800/H200. Parameter/optimizer grouping/cache storage are expected
unchanged but require actual full-model verification.

At the boundary both architecture and bounded HP were reviewed. Fixed branch
scaling was the deferred distinct architecture question. Historical character
static0.1 and learned0.1 gains worsened every fresh seed; that adverse prior remains
explicit. This question uses a different gain magnitude, tokenizer/backbone and
horizon, without pooling old scores/seeds, reopening campaigns or assuming benefit.
Global gradient clipping0.5 versus incumbent1.0 is deferred. Its actual clipping
frequency and Muon/AdamW ordering would require measured instrumentation before a
future finite experiment. The closed cooldown grid is not extended or rescued.

Frozen component001 binds three new source files and623 unchanged predecessors.
All12 unit/fixed_half cases passed at batches1/3 on CPU FP64/FP32 and MPS FP32:
explicit independent causal equations, every parameter/input gradient, selected
CPU finite differences, unchanged prefix under suffix perturbation, exactly zero
future-input gradient, identical shared weights and construction RNG. Unit outputs
exactly match the direct serial reference; the candidate intentionally differs.
Maximum forward/gradient errors were5.662441253662109e-7/1.430511474609375e-6.
Distinct normalizers and supported gain kinds are enforced. This generated
one-head attention/SwiGLU composition is not an actual MLA/full-model/cache/
optimizer/checkpoint or scientific quality claim. No corpus or scientific seed
was used. Original sources/plan/log/result remain immutable; do not rerun.

Next: narrow additive explicit full-model policy/optimizer-group/checkpoint identity,
CPU/MPS raw-model/EMA/RNG resume, final-horizon H200 and full/cache/reset integration
outside frozen namespaces; do not bypass serial/unit-only guards. Then an actual
bounded worker and independent parent full validation/inference, fresh seed census,
complete measured retained-output high-water including serialization overlap,
metadata and8GiB reserve, finite resource pilots and final quality freeze.
Prospective comparison: exactly three fresh pairs at common peakLR0.00125, equal
zero LR-search budgets, original strict gates, no screens/rescue/partial panel;
standard-recipe claim only. No new scientific seed or training plan is reserved.

Current free space is an observation, not next-panel capacity. Small source and
numerical work proceeds; authorized exact retirement/lossless archival requires
closed dependencies, allowlisted paths/sizes/hashes, shared locks/process checks,
durable journals and survivor verification. Preserve every selected/fresh model,
original failure sources/logs/configs/receipts and protected storage004/006/007
archives. Sixty historical RAW files remain losslessly recoverable, NOT retired;
all H200 files stay at original paths. Permanently retired numerical payloads
cannot be restored and no closed fixture may rerun. One worker/fourthreads, all
resource guards, closed holdout, no paid compute or subagents.
