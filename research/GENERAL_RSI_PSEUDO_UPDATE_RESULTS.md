# Common-state pseudo-update diagnostic, 2026-11-12

The first-step diagnostic does not support uniform direct pseudo-gradient harm.
Pseudo-minus-zero loss differences have mixed signs across the three parent
means. Zero-gradient optimizer advances raise loss on all three fixed training
panels, but ordinary CE steps also raise it. This early reset-optimizer state
does not isolate the cause of the completed long-run regression.

## Frozen design and execution

`tokenizer-evidence/general-rsi-pseudo-update-001/plan.json`, SHA256
`301aba9d8ff542de0a9d87067bf8058b367ef1beb9667bd8046ea75727fdbcd8`,
was frozen with input inventories and full admission before execution. Each
selected native parent, slots01/03/05, received one fresh CE update to seed its
reset optimizer. Eight cases then compared identity, explicit zero gradients,
pseudo gradients and fresh CE gradients from exactly that complete common state.
Every branch restored the same parameter, optimizer, sampler, timer and RNG
digest. All24 identity evaluations reproduced the baseline exactly.

Each parent used64 fixed training-document windows,128 targets each. Evaluation
positions were disjoint from every seed-CE, branch-CE and pseudo source range.
These are training windows already available during parent training, not a new
holdout. No validation scores, test contents or generations were opened. All96
branch records and3 parent records are retained. No model payload was saved.

The optimizer uses child weight_decay=0 and the wrapper multiplies every
parameter by1−0.00006 on each step. Thus the zero branch combines momentum
advance and independent decay. It does not separate those two mechanisms. This
clarifies the generic installed-Muon source observation in the earlier boundary
note without changing its historical evidence. Original clipping1.0 and base
LR0.00025 remain fixed. Gradient/update cosines use post-clipping gradients.

## Measured contrasts

Values below are means over eight cases, in nats/target. Positive means higher
loss for the first branch. Identity and zero repeat the same deterministic
operation within each parent; they are not eight independent observations.

| Parent | Pseudo−zero | Zero−identity | Pseudo−identity | CE−identity |
|---|---:|---:|---:|---:|
|01|−0.000149591|+0.000381660|+0.000232069|+0.000383577|
|03|+0.000020686|+0.000257678|+0.000278364|+0.000406735|
|05|−0.000128067|+0.000557322|+0.000429256|+0.000592465|

Pseudo−zero is negative in16/24 cases and positive in8/24. Its range is
−0.000401245 to+0.000125944. Pseudo−identity is positive in23/24 cases.
All24 CE−identity differences are positive. The intended positive learning
control therefore does not show an immediate benefit on these disjoint windows.
That limits attribution; it is not an implementation failure or a reason to
discard any case. No confidence interval or final-quality claim is made.

Mean Muon update norms are0.03763–0.03782 for zero,0.03720–0.03734 for pseudo
and0.03808–0.03822 for CE. AdamW norms are0.23116–0.23932,0.23904–0.24493
and0.32055–0.32346 respectively. Pseudo gradient/update cosines average
−0.2495 to−0.2618 for Muon and−0.0714 to−0.0740 for AdamW. Similar update
norms do not establish similar directions or long-run behavior. These measurements
support inspecting optimizer dynamics rather than treating a small pseudo loss
as a small update. Full values and input receipts are in `analysis.json`.

Execution took247.653s, including binding, loading, cloning and scoring, below
the467.021s forecast and600s ceiling. Peak RSS was2,046,558,208bytes; maximum
MPS driver allocation sampled at boundaries was2,497,495,040bytes. Limits were
4GiB and6GiB. The final guard observed19,939,561,472 free bytes against10GiB
reserve. The complete output/overlap allowance was32MiB. No cleanup, failure,
retry, scientific threshold change or checkpoint serialization was needed.

## Updated hypothesis and next discriminating question

The closed quality result remains ordinary-CE superiority by0.144615nats/token.
Its matched natural-batch logs also remain adverse. Their first100-batch means
are+0.034655,+0.039440,+0.027976, and the next100-batch means increase to
+0.073064,+0.075841,+0.064138. Those existing observations motivate studying
accumulation. The one-step diagnostic uses fresh seeds, separate training windows
and a reset optimizer with just one prior CE update, so it cannot quantify how
much of that longer trajectory gap comes from momentum, decay or pseudo gradients.

The next executable action is to design and admit a short accumulated-state
diagnostic with ordinary CE, CE interleaved with explicit zero-gradient advances,
and CE interleaved with pseudo updates. Use the same natural-batch prefix within
each fresh paired bundle and evaluate at matched natural-update boundaries.
This is a proposed mechanism control, not an additional quality panel or a
batch-reuse optimization arm. The first100 natural updates are an evidence-based
candidate horizon because divergence is already present in that original block.
Freeze fresh roles, exact boundaries and full cost before execution. Do not tune
the pseudo coefficient or claim a new optimum from this result.

The primary-paper basis and contrary findings remain those reviewed in
`GENERAL_RSI_PSEUDO_MECHANISM_BOUNDARY.md`: DITTO and sequence unlikelihood report
benefits under different scales and selection policies, without establishing this
local optimizer-state contrast in the inspected sections. No new literature gain
is asserted. There is currently no active worker or newly admitted quality panel.
