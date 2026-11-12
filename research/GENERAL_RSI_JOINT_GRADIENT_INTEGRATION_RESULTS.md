# Joint-gradient integration qualification, 2026-11-12

All three arms passed the new accumulation and complete-restart checks.
This is an engineering result. No validation, test or generation scoring ran,
and no quality decision or selected model changed.

- Frozen plan: `tokenizer-evidence/general-rsi-joint-integration-001/plan.json`,
  SHA256 `b60d72721a0e71ec2d8db6d5950a5eb1ae64ebdcdab593b5d96cce600446b204`.
- Acceptance: `tokenizer-evidence/general-rsi-joint-integration-001/completion-review.json`.
  Attempt01 finished normally without failures or retries. The worker ended and
  the shared nonblocking lock was reacquired. All5,145 bound sources and the
  selected parent hash remain unchanged.
- Each arm's88 pre-clip parameter gradients exactly matched the independently
  calculated weighted component sum. Maximum absolute difference was0 in all
  three arms, within the prospectively fixed rtol1e−5/atol1e−7 integration gate.
  This checks accumulation wiring, not the old FP32 arithmetic qualification.
- Full saved states and replayed endpoints matched exactly for all three arms.
  Checks include model, optimizer children, RNG, natural/pseudo samplers, history,
  component counts, elapsed clock and EMA. Each arm rejected corrupted component
  counts before mutation. All50 shared natural input batches matched.

| Arm | Original updates | Replay updates | Mean update seconds | Restart bytes |
|---|---:|---:|---:|---:|
| Full CE | 70 | 22 | 0.343274 | 445,024,181 |
| Half CE | 68 | 21 | 0.354226 | 445,023,861 |
| Half CE + half pseudo | 50 | 16 | 0.486005 | 445,039,285 |

The joint arm consumed50 natural and50 pseudo batches with50 optimizer advances.
The other arms consumed70 and68 natural batches with matching advance counts.
All losses/gradients and checked model/optimizer state remained finite. No
pseudo target saturated. This short qualification used a24-second schedule and
EMA half-life1second to exercise serialization and replay; its losses are not
quality-selection evidence. Timing is descriptive of this local run, not a
statistical speed comparison.

Total worker time was111.703seconds, below the600-second bound. Peak RSS was
2,401,894,400bytes; sampled MPS allocation was2,510,077,952bytes. Three retained
restart files total1,335,087,327bytes. Admission included all three payload caps,
a possible256MiB failure payload,512MiB serialization overlap,64MiB metadata,
10GiB reserve and another2GiB headroom. No cleanup was needed. Post-review free
space was18,588,930,048bytes. The source, receipts, logs and restart payloads remain.

The hypothesis remains unresolved: combining gradients before one clip and
advance may reduce prediction damage while preserving repetition suppression.
The qualification establishes that this implementation can test it. The next
executable action is to implement and freeze the three-parent100-natural-update
comparison of full CE, half CE and the half-CE/half-pseudo mixture. Use fresh
paired roles, rotated order, the same five probe horizons1/10/25/50/100, and fixed
disjoint training-only natural windows and synthetic repetition probes. Report
all three contrasts, clipping, natural CE, pseudo loss and repeated/previous-copy
probabilities. Freeze the actual probe inventory and aggregation before model
scores, then admit the entire diagnostic using these measured update costs.
No quality panel is admitted by this qualification. Preserve all closed results.
