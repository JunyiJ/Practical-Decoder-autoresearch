# Tokenizer cooldown-onset preparation

Current status: generated schedule-component001 PASSED and is closed. No active
worker, scientific seed reservation, pilot freeze or quality freeze exists.
Parallel-residual quality001 is CLOSED_NOT_ADOPTED; serial remains adopted.

Test linear cooldown beginning at zero-based update600 versus incumbent800 over
1,200 updates. Both use peak LR0.00125 and final multiplier0.1, no warmup and
independent per-update decay0.00006. For start s, multiplier m(t)=1 for t<=s,
otherwise m(t)=1-0.9*(t-s)/(1199-s). Keep the adopted serial D256 MLA score0.5,
SwiGLU672/SiLU, separate block RMSNorms, no product norm/local mixing and H200.
H200 remains start800/every10/half-life200, giving41 snapshots at the quality
horizon. Earlier cooldown also changes total LR exposure: multiplier sums are
1020 versus930, or1.275 versus1.1625 at base LR0.00125 (8.82% less). This is not
an equal-integral schedule-shape comparison.

At the boundary both architecture and bounded HP were reviewed. Fixed serial
residual branch scaling is a distinct deferred architecture hypothesis requiring
explicit gains, initialization/checkpoint/cache contracts. Historical character
schedule evidence at3,000 updates does not answer this subword1,200-update onset
question; do not pool historical scores/seeds or assume improvement.

The frozen schedule-component001 plan binds601 sources, including600 unchanged
predecessors. Both complete schedules match an independent rational oracle;
width-scaled group rates are exact. Generated CPU float64 SGD parameter/momentum
resume matches exactly after zero-based updates0/598/599/600/799/800/1198.
Absolute-step reapplication does not compound rates; malformed schedules are
rejected. Source/plan/log/result/guard are immutable under
[tokenizer-evidence/cooldown-preparation-001](tokenizer-evidence/cooldown-preparation-001).
This is toy-SGD evidence only, not actual Muon/AdamW/model/EMA/H200/cache evidence.
No research corpus or scientific seed was used and no panel was launched.

Next implement narrow additive explicit full-model Muon/AdamW schedule group-rate,
checkpoint identity and raw/EMA/RNG resume/final-H200/full-cache-reset integration
outside frozen namespaces. Then actual bounded worker and independent parent
validation/inference, fresh scientific seed census, measured complete retained
output high-water including serialization overlap/metadata and8GiB reserve,
finite resource pilots, and final quality freeze. Prospective quality design is
exactly three fresh pairs at common LR0.00125, zero LR-search budgets, no screens,
selection, rescue or retries, original strict gates and standard-recipe scope.
Do not weaken guards or start a partial panel to fit storage. Current free space
is not readiness; source work proceeds while any necessary exact recovery is
prepared. Preserve selected/fresh models, all original evidence and protected
storage-recovery-004/006 archives; archived RAW bytes are not retired.
