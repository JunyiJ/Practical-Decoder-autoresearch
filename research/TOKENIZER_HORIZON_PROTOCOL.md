# GPT-2 width256 GQA duration and schedule experiment

Prospective finite protocol, October27,2026 (Los Angeles). The completed GQA
round adopted width256 half-KV GQA as a scoped serving recipe at LR0.00125.
The first width256 control still showed a sampled validation minimum at1200
updates and degradation by3000 while training loss fell. The previous LR round
showed similar behavior. Test a shorter complete training schedule before
continuing MLA and3:1 KDA/MLA comparisons in a potentially unsuitable regime.
This is an evidence-driven HP round following structural GQA research. MoE,
activation, normalization, residual and readout questions remain in the backlog.

## Fixed recipe and estimand

Use the exact model, data and optimizer from
`tokenizer-evidence/gqa-001/adopted-width256-serving-recipe.json`: width256,
four dense SwiGLU21D/8 layers,8 query/4 KV heads of32, RoPE, RMSNorm blocks,
affine final LayerNorm, untied vocabulary tables, score scale0.5, no output
bias, GPT-2 BPE50257, Muon/AdamW muP base64, independent decay0.00006,
LR0.00125, batch8/context128,1024 sampled targets per update. Train each
trajectory from scratch; do not resume or select an old checkpoint.

Compare1200,1800 and3000 updates. Linear cooldown starts at two-thirds of each
horizon (800,1200,2000) and ends at multiplier0.1. H200 averages from that same
start through the endpoint every10 steps:41,61,101 snapshots respectively.
Half-life remains200 updates. Evaluation diagnostics every300 updates are not
selectors. Only final full-validation H200 CE selects candidates. Record raw
CE, BPB, training time, target exposure and the different EMA tail spans.

This is a combined duration/proportionally timed cooldown/EMA-window comparison,
not an isolated duration effect or a change in architecture. Short runs see
fewer targets (1,228,800 /1,843,200 /3,072,000); report measured time, never call
these equal-compute or equal-token experiments. The prospective choices were
informed by reused validation curves, so this is adaptive validation research.
Do not claim the observed diagnostic minimum is an optimal horizon.

## Finite selection and confirmation

One60-update resource/correctness pilot, seed2791, cooldown/EMA start40,
interval10,3 snapshots, is excluded from quality selection.

Screen one paired seed2801 at1200,1800,3000 in that order. The single screen
seed is a selection device only. A shorter candidate qualifies for confirmation
if final H200 CE is more than0.02 nats/token below the3000 control. Choose the
lowest eligible CE, tie shorter duration. If neither qualifies, retain3000
and close after the three screens. Do not extend the grid automatically.

If one qualifies, freeze that duration and run fresh paired seeds2811/2812/2813
versus3000, alternating control/candidate order. Maximum nine quality trials.
Adopt the shorter schedule only if the paired mean improves by more than0.02,
every fresh seed improves, and all conditional text-bootstrap95% upper bounds
are negative (2048/4096/8192-token blocks,2000 draws, seed20261024), followed by
artifact review. Scope adoption only to this width256 GQA LR0.00125 recipe.
The general MHA baseline and width64 are outside this experiment. A failed
candidate is preserved, not silently retried or replaced. Both endpoints and
all checkpoints remain retained. Screen/confirmation seeds do not overlap the
previous tokenizer rounds. Test loss remains closed.

Same corpus hash and raw80/10/10 boundaries, tokenize each split independently;
full validation scores34278 targets and111536 bytes with context128 resets.
Never compare character CE with subword CE. Conditional text intervals are not
population uncertainty over training seeds or an independent test guarantee.
No new inference-cost adoption claim is made: model/cache geometry is unchanged.

## Resources and transition

One serial MPS worker on this Mac, four CPU threads, no paid compute/delegation.
Freeze source/runtime/corpus/protocol/recipe/plan. Pilot process limit600seconds;
quality process limit7200seconds. Memory guards: RSS4GiB/MPS driver6GiB. Keep8GiB
free, cap pilot at1GiB and quality output at6GiB. Project nine raw/H200 checkpoint
pairs from the new pilot with50% headroom plus the8GiB reserve before launch.
Do not delete research checkpoints to create headroom. Preserve failure logs.

Verify config equivalence outside declared schedule/seed/output fields; exercise
actual schedule endpoints, EMA timing/resume, strict selection completeness and
fresh-confirmation gates. Run existing schedule/averaging regression tests and
the serial resource pilot before quality launch. Publish each completed round
and its learnings. After this round, revisit MLA/3:1 KDA/MLA and the next broad
architecture family using the new evidence; avoid an indefinite HP-only loop.
Historical closed campaigns, paused null-attention and permanent stop006 remain
closed. A human stop stops new launches and pauses the heartbeat.
