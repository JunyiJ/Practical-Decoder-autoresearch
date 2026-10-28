# Wider direct MLA/GQA comparison: human steering

October28,2026. The human clarified that this campaign is a pilot for larger
models, requested wider targets and a fresh MLA-versus-GQA comparison, and asked
whether the hybrid had received detailed HP scanning. This supersedes the proposed
next activation round. No new training has launched; executable contracts,
resource pilots and a finite source/plan freeze are still required.

## What existing evidence means

The shorter recipe means1200 optimizer updates rather than3000, a linear LR
cooldown starting at800 rather than2000 and an H200 EMA tail starting at800 rather
than2000. Both end at0.1 of peak LR; H200 is averaging with a200-update half-life.
At batch8/context128 this is1,228,800 versus3,072,000 target-token exposures, often
revisiting the small training corpus. Three fresh width256 GQA pairs favored the
combined shorter recipe by0.710852 nat and used60% less training time. This is
consistent with the observed overfitting pattern, not proof that1200 updates is
optimal at larger widths, on more data, or under a different schedule.

The closed hybrid round was bounded, not an exhaustive HP scan: one screening
seed, two peak LRs (0.00125/0.0025) for each of hybrid/all-KDA/all-gated-MLA, then
three fresh pairs versus the selected all-KDA control. It held horizon, decay,
latent dimensions, KDA gate/decay settings and3:1 layer placement fixed. The
correct conclusion is rejection of that width64 tested recipe under those budgets,
not optimality or a rejection of the hybrid family. No failed panel is reopened.
A new wider hybrid study with broader but finite staged tuning remains eligible
following the direct attention comparison and resource review.

## Next question and provisional scale

Prioritize a fresh direct comparison of half-KV RoPE GQA and quarter-latent RoPE
MLA at width512 (16 query heads of32; GQA8 KV heads; MLA query/KV rank128 and
shared rotary dimension32). Width512 is a proposed next feasibility target,
not an established large-model proxy or a claim of transfer beyond that width.
Keep four layers, SwiGLU21D/8, legacy untied GPT-2 tables and existing norm,
score-multiplier/optimizer conventions. Explicitly validate larger geometry and
checkpoint lineage rather than bypassing the current width64/256 policy.

Measure width512 training/resource feasibility before choosing the final panel.
Account for width1024 as a future scale point; do not launch it or silently drop
to256 on a guard failure. Local memory/storage/time may limit a full retained
multi-seed panel; report that constraint with measured evidence. Larger-scale
recipe claims ultimately need fresh confirmation at additional widths and more
representative data/token budgets. Tiny Shakespeare/context128 remains a pilot.

## Fair tuning and adoption design to finalize before launch

Use equal token budgets and identical data boundaries, independent bounded LR
screens for BOTH attention families, then three fresh paired confirmation seeds.
Propose three rates around the inherited recipe (0.000625/0.00125/0.0025) rather
than treating the width256 choice as optimal. The1200/cooldown/H200800 schedule
can serve as a fixed first comparison budget; its wider optimality is explicitly
untested. A separate schedule/regularization follow-up remains first-class.
Declare rates, seeds/order, selection and panel budget in an executable protocol;
this note is not a source freeze or launch plan.

Use a prospective MLA-minus-GQA quality-neutrality band and actual inference cost
for a scoped memory-saving option. Proposed +0.01 nat retains the prior engineering
convention (not a human numeric tolerance), requiring mean, every fresh seed and
all declared conditional interval upper bounds to pass. Measure actual cache and
parameter storage, cached correctness, prefill/decode latency and throughput at
stated batches/context. Freeze the minimum meaningful cost benefit before pilots
and quality; do not require a speed win for a genuine memory-saving variant.
Publish both sides of any tradeoff and allow no-winner outcomes. Do not pool old
unpaired GQA/MLA seeds or declare either universally superior.

Single worker, no paid compute/delegation, strict MPS, retained raw/H200 outputs,
8GiB disk reserve and existing sampled memory/time guards remain mandatory.
Compute the full panel storage projection and CPU correctness/resume/EMA checks
before frozen serial pilots; validate actual pilot costs before quality dispatch.
All old controllers, stop markers and cleanup manifests stay closed/authoritative.
The test holdout stays closed. Broad architecture and HP exploration remain active.

## Implementation preparation, October28

Explicit wider-only policies now cover GQA/MLA at512/1024 while historical
policies still reject those widths. Meta-device geometry/optimizer accounting
finds62,877,696 GQA and62,424,192 MLA parameters at512; at1024 these are
148,561,920 and146,606,208. This is allocation/count verification, not tested
width1024 numerical training or serving correctness. Full retained14-trajectory
projections with reserve are32,836,217,809 and65,654,375,377 bytes respectively.
Measured available space at accounting was about35.76GB; only512 fits.
See `tokenizer-evidence/wide-attention-preparation-001/storage-projection.json`.
CPU continuation fixtures use in-memory serialization to avoid consuming the
retained research-checkpoint budget. Pilot protocol is separately recorded in
TOKENIZER_WIDE_ATTENTION_PILOT_PROTOCOL.md; no quality result follows from CPU
checks or resource feasibility. Actual disk exports are verified by those pilots.
