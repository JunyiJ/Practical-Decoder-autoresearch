# Tokenizer MLA results

The ten-trial round and three serial inference comparisons are complete and
audited. Adopt width256 quarter-latent RoPE MLA at LR0.00125 with1200 updates,
cooldown/H200 start800 as a scoped option versus the equally tuned fresh MHA
control. The adopted GQA recipe remains separate: this round did not compare
MLA with GQA. General MHA and width64 recipes are also unchanged.
See [prospective protocol](TOKENIZER_MLA_PROTOCOL.md).

MLA support passed63 CPU checks (22.82seconds), then one source-frozen60-step
width256 MPS feasibility trial (11.29training seconds). Saved-artifact review
passed exact config, finite optimizer/weights,60 optimizer steps,3 EMA snapshots,
checkpoint hashes, training sampler replay, untouched test RNG and validation
arithmetic. This pilot's validation values are sanity data, excluded from quality
selection. [Pilot evidence](tokenizer-evidence/mla-pilot-001/result.json).

Full/cached FP32 MPS logits passed at batches1/8 (maximum absolute differences
3.34e-6/1.67e-6). Actual allocated latent-plus-rotary cache was196608/1572864bytes
at context128. These checks establish implementation feasibility, not comparative
quality or latency. [Cache checks](tokenizer-evidence/mla-pilot-001/cache-check.json).

The raw/H200 checkpoint pair uses559170172bytes. A new ten-trial quality storage
projection includes the larger preserved width256 MHA pair,15% overhead and a
separate8GiB reserve. [Resource review](tokenizer-evidence/mla-pilot-001/resource-review.json).
Completed temporary pytest fixtures were cleared only after saving logs and a
manifest; no research checkpoints were removed. All prior failures remain
preserved. Quality controller and MLA-specific benchmark integration are complete. The
initial41-test suite passed, followed by7 focused controller/guard checks,
including serial ten-trial progression, frozen selection, STOP behavior and
no restart of completed runs. The suites overlap; counts are not additive.
The benchmark retains GQA labels for legacy calls and uses explicit MLA labels
for the new comparison. Arithmetic/cache implementation remains unchanged
from the passed MPS pilot. Source freeze and fresh disk projection precede launch.

## Quality launch prerequisites

The quality protocol contains four width256 screening trials (one seed, two LRs
per family) and six fresh confirmation trials. Quality eligibility is separate
from inference memory/latency measurement and final adoption review. The7GiB
output cap accommodates the measured ten-pair projection plus15% overhead;
the8GiB free-space reserve is retained and checked against all remaining trials.
A global lock prevents a second training or timing worker. Source/runtime,
protocol/recipe/corpus and pilot/resource evidence are hashed into the plan.
Failure logs and all checkpoints are retained. Test remains closed.

## Launch

The ten-trial quality run started successfully on October27,2026. Frozen source
commit5276412, plan recorded in f22616a; plan SHA256
afb45c98f9185f8f659577ac7c1326140aa42349047be956c4ade06edac8ea10.
Full remaining-trial storage projection passed before launch. The controller
continues under the single-worker lock; source remains frozen.

## Screening review

Seed2901 selected LR0.00125 independently in each family: MHA H200 CE4.746541
versus4.825638 at LR0.0025, and MLA4.723452 versus4.809730. The frozen
lowest-endpoint rule therefore selects the same LR for both. Saved tensor,
optimizer, checkpoint lineage, validation arithmetic and sampler checks passed
for all four trials; the test RNG is untouched. The selected screen difference
is MLA minus MHA -0.023089nats/token, descriptive single-seed selection evidence
only. Fresh seeds2911/2912/2913, the declared neutrality gates and serial cached
inference measurements remain required before any adoption. See the
[selection audit](tokenizer-evidence/mla-001/screen-review.json).

## Final quality and inference review

All three fresh seeds favor MLA: paired H200 CE differences are -0.037222,
-0.030261 and -0.029838nats/token. Means are4.707100 MLA versus4.739540 MHA,
delta -0.032440. All frozen +0.01-nat neutrality gates pass. Separately
recomputed conditional95% text intervals are [-0.036286,-0.028357] for2048-token
blocks, [-0.037142,-0.027935] for4096 and [-0.035610,-0.029685] for8192.
These condition on the trained models and adaptively reused validation text;
they do not measure training-seed population uncertainty. Test remains closed.

Every fresh checkpoint and batch passes cached correctness and the prospective
60% cache-saving gate: actual allocated KV storage is81.25% lower. At batch1,
MHA uses1,048,576bytes and MLA196,608; at batch8,8,388,608 versus1,572,864.
The measured tradeoff is slower decode: median64-token decode is0.184746s MHA
versus0.223410s MLA at batch1 (20.93% slower), and0.188575s versus0.232976s
at batch8 (23.55% slower). Prefill medians are0.007046/0.008217s at batch1 and
0.013294/0.013547s at batch8 (MHA/MLA). Parameter storage is115,404,800 versus
113,969,664bytes. Cache savings are not equivalent to total-memory savings.

Timing uses FP32 MPS, context128,32 prefill/64 teacher-forced decode tokens
from training text, two warmups and eight alternating paired repeats per seed;
both models are resident. No production, long-context or cloud-cost claim.
All ten artifact reviews and source/runtime/protocol/recipe/corpus/resource
hash checks passed. Raw timing records and separately recomputed arithmetic
are preserved in the [final review](tokenizer-evidence/mla-001/final-review.json).
The [scoped recipe](tokenizer-evidence/mla-001/adopted-width256-mla-recipe.json)
records all three fresh checkpoints.

Next select explicit3:1 KDA/gated-MLA implementation and feasibility checks,
as prioritized by the human. This new topology needs NoPE and per-layer state
contracts; the closed all-KDA character result cannot reject it. The current
RoPE MLA recipe is not a NoPE/gated-MLA control without a new comparison.
MoE, activation, normalization, residual, depth and readout remain active
architectural hypotheses; LR/schedule/regularization remain valid standalone
directions. With about9.9GiB free after this round, another ten width256
checkpoint pairs would not fit the8GiB reserve. Resolve a prospective bounded
pilot and quality storage budget while preserving all research checkpoints.

## Reviewed trajectories

Only completed, artifact-reviewed records appear here. Test remains closed.

| Stage | Family | Seed | LR | Raw CE | H200 CE | Training seconds |
|---|---|---:|---:|---:|---:|---:|
| screen | mha | 2901 | 0.00125 | 4.742776 | 4.746541 | 204.96 |
| screen | mha | 2901 | 0.0025 | 4.934339 | 4.825638 | 202.56 |
| screen | mla | 2901 | 0.0025 | 4.907554 | 4.809730 | 219.80 |
| screen | mla | 2901 | 0.00125 | 4.719903 | 4.723452 | 219.20 |
| confirm | mha | 2911 | 0.00125 | 4.738932 | 4.744072 | 205.45 |
| confirm | mla | 2911 | 0.00125 | 4.704028 | 4.706850 | 219.75 |
| confirm | mla | 2912 | 0.00125 | 4.694593 | 4.703851 | 220.79 |
| confirm | mha | 2912 | 0.00125 | 4.727290 | 4.734112 | 204.69 |
| confirm | mha | 2913 | 0.00125 | 4.740290 | 4.740438 | 204.09 |
| confirm | mla | 2913 | 0.00125 | 4.713418 | 4.710600 | 219.45 |

Frozen family LRs: {'mha': 0.00125, 'mla': 0.00125}.

- Seed2911, MLA minus MHA H200 CE: -0.037222.
- Seed2912, MLA minus MHA H200 CE: -0.030261.
- Seed2913, MLA minus MHA H200 CE: -0.029838.
