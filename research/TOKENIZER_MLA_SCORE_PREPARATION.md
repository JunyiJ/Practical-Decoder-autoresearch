# MLA attention-score multiplier transfer

Status: prospective direction selected; component arithmetic passed. No new
scientific seeds, resource-pilot plan or quality plan are frozen. No quality
training has started. Retain the reviewed D256 MLA/SwiGLU672/SiLU/RMSNorm/H200
recipe unless a new finite comparison passes its prospective rule.

## Why this question now

Review at the completed FFN boundary considered architecture and bounded tuning:

- Internal Q/K normalization: historical MHA QK-RMS fresh deltas were -0.005274
  at D64 and +0.007981 at D256, with the local screening gate already failed.
  Added FFN hidden normalization also missed its gate. Neither supplies a strong
  reason to add new normalization parameters immediately after the block-norm
  comparison. These remain closed character studies, not tokenizer results.
- Position/local mixing: historical NoPE and ALiBi both lost against RoPE at
  both widths; a short-convolution addition remains distinct but introduces a
  new cache/history contract. Defer it, without claiming a subword negative.
- Sparse MoE remains an architectural candidate, with the wider-dense control
  required and its prior full storage projection larger than this two-arm study.
  Closed activation, block-normalization, EMA and FFN comparisons are not reopened.
- Bounded HP: select **attention multiplier 0.5 versus 1.0**. The 0.5 choice was
  adopted with character MHA; historical mean gains were 0.006210/0.009851 at
  D64/D256, with a different selected LR for half-strength scores. The current
  tokenizer MLA uses compressed projections and concatenated content/rotary
  coordinates. Its score multiplier has not been independently tuned in the
  reviewed tokenizer inventory. This is a transfer question, not a rerun of the
  character campaign, a bug fix, or a claim that weaker/stronger scores are better.

Source inspection confirms that current full and cached MLA **do consume 0.5**.
The attention roadmap's earlier missing-multiplier observation describes the
historical pre-integration state and is already resolved in the adopted runtime.
No entropy-collapse or clipping diagnosis has been established. Generic warmup,
clipping and decay sweeps are deferred in favor of this exact inherited choice.

## Exact contrast and prospective finite design

For each head and causal token pair, let content width and rotary width both be32:

`score[t,s] = alpha * (q_content[t]·k_content[s] + RoPE(q_rot[t])·RoPE(k_rot[s])) / sqrt(64)`.

Apply the causal mask after content scaling, then softmax, value weighting and
output projection. Control alpha=0.5; candidate alpha=1.0. Scale both terms
jointly; keep rotation frequencies, low-rank geometry, learned biases and all
other inherited operations unchanged. No trainable temperature or new tensor.
The intended full model has28,492,416 parameters in both arms, identical parameter
shapes, initialization draws and Muon/AdamW groups. Full integration must verify
these properties rather than infer them solely from the equation.

Keep four layers, D256/head32, query/KV ranks64, shared rotary key width32,
SwiGLU672, block RMSNorm, final affine LayerNorm, grouped-embedding v2, untied
GPT2 tables, dropout0, context128, batch8, paired512 sampling,1200updates,
cooldown800 and H200 from800 every10updates. Same data/splits and closed test.

Proposed two20-update resource pilots, then equal three-LR screens per arm
(0.000625/0.00125/0.0025), then three fresh paired seeds at each arm's independently
selected minimum H200 full-validation CE (lower-LR tie break). Six screens and
six fresh trajectories; no boundary extension or alternative winner. Pair order
will alternate, with exact seeds/order frozen after census and prerequisites.
Retain two selected screen pairs and all six fresh raw/H200 trajectories.
Any successful-pilot/losing-screen retirement needs exact audited boundaries.

Adopt only if candidate-minus-control mean H200 CE is below-0.02, every fresh
paired delta is below0, and simultaneous conditional-text upper bounds are below0
for2048/4096/8192-token blocks,2000 common paired PCG64 draws and linear quantile
1-0.05/3. No cost-only route. Full/cache correctness and three fresh paired
inference reviews remain required. Reused validation text gives adaptive local
evidence, not test-holdout or training-seed-population confidence. Independently
selected rates make this a recipe comparison, not an isolated causal score effect.

## Completed narrow numerical prerequisite

`tokenizer-evidence/mla-score-preparation-001/component001.py/json` extracts an
actual incumbent attention module, makes private copies and overrides only their
scalar for this component test. It does not bypass or change full-model validators.
An independent complex-RoPE oracle contracts content and rotary scores separately
and explicitly normalizes exponentials. Both multipliers passed FP64 CPU forward
and all input/parameter gradients; maximum errors7.78e-16 and8.89e-15 respectively.
CPU/MPS FP32 forward errors were at most4.48e-7 and gradient errors4.77e-6.
Uneven3/1/5/7-token caches, nonzero continuation offsets, reset and exact causality
passed on both devices. Cache allocation for one layer and batch2 is98,304B.

The frozen full-model contract correctly rejects alpha1.0. New explicit model,
optimizer/checkpoint/cache identities must be implemented outside the424 frozen
sources, with bitwise incumbent initialization/output checks, serialization/resume
and raw/H200 worker/parent acceptance before scientific training. No corpus,
checkpoint weights or test split were read by this synthetic arithmetic run.
No quality claim follows from numerical correctness.

## Capacity and next executable work

The completed FFN pilots are now eligible for exact policy cleanup because all
of their dependent quality and inference work closed. Their original protection
was scoped to that campaign; selected screens, every fresh pair and failed
artifacts remain protected. A separate manifest and journal record retirement;
historical closure and source snapshots remain immutable.

The next capacity review must bind actual serialized raw/H200 sizes, metadata
and serialization overlap, plus the unchanged8GiB reserve. Do not treat the old
FFN projection or this component check as a new launch authorization. Complete
the small full-model integration while auditing eligible obsolete artifacts;
no partial quality panel or reduced confirmation is permitted to hide a deficit.

## Full-model and serialized resume integration completed

The earlier component-only status is superseded by the isolated
`research/mla_score_v1` runtime and model002/checkpoint-resume001 evidence. Only
block-normalization policy resolution and constructed architecture validation
were extended; unchanged implementation modules resolve under the new namespace.
The named score policy permits exactly float0.5/float1.0 on the incumbent backbone.
Both arms retain28,492,416 parameters, bitwise-identical initial weights and
identical Muon/AdamW groups. Control logits match the incumbent bit for bit.
Cross-score checkpoint/config/cache identities, mutated module scalars and
unsupported settings are rejected. CPU/MPS logits, gradients, uneven/interleaved
cache, reset and causality passed; no quality claim follows.

The first model001 harness stopped before training because it used the new
optimizer validator on the legacy model. Its source/snapshot/failed guard remain
intact. Separately recorded model002 uses each namespace's own optimizer factory;
no historical failed fixture was rerun or validator weakened.

Generated-text strict-MPS tests completed three updates per uninterrupted and
repeated trajectory, plus interrupted replay, for each arm. Model, optimizer,
EMA and RNG comparisons were bitwise exact; raw/H200 export/load and cached
inference passed at batches1/8. Actual serialized sizes are445,082,407B per raw
and114,074,435B per H200. These numerical fixtures are now exact members of the
verified `mla-score-synthetic-001` archive; their original hashes remain valid.
Original-path consumers must use verified archive resolution or restore exact
bytes with adequate space. Scientific worker/parent acceptance remains pending.

The new measured prospective projection is13,690,208,256B: eight retained
trajectories at576MiB each (448MiB raw +112MiB H200 +16MiB per-trajectory metadata),
256MiB round metadata and8GiB reserve. It assumes audited successful-pilot
retirement before screening and losing-screen retirement after saved selection.
The worker must enforce these caps and tested publication/retention boundaries
before this becomes a frozen launch contract. No scientific scope or global
guard is reduced. Later prerequisite output consumption must be charged again.

Completed EMA-pilot retirement reclaimed787,232,443B. Lossless archival of twelve
successful FFN/score numerical checkpoints reclaimed another2,199,083,259B net;
all selected/fresh/failed evidence remained unchanged. Observed free13,986,324,480B
exceeds the prospective projection by296,116,224B at that observation only.
Next: real bounded worker/parent validation and inference, then updated capacity,
fresh seed census, finite pilot freeze/pilots and final quality freeze.

## Real worker, independent parent and trained inference prerequisites passed

The score half/unit arms each completed two actual generated-text run_training
updates through the bounded serial child process and output sink. Independent
parent acceptance verified optimizer/RNG/resume semantics, both raw/H200 identities,
all88 EMA tensors and two snapshots,2,048 processed training targets, and all3,014
validation targets. CPU/MPS validation CE differed by at most2.775e-7. Both raw
checkpoints measured445,084,199B and both H200 exports114,074,819B. Existing streaming
publication and new individual448/112MiB limits passed; no duplicate payload or
temporary checkpoint remained. These are generated-data engineering checks only.

The resulting trained pair passed full/cache/reset checks at batches1/8 before
warmups and eight alternating timing repeats. Cache bytes were196,608 per item.
Timing is descriptive for this two-update fixture; it establishes no score-arm
quality or deployment-performance advantage. The research corpus and test stayed
closed. All frozen and worker-bound sources were rechecked unchanged.

Six successful worker checkpoints (both new score pairs and the prior successful
ungated FFN worker003 pair) are now losslessly archived in
`research_archives/mla-score-worker-001/successful-workers.tar.zst`. Full member
verification preceded loose-copy removal, and26,524 other files were rehashed.
Original1,677,471,074B became849,909,661B, recovering827,561,413B net. The failed FFN
worker002 control and every selected/fresh/failure artifact remain unchanged.
Use worker-archive001's exact manifest to resolve or restore absent loose paths.

Capacity003 corrects duplicate metadata accounting: the enforced256MiB complete
campaign budget already includes8.25MiB per trajectory, below its16MiB allowance,
including temporary metadata publication. Thus the complete proposed high-water
is eight560MiB weight allowances +256MiB metadata +8GiB reserve =13,555,990,528B.
The active serializer is included in its weight allowance and publishes by
same-inode link/unlink. No guard or confirmation requirement was reduced. Observed
free13,654,347,776B gives98,357,248B headroom at that observation only. The earlier
13,690,208,256B estimate double-counted128MiB of trajectory metadata.

Next execute the narrow prospective pilot/losing-screen retention-boundary checks,
then fresh seed census, finite pilot freeze, guarded resource pilots and final
quality freeze. Recheck actual free space and charge all new output before dispatch.
No scientific pilot/quality plan is frozen and no scientific worker is running.
Evidence: mla-score-preparation-001/trainer-worker001, trained-inference001,
worker-archive001, capacity003 and worker003-review.json; implementation in
research/mla_score_v1. No new ledger framework was introduced.
