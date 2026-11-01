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
