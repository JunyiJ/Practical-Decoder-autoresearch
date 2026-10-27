# Tokenizer MLA preparation and finite feasibility pilot

October27,2026. The horizon round is complete and audited. Width256 GQA at
LR0.00125 now uses1200 updates with cooldown/EMA start800. Next structural
question: can compressed MLA offer useful inference memory savings at neutral
quality under a fairly tuned tokenizer comparison? No MLA quality adoption is
made by the prerequisite work below. General MHA and all historical studies
remain separate; test stays closed.

## Explicit implementation

New policy `tokenizer_mla_quarter_rope_v1` permits MLA only, widths64/256,
head32, query and KV latent ranksD/4, four layers, decoupled rotary features32
per head with a shared rotary key. It retains the GPT-2 vocabulary, untied
readout, no output bias, block RMSNorm/final affine LayerNorm, serial unit
residuals, dense bias-free SwiGLU21D/8, Muon/AdamW muPbase64 and independent
decay0.00006. It preserves the exact old MHA/GQA policy and rejects additional
geometry overrides. The existing quarter-rank MLA optimizer treatment is used;
all block matrices including the fixed-row rotary-key projection follow the
historical hidden-matrix learning-rate rule, while biases stay at base LR.

Scores are0.5 times the sum of content and rotary dot products divided by
sqrt(32+32). Cached inference stores the KV latent and shared rotated key;
expanded keys/values are recomputed. This is explicit RoPE MLA, not the NoPE
MLA needed for the later3:1 Kimi hybrid. No absorption optimization, output gate,
latent normalization or new kernel is introduced. Legacy MLA without a declared
score multiplier remains unscaled. The new contract binds checkpoint identity.

## Completed CPU prerequisite checks

63 tests passed in22.82seconds across new tokenizer MLA tests, existing MHA/GQA
tokenizer tests, historical MLA, rotary/cache and generic cache suites. They
cover independent attention arithmetic and all projection gradients, cached
nonzero offsets, causality, reset and separate requests, allocated cache sizes,
optimizer groups, rejected policy/shape mutations, exact training resume and
EMA/tokenizer checkpoint loading. Initial lint found five style issues,
including existing style in touched files; import and literal rewrites resolved
them. Logs are preserved. CPU checks are correctness evidence, not quality.

## Frozen next finite experiment: one MPS feasibility trial

One width256 MLA trial, seed2891, LR0.00125,60 updates, cooldown/EMA start40,
H200 every10 updates (3 snapshots), batch8/context128, diagnostic evaluation
at60. The config copies the adopted GQA recipe, changing only declared MLA
policy/geometry, seed, horizon/schedule and output fields. It scores the same
full validation panel for sanity, excluded from quality selection. It then
checks full versus cached logits on the first12 training tokens at batches1/8.
Expected actual FP32 cache allocation:196608 and1572864bytes. No test scoring.
No latency/adoption inference is made from this pilot.

Freeze source, runtime, helper, recipe, protocol and corpus hashes. Launch only
from a clean committed tree with the global single-worker lock. The finite
helper is `research_state/tokenizer-mla-validation/run_pilot.py` (portable copy
in the evidence directory). Timeout600seconds including training/cache checks,
RSS4GiB and MPS driver6GiB memory guard inherited from the training worker,
output cap1GiB, free disk reserve8GiB plus the entire1GiB cap before launch.
No silent retry; preserve checkpoints, logs and failures. A STOP file before
launch prevents training; a signal stops the active worker. Pilot artifact review
must verify optimizer/sampler lineage, EMA snapshots and cache evidence.

After the pilot, use measured checkpoint size for a new finite quality budget.
Proposed comparison is width256 MHA versus MLA at1200 updates, bounded equal LR
tuning, fresh paired confirmation, +0.01-nat neutrality convention and measured
memory savings with latency reported separately. Freeze exact seeds, LR grid,
number of runs and budget in a separate quality protocol before any quality
training. The shorter GQA schedule is motivation, not proven transfer to MLA or
MHA. Preserve all research checkpoints; only identified completed temporary
unit-test fixtures may be cleaned after logs and a manifest are saved.

The3:1 KDA/MLA layout remains next; its NoPE/layerwise state contracts need
separate work. MoE, activation, norm, residual, depth and readout studies remain
active research directions after these structural priorities.
