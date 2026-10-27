# Prospective width256 tokenizer MLA quality and inference experiment

October27,2026. Preparation is in TOKENIZER_MLA_PREPARATION.md. The CPU and
60-step MPS feasibility checks passed. This protocol fixes the next quality
question before any quality trial; the controller and inference benchmark
extension must still pass tests and source freeze before launch.

Compare fresh MHA and quarter-latent RoPE MLA on the inherited width256 dense
backbone, GPT-2 BPE50257, four layers, head32, attention multiplier0.5, block
RMSNorm, final affine LayerNorm, serial unit residuals, bias-free SwiGLU21D/8,
untied tables/no output bias, batch8/context128. MLA uses query/KV rank64 and
shared rotary-key dimension32; scores have dimensional normalization sqrt64
instead of MHA sqrt32, both with the explicit0.5 multiplier. No other model
changes or latent/gate/kernel sweep in this first structural comparison.

Both families train from scratch for1200 updates with cooldown/EMA start800,
final LR multiplier0.1, H200 every10 steps (41 snapshots), Muon/AdamW muPbase64,
independent decay0.00006. The horizon result motivates this equal recipe but
only established it for GQA; this comparison does not assume transfer of that
quality gain to either family. Equal update/token budget, not equal FLOPs,
parameters or time. Architecture changes alter initialization RNG consumption.

Screen exactly one seed2901 for each family at LR0.00125 and0.0025: four trials.
Order MHA low/high, then MLA high/low. Each family independently selects lowest
final full-validation H200 CE, tie lower LR. No screen result can establish
adoption. Freeze both LRs, then train fresh paired seeds2911/2912/2913,
MHA/MLA order alternating by seed. Six confirmation trials, maximum ten quality
trials. Preserve every trial and checkpoint; a failure stops the finite round
for diagnosis, never silent retry or expansion. Raw CE/BPB/time are secondary;
no sampled minimum or best intermediate checkpoint selection.

Quality eligibility requires MLA minus MHA paired mean, each fresh seed delta,
and all conditional text-bootstrap95% upper bounds <=+0.01nats/token. Use
2048/4096/8192-token blocks,2000 resamples, RNG20261024. The margin is an agent
engineering convention, not an explicit human numeric tolerance. Conditional
text intervals are not training-seed population uncertainty or independent-test
evidence. All three fresh pairs and artifact review are required.

After quality training, benchmark each fresh MHA/MLA checkpoint pair serially
with no other training/timing worker: FP32 MPS, context128, first96 fixed training
tokens,32 prefill and64 teacher-forced decode, batches1/8, two warmups/eight paired
repeats with alternating order. Verify full/cached logits first. Measure actual
allocated cache, parameter bytes, observed MPS allocations, prefill/decode times
and throughput; distinguish allocator observations from single-model peak memory.
Each process capped600seconds, raw records retained. Source-frozen implementation
must explicitly accept MLA instead of mislabelling it GQA.

Adopt only a scoped MLA option versus this equally tuned MHA if quality is
eligible and every fresh-checkpoint/batch measurement saves at least60% actual
allocated KV cache versus MHA, with cached correctness and complete artifact
review. This memory route does not require a latency win; report slower decode
honestly. No production/long-context/cloud-price claim. This round does not
freshly compare MLA with the adopted GQA recipe, so it cannot replace GQA or
claim superiority over it. General MHA LR0.0025/3000 and width64 remain separate.

Same Shakespeare raw80/10/10 boundaries and SHA256
86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed;
full validation34278 targets/111536bytes, reset context128. Test closed; no
character/subword CE comparison. Historical stop markers remain closed.

One serial local MPS worker/four CPU threads. Freeze source/runtime/protocol/
recipe/corpus/plan. Quality process cap7200seconds, RSS4GiB/MPS driver6GiB,
round output cap7GiB and free disk reserve8GiB. Before launch project all ten
raw/H200 pairs using the larger measured MLA pilot or preserved width256 MHA
pair, plus15% storage overhead and the separate8GiB reserve. Recheck full
remaining storage and one-trial headroom at each launch. Do not delete research
checkpoints for capacity. Log and stop on a resource violation. Current pilot
resource projection is evidence, not authorization to ignore later disk changes.

Next structural priority remains3:1 KDA/MLA with explicit NoPE and per-layer
cache/state semantics. Broader MoE, activation, normalization, residual, depth
and readout studies remain active alongside HP tuning.

Prelaunch storage clarification: the cap is7GiB because the measured ten-pair
projection plus15% overhead is6.05GiB, slightly above a6GiB cap. The separate
8GiB free-space reserve is unchanged. This correction precedes all quality trials.
