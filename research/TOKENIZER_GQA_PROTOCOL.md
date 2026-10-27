# GPT-2 GQA quality and cache-cost comparison

Prospective finite round, October26,2026 (Los Angeles). The first tokenizer
round adopted LR0.0025 after18 reviewed trajectories. Use that architecture and
data contract as the starting point. The next structural hypothesis is that
half as many KV heads can retain quality while reducing inference cache storage.
This follows the human's request to consider quality-neutral GQA/MLA for cost.
MLA and the3:1 KDA/MLA hybrid remain subsequent priorities, alongside the observed
horizon/schedule problem and the broad architecture agenda.

## Model and data

Four serial dense SwiGLU21D/8 blocks, RMSNorm blocks, affine final LayerNorm,
RoPE, attention score multiplier0.5, unit residuals, local values, no gates or
extra FFN/QK normalization, untied vocabulary tables and no output bias. Retain
Muon/AdamW with muP base64 and independent decay0.00006. GPT-2 BPE50257 and the
same raw80/10/10 corpus boundaries, context128 and1024 targets/update. Test loss
stays closed. Preserve all old campaign stop markers.

MHA uses head32. GQA uses half its KV heads: at width64, two query heads and one
KV head (the MQA special case); at width256, eight query heads and four KV heads.
This is a fixed two-to-one grouping, not a search over group sizes. QKV projection
parameter counts change. Report total and attention parameter counts, training
time and memory; equal tokens are not equal compute or equal parameters.

The new named tokenizer architecture contract is separate from the historical
MHA-only policy chain. Its exact allowed fields fix the inherited behavior and
reject unreviewed combinations. Check numerical parity of the new MHA control
with the adopted MHA, including initialization, gradients and optimizer steps.
Verify GQA's score equation with an expanded-KV reference, gradients, causal
masking, cached nonzero offsets, reset/interleaved requests, checkpoint/resume,
EMA and tokenizer identity. Keep historical validators and defaults intact.

## Finite quality plan

1. Four separate40-update serial MPS pilots: MHA/GQA at widths64/256, seed2691,
   LR0.0025. They are correctness/resource evidence only, never quality selectors.
2. Eight width64 screen trajectories: each family at peak LR0.0025 and0.00125,
   seeds2701/2702, reverse family/LR ordering for the second seed. Both families
   receive the same tuning budget. Select each family's lowest mean final H200
   CE (tie: lower LR), then freeze both choices before confirmation. Report a
   boundary choice as unresolved, not an optimum; no automatic grid extension.
3. Twelve fresh confirmation trajectories: each selected family at widths64/256,
   seeds2711/2712/2713. Alternate family order by seed. Total20 quality trajectories.
   No width256 reselection. All train3000 updates, cooldown from2000 to multiplier
   0.1, H200 sampled every10 steps from2000 through3000 (101 snapshots).
4. Full-validation H200 CE is primary; raw CE, BPB and fixed samples are diagnostic.
   Same34278 targets and context resets as the previous round. For each width,
   require GQA minus tuned MHA mean, every fresh-seed difference, and all upper
   conditional text-bootstrap bounds to be at most+0.01nats/token. Blocks2048,
   4096,8192;2000 draws; seed20261024. The margin is an agent-chosen engineering
   convention. Lack of statistical significance does not prove neutrality.

Each width can qualify separately as a serving variant; do not force a width64
result onto width256. If families select different LRs, the confirmation compares
tuned recipes, not an isolated architecture effect. Report fixed-LR screen
contrasts separately. The selected MHA LR is a comparator, not an automatic new
general incumbent without its own confirmation against the retained recipe.
Conditional intervals concern the reused validation text given these models,
not population uncertainty over seeds or an independent test set.

## Inference-cost route and decision

After quality completion, benchmark the selected fresh H200 checkpoints on this
Mac/MPS in FP32 at batch1 and8:32 fixed prompt tokens plus64 teacher-forced decode
steps, total96 positions within the trained context128. Prompts/tokens are fixed
and shared across arms. No sampled generations or test text enter timing. Use
two warmups and eight synchronized paired repetitions, alternating arm order.
Check full-versus-cached logits before timing. Record each repeat, prefill/decode
latencies, derived decode tokens/second, actual allocated KV buffer bytes and
model tensor bytes. Report sampled current/driver memory as observations, never
as a guaranteed peak. Timing two resident models must be disclosed; model loading
is outside the timed region and absolute memory is not a single-model deployment
measurement. No optimized-kernel or long-context extrapolation claim follows.

The declared adoption route is cache storage: at least40% fewer allocated KV
bytes at each tested batch, plus the quality gate above and cached correctness.
This percentage applies to KV buffers, not total application memory. Latency is
a separately reported tradeoff; no speed win is required for this memory route.
If costs render the variant impractical for the tested workload, record that
limitation and keep it as a scoped option rather than claim universal adoption.
Do not translate cache savings directly into cloud cost or throughput savings.
Quality completion alone never automatically adopts GQA.

## Resources and continuation

One local MPS worker, four CPU threads, no paid compute or delegation. Freeze
source/runtime/corpus/plan before dispatch. Worker limits: sampled RSS4GiB and
MPS driver6GiB,600s per pilot and7200s per quality trial. Reserve8GiB free and
cap this round at8GiB output; project checkpoint costs from pilots with50%
headroom before quality launch. Timing has a separate600s process bound and
the same memory guards. Preserve raw/H200 checkpoints and failures; no silent
retry or overwritten trial directories. A failure requires diagnosis and a new
reviewed plan when source changes. STOP pauses at a trial boundary; a human stop
also pauses the heartbeat. Keep the test holdout closed.

Both active training source and its plan remain frozen. Review and publish each
completed round, then select the next justified finite experiment. The known
late-validation degradation remains a reason to test training duration/schedule;
this3000-step architecture comparison only establishes behavior at that budget.
