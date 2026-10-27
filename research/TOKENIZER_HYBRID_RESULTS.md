# Tokenizer NoPE hybrid results

October27,2026. **Do not adopt this width64 hybrid.** All12 quality trials and
three serial cached-inference comparisons passed saved-artifact review. The
four-layer KDA/KDA/KDA/gated-MLA adaptation failed the frozen strict-quality
and neutrality gates against the selected all-KDA control. Existing GQA,
RoPE MLA and MHA recipes remain separate; this round does not promote all-KDA
against them. [Protocol](TOKENIZER_HYBRID_PROTOCOL.md),
[final audit](tokenizer-evidence/hybrid-001/final-review.json).

## Fresh confirmation and scope

| Seed | Hybrid H200 CE | All-KDA H200 CE | Hybrid minus control |
|---|---:|---:|---:|
|3011|4.965797|4.957274|+0.008523|
|3012|4.944646|4.930151|+0.014495|
|3013|4.924664|4.910561|+0.014103|
|Mean|4.945036|4.932662|+0.012374|

All three fresh pairs favor all-KDA. Independently recomputed conditional95%
text-block intervals are[0.003732,0.021187] at2048 tokens,
[0.004802,0.019795] at4096, and[0.005825,0.017639] at8192. These condition on the
three trained seeds and adaptively reused validation text; they do not quantify
seed-population uncertainty or independent test performance. Test loss stayed
closed. H200 means spaced-tail EMA half-life200 updates.

Six screens tuned LR0.00125/0.0025 equally for hybrid, all-gated-MLA and all-KDA
on seed3001. The frozen rule selected LR0.00125 for hybrid/all-KDA and0.0025
for all-gated-MLA; all-KDA was the stronger screened homogeneous control.
The remaining six trials confirmed only hybrid versus all-KDA. No fresh
comparison against all-gated-MLA, GQA, RoPE MLA or MHA, and no width256 transfer
claim. No grid extension or replacement seed was used.

## Measured cost tradeoff

| Batch | All-KDA state bytes | Hybrid state bytes | All-KDA median64-token decode | Hybrid median64-token decode | Ratio |
|---|---:|---:|---:|---:|---:|
|1|41,984|56,064|0.101282s|0.222981s|2.202x|
|8|335,872|448,512|0.101449s|0.227875s|2.246x|

Hybrid state is33.54% larger in every seed/batch, including recurrent matrices,
convolution histories and latent/shared-key buffers. The memory route was
inapplicable against all-KDA. Parameter bytes are26,602,160 hybrid versus
26,629,696 all-KDA, a small separate difference. Prefill medians were
0.012235s/0.012054s at batch1 and0.015024s/0.015120s at batch8 (hybrid/control).
Mean training time was171.969s versus200.743s,
so faster training did not predict faster cached decoding here.

Workload: FP32 local MPS, context128 allocation, first96 training tokens,
32-token prefill and64 teacher-forced decode, two warmups/eight alternating
paired repetitions per seed. Two models resident; allocator observations are
not single-model deployment peaks. Cache bytes are not total memory or cloud
cost. Sequential KDA and unabsorbed gated-MLA implementations do not predict
optimized-kernel or long-context performance.

## Implementation, audit and next work

This is an explicit NoPE, dense, head32 K3-style adaptation: bounded-decay KDA
with full-rank output gates, three recurrent layers followed by one gated-MLA
layer. No learned-position fallback. It retains a32-dimensional unrotated
shared key/query component, query/KV rank16, and0.5 score multiplier; it is
not a Kimi reproduction. See [preparation](TOKENIZER_HYBRID_PREPARATION.md).

The audit verified all raw/H200/config/request hashes, finite model/optimizer
state,1200updates/41EMA snapshots, exact train-sampler replay, untouched test
RNG, validation coverage/arithmetic, source/runtime/protocol/recipe/corpus
identity and final cache/timing records. This is same-agent review with separate
interval recomputation, not an independent reviewer. All24 quality checkpoints,
pilot checkpoints, raw records and preflight failure evidence are retained.
The failed control preflight was a noncontiguous synthetic-target error before
any optimizer update; its separately frozen correction passed.

Next structural hypothesis: embedding/readout sharing with explicit matched
untied control, initialization/scale, optimizer, EMA and checkpoint contracts.
The BPE vocabulary now dominates parameters, giving this old character-level
hypothesis a new capacity/storage context. Broad MoE, activation, normalization,
residual/depth and HP studies remain active. Only about0.26GiB is free above the
8GiB reserve; a new multi-seed training plan needs additional storage. Continue
CPU preparation while that resource prerequisite is resolved.

## Complete trial record

| Stage | Family | Seed | LR | H200 CE | Training seconds |
|---|---|---:|---:|---:|---:|
| screen | hybrid | 3001 | 0.00125 | 4.954479 | 168.87 |
| screen | hybrid | 3001 | 0.0025 | 4.961288 | 171.62 |
| screen | gated_mla | 3001 | 0.0025 | 5.027217 | 112.39 |
| screen | gated_mla | 3001 | 0.00125 | 5.050826 | 113.10 |
| screen | kda | 3001 | 0.00125 | 4.922120 | 202.67 |
| screen | kda | 3001 | 0.0025 | 4.954309 | 201.60 |
| confirm | hybrid | 3011 | 0.00125 | 4.965797 | 171.51 |
| confirm | kda | 3011 | 0.00125 | 4.957274 | 201.59 |
| confirm | kda | 3012 | 0.00125 | 4.930151 | 199.64 |
| confirm | hybrid | 3012 | 0.00125 | 4.944646 | 172.14 |
| confirm | hybrid | 3013 | 0.00125 | 4.924664 | 172.25 |
| confirm | kda | 3013 | 0.00125 | 4.910561 | 201.00 |
