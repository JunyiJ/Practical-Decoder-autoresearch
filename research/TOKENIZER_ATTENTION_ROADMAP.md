# Attention quality and inference-cost roadmap

Human steering, October26,2026 (Los Angeles): consider adopting GQA and MLA when
quality is better or neutral because inference cost matters; investigate Kimi's
3-KDA/1-MLA hybrid and its tunable parameters. This is the governing direction for
new attention experiments after the current tokenizer LR round. It does not
change the already-frozen LR-only plan or rewrite historical study decisions.

The subsequent human instruction also requires broad architecture research.
This attention work is one part of the [architecture agenda](TOKENIZER_ARCHITECTURE_ROADMAP.md),
which includes MoE, activations, normalization, residuals, depth and readout.
Attention parameter refinements must not indefinitely displace those families.

## Adoption objective

Treat quality and inference cost as separate measured objectives. GQA/MLA need
not beat MHA's CE to become useful serving alternatives. A quality-neutral
candidate with a useful measured cost reduction is eligible for adoption.
Retain workload-specific variants where appropriate, rather than forcing one
architecture to win at every width, batch and context.

For the next finite attention protocol, use a prospective engineering neutrality
margin of +0.01 nats per GPT-2 token (approximately1.005% perplexity) versus a
fresh, equally tuned MHA control. This number is an implementation choice, not
a tolerance explicitly supplied by the human. Freeze it before any new attention
quality results. Require the paired mean, each fresh seed's delta and the upper
bounds of the declared conditional-text intervals to stay within the margin at
the claimed deployment width. No requirement that every seed improve or that the
mean beat the old strict-improvement threshold. A failure to find statistical
significance is not evidence of neutrality; report seed variation separately
because text intervals condition on the trained models.

Cost routes must be declared before timing: memory/capacity, decode throughput,
or latency. Report actual allocated KV/recurrent/convolution state, working
memory, prefill latency, decode latency and achievable batch/throughput separately.
Choose a meaningful minimum benefit and its measurement rule in each finite
workload protocol, above observed timing noise. A memory-saving route does not
require an across-the-board speed win; disclose and assess latency costs for that
workload. Do not equate smaller cache with lower measured latency or cloud price.
Width256-only success may justify a width256 serving variant if that scoped route
was declared prospectively; it is not universal width transfer.

Retain synchronized warmups and alternating paired measurements, fixed prompts,
precision, hardware and generation lengths, repeated timings, and fresh training
seeds. Check cache correctness before timing. Short context128 Mac measurements
cannot establish long-context production savings. Longer-context claims need
models and evaluation explicitly validated at those lengths, including retrieval
or copy diagnostics where useful. Do not change the frozen current model's
context simply to manufacture a favorable benchmark.

## Paper check and version distinctions

Kimi Linear uses repeated KDA/KDA/KDA/MLA groups, with NoPE in MLA. Its KDA uses
short convolution, normalized queries/keys, low-rank decay and output gates,
and headwise output normalization. The paper compares hybrid ratios and ablates
output gating and convolution; 3:1 gives its preferred tradeoff. Its head size
is128, which differs from our local head32 adaptation. These are reported results
under the paper's scale and setup, not measured gains here.
[Primary report, sections4–5.2](https://arxiv.org/html/2510.26692v1#S4).

Kimi K3 keeps the3:1 structure with Gated MLA and an additional final global
layer. It uses NoPE MLA, bounded log-decay for KDA and full-rank output gates.
Its decay example uses a lower bound of−5. These are version-specific choices;
a small dense hybrid is an adaptation, not a full K3 reproduction.
[Primary report, sections2.1.1–2.1.2](https://arxiv.org/html/2607.24653v2#S2.SS1).

## What the current repository actually tests

The closed character-level KDA study used **four KDA layers**, head32, decay
bottleneck D/4, fixed convolution width4, bounded log-decay−5 and a full-rank
sigmoid output gate. Its mean quality regressions and slow training describe
that all-KDA sequential implementation. They do not reject a3:1 hybrid or
predict optimized-kernel inference performance. See [historical KDA results](KDA_QUALITY_RESULTS.md).

The production model currently chooses one global `attn_type` for every block;
its cache factory does the same. A faithful hybrid needs a layerwise attention
schedule and matching per-layer cache/state construction. Current KDA policy
rejects most tuning overrides, and the retained tokenizer model contracts assume
MHA. Add explicit new contracts rather than bypassing those checks. KDA must
remain non-RoPE; NoPE MLA must not silently fall back to learned position tables.

Historical local timing measured GQA cache reduction50% with mixed decode speed,
and MLA cache reduction62.5–81.25% with slower decode in the tested short-context
implementation. Those are useful tradeoffs to revisit under the new objective,
not grounds for assuming either family always reduces latency. Prior quality
and timing panels do not become fresh tokenizer evidence.
[Historical inference measurements](INFERENCE_EFFICIENCY_RESULTS.md).

## Planned sequence and tuning axes

### Implementation preparation checked during the LR round

Read-only inspection of the frozen source found that MHA applies the inherited
extra `attention_logit_scale=0.5`, while `src/attention/gqa.py` and
`src/attention/mla.py` do not consume that field. Their dimensional score
normalization is not this extra multiplier. A new comparison must explicitly
implement and test the declared multiplier in full and cached attention paths,
or declare a separately controlled scaling difference. Changing `attn_type`
alone cannot establish a controlled architecture comparison.

The tokenizer no-output-bias contract calls `readout_norm_contract`, whose
inherited backbone requires MHA and rejects GQA/MLA geometry overrides. The
optimizer already has historical GQA/MLA cases, but those do not authorize the
complete current tokenizer backbone. Add narrowly defined new contracts across
these dependencies, preserving the old validators and checkpoint semantics.
Before training, test actual attention arithmetic, cache parity, parameter
grouping/scaling and rejection of unsupported combinations. This is an
implementation finding, not measured evidence against GQA or MLA; no active
source was changed or additional worker launched for this inspection.

Finish the current paired LR screen and confirmation, then resolve its audited
subword baseline. Prioritize a new inference-aware GQA/MLA comparison and a3:1
hybrid implementation/comparison. Keep the shorter-horizon warning in the
backlog; it is a distinct hypothesis, not a reason to change active trials.

| Family / question | Bounded tuning axes to resolve before a new run |
|---|---|
| GQA | KV-head count/group ratio, equal LR tuning, attention-score scaling; require valid divisibility at every width and report where a ratio collapses to MQA |
| MLA | KV-latent rank, query rank, head geometry, score scale, position choice; compare cached implementation and any absorption optimization separately |
| Hybrid layout | Start with KDA/KDA/KDA/MLA at four layers; compare full MLA and full KDA controls and, in a separate bounded ablation, a1:1 layout; keep total depth/FFN fixed |
| KDA recurrent capacity | Head size and value-state dimension, decay bottleneck rank, parameter/state counts and width-scaling rules |
| Decay and gates | Explicit Linear-style versus K3-style formulation, bounded-decay value and initialization, decay bias, write-gate initialization, low-rank versus full-rank output gate |
| Local mixing / positions | Convolution width (including width4 anchor), NoPE global attention versus a separately specified positional alternative, cache-history length |
| Training | Equal-budget family-specific LR tuning; separate warmup, decay, clipping, horizon and EMA studies when motivated by instability or validation curves |
| Implementation | Sequential reference versus validated chunkwise implementation, chunk size and state precision; measure backend-specific costs without attributing Python-loop overhead to the architecture |

These are proposed local experiment axes, not a claim that the paper tuned every
one or that all are implemented. Use staged finite searches, not an unbounded
Cartesian sweep. Freeze geometry and version semantics first, then tune a small
set with equal budgets and fresh confirmation. Mark intentional departures from
the paper, especially small widths, dense FFNs, head sizes and short contexts.

Before hybrid training: equation/gradient tests, full versus cached/chunked
outputs, causality, nonzero offsets, reset/interleaved requests, per-layer state
shapes, tokenizer/optimizer/checkpoint/resume semantics, and serial MPS resource
pilots. An optimized kernel requires numerical agreement with the reference;
its speed is measured only on hardware/backend that actually supports it.
Preserve all failed attempts and closed studies; keep test loss closed.
