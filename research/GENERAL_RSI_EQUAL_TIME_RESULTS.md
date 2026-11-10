# Equal Mac training-time comparison: complete

Native passes all frozen validation-loss gates across three fresh paired seeds. At
7,200 synchronized training seconds per arm, mean time-EMA CE is4.968966040810
for native and5.091067701052 for byte. Native minus byte is−0.122101660243
nats/token. The one-sided99% paired document-bootstrap upper bound is
−0.105244791590. Both source means favor native. No test data was opened.

The adopted recipe retains all three native time-EMA checkpoints without
best-seed selection. Both arms and all RAW checkpoints remain retained.
Generation diagnostics show substantial repetition and drift in both arms.
This is a validated loss result for this Mac, corpus, architecture size and budget.
It is not evidence of reliable generation or broader-domain superiority.

## Protocol and execution

Frozen plan: `tokenizer-evidence/general-rsi-equal-time-001/plan.json`
(SHA256 `4f51d54569d8b649c9609d25d42ba5bbaefdfd390db99b3283d5efc1c2c1165d`).
Two LRs received1,800 seconds per arm. The frozen within-arm screen selected
LR0.0025 for both. Six fresh confirmations then received7,200 seconds each.
All ten execution reviews passed; paired sampler prefixes match. All overshoots
were within one update. No run required failure recovery.

Training time includes synchronized sampling, device transfer, forward/backward,
clipping, optimizer, EMA and durable update-start/hash accounting. Checkpoint I/O,
validation, generation, initialization and other logging are excluded from that
clock and included in the separate guarded worker totals. Each schedule cools
down over its own final third to0.1. EMA uses the frozen127.184813977-second
half-life, initialized at the first update start at or beyond two thirds.

| Slot | Phase | Arm | LR | Updates | Training seconds | Time-EMA CE | RAW CE | Guard seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00 | screen | native | 0.00125 | 5,338 | 1800.200685 | 5.6160593260 | 5.6100533674 | 1866.154 |
| 01 | screen | raw_bytes | 0.00125 | 2,975 | 1800.220744 | 5.9043330308 | 5.8894565185 | 1917.650 |
| 02 | screen | native | 0.0025 | 5,339 | 1800.299251 | 5.4689027207 | 5.4637743363 | 1866.499 |
| 03 | screen | raw_bytes | 0.0025 | 2,975 | 1800.479829 | 5.7425720106 | 5.7293537405 | 1918.304 |
| 04 | confirmation | native | 0.0025 | 21,348 | 7200.237572 | 4.9757529521 | 4.9846985006 | 7380.482 |
| 05 | confirmation | raw_bytes | 0.0025 | 11,919 | 7200.587297 | 5.0788650620 | 5.0838756420 | 7449.937 |
| 06 | confirmation | native | 0.0025 | 21,397 | 7200.071227 | 4.9719577012 | 4.9829403235 | 7382.354 |
| 07 | confirmation | raw_bytes | 0.0025 | 11,873 | 7200.600330 | 5.1067060895 | 5.1130765487 | 7447.069 |
| 08 | confirmation | native | 0.0025 | 20,894 | 7200.275345 | 4.9591874691 | 4.9691853944 | 7390.726 |
| 09 | confirmation | raw_bytes | 0.0025 | 11,681 | 7200.502229 | 5.0876319517 | 5.0953980789 | 7450.279 |

Total guarded worker time: 52,069.453s of the72,000s tranche.
Retained final weights: 1,794,399,184bytes. Free space at completion:
25,606,782,976bytes. The shared lock was available and no training worker
was active. No cleanup was needed.

## Frozen acceptance result

| Pair | Native time-EMA CE | Byte time-EMA CE | Byte minus native | RAW difference |
| --- | --- | --- | --- | --- |
| 1 | 4.975752952122 | 5.078865061978 | +0.103112109855 | +0.099177141401 |
| 2 | 4.971957701231 | 5.106706089479 | +0.134748388248 | +0.130136225203 |
| 3 | 4.959187469075 | 5.087631951701 | +0.128444482625 | +0.126212684495 |

All three deltas favor native, the mean exceeds the0.02 improvement threshold,
and the99% upper bound excludes zero. Native-minus-byte source deltas are
−0.119949133157 for FineWeb-Edu and−0.130723141723 for Wikipedia, within
the0.05 source-regression guard. Mean RAW CE is4.978941406174 versus
5.097450089874. RAW is a secondary endpoint.

The bootstrap resamples108 paired documents10,000 times with frozen PCG64
seed62110700999, averages loss sums across the three seeds before resampling,
and divides by sampled token counts. It is conditional on these documents and
seeds, not a population-seed confidence statement. All100,416 validation targets
and their per-document inventories agree across arms and endpoints.

## First-generation evaluation

All144 first outputs were retained:12 prompts ×2 fixed decoders ×6 models.
Each uses at most64 new tokens and seed42 reset per prompt/decoder. No rerolls
or decoder tuning occurred. Arm, seed and checkpoint labels were removed and
rows shuffled with seed62110500998. All annotations were saved before unblinding.
The annotator was the Codex assistant. These are exploratory agent ratings,
not independent human validation. Consistency is not exhaustive factual accuracy.

Each cell below covers36 outputs. Relevance and consistency are0/1/2 ordinal
ratings;1 means partial relevance or ambiguous consistency. Semantic means give
equal weight to prompts with assessable ratings. Unassessable outputs are excluded
from means and shown separately; they are never passes.

| Arm | Decoder | Repeat trigram fraction | Terminal loop | Mean content tokens | EOS | Token cap | Relevance mean | Consistency mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native | greedy | 0.5076 | 41.7% | 64.00 | 0.0% | 100.0% | 0.9167 | 0.8788 |
| native | T0.8/top-k40 | 0.0197 | 0.0% | 62.03 | 5.6% | 94.4% | 0.7500 | 0.9722 |
| raw_bytes | greedy | 0.5421 | 33.3% | 64.00 | 0.0% | 100.0% | 0.6111 | 0.9697 |
| raw_bytes | T0.8/top-k40 | 0.0237 | 0.0% | 64.00 | 0.0% | 100.0% | 0.7222 | 0.9722 |

| Arm | Decoder | Relevance counts0/1/2/U | Consistency counts0/1/2/U | Consistency U fraction | Assessed consistency prompts |
| --- | --- | --- | --- | --- | --- |
| native | greedy | 4/31/1/0 | 5/25/2/4 | 11.1% | 11/12 |
| native | temperature_0.8_top_k_40 | 9/27/0/0 | 1/34/0/1 | 2.8% | 12/12 |
| raw_bytes | greedy | 15/20/1/0 | 2/30/1/3 | 8.3% | 11/12 |
| raw_bytes | temperature_0.8_top_k_40 | 10/26/0/0 | 1/35/0/0 | 0.0% | 12/12 |

Relevance has no unassessable ratings. All repetition metrics are assessable;
no unexpected early stop occurred. Greedy output is highly repetitive in both
arms. Sampling sharply reduces repeated trigrams but still produces fragmentary
or drifting content. Native has higher descriptive relevance means, especially
under greedy decoding, but terminal loops and consistency do not uniformly favor
native. Only two of144 outputs received relevance2, and only three received
consistency2. These diagnostics support neither reliable generation nor a broad
generation-quality superiority claim. They were not used to change the CE gate.

### Per-seed diagnostics

| Arm | Decoder | Seed bundle | Repeat fraction | Loop fraction | Relevance mean | Consistency mean | Consistency U count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native | greedy | 1 | 0.5739 | 0.5000 | 1.0000 | 1.0000 | 2 |
| native | greedy | 2 | 0.5134 | 0.3333 | 0.9167 | 0.8182 | 1 |
| native | greedy | 3 | 0.4355 | 0.4167 | 0.8333 | 0.9091 | 1 |
| native | temperature_0.8_top_k_40 | 1 | 0.0202 | 0.0000 | 0.6667 | 1.0000 | 1 |
| native | temperature_0.8_top_k_40 | 2 | 0.0188 | 0.0000 | 0.7500 | 1.0000 | 0 |
| native | temperature_0.8_top_k_40 | 3 | 0.0202 | 0.0000 | 0.8333 | 0.9167 | 0 |
| raw_bytes | greedy | 1 | 0.4664 | 0.3333 | 0.7500 | 1.0000 | 1 |
| raw_bytes | greedy | 2 | 0.5726 | 0.3333 | 0.4167 | 0.9091 | 1 |
| raw_bytes | greedy | 3 | 0.5874 | 0.3333 | 0.6667 | 1.0000 | 1 |
| raw_bytes | temperature_0.8_top_k_40 | 1 | 0.0215 | 0.0000 | 0.7500 | 1.0000 | 0 |
| raw_bytes | temperature_0.8_top_k_40 | 2 | 0.0188 | 0.0000 | 0.6667 | 0.9167 | 0 |
| raw_bytes | temperature_0.8_top_k_40 | 3 | 0.0309 | 0.0000 | 0.7500 | 1.0000 | 0 |

Full lengths, termination, ordinal counts, pooled assessable means and prompt
means for each seed are in `generation-evaluation/summary.json`. Every prompt,
exact continuation, evidence quote and rationale is retained in `blind.json` and
`annotations.json`; `mapping.json` links them to all original token-level outputs.

## Interpretation and next action

Measured: native completes20,894–21,397 updates in the same budget in which byte
completes11,681–11,919. All paired loss and source results favor native. Inference:
the extra updates enabled by native throughput outweigh byte’s equal-update
advantage at this budget. This comparison includes the full native/byte parameter
and implementation tradeoff; it does not isolate a parameter-count mechanism.
The closed equal-update result favoring byte and longer-byte-training result
remain valid for their different questions.

Next: qualify a prospective broader-domain evaluation of all six time-EMA
checkpoints, with no further training. Freeze source versions/licenses, domains,
selection, token budget, exclusions, train/validation deduplication, aggregation
and checkpoint hashes before scoring new text. Keep the original test unopened.
This lower-cost transfer question comes before another model-size training panel.

## Evidence

All paths below are under `tokenizer-evidence/general-rsi-equal-time-001/`:
- `decision.json`: frozen gates, paired deltas, bootstrap and bound evidence hashes.
- `slot-00` through `slot-09`: original logs, clocks, validation, execution reviews and receipts.
- `adopted-recipe.json`: all three native reference weights and all three byte transfer comparators.
- `generation-evaluation/`: all144 blind ratings, mapping, metrics and complete summaries.
- `analysis-checks.json`: signed/unequal-count bootstrap fixtures, prompt aggregation fixture and all144 original-output bindings.
- `completion-resource.json`: measured retained payloads, free space and available lock.
