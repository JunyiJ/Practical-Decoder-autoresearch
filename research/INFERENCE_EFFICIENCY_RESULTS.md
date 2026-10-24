# Saved-checkpoint inference efficiency results

The 24 paired workload cells completed and passed review on source
`9dc2edd94eeea3d22c1a4b45404018b70cadb7f3`. All 12 immutable checkpoint hashes, 24 requests/results,
runtime and cache-size contracts, paired timing aggregates, report, source,
corpus and full knowledge snapshot were verified. No training or held-out test
access occurred. Historical quality selections remain unchanged.

Two-layer GQA is compared with its selected stage2D tuned MHA reference using
seed 444; four-layer MLA is compared with stage2F tuned MHA using seed 1011.
Widths64/128/256, batches1/8 and prompt/decode lengths32/32 and64/64 give24 pairs.
Both models were resident. Each model had 3 unscored warmups and 9 paired measured
repetitions, alternating execution order. The benchmark used FP32 MPS and 4 CPU
threads. Pilot samples are excluded. Positive decode change means slower;
negative means faster. Each cell is the median of the 9 paired time ratios.

| Pair | Width | Decode change: B1/P32 | B1/P64 | B8/P32 | B8/P64 | KV-cache reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GQA vs MHA (2L) | 64 | +1.57% | -3.93% | -3.95% | -1.21% | 50.00% |
| GQA vs MHA (2L) | 128 | -0.27% | -0.12% | +5.86% | +2.74% | 50.00% |
| GQA vs MHA (2L) | 256 | +4.93% | -3.11% | +1.48% | +4.27% | 50.00% |
| MLA vs MHA (4L) | 64 | +21.16% | +18.22% | +15.37% | +23.71% | 62.50% |
| MLA vs MHA (4L) | 128 | +24.03% | +13.03% | +23.90% | +19.93% | 75.00% |
| MLA vs MHA (4L) | 256 | +14.46% | +29.65% | +24.10% | +24.06% | 81.25% |

GQA halves allocated KV-cache storage at every width; its decode changes range
from 3.95% faster to 5.86% slower across these workloads. This does not establish a
consistent speed gain. MLA reduces allocated cache by 62.5%, 75%, 81.25% across the
three widths but decodes 13.03% to 29.65% slower. It offers a measured cache-capacity
tradeoff here, not a decode-speed improvement. Prefill can differ: the width256
MLA/batch8/prompt64 cell is 15.08% faster, while many other prefill cells are slower.
The full per-cell prefill results are retained rather than pooled with decoding.

Cache reductions come from actual allocated buffers at capacity128, not model
parameter counts or estimates of total device memory. MLA caches latents plus
rotary keys. Temporary expanded keys/values and other workspaces are outside
these cache-buffer counts. MPS current allocation was recorded for both resident
models together; true peak device memory was unavailable and is null in results.

Prefill is reported separately in each raw result and summary. Timing covers
synchronized model prefill and sequential teacher-forced single-token forward
calls. It excludes sampling, tokenization, cache allocation/reset, model loading,
input copying, and serving/network overhead. These are measurements of this
implementation on this Mac at context128, not predictions for an optimized kernel,
long context, another device, or a production serving system. Repeated timings
are not independent hardware replicates or training-seed evidence.

The existing full-validation MLA-minus-tuned-MHA quality deltas at widths64/128/256
are+.013453/+.006146/-.007100 CE nats/token. Against the human's+.01 quality ceiling,
width64 exceeds it on average; width128 has one paired seed at+.011711 and a
4096-token conditional text interval ending+.010028, so its lower mean alone
is insufficient for a confident near-neutral claim. Width256 has all3 paired
seeds better and a conditional text interval below zero. These remain adaptive
validation diagnostics, not an independently confirmed non-inferiority test.
GQA quality evidence still uses its original sampled panel; this timing study
does not supply broader validation or test evidence.

The quality incumbent stays four-layer MHA. The measured tradeoffs inform the
next prospective efficiency/robustness protocol; this diagnostic does not replace
its missing uncertainty/benefit acceptance rules or reopen historical selection.

Evidence: [plan](../research_runs/inference-efficiency-001/plan.json), [report](../research_runs/inference-efficiency-001/report.json), [review](../research_runs/inference-efficiency-001/review.json).
