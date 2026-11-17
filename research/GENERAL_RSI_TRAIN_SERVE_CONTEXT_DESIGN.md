# Fixed-weight training-context by serving-context diagnostic

Human request, 2026-11-17 Pacific: compare context128-trained and context256-trained
models at rolling128, rolling256 and rolling512. No new training, generations,
original-test access, endpoint selection or positional tuning.

Use all12 saved time-EMA endpoints from the completed D256 depth4/16 study,
three paired bundles per training length and depth. The model weights remain fixed.
The six train256 rolling128/256 cells reuse hash-verified original B/C scores;
subset their targets rather than repeat inference. Score the remaining24 cells.
There are36 model/serving cells in total, grouped into12 models.

Every cell uses the same51,457 validation target tokens from65 documents with
zero-based position p>=512, including eligible terminal EOT tokens. Exclude earlier
targets identically. The previous72,876-target results remain historical and are
not compared to these means. No cross-document history or padding is allowed.
Input for history h is tokens[p-h:p]; target is tokens[p]. Recompute each window
at stride1 with local positions0..h-1, no KV cache and no segment resets.

## Evidence and hypothesis

Local original evidence: depth-context-quality-001/decision.json and its accepted
completion-review.json establish the paired training runs. The original same-reset128
EMA differences were -0.021068/-0.012125 nats/token at depth4/16. The accepted
rolling-four-001/analysis.json and completion-review.json show fixed train256
rolling256 beats rolling128 by -0.025454/-0.029234 on its own matched subset.
Shuffle removes supported gain; no reasoning claim follows. Original score files,
pair variation, resource guards and failures are retained. Training used equal
time, not equal token exposure; optimizer, schedule and clipping remain entangled.

Primary literature read for this boundary:
- Khandelwal et al., arXiv:1805.04623v1, sections3–4 and equation1, compare fixed
  weights under truncated histories and exclude insufficient-history targets.
  Their word-level LSTM results do not establish effects for our subword MLA.
  https://arxiv.org/html/1805.04623v1
- Chen et al., arXiv:2306.15595v2, sections2.1–2.3, equations1–4, report that direct
  RoPE extrapolation can degrade language modeling; their interpolation plus
  fine-tuning results concern much larger LLaMA models. This is contrary evidence
  to assuming that a longer serving window must help. We apply neither interpolation
  nor fine-tuning. https://arxiv.org/html/2306.15595v2

Hypothesis: training at256 may improve predictions even when serving128, while
serving256/512 may recover or exceed that benefit for train128 models. Direct
RoPE extrapolation could instead degrade either model. The direct policy contrasts
below discriminate these outcomes. A positive or unresolved contrast is retained;
there is no scientific retry, equivalence claim or posthoc context selection.
This is a diagnostic on reused development data and models, not fresh confirmation.

## Implementation and qualification

Keep frozen original sources unchanged. A separate inference view temporarily
extends only deterministic RoPE cosine/sine and triangular causal-mask buffers to512.
Use the original CPU FP32 RoPE formula and theta10000; verify exact original prefixes.
Do not alter weights, training identity, attention scale or positional frequency.
Restore every original buffer object on exit and validate the original model.
The view has an explicit input contract: FP32/no_grad, batch<=2, history128/256/512,
at most1024 input tokens. Original wrapper guards are not edited or relabeled.

Qualify all four depth/training combinations on fixed training-prefix inputs:
original-path logits at native lengths, batch2 versus batch4 and scalar inference,
last-position projection versus full logits, CE versus float64 logsumexp within1e-5,
inherited logit tolerance3e-5+3e-4 relative, causal prefix invariance at512, independent
RoPE formula and triangular-mask checks, exact parameter hashes and buffer restoration.
All finite numerical failures are preserved separately from quality outcomes.
Measure128 target evaluations per serving length and all four combinations to
forecast complete remaining work. Qualification never uses validation quality to
alter implementation or rules. No repeat of completed older qualification fixtures.

## Frozen comparisons and uncertainty

Let L(t,h) denote CE at training context t and serving history h. At each depth:
1. Training effects: L(256,h)-L(128,h), h=128,256,512 (3).
2. Serving increments: L(t,256)-L(t,128) and L(t,512)-L(t,256), t=128,256 (4).
3. Training-by-serving interactions: difference between training lengths in each
   of the two serving increments (2).
4. Direct long-training/short-serving policies: L(256,128)-L(128,256),
   L(256,128)-L(128,512), and L(256,256)-L(128,512) (3).

The last three are the user-facing priority. Negative favors longer training with
shorter serving. Report all six cell CEs, per-bundle/source results, all12 contrasts
at each depth, and12 direct depth16-minus4 differences: one36-contrast family.
Use100,000 shared paired document draws over the65 eligible documents, PCG64
seed621117310001, average three bundle loss sums before resampling, actual token
counts as denominator, family alpha0.01 and quantiles0.01/72 and1-0.01/72.
Require all three bundle signs and adjusted interval exclusion of zero for a
consistently supported direction. No equivalence margin or adoption threshold.
Intervals condition on these saved models and reused documents, not population
seed uncertainty. Report two-depth replication, not a scaling law. Freeze all
criteria before any new validation scoring. Prior B/C and four-condition intervals
remain unchanged in their original families.

## Complete cost and execution

Run one serial MPS worker under the nonblocking shared research lock. Fixed order:
depth4 then16, bundle1 then2 then3, train128 then256, histories ascending.
Reuse train256 h128/h256; score h512 for those models. Measure each cell's elapsed
scoring time and input tokens as cost diagnostics. These uncached stride-one times
do not represent optimized production serving latency.

Forecast = measured128-target cost times all51,457 targets times three bundles,
counting only new cells, plus50% margin and600seconds overhead. Admit the entire
panel only below12hours; qualification ceiling900seconds, analysis600seconds.
Preserve RSS4GiB/MPS6GiB, normal host pressure and >=1.25GiB reclaimable-page proxy
before load, >=1GiB proxy/pressure<=2 during work. Proxy includes other apps but is
not guaranteed free RAM; RSS and MPS overlap. Check load, extension, scoring,
serialization and release. Preserve12GiB disk reserve plus512MiB for all score
files, metadata, failure evidence and atomic overlap. No endpoint copies,
optimizer or EMA serialization. Initial free space17.65GB exceeds this bound;
remeasure after qualification and before full launch. Do not shrink the panel.

On completion independently review target alignment, reuse hashes, native parity,
extended-buffer controls, weight integrity, bootstrap and resource guards. Report
results and practical training/serving tradeoffs, update state, publish authorized
source/evidence and stop its monitor. No further campaign follows automatically.
