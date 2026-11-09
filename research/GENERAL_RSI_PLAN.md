# General-corpus RSI research plan — 2026-11-08

## Authority and objective

The human requested: “go ahead - form a new RSI research plan with the new data
and re-enable the research”. Resume autonomous local research under this plan,
with one training worker, until a new human stop. This is a new campaign,
`general-rsi-v1`, not a reopening of any closed Shakespeare experiment. Use
`research_state/general-rsi-state.json` as the current pointer. Historical stop
records remain evidence about their original campaigns. The unfinished output
lexical pilot and its preserved disk failure are deferred, not silently retried.

Objective: improve measured general-text prediction and inspect readable model
completions per unit of local training time. Larger data is not evidence that the
byte architecture wins, that prose is coherent, or that the model follows chat
instructions. No paid remote compute or subagents are authorized by this plan.

## Evidence and hypothesis

Measured local facts:

- `data/raw/rsi-general-10m-v1/manifest.json`, SHA256
  `b1849a5fed1fb648f507e407e5794946bba64328f519e6b0d6c360fa4e2e4fa8`:
  10,001,163 training tokens / 9,807 documents, 100,524 validation tokens /
  108 documents; roughly 80% FineWeb-Edu and 20% Wikipedia by tokens. Test
  metadata lists 101,204 tokens / 107 documents; test contents remain unopened.
- `research/tokenizer-evidence/general-data-inspection-001.json`: training and
  validation hashes, contiguous document offsets, end markers and 29 exact
  retokenizations passed; no shared normalized exact-document hashes between
  train and validation. Near-duplicate decontamination was not independently
  revalidated. The builder has source filtering and passage-overlap rejection.
- `research/tokenizer-evidence/public-inference-correction-001/adoption/recipe.json`:
  adopted raw-byte composition uses 16,351,859 parameters versus native
  28,492,416, same D256/four MLA blocks/context128/GPT-2 IDs. Its three-seed
  Shakespeare validation advantage was mean -0.099670 nats/token against native
  and -0.079202 against nonlexical codes. This is not new-domain evidence.
- `research/tokenizer-evidence/byte-generation-demo-001/result.json`: four first
  completions have theatrical formatting but broken grammar and weak coherence.
- Old slot10 original step records total 3,746.05 seconds / 1,200 updates;
  median 3.112 seconds. A later optimized execution path measured roughly
  1.1 seconds on short Shakespeare updates. Neither predicts new-data cost
  reliably. The new pilot measures it directly.
- Closed tying, plain/residual readout transforms and beta2=.95 studies did not
  beat their respective incumbents. Do not replay them under old identities.

External evidence, distinct from local results:

- Penedo et al., *The FineWeb Datasets*, arXiv:2406.17557v2 (31 Oct 2024),
  sections 3.1, 4, 4.1–4.2: controlled equal-token comparisons and educational
  filtering support studying this corpus. Contrary scope: educational filtering
  favors some domains while ordinary FineWeb has lower perplexity on several
  broader web domains. Their 1.71B models, billions of tokens and context2048
  are much larger than ours. [Paper](https://arxiv.org/html/2406.17557v2).
- Hoffmann et al., *Training Compute-Optimal Large Language Models*,
  arXiv:2203.15556v1, sections 1, 3.1–3.3: parameter count, seen tokens and
  schedule horizon must be considered together. Their billions-of-token,
  mostly >500M-parameter regime does not prescribe an optimal token ratio for
  this small finite corpus. [Paper](https://arxiv.org/html/2203.15556v1).

Inference to test: broader data should enable general-prose learning, while
byte-spelling sharing may or may not retain its quality/parameter advantage
under a different vocabulary-frequency distribution. A matched new-data
comparison discriminates transferable architecture benefit from a narrow
Shakespeare result. Native winning is useful evidence, not a harness defect.

## Frozen first comparison

Two arms: native independent input/output tables, and adopted raw-byte input
composition with independent output table. Both start from fresh initialization;
the old Shakespeare checkpoint remains a preserved demonstration reference, not
a warmed candidate compared unfairly with a fresh baseline. Keep D256, four
MLA blocks, context128, FP32, ordinary next-token CE, Muon/AdamW grouping,
beta2=.999, epsilon1e-6, clipping1.0 and independent decay .00006 unchanged.
No output-composer, weight tying or new context mechanism enters this comparison.

Training uses the existing two-random-512-token-block sampler reshaped to 8x128,
1,024 targets/update, over the entire training stream. End-of-document tokens
remain in the stream; crossing documents in training is an explicit inherited
packing policy. This randomly mixes the source-contiguous files. Exposure is
not an epoch guarantee: random sampling can revisit and miss positions.

1. **Execution pilot:** 20 updates per arm with one fresh paired seed bundle;
   fixed prior LRs native .00125 / byte .0025. Measure throughput, RSS/MPS memory,
   actual checkpoint bytes, finite gradients and loss. Verify sampler isolation,
   exact saved-weight roundtrip and CPU/MPS corrected-logit agreement on one
   training case. No loss-improvement gate, validation selection or adoption.
   Limits: 600 seconds, RSS4 GiB, MPS driver6 GiB, retained pilot weights256 MiB,
   metadata8 MiB, disk reserve10 GiB. Preserve any failed attempt unchanged.
2. **Worker qualification:** implement only general-data complete restart,
   schedule/averaging and document-wise evaluation needed for the next run.
   One short uninterrupted versus interrupted/restored comparison must agree
   at the original numerical tolerances; verify all optimizer/sampler/averaging
   state, not just weights. Verify scoring target coverage with a small oracle.
   Do not rerun the historical test suite. Same 600-second ceiling per genuinely
   new prerequisite; a failed scientific criterion cannot be renamed a fixture
   defect. Freeze code/runtime/data hashes and roles before execution.
3. **Equal-budget LR screen:** .00125 and .0025 for EACH arm, 2,048 updates
   each, one paired screen bundle, four runs total. Choose lowest final H200
   full-validation CE within each arm; exact ties choose lower LR. Screen
   results select LR only, not an architecture or a final quality claim.
4. **Fresh confirmation:** chosen LR per arm, three fresh paired seed bundles,
   10,240 updates per run (10,485,760 targets), six runs total. This modest first
   horizon is about one corpus-equivalent exposure, not a convergence claim.
   Freeze seed roles, run order, full resource forecast and scoring contract
   before any screen starts. Never replace an unfavorable seed.

Screen and confirmation both use linear cooldown from floor(2H/3) to H,
minimum ratio .1. Tail EMA half-life200 updates, interval10, first snapshot at
ceil((2H/3)/10)*10; final H200 is primary, RAW secondary. No best-checkpoint
selection. Screen schedules match their own horizon; intermediate long-run
scores are not substitutes for short-horizon screens.

## Evaluation and decision rules

Evaluate all validation documents separately, score every token after each
document's first token exactly once, including its terminal EOT. Use sequential
nonoverlapping chunks of at most128 targets with context reset per chunk;
never score across document boundaries. Expected total is100,416 targets.
Compare models with exactly the same target/chunk inventory. Report overall
token-weighted CE and separate FineWeb-Edu/Wikipedia CE, plus document-level
loss sums and counts. Use corrected FP32 inference RMSNorm and stable CPU CE
when operating on CPU; no inference_mode or autocast.

For adopting byte over native on this corpus require all three paired final
H200 deltas <0, mean delta <=-.02 nats/token, and the 99% paired document
bootstrap upper bound <0 (10,000 resamples of the108 document IDs; average
paired seed loss sums before document resampling; fixed bootstrap seed declared
before screen). Neither source may regress by >.05 nats/token on the
three-seed mean. Native superiority uses the symmetric rule. Otherwise retain
both as unresolved; do not call a single-seed winner validated. Bounds are
conditional on these documents and seeds, not population-seed confidence.
Report wall time, tokens/second, parameters and peak memory beside quality:
equal-token quality superiority is not equal-time superiority.

Use a fixed small validation probe every1,024 updates for curves; full scoring
at final endpoints, and at confirmation step2,048 for diagnostic learning curves.
No checkpoint selection from those curves. Probe membership and evaluator must
be frozen before screen. Keep test closed throughout this first campaign;
repeatedly inspecting test would turn it into another validation set.

Save all first completions for the eight prompts in the campaign configuration,
at initialization and final endpoints of the first fixed-order confirmation
bundle. Greedy and temperature.8/top-k40/seed42, max64 new tokens, no rerolls.
Report grammar, relevance, factual consistency, repetition and truncation with
examples. These are qualitative diagnostics, not a blinded benchmark or an
instruction-following claim. They cannot rescue a failed CE comparison.

## Resource contract and recovery

Only local serial MPS execution. The initial screen+confirmation tranche has an
agent-selected ceiling of36 accelerator-hours, including evaluation and a30%
timing margin, not a human-imposed lifetime budget. Pilot and prerequisites
have their own short bounds above. If the measured full forecast exceeds this
ceiling, improve demonstrated execution overhead or prepare a new prospective
resource decision BEFORE scientific results; do not quietly shorten a frozen
confirmation or launch a partial panel. Subsequent tranches require a new
evidence/cost record but no repeated human permission within existing authority.

Forecast all ten final RAW/H200 inference-weight pairs, one active resumable
checkpoint and its atomic-write predecessor/temporary overlap, EMA/state RAM,
pilot artifacts, source/log/config/metric evidence, plus10 GiB disk reserve.
The reserve includes the existing8 GiB floor plus2 GiB headroom. Initial
conservative retained-output ceiling is4 GiB including serialization overlap;
replace the estimate with measured sizes before screen admission. Do not launch
the panel merely because the first run fits. For newly produced active-run
restart snapshots, retain the last valid snapshot until its successor is
hash-verified and resume-qualified. Retire only that explicitly transient
predecessor under the exact audit/journal/worker-lock safeguards. Final
inference weights, failures and scientific evidence remain retained. Historical
cleanup continues to use AGENTS.md; no blanket deletion permission follows.

Use `research_state/tokenizer-worker-lock/.lock` nonblocking for any worker;
maintenance uses its existing coordinated locks. Honor a new human stop at
every worker boundary and during long jobs. Preserve exact failed action,
source/config/log/receipt, classify failure, make the smallest justified repair,
and verify against the unchanged requirement under a separately recorded
attempt when needed. Never change seeds, tolerances or comparisons to hide a
negative scientific outcome. Do not create unchanged-status commits or scans.

## Recursive research after the first result

Priority1: inspect full train/validation curves and source-wise errors to decide
whether additional exposure is justified. Priority2: measure quality at equal
wall time if the byte composer improves CE but is substantially slower.
Priority3: investigate context length only if continuation/error analysis
supports it (training median document707 tokens versus context128 is a reason
to inspect, not proof of benefit). Architecture/optimizer changes require new
local evidence and relevant primary literature; an old losing family requires
a genuinely changed-condition rationale. Output lexical composition is deferred
because its measured additional cost is large and general-data learning is
the current unresolved bottleneck. No arbitrary exploration quota or tuning.

After every completed round, update hypothesis, implementation and next action
from actual evidence. Keep machinery proportional to the next decisive test.
Commit/push authorized source and evidence without force. Notify on meaningful
results, failures, completion or needed input; remain quiet on unchanged state.
The existing continuation heartbeat will target this chat and this plan.
