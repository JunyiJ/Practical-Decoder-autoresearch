# Context length quality results — 2026-11-13

Context256 passes every frozen adoption gate in the six-run native comparison.
Mean time-EMA CE is4.735973328 versus4.831685245 for context128. The paired
256-minus-128 difference is−0.095711917nats/token. The one-sided99% paired-document
bootstrap upper bound is−0.083908521. All three pairs improve and both source
means improve: FineWeb-Edu−0.090657419, Wikipedia−0.115956616.
Retain context256 for this native14,400-second recipe.

| Fresh pair | Context128 EMA CE | Context256 EMA CE | Difference | Common128 difference |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4.844975685 | 4.747021581 | −0.097954103 | −0.025389930 |
| 2 | 4.827909080 | 4.732647420 | −0.095261660 | −0.022704986 |
| 3 | 4.822170971 | 4.728250982 | −0.093919989 | −0.018567885 |

The primary contrast compares complete recipes at equal synchronized training
time, with each model evaluated at its native context. Both score the same
100,416 targets from108 documents. The context256 models also improve at the
fixed common128 evaluation, mean−0.022220934. This supports a training-related
benefit beyond longer evaluation prefixes. It does not isolate context from
batch reshaping, update count or optimization differences. Common128 is secondary
and did not rescue adoption. RAW mean differences are−0.094062460 at native
contexts and−0.021639911 at common128.

The frozen bootstrap averages three paired document loss sums, then draws10,000
resamples of108 document IDs using PCG64 seed62111211010099 and the sampled
count denominator. Its bound is conditional on these documents and seeds, not a
population-seed confidence interval. Reused validation is a development set.
No claim of broad-domain transfer, long-prompt generation or size transfer follows.
The original test remains unopened.

## Fixed first generations

All144 original first outputs were retained and rated in identity-hidden order
before joining arm/seed/decoder identities. Ratings are exploratory Codex agent
judgments, not independent human validation or an exhaustive factual audit.
The12 fixed prompts, two decoders and64-token limit were unchanged. All prompts
and continuations fit within128, so this is not a context-eviction test.

| Arm / decoder | Relevance 0/1/2/U | Consistency 0/1/2/U | Terminal loops | Mean repeated trigram fraction |
| --- | --- | --- | ---: | ---: |
| 128 / greedy | 9/26/1/0 | 5/26/2/3 | 14/36 | .613351 |
| 256 / greedy | 6/30/0/0 | 9/23/1/3 | 16/36 | .577509 |
| 128 / sampled | 14/22/0/0 | 10/24/2/0 | 0/36 | .018369 |
| 256 / sampled | 15/21/0/0 | 7/29/0/0 | 0/36 | .039427 |

U means unassessable. Per-seed summaries, assessable means, equal-prompt means,
full rationales and text evidence are retained in `generation-evaluation/`.
All72 greedy outputs hit the token cap. Of72 sampled outputs,71 hit the cap and
one context128 output ended with EOS. There were no unexpected early stops.
Relevance and consistency move in mixed directions. Greedy terminal loops rise
14→16 despite lower repeated-trigram fraction. The CE gain therefore does not
establish better readable completions. Generation did not enter the adoption gate.

## Execution and cost

All six execution reviews pass. Slot-order updates are42,152,40,907,41,557,
41,582,41,881,41,617. Each completes14,400 synchronized training seconds plus
less than one update of overshoot. Total guard time is88,803.525seconds
(24.667646hours), within the129,600-second tranche. All12 RAW/EMA endpoints
remain retained, totaling1,367,985,732bytes. All six first-generation sets and
original source/config/log/metric/checkpoint receipts remain retained.

The completion audit replays the decision exactly, independently recomputes the
bootstrap through per-draw frequency counts, checks five decision counterexamples,
replays the generation aggregate and verifies all12 endpoint hashes. An initial
audit-only assertion failed because JSON converts integer map keys to strings.
Its source and failure record remain in `completion-audit-attempt-01/`; the
separate repaired audit normalizes JSON keys and passes. No numerical result,
scientific gate or frozen worker source changed. No worker is active at closure.

Evidence under `tokenizer-evidence/general-rsi-context-length-quality-001/`:
`plan.json`, six `slot-XX/review.json` files, `decision.json`,
`generation-evaluation/summary.json`, and `completion-review.json`.
Analysis sources: `general_rsi_context_length_analysis_v1.py`,
`general_rsi_context_length_generation_analysis_v1.py`, and
`general_rsi_context_length_completion_review_v1.py`.

Next select one cross-size comparison under `GENERAL_RSI_NEXT_CAMPAIGNS.md`.
Rank this completed context finding against the default byte/native equal-token
quality, parameter saving and runtime interaction. No new scientific panel is
admitted by this result.
