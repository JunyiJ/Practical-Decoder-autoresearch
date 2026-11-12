# Native pseudo-repetition continuation results

Ordinary CE wins the frozen validation comparison. The alternating CE/pseudo
recipe loses in all three parent pairs. Its mean time-EMA loss is5.005627187
versus4.861012204 for ordinary CE, a regression of0.144614982nats/token.
The symmetric ordinary-CE superiority rule passes every gate. Its one-sided99%
paired-document bootstrap upper bound is−0.129389331nats/token. This bound is
conditional on108 validation documents and these three parent/continuation pairs.
Test remains unopened. The losing recipe is closed at this setting.

| Pair | Ordinary CE | CE / pseudo | Candidate minus CE | RAW delta |
|---|---:|---:|---:|---:|
|1|4.8625809893|5.0042306204|+0.1416496312|+0.1587098485|
|2|4.8514456982|5.0067426033|+0.1552969051|+0.1713545988|
|3|4.8690099254|5.0059083360|+0.1368984106|+0.1599897171|

Candidate source regressions are+0.141734126nats/token on FineWeb-Edu and
+0.156153628 on Wikipedia. RAW mean losses are4.885536170 and5.048887559,
respectively. Neither averaging nor a single parent explains the ordering.
The authoritative results and fixed gates are in
`tokenizer-evidence/general-rsi-pseudo-repetition-001/decision.json`.
No earlier score is pooled into this comparison. This establishes a preference
between these continuation recipes, not improvement over their selected parents
or fresh pretraining superiority. Keep all selected parents and all12 new weights.

## Execution, exposure and cost

All six execution reviews passed. Each run completed its1,200-second synchronized
training budget with less than one update of overshoot. Total training time was
7,201.236368s; total guarded training-worker time was7,848.255285s, below the
8,561.53s forecast and14,400s ceiling. Guarded time includes scoring, generation,
checkpoint I/O and initialization. Subsequent CPU analysis and agent annotation
are outside that worker total. No training failures or recovery charges occurred.
All144 first outputs and token journals remain retained. There were no rerolls.

| Pair | CE control updates | Candidate CE updates | Candidate pseudo updates | Candidate natural targets | Valid pseudo targets |
|---|---:|---:|---:|---:|---:|
|1|3,493|1,732|1,732|1,773,568|1,161,507|
|2|3,506|1,734|1,733|1,775,616|1,160,520|
|3|3,512|1,733|1,733|1,774,592|1,157,619|

Candidate natural exposure is49.58%,49.46%,49.35% of the paired control.
Every ordinary-CE sampler prefix matches its control. All states and scored losses
are finite; no pseudo targets saturated the numerical cap. This is a valid quality
negative, not an optimizer, harness or resource failure.

`history-analysis.json` binds the original logs and eight fixed probe documents.
On that same8,586-target subset, candidate-minus-control RAW deltas at one third,
two thirds and the endpoint are respectively:

- Pair1:+0.145445,+0.188236,+0.210573.
- Pair2:+0.140285,+0.202311,+0.217185.
- Pair3:+0.105372,+0.153584,+0.217268.

The observed gap is present by the first probe and grows thereafter in each pair.
These are descriptive fixed-subset curves, not additional confirmation or a reason
to select an earlier checkpoint. They cannot distinguish direct penalty damage,
reduced natural exposure, or their interaction.

## Fixed generation evaluation

All144 first completions were rated after hiding arm, seed and checkpoint names
and applying the frozen shuffle. Mapping remained separate until annotations
were saved. The annotator was the Codex agent in this research chat. Prior CE
results and a few earlier generation excerpts had already been visible, so these
are exploratory identity-hidden ratings, not independent or fully naive human
validation. Consistency is not exhaustive factual accuracy. No composite score,
decoder tuning or generation-based rescue of the CE decision is used.

Each row below aggregates36 completions from12 prompts and three parents.
Semantic columns use the fixed0/1/2 rubric and equal-weight prompt means among
assessable outputs. Full ordinal counts and per-parent summaries are retained in
`generation-evaluation/summary.json`; every rating has a quote and rationale.

| Recipe | Decoder | Repeat trigram fraction | Terminal loops | Relevance | Consistency |
|---|---|---:|---:|---:|---:|
|Ordinary CE|Greedy|0.60305|17/36|0.9444|1.0278|
|CE / pseudo|Greedy|0.43772|8/36|0.7778|1.0000|
|Ordinary CE|Temperature0.8/top-k40|0.02957|0/36|0.7222|0.9722|
|CE / pseudo|Temperature0.8/top-k40|0.00717|0/36|0.5833|0.9167|

All greedy outputs reached64 tokens without EOS. Both sampled groups had2/36
EOS stops and34/36 token caps; mean content lengths were61.58 and62.61 tokens.
No unexpected early stop occurred. Two greedy outputs per arm had no assessable
consistency proposition; none were counted as consistent. Relevance was assessable
for all outputs. Most outputs only partly continue the prompt or drift away.
Lower repetition accompanies worse validation prediction and lower descriptive
relevance ratings. Reduced looping alone does not establish a model-quality gain.

## Hypothesis update and next action

The predicted validation advantage of this complete pseudo-repetition recipe is
falsified at this setting. The original rationale and literature limits remain in
`GENERAL_RSI_PSEUDO_REPETITION_NEXT.md`; this study does not refute all versions of
DITTO or unlikelihood training. Its small model, short context, heuristic spans,
reset adaptation state and equal-time exposure tradeoff differ from those papers.

No new training panel is admitted. Next perform an evidence-grounded boundary
review of the unresolved exposure-versus-penalty mechanism: read the primary
papers' mixing/exposure controls and compare them with these retained probe and
objective logs. Identify what new control would discriminate the alternatives
before proposing any new campaign. A lower coefficient or a replay of the losing
family requires a new evidence-based rationale and full fresh admission. Preserve
all closed comparisons, the original FP32 numerical failure and selected weights.
