# Tokenizer exploration allocation plan

Proposed on November 5, 2026, in response to the human request for more
exploration. Start the new allocation after the current final LayerNorm scale
panel, paired inference and final review close. Finish that frozen panel with
its original recipe, seeds and gates. This plan defines future search priorities;
it does not freeze a new numerical experiment.

The aim is to improve the best validated model while regularly testing whether
its architecture, training regime or comparison assumptions limit progress.
Small normalization and optimizer ablations remain useful, but they cannot
substitute for exploration of a different model family or interacting choices.

Human clarification on November 5: exploration must be grounded in facts,
past learnings, already collected data and research papers. The evidence rules
below apply immediately; the numerical percentage remains a proposed allocation.

## Ground every exploration in evidence

Before selecting a new scientific direction, use the
`evidence-grounded-research` skill and review the measurements behind the current
incumbent and relevant closed studies. Analyze existing loss curves, seed
variation, stability, capacity and inference measurements where they bear on
the question. Identify a measured bottleneck or an unresolved assumption before
choosing a new architecture or parameter grid.

Read the relevant primary papers, including their ablations and limitations;
save direct references to the exact version and supporting or conflicting
sections. Compare their data, scale, training budget and architecture with ours.
Paper results are external evidence for a hypothesis, not a claim that their
gains transfer here. Mark local observations, published results and our own
mechanistic inference separately. An exact precedent is not required, but a
missing precedent must be acknowledged rather than replaced with invented support.

Each candidate needs a short note in its existing boundary/design record:

1. Local evidence paths, relevant prior successes and failures, and what the
   collected data actually show, including uncertainty and contradictory results.
2. Primary-paper references, the specific supporting/conflicting result, and
   the applicability gap in our setting.
3. A mechanism, a falsifiable prediction, and why either outcome teaches us
   something that the existing studies do not establish.
4. Discriminating controls, bounded tuning/selection rules, full fresh
   confirmation, and the complete resource cost.

Choose among qualified candidates by the bottleneck addressed, evidence,
expected information and feasible cost. Novelty alone is insufficient. The
exploration quota never requires random experiments: if no candidate has a
sound basis, carry its reserved allocation forward and do the bounded data
analysis or literature work needed to qualify one. Do not count that preparation
as numerical exploration or silently spend the reserve on more local tuning.

## Reserve 30 percent for exploration

| Allocation | Share of numerical experiment time | What qualifies |
| --- | ---: | --- |
| Incumbent improvement | 70% | Nearby optimizer, schedule, normalization, scaling and other isolated recipe refinements |
| Structural exploration | 20% | A distinct capacity allocation, sparse expert design, attention/state mechanism or training context regime with a discriminating control |
| Interaction and transfer exploration | 10% | A bounded joint-change experiment or transfer test that can reveal where a locally losing choice becomes useful |

Use the next ten numerical hours as the first budget block: seven hours for
incumbent improvement, two for structural exploration and one for interaction
or transfer. These are planning allocations, not instructions to run ten hours
uninterrupted or to truncate an experiment. Track actual elapsed numerical
worker and parent acceptance time, including pilots, tuning, failed attempts,
confirmation and inference. Count the serial timeline once; do not add a
parent's worker-wait time to the same worker time. Charge the entire experiment to the category
declared before it starts. Confirmation of an exploratory candidate remains
exploration even after it looks promising. Do not count idle time, cleanup,
paper reading, implementation or metadata work as numerical exploration.

Use existing process and guard receipts for accounting. Add one small table at
each campaign boundary with category, hypothesis, forecast, actual time, outcome
and next decision. Do not build a new scheduler or tracking service for this plan.
Track engineering time separately so expensive implementation is visible.

Reserve the full forecast through confirmation before admitting a campaign.
Campaigns are indivisible: an overrun or a full panel crossing the ten-hour
boundary carries its allocation into the next block. Never stop a frozen panel
to hit an exact percentage. At the next admission, repay an exploration deficit
before admitting more local tuning. The first campaign after final-scale closure
should be structural exploration. Allow at most two consecutive local-tuning
campaigns; if the exploration share is already high, finish useful analysis
before launching another broad search solely to satisfy that count.

## Distinguish novelty from relabelled local tuning

A 256-parameter final-normalizer change or another AdamW beta is incumbent
improvement, even though it changes architecture or optimization code. An
exploration proposal must identify a materially different mechanism or a
specific assumption it can overturn, state what observation would discriminate
it, and explain how it differs from closed studies.

Keep at least three mechanism families under consideration. A queue entry is
not qualified for training until it passes the evidence review above. Rank it
by the value of resolving the question, its evidence and applicability limits,
distance from recent work, feasibility and estimated total cost. An infeasible
MoE panel does not block a different evidence-qualified candidate or useful
independent analysis.

## Initial exploration queue

These are provisional investigation questions, not a fixed launch order. Their
local-data and literature reviews can change their priority or remove them.
Exact shapes, seeds, budgets and gates require a separate finite protocol before
any training. None is admitted merely to fill the exploration percentage.

| Priority | Question and candidate | Discriminating comparison |
| --- | --- | --- |
| 1 | Would a different depth and width allocation beat the four-layer D256 basin? | Audit four-, six- and eight-layer geometries; derive widths to approximate a declared non-embedding compute budget and preserve valid attention geometry. Report total parameters, vocabulary-table cost, active compute and measured time separately. Equal total parameters alone can hide the large table cost. Select a bounded feasible design before freezing. |
| 2 | Would sparse FFN capacity help the subword model? | Revisit the prepared four-expert/top-two MoE direction against its declared dense controls. Include router and shared work in active/total cost, load balance and auxiliary-loss reporting. Recompute complete disk/RSS/time requirements; the old 18,782,075,912-byte forecast is historical, not current readiness. |
| 3 | Does the best local recipe depend on the short training context? | Prospectively compare a bounded longer-context training regime with a fresh context128 control. Keep tokenizer and text splits fixed. Declare a common quality evaluation and separate length-specific diagnostics, equal-target and measured-compute accounting, and cache costs. A new context regime needs explicit model/data contracts; it does not change the current frozen model. |
| Conditional | Does a structural change require a companion training change? | Use a small 2-by-2 design, such as baseline/deeper allocation crossed with baseline/depth-aware residual scaling. Freeze all four cells and the selection rule together. It directly tests an interaction that two isolated one-variable studies could miss. |

Do not automatically repeat KDA/hybrid, table sharing or other closed studies.
The width64 tokenizer hybrid already lost to its selected all-KDA control; a
future attention proposal needs a new, explicit scope or mechanism and fresh
controls, not the old trial replay. Likewise, old character-model failures are
priors rather than proof about a new subword design.

## Give different families a fair chance

Before screening a new family, declare either a common fixed-recipe question or
an equally bounded per-family optimization question. A common LR comparison
answers how choices behave at that LR; it does not establish each family's best
performance. For structural exploration, prefer a small balanced LR allowance
when feasibility permits, with the same number of declared candidates and
training budget per arm. Freeze the grid before any results; no rescue tuning
or extra seeds for a disappointing arm.

Use correctness and resource pilots first. Short loss probes can reject invalid
or infeasible designs under declared rules, but cannot establish final quality.
When quality ranking is needed, prospectively give each eligible arm the full
declared screening horizon so late-learning families are not eliminated simply
for being slower starters. Freeze screen retention and selection before launch.

Keep development selection and fresh confirmation separate. Reserve the full
confirmation budget before screens, publish every screened arm and select only
under the saved rule. A selected challenger still needs the complete fresh
three-pair comparison, paired inference and independent numerical review before
adoption. Screening losses and old study scores are never pooled into that result.
No candidate is owed promotion because the exploration quota needs filling.

## Test practical value without changing old decisions

Declare the comparison's primary resource axis and practical-effect threshold
before results. Report CE and actual memory, cache, prefill/decode cost and
training time where relevant. Keep a nondominated alternative in the research
queue when it offers a useful declared tradeoff; retention is not adoption.
Use the existing prospective quality-neutral attention route only where its
scope applies. Other routes need their own justified, predeclared gates.

The current mean-delta-below-minus-0.02 rule is a practical threshold, not a test
that smaller effects do not exist. The completed block LayerNorm comparison
improved all three seeds by a mean 0.00544214 nat/token but failed that threshold.
Preserve that original decision. For new experiments, choose thresholds from
the intended practical benefit and variability before observing new results;
do not lower a gate after a candidate narrowly misses it.

Keep the test holdout closed. Log adaptive validation reuse and distinguish
conditional text uncertainty from seed-population uncertainty. A claim that a
direction generalizes beyond this Shakespeare/subword setup requires a separate
prospectively defined transfer dataset/protocol, not more reuse of the same
validation text. This plan authorizes no new data acquisition or holdout opening.

## Review allocation and avoid another narrow search

At each campaign boundary, report the numerical exploration share, distinct
mechanisms tested, tuning spent per family, complete confirmations, useful
tradeoffs and the strongest falsified assumption. Adoption rate alone is not an
exploration metric. Record failed and negative studies rather than rerunning them.

After three consecutive completed local-improvement campaigns with no adoption,
temporarily use a 50/50 local/exploration allocation for the next ten numerical
hours. Keep that block's exploration split two-thirds structural and one-third
interaction/transfer. Review afterward; do not automatically extend a plateau
exception forever. A newly validated practical improvement can justify returning
to 70/30 at the next boundary, with the reason recorded.

Every future panel retains one local worker, four threads, closed holdout,
immutable sources/evidence, shared locks and full original resource protections.
Forecast all retained outputs, serialization, metadata and the existing reserve,
then target an additional 2 GiB headroom before dispatch. Run authorized recovery
on a new blocker. Do not shrink confirmations, clear STOP, rerun closed attempts,
delete selected/fresh weights or use a partial panel to fund exploration.

## Next concrete action

After final-scale closure, review existing depth/scaling and MoE evidence,
analyze relevant collected curves/costs, and read the primary literature before
prioritizing the depth/width feasibility design or refreshing the MoE capacity
audit. Choose the most informative evidence-qualified feasible structural
campaign, then freeze its full resource and scientific protocol. AdamW beta1
remains in the local-tuning queue; it does not automatically displace this review.

See [architecture agenda](TOKENIZER_ARCHITECTURE_ROADMAP.md),
[attention objectives](TOKENIZER_ATTENTION_ROADMAP.md),
[quality protocol](TOKENIZER_QUALITY_PROTOCOL.md) and
[retention contract](TOKENIZER_CHECKPOINT_RETENTION.md).
