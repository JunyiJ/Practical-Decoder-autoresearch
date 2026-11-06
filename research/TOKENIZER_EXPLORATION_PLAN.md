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

The first [geometry audit](tokenizer-evidence/exploration-boundary-001/geometry-review.md)
derives a seven-layer,width192 candidate and a four-layer,width192 FFN1024
control to separate body allocation from narrower vocabulary tables. This
updates the provisional depth question using actual counts; it is not a quality
screen, measured performance gain or numerical admission.

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

The completed structural body-allocation panel did not adopt the deeper model;
keep the incumbent and its frozen negative result. Its equal LR screening and
complete fresh confirmation count as structural exploration. The proposed
allocation remains a proposal; diagnostics and preparation do not count as new
numerical exploration or authorize quota-driven runs.

The current [boundary review](tokenizer-evidence/exploration-boundary-002/evidence-review.md)
uses original local curves, a train-only context census, contrary table/horizon
results and primary papers. Fixed-weight context/locality diagnostics give no
robust gain; a paper-inspired affine MLA core is correct but slower locally.
Neither is adopted or a reason to launch an unqualified training panel.
Next complete the bounded full train-split fit decomposition in
`train-fit-diagnostic-001`, follow its actual guard/result without duplication,
then use that evidence to rank a falsifiable next hypothesis. Any new scientific
panel needs fresh seeds, equally bounded controls/tuning and a complete current
resource forecast. No automatic beta1, MoE replay or longer-horizon study.

See [architecture agenda](TOKENIZER_ARCHITECTURE_ROADMAP.md),
[attention objectives](TOKENIZER_ATTENTION_ROADMAP.md),
[quality protocol](TOKENIZER_QUALITY_PROTOCOL.md) and
[retention contract](TOKENIZER_CHECKPOINT_RETENTION.md).

Current boundary update: full train-fit decomposition completed, mean train CE
3.612469291 versus recorded validation4.690826763. This gap alone does not prove
overfitting or justify a regularizer. Next separate train/validation frequency
shift from model-fit evidence and inspect original regularization results plus
primary papers before admitting a fresh finite design. All diagnostics are
closed, no numerical worker remains active, and no new panel budget is frozen.
See boundary002/evidence-review.md for exact evidence and applicability limits.


## Input-only regularization preparation

Frequency-composition controls and actual old dropout evidence were reviewed
with primary supporting/contrary papers. One new input-feature dropout site
(.1 versus .0) is qualified for correctness/resource preparation, as local
regularization rather than structural exploration. Original residual/attention
dropout losses are not erased. CPU/MPS model proofs, exact stochastic real-data
repeat/serialized resume, additive RNG/config preflight and finite worker-route
checks passed. Independent generated workers/pilots and scientific quality are
not yet executed; no scientific seed/freeze/adoption. All958bindings preserve
the939scientific+6prior diagnostic sources. Keep incumbent and closed holdout.

The prepared next numerical pair awaits actualfree>=11,051,937,792B after inline
recovery. The full future all-output planning bound is separate, with no future
retirement credits. Unchanged exhausted blocker:one silent hourly wake check,
no repeated broad audits/closed probes. New eligible evidence triggers recovery.
See [preparation](TOKENIZER_EMBEDDING_INPUT_DROPOUT_PREPARATION.md) and
[boundary evidence](tokenizer-evidence/generalization-boundary-001/evidence-review.md).


## Closed dropout panel; tokenizer-capacity structural preparation


The input-feature-dropout panel is CLOSED_NOT_ADOPTED: all three fresh seeds
worsened, mean candidate-minus-control CE +.014701832892. All12 endpoints and
independent review passed execution checks; retain incumbent/dropout0. The terminal
STOP closes all original commands. See [full results](TOKENIZER_EMBEDDING_INPUT_DROPOUT_RESULTS.md).

Vocabulary-capacity is the next evidence-grounded structural preparation question:
10653observed native IDs and vocabulary-dominated parameters motivate testing its
compression/context/quality tradeoff. An8K lossless merge-prefix component is a
single feasibility anchor, not an optimum selected by intrinsic metrics. Codec,
common-byte metric, CPU/MPS model/data and exact real repeat/resume proofs passed;
worker/resource/design/full-budget gates remain. No next scientific seed or panel
is frozen. Preparation is not numerical quality exploration, and70/20/10 remains
PROPOSED. Do not use CE/token across codecs, old scores as a new baseline, random
quota runs or automatic dropout/rank/horizon grids. Read [local and primary-paper
evidence](tokenizer-evidence/tokenizer-capacity-boundary-001/evidence-review.md).


## Native/8K prerequisites and resource pilots complete

Both four-update workers and twenty-update resource pilots independently passed
RAW/H200/optimizer/EMA/RNG/sampler/clipping,complete CPU/MPS common111475-byte
validation and cache/resetB1/8. Conservative1200-update projections761.14/322.90s
pass original7200s guards. This is not a tokenizer quality result. Actual equal
20480-token pilots sampled67642/native versus58020/8K bytes;raw exposure differs.
Independent nonconstant common-byte grouping/LRselection/bootstrap contracts pass.
All995current sources preserve972 original frozen bindings. Both roots have
terminal STOP;no numerical worker or scientific seed/panel/freeze is active.

Next implement/prove the full1200 quality worker/controller/all-six-fresh serving
and independent review,then unused-seed census/full LIVE budget and complete panel.
See [current preparation](TOKENIZER_GPT2_PREFIX_PREPARATION.md) and
[evidence](tokenizer-evidence/tokenizer-capacity-boundary-001/evidence-review002.md).


## Native/8K full scientific comparison launched

All full-source/state/serialization/seed/data/resource gates passed;quality001 is
frozen with1004bindings and NEW freeze46f96b0b2911843d8272299c6a7f79df85570542687f253a3558a331fd1dcfbb.
Controller serially runs six equal-LR screens and three fresh pairs,then all-six
fresh serving and independent final statistics. No complete quality result or
adoption yet. Keep incumbent and every current screen/fresh payload. Follow
actual processes/guards/STOP/anchors;never duplicate the controller or rerun an
attempted slot. See [launch evidence](tokenizer-evidence/tokenizer-capacity-boundary-001/evidence-review003.md).


## Native/8K screens complete; fresh confirmation active

All6screens passed independent1200-update acceptance. Frozen minimum-H200BPB/
lowerLRtie rule selected native.00125 and8K.0025. The original3freshpairs now
advance serially;no complete paired quality result or adoption yet. Keep every
screen/fresh payload and all1004frozen sources unchanged. See [phase evidence](tokenizer-evidence/tokenizer-capacity-boundary-001/evidence-review004.md).
