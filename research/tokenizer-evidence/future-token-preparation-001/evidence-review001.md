# New direction: auxiliary second-next-token supervision

Human steering on November7,2026: leave the closed CE branch alone and pursue
another evidence-grounded route to model improvement. Selected direction:
**two-token auxiliary supervision**, beginning with a small target/objective
component. This is a training-objective/representation experiment, not a CE
kernel change, readout-projection rescue or new FFN activation attempt. Old
CE/FFN fixtures, sources, roles and failure receipts remain closed and unchanged.

## Local evidence and choice

Read the actual current architecture/attention/exploration roadmaps and protocol,
the original real-native config referenced by
`../post-projection-boundary-001/relu-squared-worker001-review.json`, and these
existing records without model rescoring or checkpoint loading:

- `../train-fit-diagnostic-001/result.json`: mean train CE3.61246929127793 versus
  recorded validation4.69082676267658. The gap1.0783574713986506 is descriptive;
  different text/frequency distributions prevent an overfitting diagnosis.
- `../generalization-boundary-001/evidence-review.md` and
  `../input-capacity-boundary-001/evidence-review.md`: rare-token fit correlates
  with frequency; input spectral concentration matches a frequency-controlled
  Gaussian. Neither supplies evidence for a trained low-rank-input improvement.
- `../../TOKENIZER_EMBEDDING_INPUT_DROPOUT_RESULTS.md`: complete input dropout
  comparison worsens all three fresh seeds, mean+.014701832892. Do not repeat a
  generic regularizer or infer that any auxiliary task will help.
- `../context-information-003/result.json`: local128 attention at256 restores
  much of the global256 regression but gives only an inconsistent small change
  versus native128. Additional context alone is lower priority.
- Original native inventory in
  `research/relu_squared_model_v1/contract.py:parameter_shapes`:28,492,416 learned
  elements;25,731,584 belong to the two tables. Body size and corpus exposure
  motivate checking a different supervision signal, not merely adding parameters.

The measured gap and sparse vocabulary exposure leave representation learning
and sample use unresolved. They do not prove an induction bottleneck. The new
question is whether supervising a future continuation improves the deployed
next-token head more than spending the same auxiliary capacity on the next
token again. This comparison was not in the closed dropout, FFN, readout-mixture,
projection, tokenizer-prefix, depth or MoE panels. A targeted repository search
found no prior multi-token prediction design in the research notes/plans.

Ranked alternatives at this boundary: (1) the selected auxiliary-target question
offers a distinct training mechanism and an exact target-offset control;
(2) longer-context/position work has contrary fixed-weight and historical
context findings and needs a stronger transfer rationale;
(3) input factorization/sharing has contrary rank/null and table-sharing results.
No closed family is replayed to satisfy an exploration percentage.

## Primary paper and contrary results

Actually retrieved/read [Gloeckle et al., arXiv2404.19737v1](https://arxiv.org/html/2404.19737v1),
§2 equations2–3 and memory-efficient implementation, §3.1/Figure3, §3.5/3.7,
§4.1/Figure7 and §5.1–5.2. Its heads predict future offsets from a shared trunk;
sequential head backward limits live vocabulary-sized buffers. Small code models
can regress; natural-language results are mixed. The small-model induction result
is task-specific, uses different data and permits oracle epoch selection. These
are reasons for a controlled feasibility test, not evidence of local CE gains,
free training, faster inference or a new stopping rule.

Our GPT2/Shakespeare/MLA/MuonAdamW/1200 setup is materially different. We will
retain next-token validation CE, closed holdout and full fresh confirmation; no
generation benchmark, induction score, speculative speed or auxiliary loss may
replace a failed quality gate. This review does not reproduce the paper.

## Mechanism, prediction and controls

Hypothesis: a shared causal representation trained to predict the second-next
token can carry information useful to the primary next-token head. It can also
hurt by forcing ambiguous future predictions or diverting scarce capacity.
Either direction is scientifically informative under complete fresh controls.
The immediate component asks only whether future targets and gradients are
correct. It cannot test representation learning or quality.

Prospective model design, not yet implemented or admitted:

1. Native: the genuine four-block incumbent next-token path.
2. Duplicate-next control: first three native blocks are a shared trunk; the
   original fourth block/final norm feed the primary head. An independent deep
   copy of block4/final norm is a training-only auxiliary branch predicting
   the ordinary next token. The sole native unembedding W is shared by the two
   branches; it is registered once, never tied to the input table.
3. Future-two candidate: exact same auxiliary architecture, initial values,
   optimizer policies and objective weights as duplicate-next; only its target
   offset changes to two. Inference uses only the genuine primary path.

Native construction comes first. Copying the auxiliary block/norm consumes no
extra random draws and must create independent storage. No private random
auxiliary initializer is needed. All native initial values/order/CPU draws must
be exact, and the two auxiliary arms must be exactly paired. A model proof must
test causality: neither branch may see its prediction target as input at that
position. A supplied-logit objective cannot establish this property.

Let C1 be mean genuine CE(primary,y[t+1]). Native objective is C1; auxiliary arms
use0.5*(C1+Ca), with Ca predicting y[t+1] for duplicate-next or y[t+2] for future-two.
The averaging convention is a fixed local design choice, not an optimum or exact
paper reproduction. Native-versus-auxiliary includes a changed gradient budget;
the duplicate-next comparison controls the auxiliary architecture and weights.
Primary CE is always a separately named unscaled report. No extra-target loss
enters physical validation or adoption statistics.

Targets require an explicit split-local sequence of length T+2 for T inputs.
No wrapping, padding, token discard hidden in a denominator, or val/holdout
crossing is allowed. All future arms, including fresh native controls, must
share declared sampled input positions. Actual paired512 sampler integration
and edge eligibility need a separate proof before corpus work.

For a later finite panel, only future-two is a candidate. Compare with BOTH fresh
native and duplicate-next, equal finite tuning allowance, screen/confirmation
separation and three fresh cohorts. Require mean delta<-.02 against each control,
all three seed deltas negative against each, and all six simultaneous conditional
text upper bounds negative at2048/4096/8192. No old-score pooling, duplicate-head
fallback, partial adoption, early stopping or extended horizon. The precise
scientific protocol/roles are not frozen here.

## Cost and admission boundary

The **implemented tiny component** has six new literal families, B<=2/T<=9/V<=23,
three objectives and CPU64/CPU32/MPS32:54 objective/analytic-gradient comparisons,
81 planned invalid rejections and explicit alignment/output-storage checks.
`component001/plan.json` pins exact sources/cases/elementwise predicates before
execution. No model, optimizer, corpus, checkpoint, old oracle, bootstrap or
seed/RNG draw. Source AST review imports neither component nor harness.
Forecast30seconds/1GiB RSS/256MiB MPS/no tensor files/2MiB metadata. Hard gates
remain120sCPU/600sMPS/4GiB RSS/6GiB MPS/8GiBreserve. Entry includes extra2GiB,
target10,739,515,392B; observed35,133,341,696B covers it. No storage blocker or
cleanup need. Success admits only this finite target/loss component.

For the prospective actual model, algebra from the native ordered inventory
gives690,080 elements in block4 plus512 in its final norm: auxiliary adds690,592
elements/23 tensors, making29,183,008 elements/111 tensors (55Muon/56AdamW if the
inherited grouping rules apply). This is not parameter or runtime equality with
native. It is a fresh scalar forecast, not actual model construction or grouping
acceptance. Shared W gradients and complete auxiliary exposure/EMA need proof.

Prospective18 scientific trajectories plus3 prerequisite pairs at640MiB/pair,
512MiB root metadata,128MiB sidecars,128MiB consumer workspace,8GiB reserve and
2MiB entry imply floor23,490,199,552B, extra2GiB target25,637,683,200B. This is
new own count-based accounting; matching an old number supplies no output credit.
Actual archive sizes, failures/.tmp overlap and every consumer still need new
measurement before any budget freeze. No new weights or retirement now.

Naively keeping two full-vocabulary heads live increases memory; the paper's
no-overhead result does not qualify this MPS implementation. Review sequential
head backward/trunk-gradient accumulation before real source execution. Parent
CPU reads, EMA, moment staging/deep clones, all nine fresh serving consumers,
statistics and acceptance need complete implemented forecasts and measurements.
Do not launch a model that cannot fit4GiB RSS/6GiB MPS or invent measured timings.
The outer hard-time envelope for18 workers and18 parents is18*(7200+7200), plus
600 serving+120 primary stats+120 independent stats+120 closure =260,160seconds;
this is an upper envelope, not a time budget or expected runtime. Short actual
measurements must establish a useful finite campaign forecast. No source-only
number qualifies training, quality or adoption.

## Next executable actions

Execute the pinned tiny component once after its source/case/resource review;
close success or failure without retries or post-outcome edits. On success, the
next work is the smallest actual-model source integration of the three specified
arms and memory-bounded head backward, outside all frozen namespaces. Review its
exact cases, new fixture roles/census, serialization and resources before any
model construction. Keep all CE gates and closed failures unchanged. Preparation
is not numerical quality exploration; a later full auxiliary-objective comparison
belongs to structural/interaction exploration, declared before its outcomes.
