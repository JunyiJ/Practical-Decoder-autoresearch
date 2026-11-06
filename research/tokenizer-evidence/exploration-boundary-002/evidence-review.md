# Boundary review after the structural allocation comparison

The [completed body-allocation study](../../TOKENIZER_BODY_ALLOCATION_RESULTS.md)
retains the incumbent. This note distinguishes local observations, literature
and inference; it admits bounded diagnostics, not a new training panel or seed.
The evidence-grounded-research skill governs direction selection.

## Local facts and contrary evidence

- All three equally screened arms chose nominalLR.00125. Deeper-minus-incumbent
  fresh mean is+.007499234 nat/token with two worse seeds. Deeper improves over
  the width-matched shallow control by.015588741, short of the.02 practical
  threshold. It is not adopted. Its cache is45.8% larger than incumbent and its
  measured decode latency is about17.3%/14.6% slower at batches1/8. Do not infer
  that fewer parameters or more depth is a general improvement.
- [analysis001](analysis001.json) reads hash-bound accepted metadata only.
  All nine curves are non-monotone at the five saved online observations.
  The final299-step sampled validation changes average−.22859/−.24413/−.23489
  for incumbent/shallow/deeper; the corresponding599-step changes are smaller
  in magnitude. Shared sampled-batch difficulty is a possible explanation,
  not a demonstrated cause. These sparse curves do not select a horizon or
  establish either overfitting or continued gains beyond1200.
- Clipped updates are24/21/18 for the three incumbent seeds,14/12/13 for shallow
  and5/1/3 for deeper. These descriptive counts provide no clear evidence that
  excessive clipping is the deeper model's bottleneck. Do not copy an arbitrary
  depth multiplier from a paper.
- The exact train-only census covers all267,176 valid512-token starts and all512
  target offsets,136,794,112 combinations. Repartitioning the same stream from
  four128 chunks to two256 chunks raises availability of the most recent complete
  colon-label marker from66.49% to81.37%, a14.876-percentage-point difference.
  This is a syntactic proxy for context availability, not evidence of a quality
  gain or semantic speaker annotation. No holdout was tokenized/scored.
- Only10,653 of50,257 token IDs occur in the training stream. This is not a reason
  to remove vocabulary outputs or assume tying is harmless. The original
  [table-sharing final audit](../table-001/final-review.json) rejected its tested
  scaled-tied recipe after fresh confirmation; initialization/scaling controls
  also lost. Changing the current backbone alone is insufficient rationale to
  replay that study.
- The original [GQA horizon audit](../horizon-001/final-review.json) strongly
  favored the combined1200/cooldown800/H200 recipe over3000, mean−.710852. It is
  contrary evidence to automatic longer training, though its GQA/optimizer and
  coupled schedule differ from current MLA. Old scores are never pooled.

## Primary literature actually read

**Transformer-XL, arXiv1901.02860v3, sections3.1–3.2 and4.2–4.3.** Fixed segments
can fragment context; recurrence and positional treatment are experimentally
separated, and its relative-context metric uses a shared short-context baseline.
This supports discriminating extra conditioning from model/training changes.
Our dense causal MLA/RoPE, tiny Shakespeare corpus and absence of recurrence
differ from its architectures and benchmarks. Its gains are not local gains.
[Primary paper](https://arxiv.org/html/1901.02860v3).

**Scaling Data-Constrained Language Models, arXiv2305.16264v5, sections4–6 and
appendicesQ/S/W.** Modest repetition can remain useful, but returns diminish and
excess compute can hurt; the paper explicitly limits hyperparameter/architecture
transfer. Its GPT-2/C4/OSCAR, whole-epoch shuffling,2048context and Adam/BF16/cosine
setup differ sharply from our random512-window,267,688-token,FP32 Muon/AdamW
regime. Version5 also corrects fit R² reporting. We use no extrapolated optimum
or numerical scaling-law prescription. Our4.5904 target-exposure ratio is not
literal epochs. [Primary paper](https://arxiv.org/html/2305.16264v5).

**DeepSeek-V2, arXiv2405.04434v5, section2.1.2–2.1.3, equations9–19.** It describes
absorbing KV up-projections at inference and decoupling rotary features. The
current [local MLA](../../block_norm_v1/runtime/attention/mla.py) instead expands
cached latent K/V on every call. This is a concrete serving-optimization question,
not evidence of a local speedup. Our affine biases, score.5/sqrt64, short contexts
and MPS hardware need explicit algebra, correctness and measured-cost controls.
[Primary paper](https://arxiv.org/html/2405.04434v5).

## Bounded diagnostics and what they learned

[context-information001](../context-information-001/result.json) evaluates32/64
conditioning on all three already-trained incumbent fresh models, reading native
128 scores from immutable evidence rather than rerunning the closed study.
Mean CE is4.725661/4.700602/4.690827 for32/64/reference128, giving gains.025059
and.009775. This measures changed information with fixed weights, not a new
trained model or independently fresh improvement.

The isolated `mla_context_extension_v1` inference-only implementation preserves
all28,492,416 learned parameters exactly and extends only analytic RoPE/mask/cache
capacity. CPU and MPS native128 outputs match exactly;256 full/cache/reset and
causality pass at batches1/8, along with foreign checkpoint/cache, overflow,
mutation and alias rejection. It cannot train or resume raw checkpoints. Logical
cache size doubles to393,216B/item. Original939 scientific sources remain intact.

Despite the added available text,
[context-information002](../context-information-002/result.json) worsens mean CE
to4.707065 at256:−.016238 gain versus native128. No model or serving context is
adopted. This contradicts a simple “more visible text must help” story. The test
does not prove that training at256 fails; existing weights were trained at128.

## Ranked next work and discriminating controls

1. **Next bounded question: context locality versus reset fragmentation.** Plain
   extrapolation exposes attention to distances/cardinalities absent in training.
   Test exactly one additional fixed-weight diagnostic: process256 tokens but
   restrict each layer to its original128-token causal window. Native128 outputs
   must remain unchanged. Compare the same targets and all three existing models
   against the recorded native128 and full256 conditions. If this reverses the
   degradation, locality/streaming becomes a stronger training-study mechanism;
   if it does not, lower this direction's priority. This is not a rescue of the
   closed allocation study, and it cannot select/adopt a model.
2. A future **training-context × evaluation-context** comparison would train128
   and256 on identical flattened two512-block streams and1024targets/update,
   preserving parameters, initialization, rates/schedule and total targets.
   Evaluate BOTH at common128 and common256 conditioning. Never call native128
   versus native256 CE an isolated trained-model improvement. A new explicit
   training contract, correctness/resource pilots, complete cost and genuinely
   fresh confirmation remain required. No such panel is admitted/frozen yet.
3. **MLA absorption** is a grounded, low-output serving-engineering alternative:
   eliminate repeated latent K/V expansion while preserving biases, rotary math,
   masks, checkpoint/cache signatures and original numerical tolerances. Charge
   it as engineering/inference work, not new quality exploration. Prefer a
   bounded correctness/profile question before promising speed.
4. Horizon/optimizer interaction remains provisional, below the current context
   diagnostic. Do not infer an optimum from five noisy online points, automatically
   run beta1, reopen old studies or force a novelty quota. Tying/MoE need a new
   evidence-based rationale beyond smaller memory or different family names.

For the next locality diagnostic, predict that a trained-range attention window
may prevent harmful extra-key competition while retaining cross-boundary hidden
context. The mechanism is falsifiable and remains an inference. Its control is
full256 conditioning on the identical weights, plus unchanged native128. The
complete bounded cost is three serial retained models, no weight output, original
600sec/RSS4GiB/MPS6GiB/reserve8GiB and512KiB metadata caps. No training seed,
bootstrap, holdout or quality adoption is authorized by this diagnostic. A new
numerical training panel still needs its own complete live high-water budget;
do not apply the completed body's old floor as a future blocker.

## Locality diagnostic completed; priority revised

The new `mla_context_locality_v1` session passed independent literal cropped-window
attention oracles, CPU/MPS native128 identity, full/cache/reset/causality at batches
1/8, and foreign policy/checkpoint/overflow/alias/mask-mutation rejection. The
underlying model's weights and buffers stay unchanged. Both guards passed.
[context-information003](../context-information-003/result.json) then evaluated
only the predeclared local128-at256 condition on all three fixed models and all
34,278 original targets. Mean CE is **4.690095914**: gain **.016968612** versus
recorded global256, but only **.000730849** versus native128. The native128 gains
by seed are **−.000384426, +.000035399, +.002541573**. No model/context is adopted.

This is consistent with harmful distant-key competition in plain extrapolation,
but does not isolate its cause: multilayer information, boundary treatment and
absolute positions also differ. Locality mostly restores the native baseline;
it does not establish useful additional context or justify a training panel yet.
No post-hoc window sweep, sink-token rescue or selective seed reporting follows.
Lower context-training priority until there is a stronger discriminating basis.
The old body/final-scale panels remain terminal and are never rescored/reopened.

The next concrete question is the already grounded MLA-absorption alternative:
first derive the affine query/key/value/output algebra including all biases and
score.5/sqrt64, count the changed work at the actual head32/latent64 geometry,
then use a bounded attention-core oracle/profile on retained weights. Absorption
can increase latent attention dimension and prefill work even while removing
repeated cached K/V expansion; it is not an assumed speedup. Only if correctness
and measured cost support it should a separate full-model inference integration
be considered. This is serving engineering, not a new quality result or numerical
exploration allocation. No new training panel, scientific seed or capacity floor
is admitted. The complete cost of any later training proposal must be forecast
from its current retained outputs, serialization, metadata and reserve.


## Absorption core measured; full integration not justified

[Affine core review](../mla-absorption-core-001/review001-mps.json) completed all
72 declared cases: all four layers of all three retained incumbents, batches1/8,
query1 at prefixes32/128 and prefill128. CPU/MPS maximum absolute differences
are5.07e-7/3.58e-7 within unchanged original tolerances. All biases, rotary terms
and softmax row mass are retained. Original weights/buffers are unchanged.

The absorbed/original median attention-core latency ratios are1.135/1.111/1.153
for batch1 (query1-prefix32/query1-prefix128/prefill128), and1.245/1.347/1.681
for batch8. This implementation is slower in every case. These are short local
core timings with2warmups/5repeats, not full-model serving latency. Candidate
weight-stamp checking is included; precomputation/loading/down projections/
rotary evaluation/cache update/normalization/MLP are excluded. No speedup or
serving policy is adopted; do not integrate the slower core merely because the
large-scale paper uses absorption. Different fusion/hardware could change cost,
but that is not measured here and is not a reason for an arbitrary tuning loop.

At actual latent64/head32, fewer prefix projections trade against larger latent
attention/output operations. MAC counts exclude additions/softmax/copies and are
not end-to-end FLOPs or latency predictions. This negative local engineering
result is distinct from a negative quality result.

Next bounded existing-data question: full train-split fit on all three incumbent
H200s, decomposed by predeclared training-token frequency and within-segment
position, with original validation CE read only as a descriptive reference.
No validation rescoring/holdout/training/seed/tuning/selection, and no inference
that a train–validation gap alone determines overfitting or an optimal horizon.
This closes a missing local measurement before a new evidence-qualified training
design. See train-fit-diagnostic-001/plan.json; follow its actual guard/result,
never duplicate or rerun the bounded command.


## Full training-split fit completed

The [train-fit diagnostic](../train-fit-diagnostic-001/result.json) completed all
267,687 train next-token targets on each retained incumbent H200 at128-token
resets. Mean train CE is **3.612469291**, versus immutable recorded validation CE
**4.690826763**, a descriptive gap **1.078357471**. Seed gaps are
1.069285387/1.073782428/1.092004600. The600sec/RSS4GiB/MPS6GiB/8GiBreserve
guard passed in68.38seconds; no training, new seed, validation rescoring, holdout
or weight output. Train token-frequency strata have mean CE8.071962/6.199967/
4.008183/2.052791 for counts1–5/6–50/51–500/501+, with14686/43443/76170/133388
targets. This correlation is not a rare-token bottleneck intervention result.

The large gap makes generalization/data reuse a stronger question than assuming
insufficient context or adding depth. It still does not prove overfitting, an
optimal horizon or that dropout will help: train/validation text distributions
and frequency mixes differ. Before admitting a new campaign, separate the
measured unigram/frequency-shift contribution using original train/validation
IDs (no model rescoring/holdout), inspect previous dropout/regularization configs
and outcomes, and read applicable primary regularization/data-reuse literature.
A candidate needs a specific mechanism, contrary evidence, falsifiable controls,
equal bounded tuning, fresh confirmation and a new complete feasible budget.
No numerical study is admitted by the gap alone. Continue this concrete evidence
work under the human's improvement authorization; do not force beta1, novelty
or a quota, or claim a worker is running after this diagnostic has finished.
