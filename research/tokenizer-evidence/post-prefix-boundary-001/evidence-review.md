# After vocabulary capacity: next feasibility question

The native/8K study is CLOSED_NOT_ADOPTED. Read its original final review,
closure002 and existing-analysis002 in ../tokenizer-capacity-boundary-001.
All 12 endpoints and independent statistics passed; all three fresh 8K deltas
were positive, mean +.023745683009 BPB. Keep the incumbent. This is a valid
negative quality result, not a resource failure. All sources and weights stay
protected. No numerical worker or new scientific seed/panel is active.

## What the collected measurements establish

8K was cheaper but sampled about 15.1% fewer raw target bytes at equal token
updates. The mean quality deficit is positive across 26 of 28 4096-byte blocks.
Five sparse online points and final RAW/H200 differences do not resolve whether
segmentation, normalization, lexical capacity, shorter raw context, exposure,
EMA lag or their interaction dominates. Do not infer a specific cause or launch
an automatic 16K/context/longer-horizon/EMA rescue. Prior global256 context and
body-depth allocation comparisons also failed their declared quality gates.
Input-feature dropout .1 lost on all three seeds; no new dropout grid is justified.

The old MoE feasibility was deferred for capacity, not because a tokenizer MoE
quality experiment failed. Read ../../TOKENIZER_MOE_PREPARATION.md and original
moe-preparation-001/direction-decision.json without rerunning its closed tools.
Current free space near 96 GB removes that historical disk obstacle; the old
projection is not a new validated budget. The historical character study in
../../MOE_QUALITY_RESULTS.md is contrary evidence: top2-of4 beat dense at width64,
but lost by +.002608 CE at width256 and by +.007630 versus wider dense there.
Wider dense explained most narrow-model improvement. Its last-batch router
histogram was not a whole-run exposure audit. Those scores cannot be pooled with
new GPT2-tokenized MLA results.

## Primary paper actually reviewed at this boundary

[ST-MoE, arXiv:2202.08906v2](https://arxiv.org/html/2202.08906v2), sections 3.3
(equations 5-6, table 4), 5.1-5.2 (table 8), 7.2 (table 14) and 9 were retrieved
and read. The authors report router z-loss stabilization, hardware-dependent
routing tradeoffs and small top-two gains at their larger training scale.
They find little meaningful decoder expert specialization and warn that routing
improvements can reverse with scale or horizon. Their encoder-decoder,
span-corruption, Adafactor, TPU and very large data/model setup differs greatly
from our tiny decoder-only FP32/MPS/Muon-AdamW experiment. These results neither
predict a local gain nor justify adding z-loss without local stability evidence.

## Candidate review and next executable work

Select **a bounded MoE component feasibility test**, not a quality campaign.
The unresolved question is whether conditional FFN capacity is implementable
and auditable on the current native-tokenizer MLA recipe while distinguishing
it from simply adding dense FFN capacity. Tokenization/backbone/optimizer changes
since the contrary character result make this an untested transfer question;
they do not establish a benefit. The new 8K negative result is not evidence that
more FFN parameters will help.

Use the prior explicit anchor: four SwiGLU experts of width336, top two,
softmax router with renormalized selected weights, no shared experts, capacity
drop, noise or stochastic routing. Keep the native codec and fixed body.
The prospective controls remain dense672 (active FFN matrices matched) and
dense1344 (stored expert matrices matched), with router overhead disclosed.
A future quality prediction would be MoE improving independently tuned dense
controls on fresh seeds; failure against either control falsifies useful sparse
capacity under that finite recipe. A wider-dense win alone would favor ordinary
capacity instead of routing. Balanced assignments alone would not prove semantic
specialization or quality.

Next implement an additive component namespace outside all 1,004 frozen sources:
an explicit routed SwiGLU and independent all-expert dense-sum oracle. Check
forward/input/router/expert gradients, auxiliary-loss separation, exact tie
policy, zero-route experts, full token coverage and cumulative assignment counts.
Begin with generated CPU FP64 cases under 600s/RSS4GiB/8GiB reserve and four
threads, bounded 16MiB metadata, no model weights, corpus scoring, holdout or
scientific seeds. MPS/full-model/optimizer/transport/cache acceptance follows
only after this component passes. Never bypass existing dense/old-MoE guards.

No full model or scientific budget is admitted here. Exact complete serialization,
all generated workers/pilots/screens/fresh outputs, resource pilots, equally
bounded LR selection, simultaneous two-contrast quality gates and fresh-seed
census must precede a new freeze. The old draft nine-screen/nine-fresh proposal
is a cost/design starting point, not execution authority or a measured forecast.
No automatic new router-loss/expert-count/placement grid. Preparation does not
count as numerical quality exploration; proposed 70/20/10 remains proposed.

## Actual component outcome

Additive tokenizer_moe_component_v1 is now implemented. CPU component002 passed
six generated FP64 cases, including the proposed D256/H336 shape, ordinary and
exact-tie/zero-route cases. Maximum value/gradient discrepancy was
1.1102230246251565e-15. Forward/input/router/all-expert gradients, explicit
auxiliary coefficients0/.01, tie policy, two assignments per token, dormant
experts, cumulative counters and no-record behavior agreed with an independent
all-expert dense-sum oracle. The global CPU RNG and all1,004 predecessors were
unchanged. No corpus/model scoring, training or weight files. Guard passed.

Original component001 stopped on an oracle error: integer tensor division created
FP32 route fractions before FP64 aggregation. Its source/failure/guard remain
closed. Separate component002 corrects only reference arithmetic; the candidate
component is unchanged. This is not a negative quality result or an original
campaign retry. Both proof sources are immutable. The implementation namespace
has2files;2proof files plus1,004predecessors give1,008 numerical source bindings
for the next integration,which must verify their actual hashes.

Next executable action: additive MPS FP32 component versus independent CPU oracle,
including deterministic tie/zero-route behavior and device-local counter state,
under original guards and one-worker lock. Then own full-model/optimizer policy,
CPU/MPS/cache/transport acceptance and real workers/pilots before any scientific
freeze. Do not treat this CPU proof as MPS/full-model/resource/quality readiness.
