# Boundary after the functional anchor test

## Decision from the completed test

projection-quality-preparation-001/anchored-norm001/result.json and its original
four finished guards/STOP support66comparisons:33anchored forward/analytic-gradient
gates,33descriptive originalLN controls and24transactional rejections. All fixed
anchor predicates passed on CPU64/CPU32/MPS32; inputs/gradslots/globalRNG and1175
source bindings stayed unchanged. Whole.490121s/RSS354107392B/MPS11370496B, under
the120sCPU/600sMPS/4GiB/6GiB/reserve8GiB/4MiBmetadata limits. No models, optimizer,
checkpoints, corpus, random-role draws or training. All commands are now closed.

Model integration is REJECTED under the prior worse-error rule. On the SAME
narrow_gain8_low MPS case, anchor input-gradient maxerror was.000364302458053
versus original.0000427267185046; both pass the fixed elementwise tolerance.
MPSoriginal already had exact shift-pair forward/gradient equality. CPU32
translation accuracy improves, but this does not establish a useful common
kernel upgrade. Do not change the rule, split CPU/MPS implementations, tune the
anchor, or launch full-model tests to rescue it. No source is edited or replayed.
It does not explain the original failed QUALITY001 checkpoint or complete old
forward-sensitivity001/002. Their missing results stay missing.

## Local candidates reviewed

1. Anchor normalization: reject model integration for the measured contrary
   result above. A correctness pass is not a reason to demand a training panel.
2. Table sharing: reject immediate replay. TOKENIZER_TABLE_RESULTS.md records
   scaled-tied minus legacy +.133175nat/token and minus matched +.030508, every
   fresh seed worse. Matched initialization/scaling already hurts. Memory alone
   is not a new mechanism. Earlier character results are separate priors.
3. Local convolution, depth, gate, MoE and old readout mixtures: retain their
   closed outcomes. Local-mix original quality001-final-review/closure and
   architecture roadmap record mean-.00592114/oneworse, failing frozen gates.
   No new causal observation admits a replay or a random quota experiment.
4. A different FFN factorization/nonlinearity question merits SOURCE/tiny-component
   preparation only, from the exact paper distinction below. This is LOCAL FFN
   architecture followup; carry broad/interaction allocations forward.

Original ffn-structure-preparation-001/quality001-final-review.json was inspected:
ungatedSiLU1008 minus SwiGLU672 mean+.007388630366807889, two seeds worse, closed
not adopted. activation-preparation-001/inference-recovery002-closed-review.json
was inspected: GELU-gated minus SiLU-gated mean+.001550755925793319, two worse,
closed not adopted. Native and closed holdout stay unchanged. These are contrary
controls, not proof that every ungated function fails. Never pool their scores
or reuse their scientific roles in a new comparison.

## Primary literature actually read

[Primer, arXiv2109.08668v1](https://arxiv.org/html/2109.08668v1): section3 Squared
ReLU and Figure6 discussion; section4 comparator definition; AppendixA.7 insertion/
ablation discussion and A.8 training setup. Its two-matrix squared-ReLU FFN is
compared with GLU variants. ReLU-squared is equivalent to ReGLU with shared input
projections, whereas an independently gated squared-ReLU would be another function.
The paper also studies convolution; neither its combined gains nor reported
compute savings can be assigned to a local activation swap. Adafactor, long runs,
other corpora/tokenizers and attention differ materially from local Muon/AdamW,
small Shakespeare BPE and1200-step recipes. This supplies a reason to investigate,
not a local gain or a model-admission claim.

[GLU Variants, arXiv2002.05202v1](https://arxiv.org/html/2002.05202v1): section2,
equations4-6, section3.1 and Table1. Three-matrix GLUs use two-thirds the hidden
width of two-matrix FFNs for parameter/operation accounting. GEGLU/SwiGLU are
strong comparators in its T5 span-filling setup. Its task/optimizer differ from
local autoregressive training; it does not establish a local ordering.

## Concrete algebra and falsifiable future question

Current source ffn_structure_v1/runtime/models/mlp.py computes
W_down[SiLU(W_gate x) * W_up x], three bias-free matrices, hidden672 at D256.
The closed ungated control is W_down SiLU(W_up x), hidden1008.
Proposed NEW candidate is W_down [max(W_up x,0)^2], hidden1008, NO independent
gate matrix, added normalization, convolution, scale constant or router.
Merely squaring a SwiGLU gate and retaining its independent multiplicative branch
would be cubic in large positive scale and is NOT the Primer comparison.

The candidate preserves quadratic positive-branch self-interaction while removing
independent cross-feature gating. It differs from the closed ungatedSiLU's
asymptotically linear branch. Same matrix-budget counts are
3*256*672 = 2*256*1008 =516096 per layer; width/tensor factorization and pointwise
work still differ. This is not equal runtime or isolated nonlinearity causation.
Neither optimal scale nor initialization is identified by this derivation.

Falsifiable eventual quality question, IF all later gates qualify: does the
squared-ReLU two-matrix recipe beat BOTH fresh SwiGLU and ungatedSiLU controls
under equally bounded tuning and complete fresh confirmation? A win only against
ungatedSiLU is insufficient. A loss would narrow this recipe; neither outcome
overturns closed studies or proves an isolated gating mechanism. No selection,
threshold, scientific roles, full-model budget or training launch is frozen now.

## Next executable work and complete admitted scope

Prepare the smallest NEW tiny functional component plus INDEPENDENTscalar forward/
analyticbackward oracle: three equations above, D<=8/N<=16, hidden gated8 versus
ungated12 at D8 (192 matrix parameters each), no bias/dropout/rescaling. New literal
inputs only; no old FFN/component/normalization fixtures. Squared-ReLU derivative
is2*max(u,0), including zero at u=0. Source/predicates/invalid cases/resources/exact
plan SHA must be reviewed before its first execution. No model constructors,
optimizer, corpus, checkpoint, RNG draw or scientific role is admitted.

Working next tiny scope<10s/<512MiBRSS/<32MiBMPS; strict120sCPU/600sMPS/4GiB/6GiB/
8GiBreserve/2MiBmetadata/fourthreads/fallback0/fastmath0. No weights or serialization
overlap. Floor8592031744B/extra2GiBtarget10739515392B fits current~42.27GBfree;
no storage block/deletion or old-output credit. These are a tiny-scope forecast,
not a full-model/quality budget. Actual D256/V50257 integration/complete state/
serving/scoring/4-20pilots/fullquality retained-output/time remain unqualified.
