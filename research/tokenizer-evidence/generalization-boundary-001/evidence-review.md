# Generalization boundary: evidence and the next isolated hypothesis

The evidence-grounded-research skill governs this decision. The incumbent,
closed holdout and every closed study/STOP remain unchanged. This note admits
correctness/resource preparation for one input-only feature-dropout hypothesis,
not a numerical quality freeze or an adoption. Classify it as local regularization
(exploitation); do not relabel it structural exploration or launch to meet a quota.

## Measured local basis and limits

The three retained incumbent H200s have full train-split mean CE 3.612469291,
versus immutable original validation CE 4.690826763. The gap is 1.078357471,
but differing texts and adaptive validation reuse prevent calling this proof of
overfitting. [Original fit](../train-fit-diagnostic-001/result.json).
[frequency001](frequency001.json) uses only the original authorized token IDs,
no model/checkpoint loading or scoring. Train-estimated Jeffreys/Laplace unigram
gaps are .172581004/.152760591. Their JS divergence is .102463883 nats.
Validation has 957 targets (2.791878%) across 581 IDs absent from training.
Known-token coarse-bin reweighting changes train CE by .0154–.0157; it assumes
unchanged within-bin losses and excludes unknown-token mass. Neither control is
an additive/causal explanation of the model gap. They rule out treating a coarse
frequency mix as a complete explanation by assumption, not all distribution shift.

The vocabulary tables contain 90.31% of parameters, but this is not a FLOP fraction
or proof that the embeddings memorize. Context extrapolation/global256 worsened
fixed-weight CE; local128-at256 mostly restored baseline with inconsistent tiny
gain. Body reallocation/depth failed original quality gates. Table sharing and
coupled horizon changes also lost under their original conditions. None is
reopened, rescored or pooled into a future fresh panel.

Importantly, [old-dropout-evidence001](old-dropout-evidence001.json) binds the
actual original evidence and suite specs for character/MHA/Muon iterations
3/10/14/17, all 18 completed trials each, holdout unconsumed. Attention-probability
rates .05/.01 failed width-64 selection; width256 means were −.002658/−.000481,
not robust practical gains. Residual-output dropout .05/.01 worsened width256
by +.005575/+.004705; smaller widths also lost. The residual experiments dropped
attention and MLP outputs, explicitly leaving attention probabilities undropped.
They did not add a lookup-output feature mask: the original/current GPT path
reads the token embedding directly into the residual stack. These negatives
make another residual/attention dropout grid low priority. Earlier independent
zero/double decay also failed; no automatic decay or horizon sweep follows.
Scope differs: character MHA/3000 updates versus current GPT2/MLA/1200,
untied vocabulary-dominated model. Changed scope alone is not a gain claim.

## Primary literature actually read

- **Srivastava et al., JMLR15(56),2014, §4–5.1.** Bernoulli masking perturbs
  training features; deterministic evaluation approximates the ensemble.
  Optimization interacts with regularization, and their successful settings
  often include max-norm. Different networks/tasks/optimizers prevent copying
  their gains or rates. Our inverted scaling leaves eval weights unchanged.
  [Paper](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf).
- **Vaswani et al., arXiv1706.03762v7, §5.4 and table3.** The translation model
  applies .1 dropout to embeddings/positions and residual outputs together.
  This supplies a rate/site anchor, not an isolated embedding-only ablation or
  a local decoder-MLA gain. Its label smoothing explicitly hurts perplexity;
  do not substitute a BLEU benefit for our CE objective.
  [Paper](https://arxiv.org/html/1706.03762v7).
- **Merity et al., arXiv1708.02182v1, §4.2–4.3, §5, §8/table4.** The word-level
  LSTM uses input/embedding regularization with other methods; removing embedding
  dropout harms its ablation. Its word-vector dropout shares a mask by word type,
  unlike the proposed independent token/channel mask. Tying, NT-ASGD, long
  training and recurrent dynamics are major gaps; none is transplanted here.
  [Paper](https://arxiv.org/html/1708.02182v1).
- **Liu/Bauer/Manning, arXiv2505.24788v1, §3.1/4/6.** Dropout on attention/MLP
  blocks, including early dropout, hurts their single-epoch BERT/Pythia training.
  This is contrary evidence. Our tiny reused stream/input-only site differs,
  but the paper does not establish a universal repetition threshold or promise
  a gain here. No early-dropout rescue is planned.
  [Paper](https://arxiv.org/html/2505.24788v1).
- **Muennighoff et al., arXiv2305.16264v5, appendixQ/S.** They fix dropout .1
  and explicitly leave hyperparameter/repetition interactions unresolved.
  Their setup does not validate a local regularization optimum. Our 4.5904
  sampled-target exposure ratio is not literal shuffled epochs.
  [Paper](https://arxiv.org/html/2305.16264v5).

## Mechanism, falsifiable test and novel information

Test **one** new site: inverted Bernoulli feature dropout with probability .1 on
the input embedding lookup output, independently per batch/token/channel, before
the first residual block. Control .0 uses the identical new implementation.
All attention probabilities, attention/MLP residual outputs, normalized products,
readout, scale/bias, parameters, initialization, optimizer routing, schedule and
data/target budget remain as incumbent. Eval has no mask and exactly the original
equations/cache size. No mask changes token IDs, deletes targets, ties tables,
adds label smoothing or reduces the vocabulary. Rate .1 is a single literature
anchor, not an inferred optimum or a searched random perturbation.

Inference: training feature corruption may discourage reliance on particular
lexical coordinates under repeated tiny-corpus exposure without repeatedly
perturbing the residual branches that lost earlier. Predict lower fresh H200
validation CE at fixed target exposure; higher training CE is an optional
mechanistic diagnostic, not success. If gains fail practical/all-seed/conditional
bounds, retain zero input dropout. Either outcome distinguishes a new location
from the prior residual/attention losses and narrows a generalization hypothesis.
It does not prove the train–validation gap is caused by co-adaptation.

A future finite design, conditional on correctness/resource readiness, allows
three rates (.000625/.00125/.0025) equally per arm at a new shared screen seed,
then exactly three fresh paired seeds and simultaneous conditional-text bounds
at 2048/4096/8192 blocks. Candidate must have mean delta <−.02, every seed
negative and all three simultaneous upper bounds negative; no partial panel,
rescue/site/rate expansion/seed substitution/cost-only adoption. Those are
prospective design requirements, not a freeze or assignment of seeds today.
No old score is pooled. All selected recipes and fresh endpoints are protected.

Next execute isolated CPU/MPS equations/gradients, exact .0 incumbent
init/RNG/logits/two optimizer steps, own-policy checkpoint/cache rejection and
full/cache/reset/causality. Then exact stochastic fresh-repeat/serialized
optimizer/EMA/sampler/CPU-MPS RNG resume, real finite workers and resource pilots
before freezing any scientific panel. Minimal additive sources only; the 939
old scientific and six diagnostic sources remain immutable. Complete high-water
forecast must include every screen/selected/fresh output, serialization, metadata,
8GiB reserve and measured external/entry headroom, then cleanup target +2GiB.
No scientific capacity pass, numerical quality result or new seed is claimed yet.

## Actual preparation outcomes and budget correction

The new input-only model passed both CPU/MPS finite proofs (947 preserved+new
bindings). The zero control matches original initialization/RNG/logits/gradients
and two optimizer steps exactly; candidate mask equations/gradients and native
RNG repeat pass, as do cache/reset/causality and foreign policy rejection.
The 950-source transport then passed both four-update real-data fresh repeats
and serialized2+2 resumes exactly, including all state bytes/scalars/RNG/samplers/
EMA/clipping/raw/H200. No tensor files or quality seeds were created.

A separate zero-update metadata probe demonstrated that the inherited validator
accepted a too-short MPS RNG state, without calling an invalid resume. An additive
952-source preflight now rejects ten RNG/sampler/copied-config forgeries before
any destination/global-RNG mutation. Original valid numerical transport/proofs
remain immutable. The new 958-source worker contract/source review passed 18
configs and a real B8/T128 forward versus parent mask-only RNG advancement.
Independent full worker artifacts, full CPU/MPS CE and20-update pilots still await
execution. See [preparation](../../TOKENIZER_EMBEDDING_INPUT_DROPOUT_PREPARATION.md).

[capacity001](capacity001.json) distinguishes the next finite worker-pair wake
11,051,937,792B from the conservative full all-output bound19,590,082,560B.
The latter counts all16future original560MiB slot caps; no planned retirement
is credited. The earlier rough estimate assumed losing-screen stage retirement
before a new two-arm/root contract was tested. Existing code is tied to the old
three-arm/root, so it cannot be reused by bypassing guards. Refine the bound only
with a tested new contract/actual closure and live measurements; no quality
panel is frozen. The smaller worker-pair gate must not be turned into a ban on
independent numerical prerequisites or a wait for the full future-study bound.

Inline recovery001 and separate003 recovered new compiler cache bytes;
original002 stopped after one authorized removal because it queried that missing
path on the next consumer check. Its source/manifest/journal/exception stay closed
and unchanged;003 revalidated only the three remaining exact files. Storage039
losslessly compressed36newclosed metadata files and reverified48protected payload
hashes. Total exact allocated recovery29,896,704B is not the free-space delta.
Large-file-by-size inventory includes .tmp paths and found no large incomplete
write. Protected selected/fresh/history/storage004 are not deletion candidates.
Known external/app/research classes are exhausted for this blocker; no apps or
workers stopped, no permissions bypassed. After source integration is complete,
check the precise worker-pair wake once hourly, silently when unchanged. New
eligible proof/access/dependency/requirement/storage triggers inline recovery.
