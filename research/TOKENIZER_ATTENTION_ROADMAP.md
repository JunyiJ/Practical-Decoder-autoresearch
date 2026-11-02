# Attention quality and inference-cost roadmap

Human steering, October26,2026 (Los Angeles): consider adopting GQA and MLA when
quality is better or neutral because inference cost matters; investigate Kimi's
3-KDA/1-MLA hybrid and its tunable parameters. This is the governing direction for
new attention experiments after the current tokenizer LR round. It does not
change the already-frozen LR-only plan or rewrite historical study decisions.

The subsequent human instruction also requires broad architecture research.
This attention work is one part of the [architecture agenda](TOKENIZER_ARCHITECTURE_ROADMAP.md),
which includes MoE, activations, normalization, residuals, depth and readout.
Attention parameter refinements must not indefinitely displace those families.

## Adoption objective

Treat quality and inference cost as separate measured objectives. GQA/MLA need
not beat MHA's CE to become useful serving alternatives. A quality-neutral
candidate with a useful measured cost reduction is eligible for adoption.
Retain workload-specific variants where appropriate, rather than forcing one
architecture to win at every width, batch and context.

For the next finite attention protocol, use a prospective engineering neutrality
margin of +0.01 nats per GPT-2 token (approximately1.005% perplexity) versus a
fresh, equally tuned MHA control. This number is an implementation choice, not
a tolerance explicitly supplied by the human. Freeze it before any new attention
quality results. Require the paired mean, each fresh seed's delta and the upper
bounds of the declared conditional-text intervals to stay within the margin at
the claimed deployment width. No requirement that every seed improve or that the
mean beat the old strict-improvement threshold. A failure to find statistical
significance is not evidence of neutrality; report seed variation separately
because text intervals condition on the trained models.

Cost routes must be declared before timing: memory/capacity, decode throughput,
or latency. Report actual allocated KV/recurrent/convolution state, working
memory, prefill latency, decode latency and achievable batch/throughput separately.
Choose a meaningful minimum benefit and its measurement rule in each finite
workload protocol, above observed timing noise. A memory-saving route does not
require an across-the-board speed win; disclose and assess latency costs for that
workload. Do not equate smaller cache with lower measured latency or cloud price.
Width256-only success may justify a width256 serving variant if that scoped route
was declared prospectively; it is not universal width transfer.

Retain synchronized warmups and alternating paired measurements, fixed prompts,
precision, hardware and generation lengths, repeated timings, and fresh training
seeds. Check cache correctness before timing. Short context128 Mac measurements
cannot establish long-context production savings. Longer-context claims need
models and evaluation explicitly validated at those lengths, including retrieval
or copy diagnostics where useful. Do not change the frozen current model's
context simply to manufacture a favorable benchmark.

## Paper check and version distinctions

Kimi Linear uses repeated KDA/KDA/KDA/MLA groups, with NoPE in MLA. Its KDA uses
short convolution, normalized queries/keys, low-rank decay and output gates,
and headwise output normalization. The paper compares hybrid ratios and ablates
output gating and convolution; 3:1 gives its preferred tradeoff. Its head size
is128, which differs from our local head32 adaptation. These are reported results
under the paper's scale and setup, not measured gains here.
[Primary report, sections4–5.2](https://arxiv.org/html/2510.26692v1#S4).

Kimi K3 keeps the3:1 structure with Gated MLA and an additional final global
layer. It uses NoPE MLA, bounded log-decay for KDA and full-rank output gates.
Its decay example uses a lower bound of−5. These are version-specific choices;
a small dense hybrid is an adaptation, not a full K3 reproduction.
[Primary report, sections2.1.1–2.1.2](https://arxiv.org/html/2607.24653v2#S2.SS1).

## What the current repository actually tests

The closed character-level KDA study used **four KDA layers**, head32, decay
bottleneck D/4, fixed convolution width4, bounded log-decay−5 and a full-rank
sigmoid output gate. Its mean quality regressions and slow training describe
that all-KDA sequential implementation. They do not reject a3:1 hybrid or
predict optimized-kernel inference performance. See [historical KDA results](KDA_QUALITY_RESULTS.md).

The production model currently chooses one global `attn_type` for every block;
its cache factory does the same. A faithful hybrid needs a layerwise attention
schedule and matching per-layer cache/state construction. Current KDA policy
rejects most tuning overrides, and the retained tokenizer model contracts assume
MHA. Add explicit new contracts rather than bypassing those checks. KDA must
remain non-RoPE; NoPE MLA must not silently fall back to learned position tables.

Historical local timing measured GQA cache reduction50% with mixed decode speed,
and MLA cache reduction62.5–81.25% with slower decode in the tested short-context
implementation. Those are useful tradeoffs to revisit under the new objective,
not grounds for assuming either family always reduces latency. Prior quality
and timing panels do not become fresh tokenizer evidence.
[Historical inference measurements](INFERENCE_EFFICIENCY_RESULTS.md).

## Planned sequence and tuning axes

### Implementation preparation checked during the LR round

Read-only inspection of the frozen source found that MHA applies the inherited
extra `attention_logit_scale=0.5`, while `src/attention/gqa.py` and
`src/attention/mla.py` do not consume that field. Their dimensional score
normalization is not this extra multiplier. A new comparison must explicitly
implement and test the declared multiplier in full and cached attention paths,
or declare a separately controlled scaling difference. Changing `attn_type`
alone cannot establish a controlled architecture comparison.

The tokenizer no-output-bias contract calls `readout_norm_contract`, whose
inherited backbone requires MHA and rejects GQA/MLA geometry overrides. The
optimizer already has historical GQA/MLA cases, but those do not authorize the
complete current tokenizer backbone. Add narrowly defined new contracts across
these dependencies, preserving the old validators and checkpoint semantics.
Before training, test actual attention arithmetic, cache parity, parameter
grouping/scaling and rejection of unsupported combinations. This is an
implementation finding, not measured evidence against GQA or MLA; no active
source was changed or additional worker launched for this inspection.

Finish the current paired LR screen and confirmation, then resolve its audited
subword baseline. Prioritize a new inference-aware GQA/MLA comparison and a3:1
hybrid implementation/comparison. Keep the shorter-horizon warning in the
backlog; it is a distinct hypothesis, not a reason to change active trials.

| Family / question | Bounded tuning axes to resolve before a new run |
|---|---|
| GQA | KV-head count/group ratio, equal LR tuning, attention-score scaling; require valid divisibility at every width and report where a ratio collapses to MQA |
| MLA | KV-latent rank, query rank, head geometry, score scale, position choice; compare cached implementation and any absorption optimization separately |
| Hybrid layout | Start with KDA/KDA/KDA/MLA at four layers; compare full MLA and full KDA controls and, in a separate bounded ablation, a1:1 layout; keep total depth/FFN fixed |
| KDA recurrent capacity | Head size and value-state dimension, decay bottleneck rank, parameter/state counts and width-scaling rules |
| Decay and gates | Explicit Linear-style versus K3-style formulation, bounded-decay value and initialization, decay bias, write-gate initialization, low-rank versus full-rank output gate |
| Local mixing / positions | Convolution width (including width4 anchor), NoPE global attention versus a separately specified positional alternative, cache-history length |
| Training | Equal-budget family-specific LR tuning; separate warmup, decay, clipping, horizon and EMA studies when motivated by instability or validation curves |
| Implementation | Sequential reference versus validated chunkwise implementation, chunk size and state precision; measure backend-specific costs without attributing Python-loop overhead to the architecture |

These are proposed local experiment axes, not a claim that the paper tuned every
one or that all are implemented. Use staged finite searches, not an unbounded
Cartesian sweep. Freeze geometry and version semantics first, then tune a small
set with equal budgets and fresh confirmation. Mark intentional departures from
the paper, especially small widths, dense FFNs, head sizes and short contexts.

Before hybrid training: equation/gradient tests, full versus cached/chunked
outputs, causality, nonzero offsets, reset/interleaved requests, per-layer state
shapes, tokenizer/optimizer/checkpoint/resume semantics, and serial MPS resource
pilots. An optimized kernel requires numerical agreement with the reference;
its speed is measured only on hardware/backend that actually supports it.
Preserve all failed attempts and closed studies; keep test loss closed.

The MLA score transfer real worker/parent validation and trained-cache inference
prerequisites now pass for both arms on generated text. No scientific quality
finding follows. Successful-worker lossless archival recovered827.6MB. The actual
writer budgets give a prospective13.556GB complete retained-output requirement
(including8GiB reserve);13.654GB was observed free. Prior metadata was charged
twice; removing that duplicate charge relaxes no guard. Next test exact stage
retirement boundaries, census fresh seeds and freeze the finite pilot plan,
then run guarded pilots before quality freeze. Recheck capacity at dispatch.
See MLA_SCORE_PREPARATION and mla-score-preparation-001/worker003-review.json.

The score-transfer resource stage is now complete: both20-update pilot002 arms
passed independent parent acceptance and paired inference within the unchanged
resource limits. A wrapper namespace failure before training is preserved as
pilot001; its separately frozen correction completed. These pilot losses provide
no quality ranking. Next freeze the finite six-screen/three-fresh-pair quality
comparison in an additive namespace and execute exact eligible pilot retirement
before screening. Retention boundaries and seed freshness passed; keep the test
holdout closed and recheck actual complete capacity after retirement.

The MLA multiplier quality001 plan is frozen and first screening slot00 has
started after74 integration checks, real pilot re-audits and exact successful-pilot
retirement. Six equal-LR screens precede three fresh paired confirmations; no
pilot loss influenced selection. The whole retained panel fits measured storage
with the8GiB reserve. No quality/adoption conclusion yet. Follow the frozen prefix
and immutable source contracts; review architecture and bounded HP alternatives
again only when this finite campaign closes.


## MLA score campaign closed; SwiGLU-product normalization next

Twelve accepted endpoints and three inference pairs completed. Multiplier1.0
improved all three fresh seeds, mean delta-0.014971502424, with all simultaneous
conditional-text upper bounds below0. The frozen-0.02 mean threshold failed;
retain multiplier0.5/SwiGLU672/SiLU/block RMSNorm/H200. Both arms selected
LR0.00125. No grid extension, cost-only adoption or holdout access.

Both architecture and HP options were reviewed at closure. Select prerequisite
work for per-token affine RMSNorm of the SwiGLU product before down projection
versus none (epsilon1e-5, gain1, no bias, proposed2,688 extra parameters). This
is a new tokenizer/MLA internal-FFN normalization question, distinct from closed
block norm and gated-versus-ungated studies. Weak character gains are only a
prior. Defer100-update warmup/peak-LR tuning: historical width256 gains were
small and mixed, without a current identified stability mechanism. Short
convolution and MoE remain distinct later questions with additional cache/resource
requirements. No new scientific panel or seed census is frozen.

Component001 failed in pre-numerical source verification and is preserved closed.
Separately frozen component002 passed12 CPU FP64/FP32 forward/gradient/causality
cases. Next explicit full-model/optimizer/checkpoint/cache/MPS integration, real
worker/parent validation/inference, then fresh census and finite guarded plans.
Forecast complete retained output and reserve, and execute eligible audited
recovery if needed; storage does not block these small prerequisites. See
TOKENIZER_FFN_NORM_PREPARATION.md and ffn-norm-preparation-001/.


The SwiGLU-product normalization full-model and serialized MPS resume prerequisites
now pass both arms. Shared initialization/optimizer groups are identical;2,688
new gains use the existing AdamW vector rule. Repeat/resume model/optimizer/EMA/
RNG differences are0, and H200 CPU/MPS cached inference passes at batches1/8.
These generated checks establish integration only. Next actual worker/independent
parent validation/inference, complete resource projection, fresh seed census and
finite pilot/quality freezes. Exact successful numerical-fixture archival recovered
2,266,198,096B net without losing checkpoint bytes or moving protected scientific
weights. Follow TOKENIZER_FFN_NORM_PREPARATION.md and current next_action.


FFN product-normalization real workers, independent parent full validation and
trained paired inference now pass both arms on generated text. The198740 finite
scientific seeds are reserved and unused. No pilot/quality plan is frozen. Next
minimal finite pilot/retention integration and actual full-panel capacity recovery;
capacity001 requires13,555,990,528B versus observed9,661,427,712B. Source work
continues independently; preserve all protected weights/archives and do not launch
a partial panel. See TOKENIZER_FFN_NORM_PREPARATION.md for exact evidence.


Finite FFN product-normalization pilot orchestration and the complete six-screen/
three-fresh-pair design now pass config, real checkpoint/parent full-validation,
CLI and exact retention/capacity integration checks. No scientific seed consumed
or pilot/quality plan frozen. Lossless archival of224 successful non-selected
worker/numerical weights recovered644,140,089B net; all retained evidence verified.
capacity002 records10,279,190,528B free versus13,555,990,528B required. Source/design
and known eligible recovery are complete; no worker is active. Wait for actual
full-panel capacity or new storage evidence, checking hourly and staying quiet
while unchanged. See TOKENIZER_FFN_NORM_PREPARATION.md and exact later archive
manifests; do not rerun closed prerequisites against intentionally absent paths.


FFN product-normalization resource pilots are complete: both20-step endpoints
passed independent parent checkpoint/full-validation acceptance and paired
full/cache/reset inference. This is resource readiness, not a scientific quality
result. Authorized obsolete numerical archive retirement reclaimed5,443,835,089B;
34,575 retained files verified unchanged. Actual post-pilot free14,676,287,488B
exceeds the complete13,555,990,528B requirement. Next additive finite quality
orchestration and real pilot integration, exact pilot retirement after dependencies
close, then freeze six equal-LR screens and all three fresh confirmation pairs.
See TOKENIZER_FFN_NORM_PREPARATION.md; the storage wait is resolved.


FFN-product normalization quality001 is now frozen and scientific screening has
started.74 integration checks and real pilot payload/full-validation/cache audits
passed; exact successful pilot retirement reclaimed1,118,391,708B with all retained
evidence verified. The finite design remains six equal-LR1,200-step screens then
three fresh pairs, all paired inference and independent final statistics. See
TOKENIZER_FFN_NORM_PREPARATION.md for the exact freeze and capacity accounting.
No adoption or new scientific quality conclusion is claimed at launch.


Local causal FFN mixing resource pilots are complete: both20-update endpoints
passed independent checkpoint/full-validation acceptance, paired cache/reset
inference and final resource review. This establishes readiness only, with no
quality/adoption claim. Exact retirement of obsolete generated prerequisites
recovered1,118,405,952B while preserving all selected/fresh evidence. Next minimal
additive finite1200-step orchestration, real pilot integration and exact pilot
retirement before full-panel capacity/seed recheck and quality freeze. See
TOKENIZER_LOCAL_MIX_PREPARATION.md for census failure preservation, corrected
entry point,521 frozen sources and reserved198970 seed identities.


Local causal FFN mixing quality001 is now frozen and the first1200-step screen
is running.75 finite checks plus real pilot payload/full-validation/cache audits
passed in additive local_mix_quality_v3; two earlier copied config-check failures
remain preserved without changing the tested model. Exact pilot retirement
reclaimed1,118,415,676B with27,372 retained files verified. The plan binds531
sources, six equal-LR screens and all three fresh confirmation pairs, with
strict original quality gates and closed test. No quality or adoption conclusion
at launch. See TOKENIZER_LOCAL_MIX_PREPARATION.md for the exact freeze and routing.


## Local mixing closed without adoption

All twelve endpoints, three paired inference reviews and independent final
statistics are complete. Both arms selected LR0.00125. Fresh none/causal3 CE:

| Seed | None | Causal3 | Candidate minus control |
|---|---:|---:|---:|
|198970011|4.6991971921|4.6884827798|−0.0107144124|
|198970012|4.7075981792|4.7101195406|+0.0025213615|
|198970013|4.7005193711|4.6909489906|−0.0095703805|

Mean delta−0.0059211438 misses the frozen−0.02 threshold; one seed worsens,
and2048/4096-token simultaneous text bounds include zero. All three quality
gates fail. Retain no local mixing with D256 MLA score0.5/SwiGLU672/SiLU/
blockRMS/no product norm/no warmup/H200. The candidate also adds3072 parameters
and8192 cache bytes per item; measured cache is196608 versus204800B. Timing
is descriptive with two resident models, not a deployment or speed win.

Terminal STOP closes training/boundary/inference/final commands. All selected
02/03 and six fresh raw/H200 pairs remain protected. Losing screens resolve
through the exact later retention manifest. Original evidence and all531 frozen
sources are unchanged; holdout remains closed. See local-mix-preparation-001/
quality001-final-review.json and quality001-closure.json for the independent
bootstrap bounds, inference provenance and closure.

At this boundary both parallel-residual topology and independent weight decay
were reviewed. Select the previously deferred0.00006-versus0 decay question at
common peak LR0.00125. Recent repeated LR selection and the user's concern
about repeated screening motivate a prospective standard-recipe comparison
with equal zero LR-search budgets, not a separately optimized best-LR claim.
This does not alter any closed campaign. See TOKENIZER_DECAY_PREPARATION.md.


## Fixed-LR decay comparison closed without adoption

All six fresh endpoints, three passing paired inference reviews and independent
final statistics are complete. Both arms used common peak LR0.00125 with zero
LR-search budgets; no screening or rescue was performed. Fresh H200 CE:

| Seed | Decay0.00006 | Zero decay | Zero minus decay |
|---|---:|---:|---:|
|199080011|4.7141558584|4.7234604882|+0.0093046298|
|199080012|4.7027188086|4.7109193908|+0.0082005822|
|199080013|4.7300105276|4.7400008942|+0.0099903666|

Zero decay worsened all three seeds. Mean delta+0.0091651929 nats/token and all
three frozen gates fail. Conditional95% text intervals are wholly positive at
all declared block sizes;2048/4096/8192 simultaneous upper bounds are
+0.0128724072/+0.0141141413/+0.0158320476. These are conditional reused-text
intervals, not population-seed or holdout confidence. Retain decay0.00006 with
the adopted D256 MLA score0.5/SwiGLU672/SiLU/blockRMS/no product norm/no local
mix/no warmup/H200 recipe. This supports retaining regularization in this fixed
recipe, not a globally optimal decay or independently optimized-LR claim.

All inference correctness/reset checks passed; both arms retain196,608B cache
per item. Timing is descriptive and provides no alternate adoption route.
Terminal STOP closes decay quality training/inference/final commands; all six
fresh raw/H200 pairs and every historical selected/fresh model remain protected.
Original pilot receipts resolve through exact pilot-retirement manifests. No
holdout scoring, closed-fixture rerun or frozen-source edits occurred.

At the boundary both architecture and bounded HP were reviewed. Select parallel
separate-norm residual topology on the subword MLA/blockRMS backbone for
prerequisites only; the negative character-MHA prior remains explicit. Defer
cooldown start600-versus800 as a separate schedule question. Continue prospective
fixed-common-LR comparisons with equal zero search budgets. Read
TOKENIZER_PARALLEL_RESIDUAL_PREPARATION.md; no new seed/training plan is reserved.

## Parallel residual comparison closed without adoption

All six fresh 1,200-update endpoints were independently accepted at common
LR 0.00125 with zero LR-search budgets. Three paired full/cache/reset inference
reviews passed at batches 1/8; both arms use 196,608 bytes of cache per item.
Independent final statistics passed provenance and arithmetic checks.

| Seed | Serial H200 CE | Parallel H200 CE | Parallel minus serial |
|---|---:|---:|---:|
|199190011|4.7176393932|4.7296534829|+0.0120140896|
|199190012|4.7007159542|4.7068419199|+0.0061259657|
|199190013|4.7184464682|4.7267622133|+0.0083157451|

Parallel composition worsened all three seeds. Mean delta is
+0.008818600140528715 nats/token; all three frozen quality gates failed.
Conditional text-bootstrap simultaneous upper bounds at block sizes
2048/4096/8192 are +0.0112559128/+0.0115174044/+0.0117277727. These reused-text
intervals are conditional on the three trained seeds, not population-seed or
holdout confidence. This is a fixed standard-recipe result, not a comparison of
separately optimized optima or a general rejection of parallel architectures.

Retain serial residual composition with D256 MLA score0.5/SwiGLU672/SiLU/blockRMS,
no product norm, no local mixing, no warmup, independent decay0.00006 and H200.
Terminal STOP closes every parallel-residual quality training/boundary/inference/
final command. All six fresh raw/H200 pairs (3,354,992,304 bytes) remain protected,
as do historical selected/fresh models and their lossless RAW archives. All600
frozen sources remain unchanged. No holdout was scored. Closure free space
10,045,964,288 bytes is an observation, not next-panel readiness.

Both architecture and bounded HP were reviewed at the boundary. Select cooldown
start600 versus800 for prerequisites; defer fixed serial residual branch scaling.
The prospective comparison retains common LR0.00125, three fresh pairs, equal
zero LR-search budgets and original strict gates. No new scientific seeds or
pilot/quality plan are reserved. See TOKENIZER_COOLDOWN_PREPARATION.md and
[tokenizer evidence](tokenizer-evidence/parallel-residual-preparation-001/quality001-closure.json).

Cooldown resource preparation has advanced to two independently accepted20-update
pilots, passing paired inference and independent resource review. This is not a
scientific quality result. The next finite comparison remains exactly three fresh
pairs at common peakLR0.00125 with zero LR screening; cooldown600/800 also changes
total LR exposure. Quality001 is now frozen with623 sources; exact pilot retirement
passed and the first fresh trajectory is active. There is no quality result yet. All selected/fresh bytes remain protected, including the lossless
storage-recovery-007 RAW archives. See TOKENIZER_COOLDOWN_PREPARATION.md for the
current source, capacity, seed and next-action evidence.

## Cooldown onset comparison closed without adoption

All six fresh1,200-update endpoints were independently accepted at common peak
LR0.00125 with zero LR-search budgets. All three paired full/cache/reset inference
reviews passed at batches1/8,196,608B cache per item. Independent final statistics
verified both provenance and a separate arithmetic/bootstrap implementation.

| Seed | Cooldown800 H200 CE | Cooldown600 H200 CE | 600 minus800 |
|---|---:|---:|---:|
|199300011|4.7127084386|4.7222150878|+0.0095066492|
|199300012|4.7216895012|4.7284279587|+0.0067384574|
|199300013|4.7107724355|4.7180821944|+0.0073097588|

Earlier cooldown worsened all three seeds. Mean delta is
+0.007851621813797643 nats/token; all three frozen gates failed.
Simultaneous conditional text-bootstrap upper bounds at blocks2048/4096/8192 are
+0.0107926732/+0.0113509813/+0.0114523509. These reused-text intervals condition on
the three trained seeds, not population-seed uncertainty or an independent holdout.
No test split was scored. This is a fixed standard-recipe comparison, not separately
optimized optima. Earlier cooldown also lowers multiplier sum1020 to930; it does
not isolate schedule shape at equal LR exposure. No rescue or grid extension.

Retain cooldown800 with D256 MLA score0.5/SwiGLU672/SiLU/blockRMS/serial residual,
no product norm/local mix/warmup, independent decay0.00006 and H200. Terminal STOP
closes all cooldown quality training/inference/final commands. All six fresh
raw/H200 pairs(3,354,990,192B), all historical selected/fresh models,
and protected storage004/006/007 archives remain intact. All623 sources unchanged.
Closure free9,726,046,208B is an observation, not next-panel readiness.

Both architecture and bounded HP were reviewed. Select fixed0.5 versus unit scaling
of both complete serial attention/FFN branches; defer global gradient clipping0.5
versus incumbent1.0. Negative character0.1-gain results are an adverse prior; no
gain is assumed, old scores/seeds are not pooled and closed campaigns stay closed.
The generated standalone branch-scaling component passed12 CPU FP64/FP32 and MPS
FP32 forward/all-gradient/causality cases, exact unit reference and identical shared
initialization/RNG. This is not full-model integration or a quality result. See
TOKENIZER_BRANCH_SCALE_PREPARATION.md and cooldown quality001-final-review.json.


## Branch-scale quality001 closed; retain unit gains

All six fresh endpoints and three paired inference reviews completed. Independent
fixed-half minus unit mean validation CE delta+0.024273102061827024 nat/token;
seed deltas+0.0287286911826099,+0.024305901472923885,+0.019784713529946707.
Every seed worsened and all original gates failed. Terminal STOP closes all branch
quality commands. Preserve all six fresh raw/H200 pairs(3,354,994,032B), historical
selected/fresh models and protected archives; all659 sources remain unchanged.
Retain the incumbent unit-gain recipe. No LR screens/rescue or old score/seed pooling.

Both architecture and bounded HP reviewed: select global norm clipping0.5-versus1.0;
defer matched-epsilon final-readout LayerNorm versus final RMSNorm with pre-branch
norms unchanged. No benefit or actual clipping frequency assumed. Frozen generated
optimizer-component001 passed four CPU FP64/FP32 real Muon/AdamW clipping/order/
serialized-resume cases; no full-model/MPS/EMA/H200/cache or quality claim. Next
narrow additive full-model clipping instrumentation and identity integration,
then real worker/validation/inference, full capacity/census/resource pilots before
any quality freeze. See TOKENIZER_GRADIENT_CLIP_PREPARATION.md.


## Clipping closed; corrected final-readout centering question

Clipping0.5 worsened all three fresh seeds: mean+0.004133149591824823 nats/token,
all original gates failed; all six endpoints and three inference pairs passed.
Retain clipping1.0. Quality002 and original publication-failed quality001 stay
closed, with every fresh raw/H200 protected, including imported slot00.

Both architecture and bounded HP reviewed. Actual accepted H200/raw inspection
corrected earlier deferred wording: the final readout already uses affine LayerNorm
eps1e-5, not RMSNorm. Select centering removal only: compare incumbent LayerNorm
with affine RMS plus bias, both512 parameters/eps1e-5; keep pre-branch RMSNorms fixed.
No benefit assumed; prior block LayerNorm failures are adverse evidence at a distinct
placement. Defer AdamW beta2 .95-versus-.999 with beta1 .9/Muon/decay unchanged;
44 accepted AdamW groups confirm current betas(.9,.999). No seed or score pooling.

Standalone component002 passed24 CPU FP64/FP32 and MPS FP32 cases with forward/all-
gradient/finite-difference/token-independence checks. Original component001 FP32
incumbent-oracle failure stays closed; separate002 adds exact native LayerNorm
forward/all-gradient parity and a documented normwise FP32 control oracle. Candidate
elementwise tolerances and all resource guards unchanged. No candidate full-model
integration, scientific seed or pilot/quality freeze. Next explicit policy/grouping/
identity/raw-EMA-RNG resume/H200/cache integration outside695 frozen sources; never
bypass the existing final-LayerNorm/MLA guards. Then worker/validation/inference,
fresh census, measured full high-water plus8GiBreserve and finite pilots before
exactly three fresh quality pairs at fixedLR0.00125/equal zero LR-search/strict gates.
See TOKENIZER_READOUT_CENTERING_PREPARATION.md.


## Readout centering full-model prerequisites complete

Both layer_affine/rms_affine_bias arms passed explicit policy/group/checkpoint/cache
identity, CPU/MPS gradients and full/cache/reset/causality. Shared matrices/RNG/groups
match exactly; incumbent LayerNorm logits are exact, candidate function differs.
Generated four-update repeat/resume raw-model/optimizer/EMA/RNG/clipping history
matched exactly; final-step4 H2003snapshots passed CPU/MPS batches1/8. All710
bindings unchanged. No scientific result/seed reservation or active worker.
Next additive actual worker/output/independent validation/inference with explicit
readout-policy routing, then closure of resume tensor dependencies, exact retirement
audit/capacity recheck. The4 generated weights1,118,317,844B remain retained.
Storage012 saved654,730,043B losslessly from6 closed branch RAWs,50,525 survivors
verified and exact restoration PASS;120 historical RAWs now remain recoverable
in protected archives, NOT retired. All H200 originals intact. Full scientific
panel capacity and census remain separate. Retain LayerNorm/clip1.0 and adopted
recipe; AdamW beta2 .95-versus-.999 deferred. See TOKENIZER_READOUT_CENTERING_PREPARATION.md.
