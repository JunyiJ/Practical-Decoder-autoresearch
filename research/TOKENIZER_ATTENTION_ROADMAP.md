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


## Readout-centering real worker and paired inference complete

Additive research/readout_centering_worker_v1 explicitly routes the readout policy
through actual run_training, data loading, bounded raw/H200 output and independent
parent full validation. Both generated four-update layer/rms_bias arms passed:
4,096 training targets,3 H200 snapshots and88 EMA tensors each. Parent validation
was exact; CPU/MPS CE differences were2.733889967743153e-7 and2.62410942752922e-7.
Complete clipping histories agree across raw/summary/metrics. Integration001 also
rejected missing/sequence/threshold/coefficient/nonfinite history forgeries and
foreign-readout checkpoints. Paired full/cache/reset inference passed batches1/8;
cache remains196,608B per item in both arms. All717 runtime/acceptance bindings
unchanged,710 predecessors preserved. This is generated-fixture engineering
evidence, not a scientific quality result or seed/pilot/quality reservation.
The prerequisite plan's inherited kind string says generated_gradient_clip_worker_parent_prerequisite;
its explicit readout configs, source hashes, requests and accepted artifacts bind
the actual readout comparison. This descriptive label does not change execution.

Resume-retirement001 independently audited and retired exactly four obsolete
generated resume weights,1,118,317,844B, after real payload/cache integration closed
their dependencies.41,499 retained files verified unchanged; original sources,
configs, logs and receipts remain immutable. Absent weights resolve through the
later retirement manifest; never restore or rerun them. Actual post-recovery
free10,253,819,904B exceeded complete worker-pair9,814,671,360B including8GiBreserve.
Both fresh generated workers then completed. Their new raw/H200 weights total
1,118,323,284B and remain retained until pilot integration closes dependencies
and a new exact audit. Current free space is not scientific-panel readiness.

Next: minimal additive finite20-update resource-pilot orchestration and complete
THREE-fresh-pair fixed-LR design outside717 frozen sources. Re-audit actual worker
payload/cache, test finite contracts, close dependencies, fresh scientific census,
measure complete retained-output high-water including serialization overlap,
metadata and8GiBreserve, and take exact authorized recovery if necessary before
finite pilots and final quality freeze. CommonLR0.00125/equal zero LR-search,
original strict gates/no rescue/selection/partial panel. Retain final LayerNorm
and the adopted recipe; AdamW beta2 .95-versus-.999 remains deferred.
Preserve120 recoverable historical RAW archives, all H200 originals, storage004
and every selected/fresh byte. All earlier campaigns/fixtures remain closed.

Evidence: tokenizer-evidence/readout-centering-preparation-001/trainer-worker001.json,
trained-inference001.json, worker-integration001.json and resume-retirement001/completed.json.


## Readout-centering resource pilots complete

Both20-update layer/rms_bias arms passed independent acceptance, full/cache/reset
inference batches1/8 and independent final003 resource review. All729 source
bindings unchanged; no quality result or active worker. Exactly three fresh pairs
at commonLR0.00125/equal zero LR-search/original strict gates remain prospective.
Next additive quality orchestration, actual pilot re-audit/dependency closure,
exact retirement and capacity/unused-seed recheck before freeze. AdamW beta2 remains
deferred; retain final LayerNorm and adopted recipe. Import-only integration001
failure is closed; separate additive v2/integration002 passed. Worker retirement
and lossless storage013 resolved capacity;147 historical RAWs remain recoverable,
NOT retired, with all H200 originals intact. Details and exact evidence are in
[TOKENIZER_READOUT_CENTERING_PREPARATION.md](TOKENIZER_READOUT_CENTERING_PREPARATION.md).


## Readout-centering quality001 frozen and launched

All737 sources frozen,729 predecessors unchanged;24 finite-contract/publication
tests and actual retained pilot tensor/validation/cache audits passed. Separate
exact pilot retirement reclaimed1,118,335,248B with28,586 survivors verified.
Freeze free12,691,947,520B exceeds complete12,381,585,408B including8GiBreserve.
First fresh slot00 LayerNorm is running under exact freeze`409de49f511b3d94733ff788ef0aec18a26291843b6957e0ed95e539bec9546c`.
No scientific result. Complete exactly three paired1200-update comparisons at common
LR0.00125 with zero LR-search, then all three inference reviews and independent
statistics. Candidate is affine RMS WITH bias, sameepsilon1e-5/512parameters;
eight pre-branch RMSNorms fixed. Retain incumbent LayerNorm and adopted recipe pending
the original strict gates. AdamW beta2 remains deferred. All fresh pairs and147
losslessly archived RAWs stay protected. Current live state and exact next action
are in [TOKENIZER_READOUT_CENTERING_PREPARATION.md](TOKENIZER_READOUT_CENTERING_PREPARATION.md)
and the resume index; current processes/anchors supersede launch snapshots.


Readout quality001 now has two accepted endpoints and slot02 running. A pre-dispatch
capacity shortfall was resolved with storage014: six closed clipping fresh RAWs
preserved losslessly,654,173,784B reclaimed,51,335 survivors unchanged and exact
restoration PASS. No scientific retry or guard change. All153 historical RAW archives,
H200 originals and current readout pairs remain protected. Current next action and
resolver details are in TOKENIZER_READOUT_CENTERING_PREPARATION.md and resume state.


## Readout-centering quality001 closed, not adopted

All six1200-update endpoints were independently accepted, including1,228,800targets,
41H200snapshots,88EMA tensors and complete clipping history per endpoint. All three
paired full/cache/reset inference reviews passed. Independent statistics reproduced
the frozen reduction, bootstrap and gates; all737 sources remained unchanged.

| Fresh seed | LayerNorm H200 CE | RMS with bias H200 CE | RMS minus LayerNorm |
| --- | --- | --- | --- |
|199630011|4.723623907453187|4.722338028546402|-0.0012858789067847274|
|199630012|4.732624464205026|4.732265809794470|-0.00035865441055628946|
|199630013|4.705603105991945|4.706875608092376|+0.0012725021004309565|

Mean delta is-0.00012401040563668678 nats/subword token. Two seeds improve slightly;
one worsens. Mean<-0.02, every-seed improvement and simultaneous conditional-text
upper-bound gates all fail. Upper bounds for2048/4096/8192-token blocks are
+0.0008101435587669836/+0.0008345895539876811/+0.0009522751174451204.
Intervals concern reused validation text, not population seeds or held-out test.
This fixed-common-LR/zero-search comparison does not establish separately tuned
optima. No RMS-with-bias adoption, cost-only route, rescue or partial-panel decision.

Terminal STOP closes all quality training/inference/final commands. Retain final
affine LayerNorm and the adopted recipe. Preserve all six fresh raw/H200 pairs,
3,355,571,184B, all original evidence and153 recoverable historical RAW archives.
Storage014's clipping-import resolver remains protected; permanently retired
numerical tensors cannot be restored. Closure free9,388,134,400B is an observation,
not capacity approval for another panel.

At this boundary both architecture and bounded HP were reviewed: select the
previously deferred AdamW beta2 .95-versus-.999 with beta1 .9/epsilon1e-8/Muon and
outer decay fixed. Defer final LayerNorm learned bias versus fixed zero bias.
No gain assumed or prior score/seed pooling. Read TOKENIZER_ADAMW_BETA2_PREPARATION.md.

Evidence: tokenizer-evidence/readout-centering-preparation-001/quality001-final-review.json,
quality001-closure.json and quality001-fresh-inference-pair0/1/2-review.json.

## Full-model optimizer and resume integration complete

Model003 and model-resume001 passed both beta2 .999/.95 arms. Initial model state,
logits and construction RNG match exactly, with28,492,416parameters/88EMA tensors.
All Muon/AdamW group membership, order and width-scaled rates match; only declared
AdamW beta2 differs. CPU/MPS forward/gradients and full/cache/reset/causality passed.
Foreign configuration and saved group betas are rejected before state loading.

Original model001 launch failed on an absent plan before numerical work; its config
lookup used a nonexistent portable path. Model002 then failed an absolute CPU/MPS
post-update tolerance in the incumbent beta2 .999 arm. Both failures remain closed.
The separate model003 contract uses an exact full-parameter native-child update
oracle on each backend: error0. Absolute CPU/MPS post-update discrepancy is
0.00020608305931091309 for both arms; beta2 adds only2.9802322387695312e-8,
within the preregistered differential bound. Forward/gradient tolerances and resource
guards are unchanged. This does NOT establish absolute backend-update parity.

Each four-update MPS fresh repeat and serialized resume matched raw model,
optimizer, EMA, RNG and complete clipping histories exactly (maxerror0).
Step2 was serialized in capped memory; final-step4 raw/H200 exports have3snapshots.
CPU/MPS inference batches1/8 passed full/cache/reset with196608B cache per item.
All753 source/config bindings unchanged,738 predecessors preserved. This is generated
correctness evidence, not scientific quality, a pilot, or a seed reservation.

Storage015 recovered capacity without retiring any selected/fresh bytes: transparent
lossless compression of1643 historical text files reclaimed259,952,640B and verified
49,952 protected files; archival of16 closed GQA selected/fresh RAWs reclaimed
916,750,821B and verified51,579 other files. Exact restoration passed. Original
copy-only probe stopped before compression on actual capacity; separate002 passed
after text recovery. All169 RAW archives throughstorage006–015 remain protected,
recoverable, NOT retired; all H200 originals and storage004 remain intact.

Next: minimal additive actual run_training/output/worker and independent parent
full-validation/inference adapters using these tested primitives, including beta2
identity and complete clipping histories. Re-audit actual retained resume payloads,
close their tensor dependencies, then take a NEW exact authorized retirement audit
if needed. Four successful generated resume weights total1,118,319,956B and stay
retained meanwhile. Capacity001 free9,222,590,464B versus complete worker-pair
9,814,671,360B conditionally yields10,340,910,420B after dependency closure/audit;
this is a recovery route, not dispatch readiness or a scientific-panel forecast.
Source work proceeds. Then complete scientific census/output high-water/finite
resource pilots before any three-fresh-pair quality freeze. Retain beta2 .999.

Evidence: tokenizer-evidence/adamw-beta2-preparation-001/model003.json,
model-resume001.json, capacity001.json; storage-recovery-015/raw-completed.json,
compression-completed.json and restoration-check.json.

## AdamW beta2 generated worker integration complete

Both fresh generated four-update beta999/beta95 arms passed independent parent
acceptance:4096 training targets,3 H200 snapshots and88 EMA tensors each. Actual
optimizer states contain44 AdamW groups at(.9,.999)/(.9,.95),epsilon1e-8 and zero
child decay, with44 unchanged Muon groups and outer independent decay0.00006.
Full validation recomputed exactly in the parent; CPU/MPS CE differs by less than
2.9e-7. Complete clipping histories agree across raw/summary/metrics. Independent
missing/sequence/threshold/coefficient/nonfinite history, foreign beta2 config/group
and wrong-target-coverage probes were rejected. Trained-inference001 full/cache/reset
passed batches1/8,196608B cache per item. All760 runtime/acceptance bindings unchanged,
753 predecessors preserved. No scientific quality result or seed reservation.

Worker-integration001 first re-audited both real retained resume raw/H200/optimizer,
EMA/RNG/clipping histories/full-validation/cache and closed tensor dependencies.
A separate exact resume-retirement001 audit then reclaimed1,118,319,956B from four
obsolete successful generated weights;42,544 survivors verified unchanged. Original
receipts remain immutable; absent tensors resolve through the retirement manifest,
never restoration or old-fixture reruns. Actual free10,517,405,696B exceeded complete
worker-pair9,814,671,360B including8GiB reserve, enabling both fresh worker arms.
New generated worker raw/H200 weights1,118,325,460B stay retained until pilot
integration re-audits and closes dependencies, followed by a NEW exact cleanup audit.
All169 protected historical RAW archives, H200 originals, selected/fresh bytes,
storage004 and original failures remain intact.

Next: minimal additive finite20-update resource-pilot orchestration and complete
THREE-fresh-pair fixed-LR design outside760 frozen sources. Verify actual worker
payload/cache and finite contracts, close dependencies, fresh scientific census,
measure complete retained-output high-water with serialization overlap/metadata/
8GiB reserve, take exact authorized recovery if needed, then finite pilots and final
quality freeze. CommonLR0.00125/equal zero LR-search/original strict gates/no rescue,
selection or partial panel; beta1 .9/epsilon1e-8/Muon/model/schedule/H200 fixed.
Current free space is not new-panel readiness; source work proceeds and additional
lossless recovery is inventoried if the complete forecast needs it.

Retain incumbent beta2 .999. Final LayerNorm learned bias versus fixed zero is
deferred. Prior model001 missing-plan and model002 absolute CPU/MPS post-update
failures remain closed; model003 exact native updates/differential check is no claim
of absolute backend-update parity. Generated observations do not select quality.

Evidence: tokenizer-evidence/adamw-beta2-preparation-001/trainer-worker001.json,
trained-inference001.json, worker-integration001.json, worker-capacity001.json,
and resume-retirement001/completed.json.

## November2 — AdamW beta2 finite resource pilots complete

Both20-update arms independently passed actual raw/H200/optimizer/EMA/RNG and
complete clipping-history acceptance:20,480targets,8H200snapshots,88EMA tensors.
Actual44AdamW groups differ only in declared beta2(.999/.95);44Muon groups,
beta1 .9,epsilon1e-8 and outerdecay .00006 remain fixed. Paired full/cache/reset
inference batches1/8 passed at196608B/cache item. Independent final001 resource
review passed; all766 campaign sources unchanged,760 runtime predecessors preserved.
No scientific quality result. Pilot losses were not used for selection.

The prospective design fixes exactly three fresh pairs/six1200-update trajectories,
commonLR .00125,equal zero search and original strict gates. Panel199740 pilot/eval
used;fresh011/012/013/bootstrap004 remain unused. Next prepare additive
`research/adamw_beta2_quality_v1`, re-audit retained pilot payloads/cache and close
dependencies before a new exact pilot retirement audit and final quality freeze.
Retained pilot weights:1,118,337,488B. Observed free:11,507,736,576B;
conditional post-retirement:12,626,074,064B
versus complete12,381,585,408B including8GiB reserve. Remeasure before dispatch.

Worker retirement reclaimed1,118,325,460B and verified42,735
survivors. Storage016 losslessly archived13 closed accepted readout/wide RAWs,
reclaimed2,236,324,199B and verified52,040
other files unchanged. Exact restoration passed. All182 archived RAWs remain
protected/recoverable,NOT retired;all H200 originals and storage004 remain intact.
Original worker evidence stays immutable; absent generated tensors resolve through
`adamw-beta2-preparation-001/worker-retirement001/manifest.json`,never restore/rerun.
Archived RAWs resolve through `storage-recovery-016/raw-completed.json` and
`research_archives/protected-raw-storage-016/*.tar.zst`;never purge.

Evidence: `adamw-beta2-preparation-001/pilot-integration001.json`,
`pilot001-plan.json`, `pilot001-review.json`, `pilot001-final001-review.json`,
`campaign-capacity001.json`, `scientific-seed-census-001/result.json`;
`storage-recovery-016/raw-completed.json` and `restoration-check.json`.
Retain beta2 .999 and the adopted recipe; final LayerNorm bias ablation deferred.

## November2 — AdamW beta2 quality001 frozen and first fresh arm dispatched

The additive `research/adamw_beta2_quality_v1` binds774 active sources and preserves
all766 campaign predecessors.24 finite contract/publication tests passed, plus
actual retained pilot raw/H200/optimizer/EMA/RNG/clipping-history/full-validation/
cache re-audits. Pilot-to-quality forgery rejected. Declared44AdamW and44Muon
group identities are checked at payload acceptance and each accepted prefix.

Exact pilot retirement reclaimed1,118,337,488B with28,769
retained research files verified unchanged. Original receipts remain immutable;
absent pilot tensors resolve via `quality001-pilot-retirement/completed.json`.
No permanently retired tensors may be restored or closed prerequisites rerun.

Freeze SHA:`b71b2e692637f3599b8c860f63abb081fcfd2d1278ac0e1db2a5a01c63b78a49`.
Observed free12,619,821,056B exceeded complete12,381,585,408B including8GiB
reserve,serialization overlap and metadata. Every dispatch charges retained fresh
outputs and rechecks the remaining full-panel capacity. All182 historical RAW
archives remain lossless/protected,all H200 originals/storage004 remain intact.

Exactly six1200-update trajectories:beta999/beta95,beta95/beta999,beta999/beta95,
commonLR .00125,equal zero LR search,original strict gates,no rescue/selection/
retry/partial panel. Only AdamW beta2 changes;beta1 .9,epsilon1e-8,Muon,grouping,
width rates,outerdecay .00006,model,schedule and H200 remain fixed.
Slot00 dispatched;initial parent81253/worker81264,exec98811. Current live
progress/processes/anchors supersede these launch identifiers. Fresh199740011 is
in use;fresh012/013 andbootstrap004 unused at launch. No quality conclusion yet.
Next poll the actual worker,independently accept each endpoint,then dispatch only
the next frozen slot. After all six:three paired inference reviews and independent
statistics through the frozen `quality001-fresh-inference001.py` and
`quality001-final-review001.py`. Retain beta2 .999 pending all original gates.

Evidence:`adamw-beta2-preparation-001/quality-integration001-review.json`,
`quality001-plan.json`,`quality001-freeze.json`,`quality001-preflight.json`,
`quality001-capacity.json`,`quality001-seed-use.json`,
`quality001-pilot-retirement/completed.json`.

## November2 — AdamW beta2 quality closed without adoption; next final-bias prerequisite

All six1200-update endpoints independently accepted, with1,228,800 targets,
41H200 snapshots,88EMA tensors,44AdamW/44Muon groups and complete clipping
histories each. Three paired full/cache/reset inference reviews passed. Independent
mean beta2 .95-minus-.999 CE delta is +0.11006817407836185; seed deltas are
+0.11153406596318402,+0.10669633482953769,+0.11197412144236384. Every seed
worsens and all original strict gates fail. Retain beta2 .999; no adoption.
This is three paired seeds on reused validation text, fixed commonLR .00125 and
equal zero LR-search budgets; no holdout result or separately tuned optimum.

Terminal STOP closes every beta2 quality command. All774 bound sources unchanged.
Preserve all six fresh raw/H200 pairs3,355,577,712B,182 historical lossless RAW
archives,all H200 originals/storage004 and original failure receipts. Closure free
9,203,707,904B is an observation,not new-panel readiness. No deletion performed.

Both architecture and bounded HP reviewed: select final LayerNorm learned bias
versus fixed zero; defer AdamW epsilon1e-6 versus1e-8 at beta1 .9/beta2 .999.
Keep centering,learned scale,epsilon1e-5 and eight pre-branch RMSNorms unchanged.
The prospective candidate removes256 parameters; this is not parameter-matched.
Standalone component001 passed24 CPU FP64/FP32 and MPS FP32 cases at batches1/3,
widths7/256:explicit equations/gradients,exact native-control parity,initial function
and RNG equality,bias-gradient sum,finite differences and causal independence.
All777 source bindings unchanged,774 predecessors preserved. No candidate full-model,
optimizer,EMA/cache or scientific quality claim; no scientific seeds or panel reserved.
Next explicit full-model policy/optimizer/checkpoint/cache and repeat/resume integration
outside frozen namespaces,then bounded workers,pilots and complete capacity review.
Never bypass existing learned-bias/final-LayerNorm policies.

Evidence: `adamw-beta2-preparation-001/quality001-final-review.json`,
`quality001-closure.json`, `quality001-fresh-inference-pair{0,1,2}-review.json`;
`final-ln-bias-preparation-001/direction-decision.json`, `component001-plan.json`,
`component001-result.json`. See `TOKENIZER_FINAL_LN_BIAS_PREPARATION.md`.

## November2 — Final LayerNorm bias full-model and serialized resume passed

Both learned/fixed-zero policies passed model001 and model-resume001. Shared
initialization/logits/RNG and all unrelated groups match exactly; only final bias
is removed. Parameter counts28,492,416/28,492,160;EMA inventories88/87,Muon44
groups both andAdamW44/43. Explicit actual readout equations,CPU/MPS gradients,
full/cache/reset/causality and foreign-bias checkpoint/cache rejection passed.
Both four-update MPS fresh repeat and serialized resume raw-model/optimizer/EMA/
RNG/clipping histories match exactly(maxerror0). Step2 serialized in capped memory;
final-step4 RAW/H200 exports have3 snapshots,CPU/MPS batch1/8 inference passed.
All791 source/config bindings unchanged,777 predecessors preserved. This is
generated integration evidence;no scientific quality result,seeds or pilot/quality
reservation. Next actual trainer/output/worker and independent parent full-validation
acceptance/inference adapters outside frozen sources,with87/88 tensor acceptance.

Storage017 losslessly archived six closed beta2 fresh RAWs, reclaiming651,815,376B
and verifying52,616 survivors unchanged. Exact restoration passed. The original
APFS copy-only probe kept bytes unchanged but gave no useful compression; the
separate streaming path uses a stricter kernel-enforced512MiB output cap and
complete staging forecast,with unchanged8GiB reserve,time/RSS limits and one
compressor thread. Output-cap self-test passed. Original RAWs resolve through
`storage-recovery-017/raw-completed.json` and `protected-raw-storage-017/*.tar.zst`;
bytes recoverable,NOT retired. All188 historical RAWs and H200 originals/storage004
remain protected. No source,receipt or closed scientific experiment was rerun.

Capacity002 observed9,943,003,136B versus9,781,116,928B complete resume-pair
requirement, enabling both arms. Four successful generated resume weights now
total1,118,313,377B and remain retained until downstream tensor dependencies close
and a NEW exact audit. Capacity003 observed8,625,647,616B versus9,814,671,360B
complete worker-pair requirement;conditional retirement leaves70,710,367B deficit.
An additional118,112,256B of allocated historical text candidates was inventoried
for exact lossless original-path compression if needed;this is not an allowlist or
dispatch readiness. Source integration proceeds;remeasure actual complete capacity
after authorized recovery. No partial panels or relaxed guards.

Retain learned final bias and the adopted beta2 .999 recipe. AdamW epsilon1e-6
versus1e-8 deferred. Prospective scientific design remains exactly three fresh
pairs at commonLR .00125,equal zero LR search,original strict gates,no rescue.
Evidence: `final-ln-bias-preparation-001/model001-plan.json`, `model001.json`,
`model-resume001-plan.json`, `model-resume001.json`, `capacity001/002/003.json`;
`storage-recovery-017/raw-completed.json`, `restoration-check.json`.

## November 2 — Final-bias worker integration completed

Final LayerNorm learned/fixed-zero bias now passes actual generated worker output,
independent full validation and batch1/8 full/cache/reset inference. Arm-specific
88/87 EMA and44/43 AdamW inventories are enforced. Original001 inter-slot capacity
failure happened before slot01 launch; separate002 imported accepted slot00 with
zero repeat updates/copies and completed only the unrun arm. No quality finding.
Next finite20-update pilots and complete three-fresh-pair design at commonLR.00125,
zero LR-search and original gates, after real payload/finite-contract audits and
full high-water forecast. Keep learned bias; AdamW epsilon ablation stays deferred.
See TOKENIZER_FINAL_LN_BIAS_PREPARATION.md and trainer-worker002 evidence.

## November 2 — Final-bias finite pilot path prepared

Pilot integration/retention passed with804 unchanged bindings and explicit87/88
EMA and43/44 AdamW inventories. Exactly three fresh pairs at commonLR.00125,
zero LR-search,original gates remain prospective. No scientific seeds or freeze.
Exact retirement and lossless recovery executed; complete-panel capacity still
lacks2,090,844,160B at the recorded observation. Prepared pilots await the precise
storage wake condition in TOKENIZER_FINAL_LN_BIAS_PREPARATION.md. No active worker
or quality result. Keep learned final bias; epsilon ablation remains deferred.

## November 2 — Final LayerNorm bias resource pilots completed

Both frozen 20-update arms passed independent parent acceptance: 20,480 training
targets, 8 H200 snapshots, and complete clipping histories. Learned/fixed-zero
arms enforce 88/87 EMA tensors and 44/43 AdamW groups, with 44 Muon groups each.
Both ran at common peak LR 0.00125; no pilot loss selected a configuration.
Full/cache/reset inference passed at batches 1 and 8, using 196,608 cache bytes
per item in each arm. The separately bound final001 resource review passed.
All 804 campaign sources remain unchanged; 798 runtime predecessors preserved.

Peak observed MPS driver allocation was 2,510,094,336B for each arm; maximum
RSS was 1,449,328,640B. Conservative 1200-update projections with 2x headroom
were 5,588.74 and 6,335.95 seconds, below the unchanged 7,200-second cap. These
are resource observations, not full-run measurements or scientific quality gains.
Neither pilot clipped any of its 20 updates; this short observation does not
establish clipping frequency in the eventual quality runs.

Panel 199850 pilot001/eval003 are used; fresh011/012/013 and bootstrap004 remain
unused. Both raw/H200 pairs (1,118,330,973B total) remain retained. Final review
observed 12,324,999,168B free; adding those pilots conditionally yields
13,443,330,141B, above the unchanged 12,381,585,408B complete requirement.
Do not delete pilots until the next integration closes dependencies and a NEW
exact retirement audit passes; remeasure actual capacity before dispatch.
This is a concrete recovery path, not a renewed capacity wait for source work.

Next implement finite quality orchestration in NEW research/final_ln_bias_quality_v1
outside frozen sources. Re-audit actual pilot tensors, optimizer/EMA/RNG/clipping,
validation and cache; test exactly six 1200-update trajectories, then audit pilot
retirement and freeze three fresh pairs at common LR 0.00125, zero LR-search,
original strict gates, no rescue/retry/selection/partial panel. The candidate
removes 256 parameters; centering/scale/epsilon1e-5 and pre-branch norms stay fixed.
Keep learned final bias and the adopted recipe. AdamW epsilon ablation remains
deferred. No active worker or scientific quality result; all older campaigns closed.

Evidence: final-ln-bias-preparation-001/scientific-seed-census-001/result.json,
pilot001-plan.json, pilot001-freeze.json, pilot001-accepted-00/01.json,
pilot001-review.json, pilot001-final001-review.json and pilot001-evidence/.

## November 2 — Final-bias quality001 frozen and first fresh trajectory launched

Eight new modules in research/final_ln_bias_quality_v1 preserve all 804 pilot
predecessors; all 812 active source bindings are frozen. The 24 finite-contract
and publication tests passed. Independent real-pilot audits revalidated raw/H200,
optimizer/EMA/RNG, arm-specific 88/87 gradient/EMA and 44/43 AdamW inventories,
complete clipping histories, full validation and cache/reset behavior. A pilot
bundle forged as a quality endpoint was rejected. Large reports use existing
2MiB resource/event groups; no cap increased.

After dependencies closed, the new independent quality001-pilot-retirement audit
retired exactly four successful pilot weights totaling 1,118,330,973B and verified
28,961 survivors unchanged. Original source/config/log/receipts remain immutable;
absent pilot tensors resolve through that later manifest and cannot be restored.
All selected/fresh checkpoints and 191 protected RAW archives remain protected.

Freeze free 13,457,494,016B exceeded the complete 12,381,585,408B requirement,
including metadata/publication overlap and 8GiB reserve. All three fresh seeds
were unused at freeze. Exactly six 1200-update trajectories at common LR0.00125
follow learned/zero, zero/learned, learned/zero order, with equal zero LR-search
budgets, no screens/selection/rescue/retry/partial panel or pilot-loss selection.
Each uses 1,228,800 targets and 41 H200 snapshots. Every fresh raw/H200 pair stays
protected. Fixed-zero removes256 parameters and one final-bias AdamW/EMA entry;
centering, learned scale, epsilon1e-5 and all eight pre-branch RMSNorms stay fixed.

Slot00 learned is running; config, initial metric and guard verified. Launch
snapshot parent2372/worker2384, exec34278 is not authoritative after this moment:
poll actual processes/logs/guards/accepted anchors before any next dispatch.
Fresh199850011 is now in use;012/013/bootstrap004 unused. No quality conclusion.
After all six accepted endpoints, run all three paired inference reviews then
the independently implemented final statistics review with the exact freeze.
Keep learned final bias/adopted recipe pending original strict gates; AdamW
epsilon remains deferred. No old score pooling, holdout scoring or closed reruns.

Freeze SHA256: b52e303992677705891412ad410e03539ee28a1b0a2ae5810729b9a5186a7297.
Evidence: quality-integration001-review.json, quality001-pilot-retirement/,
quality001-capacity.json, quality001-seed-use.json, quality001-freeze.json,
quality001-plan.json, quality001-preflight.json and quality001-launch.json under
final-ln-bias-preparation-001.
