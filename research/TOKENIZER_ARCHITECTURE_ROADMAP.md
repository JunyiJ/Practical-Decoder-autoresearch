# Tokenizer architecture research agenda

Human steering, October 26, 2026 (Los Angeles): research must explore model
architecture, including attention, MoE, activation, normalization and residual
layers, alongside hyperparameter tuning. The human clarified: "not only focus
on HP tuning." Both are first-class research directions; this is not a mandate
to replace tuning with architecture work. This governs future
rounds alongside the [attention roadmap](TOKENIZER_ATTENTION_ROADMAP.md).
The running, frozen LR transfer experiment finishes unchanged to establish a
reviewed subword baseline. Choose subsequent architecture or hyperparameter
experiments from evidence, while ensuring meaningful architectural exploration.

## Scheduling and decisions

The proposed architecture sequence is inference-aware GQA/MLA, followed by the Kimi
KDA/KDA/KDA/MLA hybrid. Then prioritize a finite sparse-versus-dense FFN study
and a finite activation/normalization/residual study, choosing each specific
hypothesis from reviewed evidence and implementation feasibility. Do not let
attention-only refinements indefinitely displace the other architecture families.
After each completed architecture round, explicitly review the next untested
family below. Record why it is selected, deferred or rejected; this agenda is
not permission to launch all candidates at once.

Each round needs a concrete hypothesis, exact configuration and, for architecture
changes, equations and layout,
matched controls, resource pilots, a frozen finite plan, fresh-seed confirmation
and an adoption rule declared before results. Use equally bounded optimization
tuning to avoid handicapping a family. Standalone hyperparameter rounds remain
valid when motivated by evidence, including LR, schedule, horizon and regularization
questions. No architecture-only ordering or extra permission is required. Review
both kinds of hypotheses at each round boundary and record the selection rationale.
Fix correctness or resource failures before continuing research;
preserve the failed evidence. Implementation tests do not establish quality.

## Architecture families and discriminating comparisons

These are local proposals, not claims of existing tokenizer results or paper
reproductions. Freeze exact candidate values separately before training.

| Family | First structural question | Essential controls and measurements |
|---|---|---|
| Attention | Can GQA or compressed MLA retain quality with cheaper inference? | Fresh MHA control; actual KV memory, prefill/decode timing and backend; use the attention roadmap's prospective neutrality and cost rules |
| Recurrent/global hybrids | Does three KDA layers plus one MLA layer preserve global recall better than all-KDA at useful state cost? | Full MLA and full KDA controls, explicit Linear/K3 semantics, layerwise caches; version/ratio ablations follow the initial 3:1 comparison |
| Sparse MoE | Does routed expert capacity improve subword quality for a comparable active FFN budget? | Dense SwiGLU control; start from a small four-expert/top-two proposal, derive expert widths before launch; report active and total parameters, router/shared-expert overhead, actual time/memory, routing entropy/load, unused experts and auxiliary loss separately from validation CE |
| Shared experts and layer placement | Is shared dense capacity useful alongside routed experts, or should only selected layers use MoE? | Separate follow-up to the sparse baseline; account for always-active shared work and vary one placement/design choice at a time |
| Activation and FFN structure | Does the tokenizer change the relative behavior of SwiGLU, a GELU gate or an ungated dense FFN? | Isolate gate nonlinearity with identical shapes first; a gated-versus-ungated comparison separately matches matrix budget as closely as possible and reports rounding/bias differences |
| Normalization | Does block RMSNorm versus LayerNorm, or additional Q/K or FFN normalization, improve stability and quality? | Separate block norm, final norm and internal norm changes; specify placement, epsilon, affine parameters and optimizer treatment; track activation/gradient scales without assuming they explain a CE gain |
| Residual topology | Does serial versus parallel attention/FFN composition help? | Specify which normalized stream each branch consumes and whether norms are shared; separate topology from fixed or learned branch gains |
| Residual scaling and depth | Can depth-aware branch scaling make additional layers useful? | First test scaling at fixed depth, then a bounded depth/width allocation; report total and non-embedding parameters and measured compute, not just equal token counts |
| Embedding/readout | Does tying the large BPE input/output tables improve the quality/memory tradeoff? | Untied control; explicit initialization, optimizer grouping and readout scaling contract; count saved parameters and measured memory; tying also changes expressivity |
| Position and local mixing | Do positions, local/global attention placement or short convolutions improve the selected architecture? | Separate structural ablations, declared context and cache semantics; no long-context claim from context128 results |

The existing repository contains GQA/MLA/KDA modules, routed/shared experts,
SwiGLU gate alternatives, norm and residual variants, and tying/depth helpers.
Their existence does not mean the inherited tokenizer policy permits combining
them. Validate and extend explicit model/optimizer/checkpoint contracts for each
new design. Do not disable historical guards or revive paused null-attention or
permanently stopped campaign006 to access a candidate.

## Fair comparisons and evidence

Retain the reviewed character learnings as priors. A failed character experiment
can motivate a targeted control or narrow a hypothesis; it does not establish a
subword result. Revisit it only with a stated tokenizer-related or structural
reason. Do not pool character and subword CE, or select the luckiest old seed.

Equal training tokens, active parameters, total parameters and wall time are
different budgets. Declare the main comparison axis and report the others.
MoE especially needs both active and total capacity accounting, including the
large shared vocabulary tables. Architecture changes alter RNG consumption;
paired seeds alone do not imply identical shared initial weights.

Keep the test holdout closed and record adaptive validation reuse. Confirm
selected candidates on fresh seeds and the claimed widths; report reversals,
failures and limitations. Quality-neutral GQA/MLA can qualify through the
predeclared measured inference-cost route. Other families need their own
prospective quality/cost rule; do not invent one after seeing favorable numbers.

Validate gradients, causality and the affected cache, optimizer, tokenizer,
checkpoint/resume and EMA behavior before serial MPS pilots. Freeze source
before each finite training round; preserve checkpoints and raw evidence.
Continue with one local worker and bounded disk, memory and time guards. Publish
completed iterations, substantive learnings and failures, then advance to the
next evidence-driven architecture or hyperparameter hypothesis until the human
stops the campaign.


## October31 completed activation comparison and next design review

The width256 RoPE-MLA SiLU versus exact-GELU comparison is now fully closed through
recovery002, including three accepted timing pairs and independent final review.
GELU failed all quality adoption gates (mean delta+0.0015507559nat/token,2/3seeds
worse); measured local inference is near parity and cache sizes identical. Retain
SiLU. See the final closure section of TOKENIZER_MLA_ACTIVATION_PREPARATION.md.

Next review the untested tokenizer normalization/residual families, specifying
one isolated hypothesis and controls before implementation. MoE remains deferred
under its full18,782,075,912-byte requirement. Closure left about12.99GBfree;
even the prior same-size two-arm15.17GB projection cannot presently fit. A new
panel needs fresh full high-water accounting and safe capacity or a separately
justified prospective design. Do not silently narrow controls, fresh seeds or
width, remove protected pairs, or launch a partial panel. Read-only feasibility
and bounded CPU prerequisites can proceed while this training constraint holds.


### October31 next normalization design: matched-epsilon block LayerNorm

Selected prospective two-arm block RMSNorm versus affine LayerNorm on the adopted
D256 RoPE MLA/SiLU backbone; only eight pre-branch normalizers change. The local
constructor audit found an epsilon confound: default RMSNorm uses FP32 epsilon
2^-23, default LayerNorm1e-5. Proposed LayerNorm explicitly uses2^-23 while the
control preserves its exact incumbent path. Centering and a learned bias remain
joint changes; no isolated-centering causal claim. Candidate adds2,048 parameters.

The guarded meta-shape/optimizer audit found15,170,221,264bytes required against
12,984,197,120free, a2,186,024,144byte gap. Even final tensor payload plus reserve
exceeds free space by77,657,600bytes before headers/logs/overlap. Reducing safety
allowances alone cannot make this full retained panel fit. No pilot or training
started; no protected evidence changed. Bounded CPU prerequisites can proceed.
See [prospective design](TOKENIZER_BLOCK_NORM_PREPARATION.md) and
[audit](tokenizer-evidence/block-norm-preparation-001/shape-storage-review.json).

## Trained pilot-pair inference completed, October 31, 2026

The actual20-update RMS and LayerNorm H200 checkpoints passed all four full/cache/
reset comparisons before timing. With two warmups and eight alternating repeats,
median32-token prefill/64-token decode times were8.477/291.908ms for RMS and
8.952/278.811ms for LayerNorm at batch1; at batch8 they were13.713/297.073ms
and14.222/305.529ms respectively. LayerNorm decode time was4.49% lower at batch1
and2.85% higher at batch8, with identical cache allocations196,608/1,572,864B.
This mixed single-pilot result supports no consistent speed advantage. It is
descriptive only; two models were resident, cache allocation/loading excluded,
and timing values are not independently reproducible hashes or adoption gates.

The24.224-second guarded run peaked at RSS881,606,656B and driver1,155,842,048B.
Both original pilot directories, weights and failed-process status remained
unchanged. The new RMS measurement request explicitly rebinds only the additive
continuation source; original requests and recovery evidence remain immutable.
No training, test scoring or deletion occurred.

The remaining quality projection keeps both existing pilot pairs without cleanup
credit, retains all six screens through saved LR selection, then permits only
exactly audited losing-screen weight removal. Two selected screen pairs plus six
fresh pairs define the later peak. Original per-trajectory allowances,256MiB
metadata and8GiB reserve require15,170,221,264 additional free bytes, versus
approximately17.95GB available. This projection fits, but is not launch authority:
real quality dispatch, external acceptance anchors and the exact retention
boundary must be connected and checked before final finite quality freeze.
Evidence: pilot-pair-inference-001-plan/result/review/guard.json and
quality-remaining-capacity-001.json. SiLU remains adopted; no norm quality result.

## Block-normalization comparison closed, November 1 UTC / October 31 Los Angeles

All six LR screens, six fresh confirmations and three fresh inference pairs
completed with independent acceptance. Both arms selected LR0.00125.

| Fresh seed | RMSNorm H200 CE | LayerNorm H200 CE | LayerNorm minus RMSNorm |
|---|---:|---:|---:|
| 374011 | 4.709659 | 4.706465 | -0.003194 |
| 374012 | 4.728085 | 4.720328 | -0.007757 |
| 374013 | 4.715869 | 4.710494 | -0.005376 |

Mean delta is **-0.00544214 nat/token**. Every seed improves and simultaneous
conditional-text upper bounds are negative at all declared block sizes:
2048: -0.00382454; 4096: -0.00420806; 8192: -0.00391838. The required mean
delta below -0.02 is not met. **Do not adopt LayerNorm; retain RMSNorm and SiLU.**
This is a small consistent validation improvement that fails the prospective
practical-effect threshold, not a claim of no effect. Independent grouping,
PCG64 resampling and linear quantiles agreed with the frozen arithmetic within1e-12.
Intervals condition on these trained seeds and reused text; they do not establish
population-seed or holdout confidence. Centering and learned bias changed jointly.

All three H200 inference pairs passed full/cache/reset checks for batches1/8.
Median across the three paired timing medians (milliseconds,32-token prefill /
64-token decode) was:

| Batch | RMSNorm | LayerNorm |
|---|---:|---:|
| 1 | 8.720 / 275.931 | 8.664 / 284.162 |
| 8 | 13.942 / 290.713 | 14.104 / 292.302 |

LayerNorm batch1 decode was slower in all three pairs; batch8 was mixed.
Cache allocations are identical (196608B/item). Timings are descriptive, with
two models resident and allocation/loading excluded; there is no cost-only
adoption route. All guards passed; approximately13.35GB remained free.
Every fresh raw/H200 pair, both selected screens and original failed pilot
are preserved. No additional checkpoint deletion or test scoring occurred.
A STOP marker closes quality001; do not rerun its training, inference or review.
Evidence: quality001-final-review.json, quality001-fresh-inference-* and
quality001-closure.json in tokenizer-evidence/block-norm-preparation-001/.

At this boundary, the next review compares residual topology with a bounded
schedule/horizon hypothesis. Parallel separate-norm blocks were worse in the
closed character experiments (+0.026757/+0.013555 at widths64/256); a subword MLA
comparison therefore needs a specific transfer rationale, not a presumed gain.
Scaled table sharing already failed the closed BPE GQA neutrality test; do not
repeat it solely to save disk. MoE remains deferred pending a complete feasible
capacity projection. No next hypothesis or training grid is frozen here.


## EMA horizon comparison closed: retain H200

The finite H100-versus-H200 panel completed all three fresh1200-update online
trajectories with41snapshots per EMA. Both averages used each same online model;
only averaging half-life differed. All raw resume/export audits, independent
parent full-validation recomputations and fresh inference/cache/reset checks
passed. No source, LR, schedule, seed or grid change occurred after freeze.

| Fresh seed | H200 validation CE | H100 validation CE | H100 minus H200 |
|---|---:|---:|---:|
| 187610011 | 4.726289522 | 4.715769963 | -0.010519559 |
| 187610012 | 4.691822251 | 4.686677891 | -0.005144360 |
| 187610013 | 4.700717490 | 4.689580086 | -0.011137404 |

Mean delta is **-0.0089337743 nat/token**. Every fresh seed improves; simultaneous
conditional-text upper bounds are negative at2048(-0.0064582934),
4096(-0.0058233907) and8192(-0.0049000548) tokens. The declared mean delta
below-0.02 is not met. **Do not adopt H100; retain H200, RMSNorm and SiLU.**
This is a consistent small validation gain below the prospective practical-effect
threshold, not evidence that the two averages are identical. Independent
contiguous grouping, PCG64 resampling and NumPy linear quantiles agree with
frozen arithmetic within1e-12. The intervals condition on three trained seeds
and reused validation text; they are not population-seed or holdout confidence.
The experiment is consistent with a smaller averaging lag helping this400-update
tail, but it does not isolate lag as the cause or establish a new optimal half-life.

All three inference pairs passed. Median across paired medians, milliseconds
for32-token prefill/64-token teacher-forced decode:

| Batch | H200 | H100 |
|---|---:|---:|
| 1 | 8.212 / 280.984 | 8.839 / 276.767 |
| 8 | 13.918 / 293.076 | 13.861 / 288.526 |

Per-seed decode timing was mixed; cache196608B/item is identical. This is
descriptive timing of two resident models, excluding loading/cache construction;
there is no cost-only adoption route. All12 raw/H100/H200 checkpoint files
(pilot plus three fresh triples) remain verified and protected. About10.19GB
was free at closure, above the8GiB reserve. No deletion or test scoring occurred.
The inherited trainer materialized test tokens; validation used only train/val.
STOP closes tokenizer-ema-horizon-v1-001; do not rerun training, inference or
this final review. No new control checkpoint is promoted from these results.

Evidence: tokenizer-evidence/ema-horizon-preparation-001/quality001-final-review.json,
its executable source/guard, all three slot reviews/inference reports, and
quality001-closure.json. Final quality source inventory remains399 files,
including395 byte-identical inherited files.

Next boundary: review a distinct structural FFN question (gated versus ungated
at a declared comparable matrix budget), residual scaling at fixed depth, and
MoE feasibility against the recorded evidence. Do not repeat the closed gate-
activation, block-normalization, wider-hybrid, tying or EMA campaigns. Choose
one concrete architecture or HP hypothesis with a full retained-output capacity
projection before implementation/training; no next grid or launch is frozen yet.


## November1 boundary: ungated SiLU FFN selected for preparation

After the closed EMA comparison, select a distinct FFN-structure question:
SwiGLU hidden672 versus ungated SiLU hidden1008 at equal516,096 matrix parameters
per layer, width256 MLA. SiLU stays fixed; hidden width and multiplicative
factorization change jointly. No isolated gating-mechanism claim. Historical
residual attenuation worsened all fresh seeds; defer it and the still-unfunded
three-arm MoE storage proposal. The standalone CPU analytic forward/gradient
oracle passed, but the candidate full-model policy remains unsupported.

The complete equal-LR-screen/three-fresh-pair proposal needs15.170GB free.
Audited successful norm-pilot cleanup reclaimed0.559GB and verified all protected
evidence. About10.740GB is free; another4.430GB is required. No
seeds/grid/launch frozen and no training started. See
[TOKENIZER_FFN_STRUCTURE_PREPARATION.md](TOKENIZER_FFN_STRUCTURE_PREPARATION.md)
and ffn-structure-preparation-001/direction-decision.json. Preserve all frozen
source namespaces; resolve full retained-output capacity before new dispatch.


## FFN integration advanced after capacity recovery

Renewed authorized obsolete-screen retirement reclaimed9.828GB; capacity now
exceeds the proposed15.170GB panel requirement. The earlier storage blocker is
resolved against that projection. Full-model CPU and strict MPS tests now pass
for SwiGLU672 and ungatedSiLU1008, including gradients and cached decoding.
No scientific training or quality claim yet. Continue serialized checkpoint/
resume/H200 and actual worker acceptance; see TOKENIZER_FFN_STRUCTURE_PREPARATION.md.


## FFN resource gate passed

Both matched-parameter FFN arms completed the finite scientific-corpus20-update
resource pilots with exact independent full-validation acceptance and trained
cache/inference checks. Both pilot pairs remain retained; the remaining full
quality projection fits current capacity with all reserve/output guards. No
quality result yet. Next is narrow full-duration/retention integration and the
finite3LR-per-arm,3freshpair quality freeze; see TOKENIZER_FFN_STRUCTURE_PREPARATION.md.


## FFN screening complete; fresh confirmation ready

The six frozen FFN screens and independent acceptance are complete. Both arms
select LR0.00125; selected-rate H200 CE is4.712841 for SwiGLU672 versus4.722775
for ungatedSiLU1008. The high-rate0.0025 ordering reverses, so the screening
comparison depends on the rate. No adoption conclusion from one screening seed.
Exact audited losing-screen cleanup reclaimed2.237GB and preserved selected
pairs, pilot inventories and all non-weight evidence. The remaining finite
projection fits current capacity. Next: three fresh pairs at frozen selected
rates, paired inference and final review; see TOKENIZER_FFN_STRUCTURE_PREPARATION.md.


## FFN structure completed; next distinct hypothesis review

All six screens, three fresh pairs, three paired cache/inference measurements
and independent statistical review are complete. At selected LR0.00125 for both
arms, ungatedSiLU1008 is worse by0.007388630367nat/token on average and on two
of three fresh seeds. All predeclared quality gates fail. Retain SwiGLU672,
SiLU/RMSNorm/H200 and the adopted D256 MLA backbone; close this FFN comparison.
Cache sizes match; small local decode timing differences do not establish a
cost-only adoption case. Preserve all selected/fresh/pilot and failure evidence.

Next executable work: review remaining internal-normalization/position/local-
mixing architecture questions and bounded attention-score/optimizer HP questions
against their original character and current-tokenizer evidence. Select one
distinct hypothesis with an explicit rationale and finite prospective design;
reject reopening any already closed comparison. Recompute complete retained
output plus serialization/metadata/reserve for that design before any freeze.
About12.06GB was free at FFN closure; this is an observation, not proof that a
new panel fits. Small source/numerical work can proceed while eligible recovery
is assessed. Do not revive the deferred MoE proposal without its full capacity.
See TOKENIZER_FFN_STRUCTURE_PREPARATION.md and quality001-final-review.json.

## Next boundary review: attention multiplier transfer on adopted MLA

After the closed gated/ungated FFN comparison, select bounded HP alpha0.5 versus1.0
in the D256 MLA attention scores. The inherited0.5 was selected in character MHA;
current subword/low-rank geometry provides a distinct transfer question. Current
MLA already applies the multiplier correctly. Review deferred internal norms
(weak or mixed historical evidence), position alternatives (historically worse),
short-convolution mixing (distinct new cache contract) and MoE (larger complete
retained-output requirement). This HP round follows several completed structural
rounds; both architecture and HP remain candidates at its next boundary.

The synthetic attention arithmetic gate passed, including CPU/MPS gradients and
uneven/reset/causal caches. No new quality finding yet. The exact proposed controls,
equal LR budget, three fresh pairs and strict adoption rule are in
[TOKENIZER_MLA_SCORE_PREPARATION.md](TOKENIZER_MLA_SCORE_PREPARATION.md).
Next implement only the full-model/checkpoint/inference prerequisites outside
frozen sources. Audited obsolete FFN pilot retirement reclaimed1.118GB; capacity
still needs actual new serialization accounting and further eligible cleanup.

The score-transfer full-model and serialized-resume prerequisites now pass for
both multipliers, including identical initialization/grouping, strict cross-arm
identity rejection and exact model/optimizer/EMA/RNG replay on generated text.
This is engineering evidence, not a quality improvement. Completed EMA-pilot
retirement plus lossless successful-numerical archival recovered2.986GB while
preserving protected evidence. The measured prospective13.69GB requirement fits
the13.99GB observation, subject to actual worker caps, tested stage retention and
fresh capacity checks. Proceed to real worker/parent validation/inference before
scientific seed and finite-plan freezes; do not repeat stale storage blockers.

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
