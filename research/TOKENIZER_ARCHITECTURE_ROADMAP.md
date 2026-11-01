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
