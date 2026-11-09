# General RSI: prospective exposure comparison

2026-11-09. Status: evidence and design selected; implementation/source/seed/resource
bindings must be frozen before launch. This is new work after panel001, not an
extension or replay of its completed scientific comparisons. No worker is active.

## Local evidence and uncertainty

`tokenizer-evidence/general-rsi-panel-001/decision.json` and the original slot04–09
scores/steps show the adopted byte recipe improves equal-token H200 CE by0.052604
nats/token across three fresh pairs. All original gates pass. However, its
training wall time is about1.84× native and preserved first outputs remain weak.
`general_rsi_analysis_v1.py` derives the same-document probe curves from original
validation files. Byte mean probe CE decreases5.172979→5.033638→4.937302 over
steps8192/9216/10240. Full RAW CE falls6.049977→5.150732 from2048→10240.
No late validation reversal is visible in these observations. The LR also decays
in this interval, so exposure alone is not identified. Probe size is only8docs;
full endpoints cover108docs. None of these observations proves longer training
will improve readable output or generalize outside this corpus.

## Primary literature and limits

Muennighoff et al., *Scaling Data-Constrained Language Models*,
[arXiv:2305.16264v3](https://arxiv.org/html/2305.16264v3), sections4–6 and appendixQ
(hyperparameter sensitivity): repeated data can improve loss, with diminishing
returns and eventual degradation. Unique data remains more efficient; their
fits miss some overfitting failures. They use shuffled full epochs, GPT-2 models,
C4 and budgets starting at100M unique tokens. Our10M-token random-block sampler,
byte-composed inputs, MLA/Muon optimizer and short context differ materially.
Their repetition thresholds are not prescriptions for this model.

Hoffmann et al., *Training Compute-Optimal Large Language Models*,
[arXiv:2203.15556v1](https://arxiv.org/html/2203.15556v1), appendixB/FigureA1:
misaligned cosine schedule horizons can impair endpoint loss. We retain our
linear cooldown but match it to each arm's horizon. Their experiments do not
supply a local optimum or prove the effect of repeated data here.

## Hypothesis and decision value

Inference: the current byte model may still be limited by training exposure.
Prediction: doubling updates with a cooldown matched to that longer horizon will
improve final H200 validation CE by at least0.05nats/token on average across fresh
paired seeds, without a source regression. A failed gate is a scientific negative
for this recipe/budget, not permission to reroll or extend it until it wins.
This tests the practical longer-training recipe, including its schedule; it does
not isolate token exposure from cooldown, establish a compute optimum, or prove
that a larger dataset is unnecessary. A positive result motivates longer-training
quality work; a negative result shifts attention to equal-time native controls
or diagnosed generation failures. Old scores are context only.

Ranked alternatives: longer exposure first because all saved late curves still
improve; equal-time native comparison second because byte is substantially
slower; context change third because no context-specific error analysis yet
establishes a mechanism. No architecture/optimizer sweep or output-lexical retry.

## Prospective scientific contract

Use the adopted raw-byte model and LR.0025, identical dataset/model/optimizer/
clipping/sampler/scorer and context128 as panel001. Compare fresh initialization
at H10240 versus H20480; same initializer and training sampler within each pair,
three entirely fresh bundles, six serial runs in short/long order. Do not warm
start old endpoints or pool old confirmations. No LR search: this is one fixed
recipe hypothesis, not a claim that either horizon has optimal LR.

Each arm uses its own floor(2H/3) cooldown to ratio.1 and the unchanged H200
half-life200/interval10 tail averaging rule. Compare final H200 primary, RAW
secondary; no best-checkpoint selection. Every run scores the same108validation
docs/100416targets. Retain the existing eight-doc probe every1024updates and
full diagnostic at2048. First fresh pair gets the same eight generation prompts,
initial/RAW/H200, greedy and seeded sampling, preserving every first output.
The longer arm's first10240 training batch hashes must equal its paired short
arm; model trajectories differ when their schedules diverge.

Adopt longer horizon only if all three long-minus-short H200 deltas are negative,
mean<=−.05nats/token, the one-sided99% paired document bootstrap upper bound<0,
and neither source mean regresses by>.05. Use10000resamples with one new seed
fixed before launch, average paired seed loss sums before resampling108IDs, and
divide by resampled target counts. Bounds remain conditional on these docs/seeds.
Otherwise retain H10240; no seed replacement or gate change. Report wall time,
parameters/memory and first samples alongside CE. Test stays unopened. Fresh
role IDs and source hashes must be audited and bound before any execution.

## Complete resource forecast and implementation boundary

Total92160updates /94,371,840sampled targets across6runs. Each longer arm sees
20,971,520targets (~2.10corpus-equivalents, not guaranteed epochs). Conservative
measured baseline is the slowest completed byte guard6975.039s per10240updates.
Nine such units,30%margin and an extra hour for evaluation/generation yield
85,207.96s (~23.67hours). Set a new tranche ceiling30hours; per-run ceiling5hours.
Keep one local MPS worker, RSS4GiB/MPS6GiB and10GiB disk reserve. No paid compute.

Full retained-output forecast:6 final RAW/H200 pairs at the earlier conservative
132,975,578bytes/pair, two simultaneous restart generations529,357,538bytes,
plus64MiB metadata =1,394,319,870bytes. The restart allowance includes16MiB extra
history/RNG per generation; at~0.2MiB growth/1024updates, doubling to20480fits.
The latest measured free space after panel001 is about12.77GB, above the
12,131,738,110byte requirement including10GiB reserve. Recompute actual space and
forecast at freeze/launch; do not admit a partial panel if capacity changes.
All selected/fresh panel001 weights remain protected. Transient optimizer
snapshots alone follow the existing verified successor/completion retirement.

Implement only a new panel/worker namespace supporting this prospective contract.
Reuse qualified immutable training/data/scoring code. Inspect differences and
check horizon/EMA counts, paired roles, paths and capacity arithmetic before
launch. No historical test-suite replay: an additional numerical prerequisite
is warranted only if a changed execution path invalidates existing qualification.
This plan does not authorize modifying frozen panel001 sources or continuing its
retired optimizer checkpoints.
