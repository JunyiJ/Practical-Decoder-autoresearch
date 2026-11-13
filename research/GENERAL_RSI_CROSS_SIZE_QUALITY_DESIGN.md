# Cross-size equal-token quality protocol

## Evidence and question

The selected contrast is existing byte composition versus native input tables,
at D256 depths4,8,16. The reference evidence and competing context finding are
ranked in `GENERAL_RSI_CROSS_SIZE_BOUNDARY.md`. Local equal-token byte quality
improved0.052604nats/token, with42.61% fewer parameters and about1.84× runtime.
Equal-time native won. The completed context256 finding is strong, but its full
cross-size matched-time comparison costs more. ComposerFP32 and prior broader
transfer were inconclusive. None of those old scores enters this fresh panel.

The mechanism predicts dilution of the fixed composer cost as body depth grows.
The native body share rises9.69% to30.03%, and arithmetic parameter saving falls
42.61% to33.01%. Quality may persist, vanish or reverse independently of runtime.
A persistent equal-token benefit would support fixed-recipe transfer over this
range. A loss or reversal would bound that benefit and motivate error analysis.
Either result is useful. Three shallow sizes with large vocabulary tables do
not identify a scaling law or establish larger-LLM relevance.

Primary literature and exact sections already reviewed are recorded in the
boundary: Yang2203.03466v1 §§3–4,6.1, Pagnoni2412.09871v1 §§5.1,5.3/Table2,
Hoffmann2203.15556v1 AppendicesB–C. Their parameterizations, architectures and
budgets differ. FixedLR transfer here is not a test of each size's optimum.

## Fresh controlled panel

- Three fresh paired bundles, each shared across all three sizes. Each has four
  private core roles and three independent extra four-block initializer roles.
  Exact integer roles and serial order are frozen in plan.json before launch.
- Each size/arm receives10,240updates ×1,024targets =10,485,760 sampled targets.
  All18runs use the same horizon. Corpus size is10,001,163 training tokens.
  Exposure is1.048454 corpus-equivalents per run, with random revisits and missed
  positions. It is not a guaranteed epoch. Historical and engineering seeds are
  excluded. No LR screen or warm start enters the panel.
- Width256,8heads,50257GPT2 IDs, context128, original two512-block sampler
  reshaped8×128, ordinaryCE, independent readout, original composer math,
  original Muon/AdamW and mixed internal precision stay fixed. Parameter tensors
  areFP32. The qualified Muon internalBF16 path stays unchanged.
- LR.0025, Muon multiplier.5, AdamW betas(.9,.999),epsilon1e-6,
  globalclip1, independentdecay.00006. Linear cooldown from zero-based update6826 to10239,
  minimumratio.1. H200 EMA startsstep6830, every10steps,342snapshots.
  RAW is secondary. No best-checkpoint selection.
- The first four blocks and output head share the original native initializer
  within a bundle. Extra groups use private independent native draws. Native
  and byte have identical body/head initialization and sampler streams. Added
  groups do not clone or share blocks. Common prefixes are shared across sizes.
- Rotate size order across bundles and alternate arm order as listed in plan.
  Serial order cannot eliminate thermal or system-load confounding. No runtime
  extrapolation to GPUs, no equal-time endpoint claim.

## Frozen quality and interaction decisions

Let delta[d,b] be finalH200 CE(byte) minus CE(native) for depthd and bundleb.
Primary development evaluation covers all108documents and100,416targets. Use
sequential nonoverlapping chunks128, every target after the first once, including
original EOT, never across documents. CorrectedFP32 inference, no autocast or
inference_mode. Preserve per-document loss sums/counts and source means.

Four contrasts form one declared family: mean delta at each of4/8/16 and the
mean interaction delta16 minus delta4. For each, average paired loss sums over
three bundles before100,000 paired document bootstrap draws. Resample the same
108document IDs across all arms/sizes/seeds with the plan's fixed bootstrap seed.
Use token-weighted aggregation within each draw. Report the percentile interval
[0.00125,0.99875], two-sided99.75%, as Bonferroni allocation of1% across four
contrasts. This is approximate bootstrap coverage conditional on the chosen
seeds/documents, not confidence over the population of training seeds. Report
all three seed deltas, range and standard deviation explicitly. Three seeds
cannot estimate population-seed uncertainty reliably. Reused validation remains
a development set, not independent external confirmation.

At a given size accept byte superiority only if all three deltas<0, mean<=−.02,
upper interval<0, and neither FineWeb-Edu nor Wikipedia three-seed mean regresses
by>.05nats/token. Native superiority is symmetric. Otherwise unresolved.
Call a confirmed byte benefit transferred to a larger tested size only if the
fresh reference and that size both pass the byte rule. Do not pool old pairs.

Declare a practical cross-size change only if all three seed interactions have
the same strict sign, absolute mean>=.02, and the two-sided interaction interval
excludes0 in that direction. Positive means byte's relative loss advantage
weakens; negative means it strengthens. Depth8 interaction and RAW analyses are
descriptive. Failure to detect an interaction does not establish equivalence.
Separate significance tests do not establish an interaction.

Broader dataset002 is a prospective narrow probe:8PG19 books and8Python prose
parents,32,768targets, selected without model scores after excluding old64parents
and exact32-token train/validation/selected-excerpt spans. The failed initial
filter remains evidence. The successful manifest uses the existing
`data/raw/rsi-cross-size-eval-001/` payload directory; manifest hashes, not its
folder suffix, identify dataset002. Python paragraph joins are artificial and
exact-span exclusion does not detect paraphrases. Report both domains, every
seed/parent, RAW/H200, and descriptive95% paired parent-bootstrap intervals.
Broader scores do not alter the four primary gates or support universal claims.

## Accounting and diagnostics

The fixed probe is documents0 and1, every1,024updates before the terminal step.
Retain train loss and synchronized update time every step. Plot probe curves
against targets, estimated FLOPs and accumulated synchronized training seconds.
Intermediate points do not replace terminal horizon-matched comparisons.

`general_rsi_cross_size_flops_v1.py` derives dense matrix forward/backward and
Muon Newton–Schulz work from bound parameter shapes and source. One multiply-add
is2FLOPs; ordinary dense forward plus both gradients is3×forward. Attention
executes the full masked square. Byte composition executes130 padded positions
for all1,024 occurrences. Input embedding lookup is not a dense vocabulary
matmul. Separate partial operator estimates from actual wall time. Explicitly
uncounted work includes activations, norms, softmax/CE, lookup/scatter, clipping,
AdamW, decay, averaging, reductions, validation and I/O. Report failures/replays
separately in tokens, operator estimates and time. No6NT or compute-optimal claim.

For runtime report every paired training-time ratio and log-ratio, median update
latency after warmup, parameter counts and peakRSS/MPS. The descriptive dilution
prediction is met if all three ratio16/ratio4<1 and their geometric mean<=.9.
This10% practical threshold has no independent hardware-population significance
claim. Total worker/guard time includes evaluation, serialization and failures.

Every finalH200 endpoint produces the same12prompts ×2decoders =432first outputs.
Use the existing frozen prompts/rubric, greedy andtemperature.8/top-k40,seed42
reset per prompt/decoder,max64newtokens,EOS50256,no rerolls. Keep token journals,
exact text and source/checkpoint hashes. Report repetition, length, termination
and identity-hidden agent relevance/consistency ratings separately by decoder.
Agent ratings are not independent human validation and cannot rescue failedCE.

## Execution and resources

The engineering wrapper, all new-depth full-state replay checks and training-only
scoring oracle are qualified. Integration tests only new worker/endpoint paths.
Keep each failed attempt and all first outputs. Do not repeat closed prerequisites.

One serial worker owns the nonblocking shared research lock. Every slot needs
prior accepted execution reviews and exact source/data/runtime bindings. Stop
requests are checked per update and evaluation/generation chunk. Checkpoint
publication verifies the complete loaded-state digest before retiring its exact
transient predecessor with size/hash/consumer checks and durable deletion journal.
At completion retain RAW/H200 and retire the last transient only after all metrics
and generation are durable. Automatic resume requires a recorded clean boundary
and exact full-state restoration. Uncheckpointed work or partial endpoint output
requires separately audited recovery, preserving failures and first outputs.

The prospective54-hour tranche replaces the initial36-hour planning bound before
scientific outcomes. Full admission includes all18runs,36endpoints, all scoring,
probes,432outputs, checkpoints and30% timing margin. Storage includes both restart
serialization slots, metadata, existing engineering payloads and2GiB failed-write
allowance, plus10GiB reserve and2GiB extra headroom. RSS4GiB/MPS6GiB guards remain.
Use the measured integration cost to refresh admission. No partial panel, reduced
confirmation, relaxed gate or result-dependent schedule is permitted to fit cost.
The original test remains unopened. State and evidence must reflect actual work.
