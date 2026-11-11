# Composer FP32 training-quality comparison

This is the separately human-authorized study
`general-rsi-composer-fp32-quality-001`. Native-horizon confirmation and all144
blinded generation ratings are complete. The original failed numerical
qualification is unchanged. No batch-reuse arm enters this study.

## New integration evidence

`tokenizer-evidence/general-rsi-composer-fp32-quality-001/qualification/` contains
the frozen fixture, installed-source audit, first-step routing, FP64 diagnostics,
complete restart checks, worker log and guard. Both arms passed in68.91seconds.
All initial weights and optimizer groups were identical. At the first update,
all momentum/Adam states and all noncomposer parameters were exactly equal.
Only the nine composer Muon matrices use FP32 instead of BF16 Newton–Schulz
arithmetic. Normalization clamp1e-7, five iterations, coefficients
(3.4445,-4.775,2.0315), momentum.95, Nesterov, shape scaling, independent decay,
AdamW and all model/data policies stay unchanged. The installed torch2.14.1
source is hash-bound. No global optimizer patch or batch reuse is used.

Both arms restored the entire step35 state exactly and replayed to step40 with
exact final parameters, optimizer, two EMA snapshots, sampler/RNG and history.
Nonfinite momentum and wrong-arm checkpoint loads were rejected before mutation.
Checkpoint size was247901617bytes for either arm. Peak process RSS was1.47GB.
Median synchronized whole-update times were.62865s existing and.62331s FP32;
this short fixture does not establish a speed win. FP32-versus-CPU-FP64 direction
relative errors on nine first clipped training-gradient inputs ranged from
7.30e-7 to2.63e-5. These are diagnostics only. The original failed rank-one case
was not rerun or reclassified. Corpus, scoring and generation qualifications
were reused without repeating their completed numerical tests.

## Frozen scientific contract

`tokenizer-evidence/general-rsi-composer-fp32-quality-001/plan.json` binds all
sources/runtime, roles, complete costs and acceptance rules before training.
Three fresh paired bundles compare existing byte with composer FP32. Each arm
gets10240updates of1024targets, LR.0025, final-third linear cooldown to.1 and
H200 averaging every10updates from6830through10240. RAW is secondary.
Initial weights and complete sampler streams match within each pair. Bundle2
runs FP32 first; bundles1/3 run existing first. No historical score is pooled.

The symmetric superiority rule requires all three paired H200 deltas to favor
the winner, mean gain at least.02nats/token, one-sided99% paired-document
bootstrap upper bound below zero in that direction, and no source mean
regression above.05. Resample all108validation document IDs10000times with
PCG64seed62111118999, averaging paired seed loss sums before resampling and
weighting by sampled target counts. Bounds are conditional on these documents
and seeds. Failure of both directions is inconclusive, not equivalence.
No FP64 or cross-precision arithmetic threshold can veto training quality.

Score all100416validation targets. Keep test unopened. Use all12fixed prompts,
greedy and temperature.8/top-k40, seed42 per prompt/decoder, max64tokens,
no rerolls, on all six H200 endpoints. Save all144 first outputs and rate them
blind to arm/seed with the existing rubric. Report actual annotator identity,
separate decoders, per-seed statistics, repetition, EOS and truncation. Generation
cannot rescue a failed CE gate.

## Resource and recovery contract

One serial local worker. Fresh confirmation costs61440updates. The slower
fixture mean predicts39295.97training seconds. A1.6timing multiplier plus7200s
for I/O, evaluation, generation and failure overhead forecasts70073.55s, below
the frozen24h total guard. Each slot has a cumulative4h guard across at most
two implementation/resource attempts. Genuine training divergence is a recorded
outcome; it does not permit changing seeds or learning rate.

At admission, actual free space was20145291264bytes. Required future free space
is18253611008bytes: twelve64MiB final weight caps, two512MiB restart slots
including serialization overlap,256MiB metadata,5GiB failed-write allowance and
10GiB reserve. The two retained qualification snapshots and all earlier models
already consume measured disk space. All12fresh native weights remain retained.
No cleanup was needed. Recheck the complete remaining requirement at every
launch; exact audited cleanup applies to any new capacity constraint.

Every1024updates, publish/hash/read a complete restart before retiring its
predecessor through the qualified exact journal and consumer check. Retain final
RAW/H200 weights and failure/generation evidence. Include sampling, transfer,
forward/backward, clipping, optimizer, averaging and batch hashing in synchronized
update timing. Durable start-journal cost is included equally in both arms;
end-journal write, checkpoints, validation and generation are outside that clock
and inside worker time. Retain lost-work charges through same-boot supervisor
receipts and the qualified interval accounting. Endpoint recovery must preserve
all first generation prefixes and needs a separately identified audit.

## Hypothesis and interpretation

The local evidence and primary-literature rationale remain in
`GENERAL_RSI_COMPOSER_FP32_QUALITY_NEXT.md`. Reduced rounding sensitivity could
improve, worsen or leave quality unresolved. Integration/restart success is
engineering evidence only. A positive fresh result would justify a later
practical comparison; a negative result rejects this recipe at this setting;
an inconclusive result does not prove harmlessness. This byte-only matched-update
ablation cannot establish superiority over native or an equal-time advantage.
