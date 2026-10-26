# Vocabulary output-bias quality results

Vocabulary output-bias comparison completed 42 physical trajectories and 84 correlated raw/H200 outputs, independently reviewed. Common warmup 0 updates. Frozen eligible winner: incumbent_unchanged. Fresh H200 width64/256 deltas versus tuned no-bias baseline: learned-zero +0.001232/+0.003492; learned train-unigram -0.004870/+0.008795. Retain the prior incumbent under the complete frozen rule.

Historical source `3013f45a37cd5889bf340be54cf027ccc91a30a8`. Result `kb_4155e219df0a530c36865599863bd0eb55a28bfb4e227486601ac246090a8d89`. All 84 checkpoint hashes/payloads, requests/results, full coverage, optimizer/RNG/schedule/H200 contracts, frozen selection and complete report/bootstrap were independently reviewed. No training/evaluation rerun or holdout access.

Controller active time 3,398.87 seconds; output before review 687.04 MiB. [Frozen protocol](OUTPUT_BIAS_PROTOCOL.md).

## Interpretation and frozen decision

Retain the no-bias, peak-LR 0.005 incumbent. Screening chose no-bias and learned-zero at LR 0.01 (the fixed upper boundary), and train-unigram at LR 0.005 (interior). Train-unigram was best, but its mean gain against tuned no-bias was 0.004951322, below the required 0.005. Both screening seeds improved; the threshold still failed. Retention was frozen before confirmation. No boundary extension, threshold relaxation or alternative promotion is permitted.

On fresh seeds 2351/2352/2353, train-unigram improves the width-64 mean versus tuned no-bias by 0.004870, but seed 2352 worsens by 0.002651. Its negative conditional text interval does not estimate training-seed uncertainty. At width 256 it worsens every seed versus tuned no-bias, by 0.005533 / 0.010155 / 0.010696. Learned-zero also worsens every width-256 seed versus tuned no-bias.

The direct unigram-minus-zero contrast reverses across widths: mean −0.006102 at width 64 (every seed better), versus +0.005303 at width 256 (every seed worse). These selected recipes use different LRs, so this is not an isolated initialization effect. The fixed-LR screening contrasts remain separate.

The tuned no-bias LR 0.01 recipe improves every width-256 seed versus fixed LR 0.005, with mean −0.008625 and negative text intervals. At width 64 its mean is +0.000244 with mixed seeds. This width-specific LR result does not justify adoption or retrospective reselection. Learned-zero's width-256 gain versus the fixed incumbent similarly does not override its loss versus tuned no-bias. Train-unigram versus the fixed incumbent improves every width-64 seed, but its mean gain 0.004626 is too small; width 256 is mixed and nearly neutral (+0.000170).

All 42 trajectories are distinct. There are no reference aliases: six fixed-incumbent confirmation runs are fresh controls, and historical controls are not pooled. Raw/H200 outputs from each trajectory are correlated. Measured training-time differences below 0.6% versus tuned no-bias are descriptive, not speed equivalence. Equal 3,000 updates and 3,072,000 targets do not imply an isolated mechanism, optimal muP transfer or inference benefit.

## Fresh confirmation means

Seeds 2351/2352/2353; 111,538 validation targets once each. H200 primary; raw correlated diagnostic.

| Family | Selected peak LR | H200 CE64 | H200 CE256 | Raw CE64 | Raw CE256 |
|---|---:|---:|---:|---:|---:|
| none | 0.01 | 1.513881 | 1.423426 | 1.518721 | 1.436052 |
| learned_zero | 0.01 | 1.515113 | 1.426918 | 1.519072 | 1.439189 |
| learned_train_unigram | 0.005 | 1.509011 | 1.432221 | 1.512911 | 1.446010 |
| incumbent_reference | 0.005 | 1.513636 | 1.432051 | 1.517863 | 1.445333 |

## Paired contrasts

Positive is worse. Intervals use 4,096-token blocks; all three block sizes were audited.

| Endpoint | Width | Contrast | Mean delta | Seed deltas | Conditional 95% text interval | Primary support |
|---|---:|---|---:|---|---|---|
| H200 | 64 | learned_zero minus none | +0.001232 | +0.001622 / +0.003736 / -0.001662 | [-0.000785, +0.003336] | False |
| Raw | 64 | learned_zero minus none | +0.000352 | +0.000187 / +0.003502 / -0.002634 | [-0.001538, +0.002230] | Diagnostic |
| H200 | 64 | learned_train_unigram minus none | -0.004870 | -0.002768 / +0.002651 / -0.014494 | [-0.008362, -0.001841] | False |
| Raw | 64 | learned_train_unigram minus none | -0.005810 | -0.004926 / +0.002196 / -0.014699 | [-0.008876, -0.003204] | Diagnostic |
| H200 | 64 | learned_train_unigram minus learned_zero | -0.006102 | -0.004390 / -0.001085 / -0.012832 | [-0.010487, -0.002062] | True |
| Raw | 64 | learned_train_unigram minus learned_zero | -0.006162 | -0.005113 / -0.001306 / -0.012065 | [-0.009841, -0.002607] | Diagnostic |
| H200 | 64 | none minus incumbent_reference | +0.000244 | -0.002288 / -0.003362 / +0.006384 | [-0.001510, +0.001933] | False |
| Raw | 64 | none minus incumbent_reference | +0.000858 | -0.000729 / -0.003240 / +0.006542 | [-0.001246, +0.003176] | Diagnostic |
| H200 | 64 | learned_zero minus incumbent_reference | +0.001476 | -0.000666 / +0.000374 / +0.004721 | [-0.001162, +0.004226] | False |
| Raw | 64 | learned_zero minus incumbent_reference | +0.001209 | -0.000542 / +0.000262 / +0.003908 | [-0.001628, +0.004165] | Diagnostic |
| H200 | 64 | learned_train_unigram minus incumbent_reference | -0.004626 | -0.005056 / -0.000711 / -0.008111 | [-0.007424, -0.002068] | False |
| Raw | 64 | learned_train_unigram minus incumbent_reference | -0.004952 | -0.005655 / -0.001045 / -0.008157 | [-0.007472, -0.002755] | Diagnostic |
| H200 | 256 | learned_zero minus none | +0.003492 | +0.004159 / +0.001323 / +0.004993 | [+0.000597, +0.006312] | False |
| Raw | 256 | learned_zero minus none | +0.003137 | +0.004641 / +0.000885 / +0.003884 | [-0.000695, +0.006845] | Diagnostic |
| H200 | 256 | learned_train_unigram minus none | +0.008795 | +0.005533 / +0.010155 / +0.010696 | [+0.006440, +0.011160] | False |
| Raw | 256 | learned_train_unigram minus none | +0.009958 | +0.006522 / +0.014142 / +0.009211 | [+0.007274, +0.012797] | Diagnostic |
| H200 | 256 | learned_train_unigram minus learned_zero | +0.005303 | +0.001374 / +0.008833 / +0.005703 | [+0.002000, +0.008746] | False |
| Raw | 256 | learned_train_unigram minus learned_zero | +0.006822 | +0.001881 / +0.013257 / +0.005327 | [+0.002621, +0.011227] | Diagnostic |
| H200 | 256 | none minus incumbent_reference | -0.008625 | -0.009408 / -0.002943 / -0.013524 | [-0.011523, -0.005812] | True |
| Raw | 256 | none minus incumbent_reference | -0.009281 | -0.009751 / -0.003284 / -0.014809 | [-0.012575, -0.006190] | Diagnostic |
| H200 | 256 | learned_zero minus incumbent_reference | -0.005133 | -0.005249 / -0.001620 / -0.008531 | [-0.007960, -0.002526] | True |
| Raw | 256 | learned_zero minus incumbent_reference | -0.006144 | -0.005110 / -0.002398 / -0.010924 | [-0.009384, -0.003029] | Diagnostic |
| H200 | 256 | learned_train_unigram minus incumbent_reference | +0.000170 | -0.003874 / +0.007213 / -0.002828 | [-0.002731, +0.003052] | False |
| Raw | 256 | learned_train_unigram minus incumbent_reference | +0.000677 | -0.003230 / +0.010858 / -0.005597 | [-0.002674, +0.003892] | Diagnostic |

## Fixed-LR screens

| Contrast | H200 mean delta | Two seed deltas | Raw mean delta |
|---|---:|---|---:|
| LR0.0025: learned_zero minus none | +0.003800 | +0.003315 / +0.004285 | +0.003675 |
| LR0.0025: learned_train_unigram minus none | -0.003949 | +0.000357 / -0.008256 | -0.005368 |
| LR0.0025: learned_train_unigram minus learned_zero | -0.007749 | -0.002959 / -0.012540 | -0.009043 |
| LR0.005: learned_zero minus none | +0.000064 | +0.001753 / -0.001626 | +0.000879 |
| LR0.005: learned_train_unigram minus none | -0.008938 | -0.008939 / -0.008936 | -0.009585 |
| LR0.005: learned_train_unigram minus learned_zero | -0.009001 | -0.010693 / -0.007309 | -0.010464 |
| LR0.01: learned_zero minus none | +0.001622 | +0.004151 / -0.000907 | +0.000309 |
| LR0.01: learned_train_unigram minus none | -0.003103 | +0.003804 / -0.010011 | -0.004063 |
| LR0.01: learned_train_unigram minus learned_zero | -0.004725 | -0.000346 / -0.009104 | -0.004371 |

## Measured training costs

| Family | Width | Training seconds | Wall seconds | Ratio versus tuned none | Sampled RSS MiB |
|---|---:|---:|---:|---:|---:|
| none | 64 | 54.50 | 59.02 | 1.000 | 474.74 |
| none | 256 | 125.16 | 133.76 | 1.000 | 483.41 |
| learned_zero | 64 | 54.67 | 59.13 | 1.003 | 476.30 |
| learned_zero | 256 | 125.89 | 134.25 | 1.006 | 473.20 |
| learned_train_unigram | 64 | 54.64 | 59.13 | 1.003 | 476.70 |
| learned_train_unigram | 256 | 125.86 | 134.65 | 1.006 | 485.28 |
| incumbent_reference | 64 | 54.48 | 58.95 | 1.000 | 474.61 |
| incumbent_reference | 256 | 125.62 | 134.47 | 1.004 | 483.30 |

## Limits

- Adaptive development on reused validation; conditional text intervals are not seed-population uncertainty, independent holdout or adjusted significance.
- 42 physical trajectories yield 84 correlated raw/H200 endpoints. There are no reference aliases; all six fixed-incumbent confirmation runs are distinct, with no historical pooling.
- Only the frozen eligible width64 winner can be adopted; no alternate fallback, larger-width reselection, grid extension or completed-trial rerun.
- Equal finite LR tuning; selected-recipe contrasts include LR choice. A tuned none-only gain is attributed to LR selection, not output bias. Fixed-LR screens are separate.
- Equal 3,000 updates and 3,072,000 targets, common reviewed warmup/cooldown/H200, independent decay .00006. Inherited input scale is fixed. Learned offsets add V scalar AdamW base-rate parameters after existing width division. Training-unigram initialization uses only tokenized training-prefix counts, Laplace1 centered log, bound to corpus/alphabet/split/count/initialization hashes. The existing full-corpus character alphabet is disclosed; heldout frequencies/losses are never used for initialization. Prefix counts are not sampler-weighted targets. Learned-unigram versus learned-zero is a separate contrast. Shared active initialization/RNG and existing width factors do not establish an identified mechanism or optimal muP transfer.
- Measured training time is separate from quality; no equal-compute, serving-speed or peak-accelerator-memory claim.
- Final H200 tensors equal saved raw-parent shadows; endpoint evidence cannot reconstruct every historical snapshot. Earlier MPS reproducibility failures remain preserved without an identified cause.
- H200 checkpoints are inference-only; resume optimizer training only from their raw parent.
