# Composer FP32 training quality: inconclusive

The six-run study completed without execution failures. Neither arithmetic
recipe meets the frozen superiority criteria. Do not adopt FP32 as a validated
quality improvement, and do not interpret this as equivalence or harmlessness.
The original failed numerical qualification remains unchanged. Its0.01%
arithmetic threshold was not used in this quality decision.

## Fresh paired quality results

Each run used10240updates,1024targets/update, LR.0025, the same final-third
cooldown and342 H200 snapshots. Initial weights and all10240 sampler batch
hashes match within each pair. All108validation documents100416targets were
scored. Test remains unopened. No historical scores entered confirmation.

| Fresh pair | Existing H200 CE | FP32 H200 CE | FP32 minus existing | RAW delta |
|---|---:|---:|---:|---:|
| 1 | 5.151609445 | 5.142816763 | −0.008792682 | −0.008561796 |
| 2 | 5.142531945 | 5.140645409 | −0.001886536 | −0.001813297 |
| 3 | 5.132913271 | 5.135397764 | +0.002484493 | +0.003239597 |
| Mean | 5.142351554 | 5.139619978 | −0.002731575 | −0.002378498 |

FP32's mean gain is below the required0.02nats/token, one pair has the opposite
sign, and the one-sided99% paired-document bootstrap upper bound is
+0.002311393. The reverse-direction upper bound is+0.007577924. Both directions
fail the sign, mean-gain and bootstrap gates. Both pass the source guard.
FP32-minus-existing source means are−0.003968287 on FineWeb-Edu and+0.002221807
on Wikipedia. Bounds are conditional on these108documents and three seeds.

The new analysis adapter passed synthetic tests for symmetric winners,
inclusive mean thresholds, mixed/zero signs, uncertain bounds, source guards,
unequal target-count bootstrap weighting, invalid inputs, missing endpoints and
144-row generation aggregation. It reuses the qualified bootstrap calculation.
The frozen training source, seeds and criteria were not edited.

## Fixed generation diagnostic

All144 first completions were retained without rerolls. The same Codex GPT-6
agent read every full continuation with arm, seed, checkpoint and decoder labels
hidden, saved all ratings, then joined the mapping. These are exploratory agent
ratings, not independent human validation. Consistency is not exhaustive factual
accuracy. They are not calibrated comparisons with ratings from older studies.

Each row below covers36outputs,12prompts ×3seeds. R and C are equal-weight
prompt means over assessable0/1/2ratings. Unassessable outputs are excluded from
those means and reported separately. A score of1 means fragmentary or ambiguous,
not satisfactory performance.

| Recipe / decoder | R | C | C unassessable | Repeated trigrams | Terminal loops | EOS | Token cap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing / greedy | 0.778 | 0.958 | 2/36 | 0.538 | 13/36 | 0/36 | 36/36 |
| FP32 / greedy | 0.750 | 0.833 | 3/36 | 0.554 | 14/36 | 0/36 | 36/36 |
| Existing / sampled | 0.750 | 0.889 | 0/36 | 0.0197 | 0/36 | 3/36 | 33/36 |
| FP32 / sampled | 0.861 | 0.889 | 0/36 | 0.0220 | 0/36 | 0/36 | 36/36 |

Sampled means temperature.8/top-k40, seed42 reset per prompt/decoder. Maximum
length was64tokens. Mean content length was63.33 for existing sampled outputs
and64 for the other groups. No unexpected early stops occurred. Only6/144
outputs received relevance2; none received consistency2. The apparent sampled
relevance difference is a small exploratory rubric result, not a validated
quality improvement. It cannot rescue the failed CE superiority gates.

Per-seed R/C prompt means, in seed-bundle order1,2,3:

- Existing greedy: R[.750,.833,.750], C[1.000,1.000,.917].
- FP32 greedy: R[.667,.750,.833], C[.909,.800,.917].
- Existing sampled: R[.833,.667,.750], C[.833,.917,.917].
- FP32 sampled: R[.750,.917,.917], C[1.000,.833,.833].

The machine-readable summary includes all ordinal counts, assessable counts,
per-seed repetition/length/termination statistics and unassessable fractions.
All prompt-level texts, evidence quotes and rating reasons remain available.

## Cost, retention and scope

The six workers used38546.633s synchronized training time and39925.071s total
worker time, below the86400s guard. Mean training time was6434.690s existing and
6414.187s FP32. This serial run does not establish a hardware speed advantage.
All12final RAW/H200 weights, two qualification restarts, first generation files,
logs, configs, hashes and receipts remain retained. Obsolete intermediate
restarts were retired only through the qualified exact publication journal.
No additional training worker is active at completion.

This was a byte-only arithmetic ablation at one model, corpus, LR and horizon.
It does not establish superiority over native, an equal-time advantage, transfer,
or a general statement about FP32 optimizers. Better FP64 agreement in the new
integration fixture did not translate into a validated quality gain here.

Evidence: `tokenizer-evidence/general-rsi-composer-fp32-quality-001/decision.json`,
`analysis-check.json`, all six `slot-*/review.json`, and
`generation-evaluation/{preparation,annotations,annotation-receipt,summary}.json`.
The protocol and primary-literature rationale remain in
`GENERAL_RSI_COMPOSER_FP32_QUALITY_PROTOCOL.md` and
`GENERAL_RSI_COMPOSER_FP32_QUALITY_NEXT.md`.

## Retained-history diagnostic and next action

A post-hoc analysis of the new logs found mean paired training-loss differences
between−0.002483 and+0.000557 across the three update windows and three pairs.
Eight-document RAW probes gave mixed effects by seed and stage, including
−0.038689 at step3413 in pair1 and+0.040448 at step6826 in pair2. These8586-target
probes are not full validation endpoints and were not used as selection gates.

H200 improved each RAW endpoint by0.003834–0.008015nats. RAW and H200 retain the
same favorable/favorable/unfavorable paired ordering. Averaging therefore does
not explain away the failed three-pair sign gate. Clipping occurred in43.18–
61.66% of updates across runs. FP32 reduced the clipping fraction slightly in
all pairs, but that did not imply a validated endpoint gain. No numerical or
resource failure was found to repair. See `history-analysis.json` for the exact
log hashes, pairwise probes, window means and clipping statistics.

The human-directed sequence through this FP32 study is complete. Keep the
validated native long-horizon recipe and the existing byte implementation;
this study supplies no replacement. Do not repeat the closed FP32 study or alter
its gates. Next, form an evidence-grounded research boundary around the remaining
observed generation failures: inspect retained repetition/termination traces and
primary literature before deciding whether a new objective, model or data test
is justified. No new training panel is admitted by this result or diagnostic.
