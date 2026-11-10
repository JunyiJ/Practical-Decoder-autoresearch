# General RSI transfer results

**Neither arm passes the frozen two-domain transfer rule.** Books are nearly tied.
Native has lower mean loss on Python documentation, but the adjusted uncertainty
bound crosses zero. This is a completed, valid, inconclusive transfer comparison.
The accepted equal-time native advantage on the original validation mixture
remains scoped to that mixture. No broader-domain superiority is established.

## Frozen comparison and result

All six retained equal-time time-EMA checkpoints were evaluated without retraining.
Each model scored the same204,800 targets:32 parent excerpts ×3,200 targets in
historical books and the same count in CPython library documentation. Context128,
GPT-2 vocabulary, document resets, FP32 and the qualified MPS scorer were fixed.
Loss is in nats/token; lower is better.

| Paired bundle | Arm | Books CE | Python rawRST CE |
|---|---|---:|---:|
| 1 | native | 5.684493 | 7.265730 |
| 1 | raw_bytes | 5.675113 | 7.287419 |
| 2 | native | 5.659193 | 7.073691 |
| 2 | raw_bytes | 5.662160 | 7.221912 |
| 3 | native | 5.658546 | 7.110846 |
| 3 | raw_bytes | 5.671621 | 7.228098 |
| Mean | native | 5.667411 | 7.150089 |
| Mean | raw_bytes | 5.669632 | 7.245809 |

Differences below are native minus byte. Negative favors native.

| Domain | Pair1 | Pair2 | Pair3 | Mean | Adjusted one-sided upper bound |
|---|---:|---:|---:|---:|---:|
| Books | +0.009380 | −0.002968 | −0.013075 | −0.002221 | +0.049614 |
| Python rawRST | −0.021688 | −0.148221 | −0.117252 | −0.095720 | +0.028495 |

The rule requires all three pairs to favor the candidate, mean improvement of
at least0.02, and a negative adjusted upper bound in **each** domain. Native
fails all three conditions on books and the uncertainty condition on Python.
Byte passes neither domain. The equal-domain macro difference is−0.048971, with
a descriptive99% upper bound of+0.010271. The macro is not an acceptance gate.

The frozen bootstrap uses20,000 PCG64 draws, seed62111000999, paired parent
resampling within each domain and the mean of the three seed-pair loss sums.
The0.9975 domain upper quantile applies Bonferroni correction to four
arm-direction/domain claims at familywise1%. These are conditional document
bounds for the fixed models, not confidence over a population of training seeds.

## Data and scope

- PG-19:32 fixed excerpts from the publisher's50-book validation source pool,
  pinned by GCS generation and checksum. This is historical book prose, with
  publisher substitutions and historical style/bias. It is not the full PG-19
  long-context benchmark or the original RSI test.
- CPython:32 fixed excerpts from293 library `.rst` files at commit
  `60403a5409ff2c3f3b07dd2ca91a7a3e096839c7` (v3.13.0). Raw markup and code
  remain included. This is not a prose-only or executable-code benchmark.
- Acquisition retained345 verified source/license objects,24,234,703bytes.
  The original builder used strict UTF-8 and universal newline translation to
  LF. All50 book sources contained CR; Python sources did not. Raw bytes remain.
- Selection used frozen seed62111000001, deterministic parent order and one
  fixed window per parent. It rejected46 short documents and20 windows with
  overlap against previously selected excerpts. None of the candidates reached
  by selection matched the original train/validation32-token span index.
- Exact32-token overlap exclusion does not rule out shorter, reformatted or
  semantic overlap. It covers scored excerpts, not all text in their source
  documents. Original test data was never opened, including for deduplication.
- Parent resampling does not model shared book authors or dependence among
  documents from one Python repository/version. Results apply to these selected
  excerpts and three existing model pairs. They are not new training confirmations.

## Descriptive diagnostics and hypothesis update

The following analyses were added after launch. They cannot change acceptance.
`token-coverage.json` binds full training-corpus token frequencies to the selected
transfer targets. Unseen target occurrences are0.333% in books and3.299% in Python.
Full-corpus frequency is not actual sampled training exposure, semantic novelty
or evidence of contamination.

`document-diagnostic.json` combines those counts with all original document
scores. Seed-averaged native loss is lower on14/32 books and15/32 Python parents;
byte is lower on the remainder. Document differences range from−0.397 to+0.353
on books and−1.230 to+0.169 on Python. Thus mean native gains are heterogeneous.
The within-domain Pearson correlation between unseen-token fraction and
native-minus-byte loss is+0.560 for books but−0.303 for Python. The conflicting
signs do not support a simple domain-independent rarity explanation. Topic,
markup and token identities can confound these32-parent descriptive correlations.
No causal attribution or significance test is made.

The prelaunch hypothesis was that native's extra updates at equal Mac time would
improve both new domains consistently. The strong claim is not supported by the
frozen rule. This does not prove equivalent models or prove the true transfer
effect is zero. Byte's earlier equal-update advantage and the accepted longer
byte-training result remain valid within their own operating constraints.

The FineWeb and Paloma evidence and applicability limits remain in
[the prelaunch boundary](GENERAL_RSI_TRANSFER_BOUNDARY.md). Their domain-dependent
findings motivated this test; they do not establish its mechanism. No new
literature claim or model improvement is inferred from this inconclusive result.

Next executable action: quantify per-parent influence and technical-format
composition from the existing score files and selected text. Determine whether a
few code/markup-heavy documents account for the mean Python advantage before
qualifying an optimization or representation experiment. Treat that work as
post-result diagnosis, not additional confirmation. Do not expand this holdout,
change its gates, repeat completed prerequisites or launch a size panel without
a discriminating hypothesis and a complete new resource contract.

## Execution and retained evidence

All six scores completed in attempt1. Preparation/loader/evaluation guarded time
was536.710s in total; evaluation used398.956s. Evaluation peak RSS was743,981,056bytes,
and free space at completion was25,400,594,432bytes, above the10GiB reserve.
All three guards finished without failure. No new weights were produced.
All selected weights, comparator weights, raw sources, exclusions and per-document
scores remain retained. All144 first generations and their fixed evaluation from
the equal-time campaign remain unchanged; both arms still show weak generation.
There was no new generation or post-result decoder tuning. No numerical worker
remains active after completion.

Authority and evidence: `tokenizer-evidence/general-rsi-transfer-001/protocol.json`
(SHA256 `cf798226383b22b0bf115d5cea851438b19cddb01eda330f99f88dbc3bb61b89`),
`scoring-admission.json`, `dataset-review.json`, `evaluation-attempt-01/`,
`decision.json`, `completion-review.json`, `token-coverage.json` and
`document-diagnostic.json`. The admission was committed before scoring in
`38ce7927d`. The frozen analysis verified every score inventory and checkpoint
binding before applying the decision rule.
