# Input byte-composition applicability question; no model or panel admission

2026-11-07. Applied evidence-grounded-research after the original TOP model-math
fixture closed failed. This is a separate lexical-input question. It does not
relax, repair or rerun the TOP exact-zero gate, recapture its missing MPS cases,
change its controls or alter the prospective future_two panel/native incumbent.
No new neural/model/optimizer/corpus/tokenizer/scoring/oracle/RNG calls here.

## Local facts and contrary findings

Read TOKENIZER_EMBEDDING_INPUT_DROPOUT_RESULTS.md, input-capacity-boundary-001/
evidence-review.md, tokenizer-capacity-boundary-001/evidence-review.md and the
architecture/exploration plans. Existing original measured facts, not rescored:

- Native input table has50,257x256=12,865,792 elements of28,492,416 total.
- Training267,688tokens observes10,653IDs; validation2.7919% targets have
  train-unseen IDs. That is a target statistic, not the fraction of contextual
  loss caused by unfamiliar inputs; output rarity remains an alternative cause.
- Input dropout.1 loses all3 fresh seeds, mean+.014701832892nat/token. This is
  contrary to generic input-regularization claims.
- Observed-type effective rank252.93; rank128covariance energy56.58%. Frequency
  weighting gives80.93%, essentially the Gaussian null81.02%. No learned low-rank
  signal is established. Initial-table restoration costs+.007688 mean CE; a new
  random table costs+2.796917. Code identity and learned changes both matter.
- Old tied/matched-table recipes lost and were confounded; smaller8K prefix
  quality later lost+.023745683009BPB. No replay, pooling or fresh tokenizer sweep.

Re-read ALBERT1909.11942v6§3.1/4.4/Table3: its non-shared embedding ablation does
not support reducing embedding dimension for better local quality. The prior
input-capacity review had already read this and adaptive-input1809.10853v3.
Their retrieval is not novel evidence. Uniform low-rank/frozen-input quality
panels and an adaptive-input replay are NOT admitted by this boundary.

## New primary evidence, actually retrieved/read

[Kim etal.,1508.06615v4](https://arxiv.org/pdf/1508.06615v4), Model/CharCNN equations5–6,
Highway equation8, Experimental Setup/Optimization, Tables1–3/7. Inputs are
composed from within-word characters while predictions remain word-level. Matched
PTB word/character baselines report97.6/92.3 and85.4/78.9 perplexity; the highway
ablation matters (large84.6 without versus78.9 with two layers). This is not
proof for a bare convolution. Their LSTM, characters, word units,SGD,preprocessing
and1M-or-larger token corpora differ from our4block MLA/GPT2 bytes/267,688tokens/
Muon-AdamW. Unknown words are replaced during preprocessing, limiting direct OOV
transfer claims. No published gain is claimed locally.

## Mechanistic inference and discriminating question

Investigate composing EACH existing GPT2 token's raw bytes into oneD256 input
vector, retaining existing token IDs/segmentation, ordinary-text policy, context
and untied next-token output. The possible information is lexical byte-fragment
sharing across IDs, including IDs not observed as training inputs. Nonlinear
composition need not impose a uniform linear rank128bottleneck. Those statements
are hypotheses/extensions of the paper, not observations or a reproduction.

The NEXT question is only whether the exact public vocabulary has a coherent,
bounded raw-byte layout (including partial UTF8 pieces, whitespace, leading-byte
markers and special ID50256) and what composer working-set bounds follow. No
replacement-tokenizer fitting/normalization or corpus/holdout encoding is allowed.
If this mapping is not exact and bounded, reject the component before any model.
If it is feasible, it still supplies no evidence that the quality hypothesis wins.

A future scientific comparison would need BOTH genuine native and a same-composer
nonlexical-code control to distinguish byte sharing from parameter count or
initialization. A global byte-alphabet permutation preserves all substrings and
is therefore NOT a discriminating random-code control. A prospective code control
would need independently assigned unique token codes with matched lengths and
explicit collision/special-ID rules; no control, role, seed or width is assigned
now. A gain versus native alone would not isolate lexical sharing. A failure
versus the same-composer control would falsify the claimed byte-content benefit.
Full-rank native controls, own identities/initialization/grouping and all existing
correctness gates remain required prospectively. No old score or partial TOP
proof is a new control. No changed reference mode or tolerance is authorized to
rescue TOP through this question.

## Cost and ranking

Only bounded source/metadata feasibility is admitted. Read at most6new relevant
source versions<=256KiB each and512KiB source+notes; inspect at most8MiB existing
public vocabulary assets using a NEW read-only stdlib metadata plan/guard. CPU
forecast30s/256MiB; hard120s/4GiBRSS/reserve8GiB/2MiB output plus2GiB headroom.
No model/component numerical test or scientific panel is admitted.

A prospective composer parameter formula is B*e + sum_k(c_k*k*e+c_k) plus any
projection/highway parameters;B includes the explicit byte/boundary/pad/special
alphabet. Total would be28,492,416-12,865,792+Pcomposer. No dimensions are selected.
Training activation cost can scale with1024*(max_token_bytes+boundaries)*e and
filter widths/channels; fewer weights do NOT prove a time or RSS benefit.
Exact all-parameter gradients/optimizer states/EMA/CPU staging, cache behavior,
serialization overlap and complete equal-tuning/fresh-confirmation time/output
high-water remain uncomputed. Consequently no quality budget/freeze/launch.

Rank next investigative work: (1) exact byte-layout/source-cost feasibility;
(2) defer frequency-adaptive inputs without a new discriminating local rationale;
(3) defer context/depth interactions until their existing contrary evidence and
full cost justify a distinct question. TOP implementation qualification is closed
failed, scientific quality unknown. Carry exploration allocation forward; this
source/literature work is not numerical quality exploration. Do not create tiny
arbitrary model probes or source/optimizer ledgers to fill the allocation.
