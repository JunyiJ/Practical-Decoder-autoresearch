# Input capacity boundary after the complete dropout result

## Local observations

The full input-dropout panel is CLOSED_NOT_ADOPTED. Original final review and
read-only closure are in `../embedding-input-dropout-preparation-001/`.
All12trajectories passed independent acceptance; both arms selectedLR.00125.
Fresh candidate-minus-control CE deltas are +.012384021743,+.016580617333,
+.015140859599; mean+.014701832892. All strict gates fail. Keep learned final
scale/bias,epsilon1e-6 and input dropout0. All16selected/fresh RAW/H200 payloads
(4,474,027,344B) are retained. No old scores are pooled into this result.

`analysis001.py`, `plan.json`, `result.json` and `guard.json` analyze all six fresh
H200s, using train-only frequency weights and independent Gaussian controls.
For control input tables, the observed-type effective rank is252.93 and rank128
retains56.58% covariance energy. Frequency weighting raises rank128energy to80.93%,
but the same-frequency Gaussian retains81.02%. That concentration therefore does
not demonstrate learned low-rank structure. The readout differs more strongly
from the null; neither matrix covariance proves a benefit from a trained bottleneck.
Both arms' last saved online validation interval still improves; this does not
identify an optimal horizon. Masked sampled train CE is not comparable to an
unmasked full-train generalization-gap measurement.

`initial-counterfactual001` reconstructs exact native CPU initialization with
already-used seeds, preserving CPU RNG and original weights. Input-table
frequency-weighted relative changes are8.76–8.86% for controls and9.33–9.46% for
dropout. On all three controls, restoring the initial table worsens CE by
.010057/.005436/.007572, mean+.007688; a new native random table worsens by
2.899324/2.709251/2.782176, mean+2.796917. These explicit in-memory inference
perturbations show code identity and learned changes matter. They do not estimate
training with frozen inputs, a low-rank table or another tokenizer. No original
validation model was rescored, no model was exported and the holdout stayed closed.

The earlier table-sharing study (`../../TOKENIZER_TABLE_RESULTS.md`) is contrary
evidence: its matched untied control lost.102667nat versus legacy and tying added
.030508. Initialization/readout scaling/optimizer changes confound the total;
parameter savings alone did not justify adoption. Historical weights retired by
human permission are resolved through later manifests, never restored or replayed.

## Primary literature actually read

[ALBERT1909.11942v6](https://arxiv.org/pdf/1909.11942v6),§3.1/4.4,Table3:
factorization separates lexical and hidden dimensions. In the non-shared ablation,
smaller embeddings lose downstream accuracy;128 is preferred only in the shared
configuration. §4.8/Figure2b/Table8 also reports harm from dropout. Its MLM/SOP,
30Kvocabulary,16GB corpus,LAMB/TPU and downstream accuracy are far from our small
causal setup. It supplies a controlled alternative, not a local rank recommendation.

[Baevski/Auli1809.10853v3](https://arxiv.org/pdf/1809.10853v3),§3/4.4/5.2/5.3,
AppendixA/Table7: input capacity can be frequency-dependent; the head also has a
projection. Fixed64d inputs help one adaptive-output WikiText103 setup, while
BillionWord benefits from more capacity. Projection sharing sometimes hurts.
Their word vocabularies,100M/768M-token data,adaptive softmax,dropout and hardware
are different. A smaller uniform GPT2 input factorization is not that experiment.

## Decision and expected information

Do not admit a rank128/frozen-input quality panel from these covariance/movement
facts. A future factorization needs native and full-rank controls, genuine
initialization/optimizer/width rules and complete feasibility; replacing weights
or freezing a table is not already validated training. No new scientific seed or
panel is assigned. The higher-priority bounded question is vocabulary capacity:
our267,688-token corpus observes10,653 of50,257IDs;2.79% of validation targets use
train-unseen IDs. Investigate a lossless smaller-codec tradeoff and a common-byte
metric before comparing any models. This is structural preparation, not a quality
result or a relabeled quota run. See `../tokenizer-capacity-boundary-001/`.
