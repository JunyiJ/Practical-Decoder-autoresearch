# Byte-input quality question after implementation qualification

This is an evidence update, not a scientific freeze, seed reservation, quality
result or launch. The resource pilot supplies timing/capacity evidence only.
Its CE must not choose a variant, learning rate, seed or scientific threshold.

## Measured local basis

The original `input-capacity-boundary-001/result.json` records 267,688 training
tokens and 10,653 observed IDs. Its input covariance summaries do not show a
learned low-rank signal: observed-type effective rank is approximately253 and
rank128 energy approximately56.6%, close to its Gaussian control. The original
`initial-counterfactual001.json` separates restoring the original input codes
from replacing them with newly random codes. These are existing analyses, not
new model gains or evidence that freezing inputs is beneficial.

`TOKENIZER_EMBEDDING_INPUT_DROPOUT_RESULTS.md` and its original artifact-bound
results report worse H200 validation CE on all three fresh seeds with input
feature dropout.1 (mean +.014701832892 nat/token). This contradicts a generic
claim that fewer effective input degrees of freedom must improve quality.
Previously losing table/low-rank/tokenizer families remain closed.

New `training-review001` and `state-review003` establish finite implementation
correctness, while `inference-review001` and `cache-review001` qualify the actual
export/scoring/cache consumers. None is scientific quality evidence. The new
resource pilots are also excluded from quality selection and confirmation.

## External evidence and applicability

Re-read [Kim et al., arXiv1508.06615v4](https://arxiv.org/pdf/1508.06615v4), equations5–8,
Optimization, Table2 and Table7, including the negative MLP ablation and the
Discussion on learned n-grams. The paper composes character inputs with a CNN
and highway layers while retaining word-level predictions. Highway layers help
in its experiments; a simple MLP replacement performs worse. The learned
filters do not generally correspond to valid morphemes. Its LSTM, word units,
characters, SGD, corpus sizes and OOV preprocessing differ from our byte-BPE,
MLA, Muon/AdamW and small corpus. These results justify a test, not a local gain
or an isolated morphological explanation.

## Mechanistic inference and falsifiable outcomes

At fixed GPT2 IDs/segmentation/ordinary next-token output, nonlinear composition
may share useful lexical fragments across input IDs without a uniform linear
rank bottleneck. Compare raw-byte composition with both the genuine full-rank
native model and the same composer using unique length-matched nonlexical codes.
The latter matches architecture, parameter counts and composer initialization,
but not every byte marginal; it tests usefulness of this lexical coding, not
pure substring-sharing causality.

Raw bytes failing to beat the nonlexical control would fail to support the
byte-content explanation. A win against the control but a loss against native
would support lexical utility inside this composer without a practical model
improvement. A native-only win cannot isolate the lexical mechanism. A loss is
a scientific result, not grounds to alter seeds, tolerances or controls.

## Next decision

Use measured pilot costs to prepare the complete predeclared three-arm quality
comparison, including balanced tuning where justified and separate fresh
confirmation. The earlier18-trajectory output envelope is a prospective design,
not a frozen panel. Before any scientific dispatch, fix selection/outcome rules,
all roles, complete retained-output/serialization/memory/time costs, multiplicity
handling and conditional-text uncertainty scope. Preserve1200 updates and the
existing scientific data/averaging/holdout boundaries. Resolve a concrete capacity
shortfall through authorized exact cleanup, rather than reducing confirmations.
