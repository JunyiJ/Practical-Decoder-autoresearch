# Transfer diagnosis: document influence and formatting

The inconclusive transfer decision is unchanged. Existing document scores show
that native's mean advantage on Python documentation is concentrated in a few
parents. This analysis is post-result diagnosis, with no model retraining.

## Measured influence

Native-minus-byte loss is negative when native predicts better. All parents have
3,200 scored targets and all three original seed pairs contribute to each mean.

| Domain | Mean difference | Median parent difference | Leave-one-parent-out mean range |
|---|---:|---:|---:|
| Books | −0.002221 | +0.006284 | −0.013670 to +0.010516 |
| Python rawRST | −0.095720 | +0.027807 | −0.104270 to −0.059117 |

The `operator.rst` and `stdtypes.rst` excerpts contribute −0.074210 to the
original Python mean, about 78% of its net native advantage. Their grid-table
lines occupy 80.5% and 62.8% of decoded characters. Removing these two parents
would leave a mean of −0.022944; removing the four largest native-favoring
parents would reverse it to +0.004353. These are explicitly outcome-selected
sensitivity calculations, not revised results or reasons to discard documents.
The original sample and decision remain intact.

Across all32 Python parents, correlations of native-minus-byte loss with literal
grid-line character fraction and runs of at least four spaces are −0.904 and
−0.930. The correlation with doctest-prompt line count is +0.014. Thus the
observed pattern is more specific than a generic claim about executable code.
These exploratory correlations have no multiplicity correction and are not
causal evidence. Document topic, structure and repetition remain confounded.

The strongest native-favoring book excerpt, `validation/2383.txt`, is verse with
aligned inline glosses, with 9.18% of characters in long space runs. Its parent
loss difference is −0.397059. The strongest byte-favoring book excerpt,
`validation/42067.txt`, consists mostly of textual Greek-character placeholders:
98.48% of its decoded characters match that literal form, and its difference is
+0.352683. Neither excerpt was excluded after inspection. This also limits the
earlier shorthand description of the book sample as ordinary book prose.

## What is established, and what remains unknown

`general-rsi-transfer-diagnosis-001/influence.json` contains all64 parent results,
original input hashes, every descriptor and all sensitivity calculations. It
uses the existing losses; no model was evaluated. The diagnostic took less than
one second and its complete output allowance was16MiB, above a10GiB reserve.
Descriptors are literal byte/character markers, not a parsed code/prose taxonomy.
They include the first context token, whereas CE excludes that token. Source
windows can start or end mid-line. All decoded excerpts had zero replacement
characters. These properties do not change the frozen scoring inventory.

Inference: formatting targets could explain much of the mean native advantage.
The current per-document scores cannot establish which target tokens contribute.
A broad claim that byte or native handles unfamiliar language better is still
unsupported. The source-scope cautions in the prelaunch FineWeb/Paloma literature
review remain relevant; no new published or causal result is claimed here.

## Bounded token-loss attribution

`attribution-plan.json` freezes a diagnostic on the **same six checkpoints and
all64 already-scored excerpts**. It retains FP32 per-target losses using the
original inference path. Every parent aggregate must reproduce its original CE
within1e-6; unchanged aggregate agreement is an implementation check, not a new
scientific confirmation. This granularity was not retained in the original
per-document score files, which is the reason for the additional inference.

Four mutually exclusive target categories are fixed before new token-loss output:
pure ASCII whitespace tokens, other tokens fully within grid-table lines, other
tokens fully within Greek-character placeholders, and the remainder. Whitespace
has precedence. Tokens crossing a boundary remain in the remainder. Categories
use exact concatenated tokenizer bytes and exclude the first context token.
The remainder is not equivalent to prose. Report all categories, arms, pairs and
domains; contributions divided by the original domain target count must sum to
the original difference. There is no new significance test, quality gate,
post-result exclusion, generation, model update or original-test access.

Prediction: native's largest gains concentrate on whitespace/grid targets.
Alternative: most of the gain persists in the remainder. Either outcome
localizes a descriptive error difference and guides the next hypothesis; neither
identifies a causal input-representation mechanism or validates new weights.

Measured full-panel evaluation cost was399s. This diagnostic allows1,800s total
across at most two attempts, RSS4GiB, MPS6GiB and a10GiB disk reserve. All six
FP32 loss arrays total4,915,200bytes before compression. The64MiB full retained
high-water envelope includes a complete second attempt, label masks, serialization
overlap, failures, source, logs and analyses. It requires no weight copies.
The new partition boundary fixtures passed. No old training, restart or loader
prerequisite is repeated. The shared research lock remains nonblocking.

## Completed token-loss attribution

All six checkpoints completed attempt1. **All384 parent aggregates reproduce the
original losses exactly** (maximum CE difference0.0). Every first token-loss array
is retained. The original transfer acceptance decision is unchanged.

The following values are additive native-minus-byte contributions, divided by
the original102,400 targets per domain. Negative favors native. Counts refer to
targets, not characters or source documents.

| Target category | Books count | Books contribution | Python count | Python contribution |
|---|---:|---:|---:|---:|
| Pure ASCII whitespace | 9,617 | −0.031914 | 27,219 | −0.213446 |
| Non-whitespace within grid lines | 0 | 0 | 2,285 | +0.005388 |
| Non-whitespace within Greek placeholders | 3,102 | +0.010426 | 0 | 0 |
| Remainder | 89,681 | +0.019266 | 72,896 | +0.112338 |
| All targets | 102,400 | −0.002221 | 102,400 | −0.095720 |

All three individual pairs have the same sign within each nonempty category:
native predicts pure whitespace better, while byte predicts the other categories
better on average. In the remainder, category-normalized native-minus-byte CE
is+0.021999 for books and+0.157806 for Python. The remainder includes markup,
code, punctuation and mixed whitespace/text tokens. It is not a semantic-content
benchmark. These are post-result descriptive comparisons with no new uncertainty
or acceptance test. Neither arm gains a broader superiority claim.

The earlier grid-table correlation therefore localizes a formatting-heavy region,
but does not mean native predicts non-whitespace table content better. Pure
whitespace targets account for9.39% of book targets and26.58% of Python targets.
This is the discriminating information missing from the document aggregates.

## Separate structural collision audit

A static audit of the actual pinned codebook and composer widths finds28 groups
covering109 vocabulary IDs with identical sets of valid byte-symbol windows for
all widths1–6. Boundary symbols are included. Max pooling over these sets discards
window multiplicity and order, so subsequent deterministic highway/projection
layers cannot distinguish the pooled representations algebraically. This is a
sufficient condition, not a complete inventory of learned numerical collisions.
Finite-precision equality of trained representations has not yet been measured.

Examples include different long runs of hyphens or equals signs. The affected IDs
occur204 times in the10,001,163-token training stream, and2/102,400 and370/102,400
scored input positions in books and Python. Training-stream frequency is not
actual sampled exposure. No single-byte ASCII whitespace IDs collide under this
audit. These are properties of the accepted architecture, not an execution defect
or grounds to invalidate completed campaigns.

A separately identified, post-result partition used the saved losses to examine
all targets whose original causal128-token chunk prefix contains one of those
109 input IDs. It covers205 book targets and8,779 Python targets. Contributions
to native-minus-byte domain loss are−0.000178 and−0.045616. The complementary
contributions are−0.002043 and−0.050104. Thus a substantial Python native advantage
also occurs without any identified collision in the current context. This does
not isolate a causal effect of collisions or exclude effects from training.
Every original target remains included; no acceptance rule uses this partition.

The first audit exported a NumPy uint16 backing buffer in a display-only hex
field. Version2 converts integer elements to bytes and checks byte lengths and
a known six-hyphen token. Group membership and all counts are identical. The
original, corrected output and `collision-audit-repair.json` are retained. Use
`composer-collision-audit-v2.json` for byte spellings.

## Literature and next discriminating action

Kim et al., [Character-Aware Neural Language Models, arXiv:1508.06615v4](https://arxiv.org/html/1508.06615v4),
Model equations5–6 and Table2 specify convolution followed by max-over-time
pooling. This supports the structural reasoning, not a reported collision result.
Discussion/Table6 reports both useful spelling generalization and representation
mistakes. Discussion's Highway Layers and Further Observations report that more
convolutional layers did not help and combining word/character inputs slightly
worsened their PTB result. Those findings caution against an automatic enlargement
or hybrid remedy. Their word-level LSTM, character alphabet and training regime
differ from our byte-composed GPT-2 subwords and transformer. Published gains or
negatives do not establish a local outcome.

Priority1 is now the same target-level attribution on the **original validation
set**, across all six equal-time endpoints. Question: does the accepted native
advantage there persist outside whitespace-only targets? The transfer diagnosis
cannot answer that. Freeze the partition and all108 original validation parents
before retaining new losses; require exact target inventory and aggregate
agreement against the original100,416-target score files. Preserve the original
validation decision and keep test unopened. This is an error-analysis extension,
not new independent confirmation. Existing original per-document files omit the
needed token losses, so a bounded inference pass is justified.

Measured original evaluation timings project about200s across six models. Before
launch, allow900s cumulative including failures, keep the4GiB RSS/6GiB MPS limits
and10GiB reserve, and admit a complete32MiB retained-output envelope covering all
six FP32 arrays (2,409,984bytes uncompressed), one full retry, labels, metadata and
serialization overlap. Measure actual free space and bind the new adapter and
inputs first. No new training, model size, source mixture or LR choice follows
from this diagnostic. A checkpoint-level numerical collision check is secondary;
its narrow frequency and the complementary loss contribution do not yet justify
a full architecture training panel.
