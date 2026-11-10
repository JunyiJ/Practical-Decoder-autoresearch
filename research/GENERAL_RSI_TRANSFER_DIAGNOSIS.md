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
