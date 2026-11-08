# First-rate loss stratification: an exploratory diagnostic

The complete native and raw-byte screens at LR.000625 are accepted. Raw-byte
full validation CE is higher by.14320218788484324nat/token. This note asks where
that difference occurs, using only their already saved per-target losses and
the exact original train/validation token IDs. It introduces no model execution,
new loss measurements, random draws, holdout use or changes to the active panel.

`analyze.py` verifies each slot's acceptance bindings, finished guard and terminal
STOP, the common sampler hashes and target contract, and original corpus/token
hashes. It reconstructs the saved loss totals and checks every partition's counts
and contributions against the complete34,278-target result. The observed script
invocation took about.34seconds. The diagnostics were chosen after the aggregate
result was known; bins are fixed in the script before their outcomes are read.
They are exploratory, not additional confirmatory tests or acceptance gates.

Positive deltas below mean raw bytes is worse. Token frequency means the count
in the original training split, not the sampled optimization stream.

| Target-token training frequency | Validation targets | Mean raw-minus-native CE | Contribution to full CE delta |
|---|---:|---:|---:|
|0|957|-.906390|-.025305|
|1–9|2,846|+.307887|+.025563|
|10–99|6,892|+.902993|+.181558|
|100–999|8,788|+.112061|+.028730|
|1,000+|14,795|-.156024|-.067343|

The10–99frequency target group accounts for+.181558nat/token of the overall
+.143202delta; improvements in the zero-frequency and most frequent groups
partly offset it. This is not a uniform loss increase across targets.

Input composition changes representations used to predict subsequent tokens,
so the same analysis also groups by the immediately preceding input token:

| Preceding-input training frequency | Validation targets | Mean raw-minus-native CE |
|---|---:|---:|
|0|957|-.923899|
|1–9|2,846|+.027525|
|10–99|6,893|+.459368|
|100–999|8,787|+.184457|
|1,000+|14,795|+.062675|

Raw bytes has lower loss on72.62% of the targets immediately following a token
absent from training. This input-conditioned result is consistent with shared
composition helping representations of unseen input tokens. It is not proof:
frequency, token identity, neighboring tokens and difficulty are correlated,
the model uses more context than the immediately preceding token, and the
nonlexical control has not completed this rate yet. A generic shared composer
could produce some of the same effect without meaningful lexical byte content.

Length alone gives another mixed pattern. After input tokens of8+bytes the
mean delta is-.217208, while after2–3byte inputs it is+.312934. For prediction
targets themselves,1-byte targets improve by-.194321,2–3byte targets worsen by
+.420755,4–7byte targets worsen by+.207625, and8+byte targets worsen by+.072876.
Length excludes boundary symbols. These correlated groupings are descriptive;
they do not isolate a byte-sharing or morphology mechanism. Full counts, means,
lower-loss fractions and additive contributions are in `result.json`.

## Consequence for the ongoing research

Keep the unfavorable complete-screen result and finish the frozen panel. Do not
select rates, declare a practical gain, substitute a subgroup endpoint or add
targeted training runs from this diagnostic. The existing nonlexical control is
already the relevant discriminating comparison; no extra model test is needed.
Once that saved result exists, the same fixed group definitions can describe
whether the unseen-input benefit is specific to lexical codes. Any such analysis
still remains exploratory until the full selected-rate fresh comparison is
complete, and does not replace its predeclared overall acceptance criteria.

The local observation, tentative mechanism and limitation are deliberately
separate. No external-paper gain is treated as a local result, and no new
architecture campaign is admitted by this note.
