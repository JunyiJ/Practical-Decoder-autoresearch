# Exact sampler-gap applicability analysis

Frozen before execution. This is existing-data analysis, not a quality trial or
numerical exploration. No closed experiment is replayed and no loss is rescored.

Local basis: `tokenizer-evidence/byte-residual-readout-001/quality001/final.json`
and its accepted `final-review.json`; the completed
`tokenizer-evidence/byte-residual-readout-frequency-001/result.json` shows losses
in every frequency bin. Rare penalties motivate a question, not a causal claim.
The incumbent `byte_input_model_v6/optimizer.py` puts the output matrix in AdamW
with beta2=.999 and epsilon=1e-6. Historical native beta2=.95 lost all three
pairs (`TOKENIZER_ADAMW_BETA2_PREPARATION.md`); it will not be retried here.

External basis: Kar, arXiv2609.37535v1 (29 September 2026),
[sections 4.1–4.3 and 5.2–5.3](https://arxiv.org/html/2609.37535v1).
The paper relates rare-token bias to the ratio kappa of arrival gap to the
second-moment time constant. Its periodic RMSProp result assumes bias-only
gradients, zero epsilon and a nearly fixed partition function. Random-arrival
experiments depart from that formula at large kappa. Its small language-model
experiment reports substantially weaker rare-token bias at beta2=.999 than
at .95. These are external results, not local gains. Our contextual output
matrix, positive epsilon, clipping, finite 1200 updates and block sampling
prevent quantitative transfer of the fixed-point formula.

Question/prediction: does actual block sampling put a substantial set of
training-observed tokens beyond kappa=1 even at .999? If none or very few do,
the broad long-gap explanation loses support; if many do, direct gradient
evidence would still be needed before admitting an optimizer intervention.
No result alone licenses CoupledAdam, AMSGrad, coefficient selection or adoption.

Use original corpus, tokenizer version/vocabulary and train/validation ID hashes
from `byte-residual-readout-001/quality001/00/config.json`. Do not tokenize the
holdout. For L training tokens and block length K=512, the actual loader draws
two independent uniform starts from 0 through N-1, N=L-K. An occurrence at
position j is a target for the inclusive start interval
[max(0,j-K), min(N-1,j-1)]. Union these intervals separately for each token;
let C be their total length. Exact update presence is pi=2(C/N)-(C/N)^2.
Expected count/update is twice the sum of individual interval lengths divided
by N; the vocabulary total must be 1024 within 1e-10. This handles repeated
occurrences in the same block and boundary effects without an iid-token model.

Report prior fixed training-frequency bins 0,1–9,10–99,100–999,1000+:
vocabulary types, validation-target counts, min/median/max mean gaps, kappa at
.999 and .95, counts/types and validation mass with kappa>1, and expected
number of unseen types after 1200 updates using (1-pi)^1200. Separate absent
training types from any position-zero-only type; pi=0 has no finite gap/kappa.
These are expectations over the sampler, not realized histories of old seeds.
No random draws or stationary probability-bias predictions are produced.

Verification: compare interval calculations with direct exhaustive enumeration
of every ordered pair of starts on all binary sequences of length7 with K=3;
include a sequence of distinct tokens to expose both boundaries. Verify actual
coverage bounds, counts, hashes and total expected targets. This is the only
new test needed for the arithmetic; no broad model test suite.

Run `python -m research.byte_sampler_gap_v1` once with the repository environment.
Output: `tokenizer-evidence/byte-sampler-gap-001`. Acquire the existing nonblocking
research lock and require closed-panel completion/review. Budget:120 seconds,
1GiB RSS,1MiB retained output, zero model/weight outputs; entry requires the
8GiB reserve plus2GiB extra headroom plus full output allowance. Pin source,
protocol, loader, optimizer, data config and relevant completed evidence.
Preserve any failure; a demonstrated implementation/setup defect gets a separate
attempt, with the scientific question and arithmetic checks unchanged.
