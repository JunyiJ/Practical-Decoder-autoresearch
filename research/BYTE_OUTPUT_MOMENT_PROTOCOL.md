# Saved output-moment diagnostic

Frozen before loading moment values. Apply the evidence-grounded research skill
to a bounded existing-state analysis; this is not a quality experiment.

Local basis: `BYTE_SAMPLER_GAP_RESULTS.md` rejects a broad long-gap rationale
at beta2=.999. `TOKENIZER_ADAMW_EPSILON_PREPARATION.md` records adoption of
epsilon1e-6 on the earlier native model (mean validation delta−.028476 across
three pairs). That closed study is not reopened. The current byte optimizer
also uses epsilon1e-6; the actual size of its output second moments is unknown.

External basis: Kar, arXiv2609.37535v1,
[Eq1 and section7](https://arxiv.org/html/2609.37535v1), notes that positive
epsilon can suppress the adaptive rare-token effect when the second-moment
root is small. The paper's periodic bias-coordinate theorem and synthetic
experiments do not establish a contextual output-matrix effect in this model.
We use the Adam denominator equation only, without a fixed-point prediction.

Question: is epsilon negligible or material relative to the saved bias-corrected
second-moment root, and does this differ across prior frequency bins? A root
much larger than epsilon weakens the epsilon-dominated premise; roots below
epsilon establish dominance in that coordinate's denominator, not causality or
an optimal epsilon. Either outcome closes this applicability question before
any optimizer change is selected. No tuning budget or training panel is admitted.

Use the already selected representative slot10 RAW from
`tokenizer-evidence/public-inference-correction-001/adoption/recipe.json`.
It was selected by frozen order before quality outcomes. Verify recipe SHA
1a5029445e36f2701fa3e4b9d5bcf6af29ce21fbb07c54b0366bb027aa1d3823,
RAW size252854065 and SHA
c086a81207231fac943c2c68b8b7a9dcf88e485d9add058d84d58e7d6d55b853.
This is one endpoint from the adopted model, not a three-seed result or a
measurement of rejected residual models. Read CPU tensors with weights_only;
do not instantiate models, restore RNG, update parameters or score any split.
Bind exact named `lm_head.weight` to its AdamW state and saved group, shape
50257×256, FP32, step1200, betas(.9,.999), epsilon1e-6, no AMSGrad. Verify
finite nonnegative second moments and identical raw/data identity before/after.

Compute in float64 s=sqrt(v/(1-beta2^step)); report s/epsilon quantiles
0,.25,.5,.75,1 and epsilon/(s+epsilon) quantiles, means and fractions s<=epsilon.
Use fixed training-frequency bins0,1–9,10–99,100–999,1000+. Report coordinate
counts and validation-target-weighted means of each token's mean epsilon share.
Every vocabulary row participates; zero moments have share1 and ratio0.
The fraction is a denominator contribution, not fraction of loss, gradients
or learning due to epsilon. Do not extrapolate earlier steps or the H200 model.
Recover only training and validation IDs, bound to the original tokenizer and
corpus hashes; the holdout stays closed. No old sampler draws are replayed.

Check the denominator equation against hand-computed float64 cases with roots
0,epsilon,2epsilon and shares1,1/2,1/3, within1e-12. Check complete bin/coordinate
coverage and prior bin validation counts. These checks suffice; no model suite.

One attempt: `tokenizer-evidence/byte-output-moment-001`. Nonblocking shared
research lock;120seconds wall,2GiB RSS,1MiB output, zero new weights. Require
8GiB reserve plus2GiB headroom plus full output allowance at entry. Peak inputs
are one253MB checkpoint plus temporary analysis arrays, with no serialization
overlap or checkpoint output. Preserve failures and use a new identified attempt
only for demonstrated execution/implementation defects with fixed requirements.
Pin source, protocol, selected recipe, state/optimizer sources and previous
sampler results; record guard and exact input receipts. This earns no new
training or exploration credit.
