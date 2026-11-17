# Saved-model rolling history diagnostic

## Human direction and local basis

On 2026-11-16 Pacific the human authorized scoring identical targets with rolling
128-token and 256-token histories on saved models, without training. The completed
study's same-weight evaluation benefit was concentrated near segment resets.
`GENERAL_RSI_DEPTH_CONTEXT_POSITION_RESULTS.md` and its original
`tokenizer-evidence/general-rsi-depth-context-position-001/analysis.json` show
66.74%/67.06% of that benefit in the first32 targets after a common128 reset.
Later positions also improved. Those scores do not isolate information older
than128 tokens, since common128 had discarded recent history at resets.

Selected test: keep each model fixed and give both conditions a full recent128
history. The longer condition additionally receives the preceding128 tokens.
Prediction: mean CE256-minus128 is negative in all three saved bundles at each
depth. A positive effect contradicts a predictive benefit for this history range;
an interval crossing zero leaves the aggregate direction unresolved. Either
outcome resolves an ambiguity without another training run. No optimizer,
recurrence, context extension or closed campaign is reopened.

## Primary literature

Khandelwal et al., https://arxiv.org/html/1805.04623v1, sections3–4.1, vary supplied
context on fixed LSTM weights and exclude short-history targets when comparing
truncations. Context needs differ across words. Their word-level PTB/WikiText
LSTMs differ from our subword MLA model and corpus. They motivate this control,
not a local effective-context estimate.

Dai et al., https://arxiv.org/html/1901.02860v3, section3.1, describe stride-one
last-position evaluation with the longest available window, and its computation
cost. Their recurrent architecture in section3.2 is absent here. Both exact paper
versions and these sections were retrieved for this boundary. Local gains are
unknown before this test.

## Frozen scientific scope before scoring

- Six retained **context256-trained time-EMA** endpoints: depths4/16, bundles1/2/3.
  These are the completed study's primary endpoints, selected by role, not loss.
  Both histories lie within their trained context range. Context128-trained
  models would add a context extrapolation confound. RAW is outside this bounded
  diagnostic; no endpoint is selected based on new results.
- Use all108 authorized validation documents. Score each within-document target
  at zero-based position `p >= 256`, through the terminal EOT, once. This is72876
  targets per endpoint and condition. Exclude the first255 previously scored
  targets per document prospectively so every comparison is exactly256 vs128
  preceding tokens. No original test access, new generations or training.
- For target `tokens[p]`, inputs are `tokens[p-128:p]` and `tokens[p-256:p]`.
  Never include the target, padding or a previous document. Shift by one token
  for every target. Recompute each window from raw tokens with no KV cache or
  retained hidden states that could carry excluded history. RoPE positions use
  the model's ordinary local coordinates; relative distances among recent tokens
  are unchanged. The longer prefix can change their contextual representations.
- Freeze order depth4 bundles1,2,3 then depth16 bundles1,2,3. Within each batch,
  score128 then256. Batch size4, FP32 corrected RMSNorm, no_grad, no autocast.
  A last-position-only vocabulary projection avoids computing unused logits.
  Qualify it against the original full-logit API and scalar-window oracle before
  validation scoring. Preserve failures as distinct engineering attempts.
- Save aligned target IDs, global positions and both token-loss arrays per
  document. Report per-bundle, token-weighted aggregate and source deltas.
  Three primary contrasts: depth4, depth16 and depth16-minus4. Average bundle
  loss sums before shared paired document bootstrap,100000 PCG64 draws,
  seed621116210001. Bonferroni family alpha.01 over three two-sided intervals:
  quantiles.01/6 and1-.01/6. Bounds condition on reused development documents and
  these three models, not population-seed uncertainty or fresh confirmation.
  Negative upper bounds plus three negative bundle deltas support consistent
  predictive benefit. Do not call an unresolved contrast equivalent or adopt a
  new recipe. Report magnitude without retrofitting the old .02 adoption gate.

A negative result supports predictive value from supplying this older prefix
under this model and evaluation intervention. It does not identify semantic
reasoning, a specific dependency, or a depth scaling law. A null result does not
prove no long-range information use. This development-set follow-up is selected
from prior results, and cannot revise the frozen training study's decisions.
Representation, exposure, optimizer and schedule uncertainty remain.

## Qualification, admission and complete cost

Qualify deterministic synthetic window coverage (including boundaries and tails),
then both depths using retained bundle1 EMA weights and a fixed training-document
prefix, without validation quality inspection. Compare last-only vs full-logit
and batched vs scalar logits using inherited3e-5 absolute +3e-4 relative tolerance;
CE vs CPU float64 logsumexp within1e-5. Check exact parameters before/after.
Measure128 paired target windows at each depth after warmup, including memory and
serialization. Require measured full six-model forecast plus50% timing margin
and600seconds overhead <=12hours before launch. Qualification ceiling600seconds.
The measured admission record binds source, endpoint, data and design hashes.

No new model/optimizer/EMA payloads or checkpoint copies are created. Forecast
64MiB scores/metadata,64MiB failure allowance and128MiB atomic/log/publication
allowance, total256MiB beyond the existing12GiB reserve (10+2). Reuse existing
retained weights through actual free space. One serial worker holds the shared
nonblocking lock. Read actual processes first. Guard RSS4GiB, MPS driver6GiB,
disk12GiB throughout import/initialization/scoring/serialization. Check host
pressure and reclaimable-page proxy including other apps: normal pressure before
load and >=1GiB free+inactive+speculative after load and during evaluation.
These pages are a proxy, not guaranteed free RAM; RSS/MPS overlap must not be
summed. Record phases and peaks, stop on new human stop. No cleanup is needed
unless measured capacity changes materially. Failed artifacts remain distinct.
