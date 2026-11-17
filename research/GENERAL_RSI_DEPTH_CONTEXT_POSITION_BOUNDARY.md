# Completed-panel context-position analysis

## Selection before position summaries

- Local observation: the completed two-depth panel passes native-context gates at
  depths4/16 with mean EMA deltas−0.096031/−0.089993. Common128 means are smaller,
  −0.021068/−0.012125. See the original `decision.json`, `secondary-accounting.json`
  and `completion-review.json` in `tokenizer-evidence/general-rsi-depth-context-quality-001/`.
- The qualified scorer `general_rsi_context_length_evaluation_v1.py` partitions
  each document into nonoverlapping segments. In a256 segment, positions1–128
  have the same prefix as common128. Positions129–256 have128 additional prefix
  tokens. This follows source inspection, not the unseen position summaries.
- Question: is the same-weight native256 versus common128 gain concentrated near
  the common128 reset, or does it persist through its later positions?
- Hypothesis: reducing fragmentation gives larger per-target benefit immediately
  after the removed reset. Prediction: benefit per target is larger in the first32
  positions after that reset than in the remaining96. A reversed pattern refutes
  this concentration prediction. Persistent late benefit would limit a claim that
  the effect is exclusively a short-prefix reset cost. Neither pattern identifies
  semantic reasoning, useful dependencies beyond128 tokens, or a causal training mechanism.

## Primary literature and limits

- Khandelwal et al., [1805.04623v1](https://arxiv.org/html/1805.04623v1), §§3–4.1,
  perturb context at evaluation on fixed LSTM weights. Model-size changes preserve
  the context-usage trend; token classes differ in effective context needs. Their
  truncation excludes early targets with insufficient history. Our subword MLA
  models and nonoverlapping scoring keep those targets, so their effective-context
  estimates do not describe this experiment. This supports inspecting target
  coverage and available history before extrapolating a depth trend.
- Dai et al., [1901.02860v3](https://arxiv.org/html/1901.02860v3), §§3.1–3.2,
  describe missing history at fixed-segment boundaries. Their sliding evaluation
  supplies a full available window for each prediction, whereas our scorer uses
  nonoverlapping segments. Their recurrence remedy is absent here. This analysis
  diagnoses a coverage pattern; it does not qualify recurrence or import its gains.
- Both exact versions and the cited sections were retrieved and read for this
  review. Published results motivate the question; they are not local evidence.

## Candidate ranking and controls

1. Selected: saved token-loss position decomposition. It resolves an evaluation
   ambiguity with no additional training, scoring, generations or payloads.
2. Defer a clipping/optimizer intervention. Final-bin clipping differs sharply
   across depths, but the mixed optimizer and other recipe differences prevent a
   causal claim. That observation alone does not justify a full new sweep.
3. Defer longer-context or recurrence training. The completed native/common128 gap
   has not yet been separated by supplied-history position. No new panel is admitted.

Use only the current12-run panel, all three fresh paired bundles, RAW and EMA,
all108 development documents and100416 targets. Do not repeat the closed
cross-size byte attribution or prior context panel. Match `(document,target_start
+ token index)` and token IDs exactly across retained score files. Decompose each
native contrast into same-common128 weight difference plus same-weight evaluation
context difference. Report per-bundle, source, document and position summaries.

Predeclare four disjoint position bins in each256 segment:1–128 (shared prefix),
129–160 (first32 after the common128 reset),161–192 (next32),193–256 (last64).
Also combine the final two bins as remaining96. Use actual target counts, including
short document tails. Shared-prefix differences are a numerical control, not a
quality outcome. Report their maximum absolute token difference. Reconcile the
weighted decomposition with original aggregate loss to1e-10. Do not change any
frozen scientific gate or claim multiplicity-controlled significance from this
posthoc descriptive analysis.

Cost: read36 existing score files serially, CPU only, no model/MPS allocation,
no test reads or new corpus access. Budget300 seconds, RSS4GiB, output16MiB,
64MiB total analysis/publication allowance, preserve12GiB reserve. Read actual
process state and acquire the shared lock nonblockingly. All scientific endpoints,
first outputs, failure evidence and frozen sources remain immutable. If the
position hypothesis fails, retain the negative description and reject a
reset-concentration explanation. If it holds, use that limitation when planning
future evaluation; it alone does not authorize new training.
