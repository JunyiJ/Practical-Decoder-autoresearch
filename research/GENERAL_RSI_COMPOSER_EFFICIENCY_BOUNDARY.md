# Composer efficiency: measured redundancy before an implementation test

The next engineering candidate is within-batch reuse of identical token
compositions. No runtime change or training panel is admitted by this note.

## Measured basis

- The accepted equal-time panel favored native by0.122102 nats/token, while the
  earlier equal-update panel favored byte by0.052604. Byte used about1.83 times
  the update time in that earlier comparison. These distinct constraints remain
  separate scientific results, not pooled confirmations.
- `byte_input_model_v6/model.py:_composed_hidden` composes every input occurrence
  before applying the transformer. It does not deduplicate whole token IDs.
- `general-rsi-composer-efficiency-preparation-001/batch-reuse.json` reconstructs
  the first128 training batches from each of the three equal-time paired seed
  bundles using the original sampler. All384 batch hashes match both historical
  arms' logs. Each batch contains1,024 positions but only310–595 distinct IDs,
  mean491.951 and median496. Thus51.96% of composer rows repeat an input already
  present in that batch. This is a measured row count, not a speedup estimate.
- The completed transfer diagnosis shows that byte can retain descriptive
  advantages outside whitespace in new domains. It still passed no transfer
  adoption gate. It is therefore useful to investigate its execution cost
  before assuming that a larger native model or more composer parameters is
  the appropriate next investment.

No new model, gradient, seed confirmation or test evaluation was used for this
census. It completed in under one second. The exact source/log bindings and
all384 counts are retained. These initial sampler prefixes need not represent
every future batch or sequence length.

## Mechanism and falsifiable engineering question

For fixed composer parameters, the same token ID produces the same deterministic
composition. Compute it once per distinct ID in the current update, then map
those outputs back to original positions. In real arithmetic, backpropagation
must sum all position gradients for an ID before differentiating its single
composition. This preserves the represented function; it does not share outputs
across optimizer updates or reuse stale learned embeddings.

Prediction: fewer composition rows reduce synchronized composer forward/backward
time enough to justify integration. Alternative: grouping, gradient accumulation,
validation checks or small matrix kernels consume the expected saving. The common
transformer, output head and optimizer remain work that this change cannot remove.
A row-count reduction alone is insufficient to qualify a speed claim.

Floating-point execution is a material concern. Historical evidence in
`TOKENIZER_QUALITY_RESULTS.md`, “Isolated MPS embedding and Muon diagnostics,”
shows repeated-index embedding-backward variation and optimizer sensitivity to
small gradient perturbations. Current `byte_embedding_v1.py` uses a deliberate
repeatable grouped reduction. A naive repeated-index gather backward can
reintroduce the old problem. Any new reduction order must be explicit, qualified
at fixed predeclared tolerances, and bound to a separate implementation identity.
Do not edit frozen sources or reinterpret accepted checkpoints.

## Literature, contrary findings and scope

Kim et al., [arXiv:1508.06615v4](https://arxiv.org/html/1508.06615v4),
“Further Observations,” reports slower character composition than input lookup
and discusses precomputed vocabulary representations for scoring. This motivates
examining repeated work, but does not validate within-update autograd reuse.
The same discussion reports a worse combined word/character model, and the
Highway Layers section reports no benefit from additional convolutional layers.
Their word-level LSTM and hardware differ from this local byte/subword transformer.
No published speed or quality result is treated as a local measurement.

## Next executable action and complete cost boundary

Freeze a **standalone composer-only** numerical and timing protocol in a new
namespace before numerical execution. Use fixed synthetic controls for all-unique,
all-repeated and alternating IDs, plus the exact historical batch inventory.
Compare outputs, every composer parameter gradient, repeated backward
reproducibility, parameter/RNG immutability and a deterministic accumulation
oracle. Do not claim full-model replay from component tests. Include grouping,
inverse mapping and gradient accumulation in synchronized timings. Fix timing
order, warmup, measurement count, tolerances and a practical benefit threshold
before observing prototype results. No quality/validation selection enters this
engineering qualification.

Use a prospective900-second cumulative ceiling across at most two attempts,
RSS4GiB, MPS6GiB and10GiB disk reserve. A64MiB complete retained-output cap must
cover source/configuration, logs, comparison tensors or failure tensors,
serialization overlap and both attempts. No full-model weight copies or new
trained checkpoints are needed for this component test. Replace these allowances
with exact tensor-shape accounting and current free-space admission before run.
Only one numerical worker may run under the shared nonblocking research lock.

If correctness/reproducibility or practical benefit fails, retain the failure and
stop this candidate rather than tune around the gate. If it passes, separately
qualify a versioned full-model execution path, including actual optimizer and
restart behavior, and measure whole-update cost. Only then design and forecast
any fresh equal-time quality confirmation, with all outputs, reserve and matched
controls. No final scientific panel is admitted by this note.
The collision-length hypothesis remains secondary: its sparse occurrence and
lack of causal attribution do not justify a full architecture panel now.
