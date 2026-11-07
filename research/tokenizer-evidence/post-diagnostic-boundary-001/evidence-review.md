# Native MLA elementwise output gate: direction and generated prerequisites

The objective is lower complete native-token validation loss. The gradient and
optimizer-response branch remains closed and inconclusive for batch selection.
This direction changes the model's representation and can be judged directly
against equally tuned controls. There is no measured quality gain yet.

## Evidence and choice

`direction001.json` binds the original local results, acceptance/final records
and roadmaps and ranks the remaining architecture and recipe hypotheses. The
selected prerequisite is a query-dependent, per-channel sigmoid gate on the
native MLA weighted values, before concatenation and the output projection.
The output projection bias is outside the gate. The unchanged and fixed-half
arms distinguish a learned gate from changing attention-branch amplitude.

The historical character/MHA headwise gate is contrary evidence: its D256 mean
change versus unchanged was only -0.000689 and one fresh seed was worse; the
study did not qualify for adoption. Its stronger comparison against fixed-half
does not establish a gain over unchanged. This test changes gate granularity,
the native tokenizer/MLA backbone and recipe, and will not pool those scores.
Other reviewed local architecture, vocabulary, dropout and MoE results do not
justify automatic replay or a rescue grid.

External basis: Qiu et al., [arXiv:2505.06708v1](https://arxiv.org/html/2505.06708v1),
§2.2 equation 5, §3.1, §3.2 tables 1–2 and §4.1–4.2 equations 6–8/tables 3–4.
The paper supports post-attention, query-dependent gating and reports a dense
model preference for elementwise gating. Its GQA models, data scale, context,
optimizer and precision differ substantially from this small native MLA model.
It supplies a reason to test, not a local improvement or diagnosed attention sink.

Our inference is that query-local channel modulation could help the compressed
value/output path. The falsifiable prediction is that the learned arm beats
both unchanged and fixed-half controls under complete fresh confirmation.
Failure retains the native incumbent. Neither outcome alone proves an
attention-sink, sparsity or rank mechanism.

## Completed correctness work

- `gate-component001` is complete and closed. Independent scalar forward and
  gradient equations covered CPU FP64/FP32 and MPS FP32, including zero,
  ordinary and saturated gates. Controls, native B1/B8 geometry, causality,
  unscaled output bias and 22 malformed-input/state cases passed. Maximum
  scalar discrepancy was 1.3668e-8. Guard: 1.266 s, RSS 390,234,112 B,
  MPS 17,530,880 B. No language model, corpus or optimizer was used.
- `gate-model001` is complete and closed. Own native model/config/cache,
  optimizer and weights policies preserve the genuine incumbent backbone.
  The unchanged arm matched native initialization, generated outputs,
  gradients and two same-device MPS optimizer steps exactly. The zero learned
  gate matched fixed-half outputs and all shared gradients; all four gate
  matrices received nonzero gradients. CPU/MPS generated forward/gradient,
  complete learned-parameter exposure, transactional rejection and cache/reset
  B1/B8 checks passed. Guard: 8.233 s, RSS 1,233,977,344 B,
  MPS 1,169,965,056 B. These are generated integration checks, not real-data
  training or cross-device optimizer-equivalence acceptance.
- `review001.json` independently checked the closed metadata, unchanged source
  hashes, declared tolerances, inventories, terminal guards and full prospective
  budget without replaying either proof or loading a model. It passed; its
  120 s CPU guard finished in 0.040 s at RSS 220,889,088 B.

The new implementation has 1,062 bound source files, preserving all 1,054
predecessors and the 1,035 original MoE scientific sources. Successful new
namespaces/proofs/review are immutable and closed. Fixture 600511 was audited
unused before generated initialization and is now used. Global CPU/MPS RNG
was preserved. No corpus, validation/holdout, tensor file or quality seed was
used. No numerical worker remains active.

## Prospective test and cost

`quality-design001.json` is a prospective design, not a quality freeze. The gate
adds 262,144 parameters: 28,754,560 versus 28,492,416 for both controls. All shared
weights and construction RNG draws match. The four new matrices use the native
body Muon policy; this policy is declared, not claimed optimal.

The bounded proposal is nine LR screens followed by nine fresh trajectories,
1200 updates each, with equal finite tuning and the original strict two-control
adoption gates. All scientific and future pilot roles are unassigned. All
outputs would be retained. Including six preceding worker/pilot pairs, metadata,
sidecars, external allowance, entry margin and 8 GiB reserve, the complete floor
is 25,227,227,136 B; target with extra 2 GiB is 27,374,710,784 B. Approximately
80.64 GB free covered that target in an inline, nonblocking locked capacity
pass; no inventory or deletion was needed. The conservative 29,605 s total
planning forecast is not measured gate runtime or a promise. Actual complete
serialization layouts and resource pilots must replace it before any freeze.

Next: implement only the own real-data resume/EMA/checkpoint prerequisite in a
new additive namespace. Prove exact sampler/RNG and per-parameter exposure,
RAW/H200 metadata, continuation and serialization before generated workers,
resource pilots and any scientific campaign. Preserve all closed studies and
the incumbent. No batch prescription or further optimizer-response detour.
