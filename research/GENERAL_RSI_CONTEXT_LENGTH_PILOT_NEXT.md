# Context128/context256 implementation and pilot boundary

## Evidence and hypothesis

The completed common-target prefix diagnostic found useful added natural context
within the present limit: CE96-minus-CE48 is negative in all three selected models
and both sources, mean −0.014213562 nats/token. Added benefit diminishes relative
to CE48-minus-CE16, and many individual windows worsen. See
`GENERAL_RSI_CONTEXT_PREFIX_RESULTS.md` and its original per-token records.
This supports checking a longer-context path, not assuming a training gain.

The train document medians exceed 600 tokens in both sources. Existing short
outputs never evict context, and all recent fresh-optimizer continuation variants
failed to improve the selected EMA parents. Therefore a future quality comparison
should use fresh paired training, with its own control and resource contract,
rather than silently extending a closed run or replaying its selected optimizer.
The current step is implementation qualification and timing only.

The context-truncation method and applicability limits from Khandelwal et al.,
[arXiv:1805.04623v1](https://arxiv.org/html/1805.04623v1), sections 3–4.1, remain
relevant. Dai et al., [arXiv:1901.02860v3](https://arxiv.org/html/1901.02860v3),
sections 3.1–3.2 and 4.3, motivate separating context fragmentation from dependency
length and using matched baselines. Neither paper supplies a local gain or an
optimal length for this small MLA/RoPE model.

Inference: merging adjacent 128-token segments into 256-token segments could
reduce loss from missing context at internal boundaries. It could also cost more
per target, add unhelpful context or train less effectively at a fixed time budget.
A later fresh comparison must distinguish predictive benefit from extra compute.
No repetition improvement is inferred from this mechanism.

## Concrete source audit

The current native wrapper is not a generic length switch:

- `byte_input_model_v3/contract.py` binds context128 in its immutable identity
  and native configuration. `byte_input_model_v6/model.py::_ids` rejects longer
  inputs. Its optimizer checks the exact model class and immutable ownership.
- The imported GPT implementation resolves to
  `ffn_structure_v1/runtime/models/gpt.py`, through package path extensions.
  It enforces the configured maximum length. The resolved architecture contract
  is `readout_centering_model_v1/runtime/models/tokenizer_architecture.py`, whose
  fixed base configuration includes block_size128.
- `block_norm_v1/runtime/attention/mla.py` allocates a causal mask from block_size.
  `final_ln_scale_model_v1/runtime/models/block_norm.py::validate_caches` has
  fixed128 cache-shape checks. These historical checks must not be weakened.
- The existing paired-block sampler already reshapes the same two sampled
  512-token blocks into 8×128 or 4×256. Both layouts contain exactly 1,024
  targets in identical flattened order. The grouped embedding backward limit
  remains 1,024 targets; it need not increase.

Implement a separate native-only context128/context256 contract and wrapper,
with explicit context in model identity, immutable buffer inventory and receipt
binding. Preserve parameter shapes/counts, FP32, four MLA blocks, RoPE math,
normalization, canonical Muon/AdamW, and the original ordinary-CE objective.
Reuse unchanged implementations where their contracts apply. Put any required
runtime contract overrides in the new namespace. Do not mutate a live historical
model's config or monkey-patch frozen modules to bypass validation.

The first pilot uses uncached logits. New cached generation is out of scope
until a later quality design needs it. Keep existing cached128 behavior unchanged.
Use 8×128 versus 4×256 with the same two512-block draws and total1,024 targets;
this is context grouping, with no repeated optimizer use of a batch.

## Only new checks and pilot work

Before numerical execution, freeze the new source, fresh qualification roles,
engineering tolerances and complete resource admission. Required checks are:

1. New128 construction must match the existing128 initializer, parameter ordering,
   logits and one optimizer update on a fresh paired fixture. This checks the new
   adapter, not a repeat of the closed model qualification suite.
2. At256, verify parameter shapes remain identical, model identity and causal-mask
   buffers are distinct and correct, short-prefix behavior agrees within frozen
   FP32 tolerances, and changing future tokens cannot alter earlier predictions.
   Check all sampler target IDs/order across the two layouts exactly.
3. Run one bounded timing/finite-state pilot with 24 ordinary CE updates per arm
   from fresh initialization at LR0.0025, clip1, inherited decay0.00006. Freeze
   paired input draws; record actual target count, loss, group state, memory and
   timing. No quality selection or validation access. The larger-context path is
   a new training layout, not another selected-parent continuation attempt.
4. Verify complete checkpoint roundtrip and one next-update replay for the new256
   identity/layout, including optimizer, sampler, RNG and any clock/averaging state
   actually used. Reject wrong-context payloads before mutation. This is required
   because model/buffer identity and sampling shape changed; do not repeat old128
   restart or document-scoring qualification without a demonstrated new need.
   A later time-budget/EMA quality trainer needs its own changed-path checks only.

Keep these checks proportional to this pilot. Do not add a learning-rate screen,
long run, generation rerolls or validation evaluation. Preserve every failure
under its own attempt. A failed numerical check is not a negative quality result;
a slower256 path is a measured cost, not a reason to change a frozen comparison.

## Complete resource forecast and next decision

Parameter count remains28,492,416 by design, but actual new output sizes must be
measured. Use conservative ceilings for the whole pilot and new restart work:
two512MiB complete restart payloads, up to four128MiB inference payloads, one512MiB
serialization overlap,64MiB metadata,10GiB reserve and2GiB extra headroom.
This totals14.0625GiB (15,099,494,400 bytes) of required free space before launch.
It includes retained outputs, not merely the active serialization. Existing
selected/fresh weights remain protected. Do not write unnecessary model copies;
unused allowance does not require producing those artifacts. Any changed size or
extra failed payload requires a fresh full admission and authorized exact cleanup.

Provisional numerical ceiling is600seconds for this bounded new-path qualification
and pilot, RSS4GiB/MPS6GiB. Current128 timing anchors are approximately0.35–0.49s
per1,024-target update;256 throughput and new serialization remain unmeasured.
Do not present those estimates as measured256 costs. Establish an early bounded
256 step/tensor-size check under the same guard, then forecast all remaining
24-step arms, replay, I/O and checks before admitting that complete tranche.
If the complete forecast exceeds the guard, revise the prospective admission
before pilot training rather than shorten a launched arm or omit required state.

The next executable action is the isolated model/contract and matched sampler
implementation, followed by the new-path checks above after admission. No worker
or quality campaign is active at this boundary. Only after measured timing,
finite behavior and complete new-state recovery pass should we design fresh
paired training roles, common-target quality evaluation, equal-budget schedules,
acceptance criteria and complete retained-output costs for a scientific panel.
No such panel or model adoption is admitted here.
