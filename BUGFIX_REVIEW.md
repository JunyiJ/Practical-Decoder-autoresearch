# Code review and fixes — 22 October 2026

Reviewed local main at `6b28425`. Changes are local and uncommitted.

## Confirmed bugs fixed

| Priority | Finding | Fix |
|---|---|---|
| High | RoPE used `2*i/dim` although `i` already advances by two. Position frequencies were incorrect. | Use `i/dim`; test known angles rather than only rotation norm/identity. |
| High | MLA concatenated flattened content and rotary features before splitting heads, mixing head features. | Reshape projections into heads first, then concatenate within each head; check against an independent attention reference and cached/chunked decoding. |
| Medium | Validation and training shared the global RNG, so evaluation cadence and model construction affected sampled training data. | Separate seeded sampling generators; seed model initialization; reset validation sampling for comparable evaluations. |
| Medium | `training.grad_clip` was ignored. | Apply configured gradient clipping before optimizer updates; reject nonfinite loss/clipped gradients. |
| Medium | MoE training-usage logs actually described the last validation batch. | Capture training routing histograms before evaluation overwrites them. |
| Medium | Both data loaders excluded the last valid window and rejected a split containing exactly one window. | Correct the exclusive upper bound of the sampling range. |
| Medium | Accelerator timings could measure asynchronous dispatch, and batched generation throughput omitted batch size. | Synchronize CUDA/MPS around measured work and count all generated batch tokens. |
| Medium | Benchmark warmup always generated five tokens, even near cache capacity; invalid generation inputs failed obscurely. | Bound warmup by requested length, validate empty prompts/top-k/cache capacity, and reject nonpositive benchmark budgets. |
| Low | Requirements allowed PyTorch versions predating `nn.RMSNorm`, used by the default config. | Raise the minimum to 2.4. |
| Low | Invalid MoE expert counts/top-k and zero evaluation iterations were unchecked. | Add explicit validation. |

Also added tokens-processed metrics and relabeled the generation memory delta so it is not advertised as peak cache memory.

## Validation

- Original suite: 17 passing tests.
- Initial regression suite reproduced five failures before fixes.
- Final suite: 30 passing tests on CPU, including thirteen new regression tests.
- A tiny three-step MoE+MLA training test checks clipping, checkpoint loading, and identical final model weights/validation loss across two evaluation cadences, with dropout enabled.
- MPS is unavailable in this sandbox; no claim of accelerator execution or performance verification.
- No full training run or quality benchmark was performed.

## Follow-up fixes

- Added optional cumulative training-step time budgets, stopping after complete
  updates and recording the actual step count and stop reason. Updated program.md.
- Added versioned atomic checkpoints with optimizer, CPU/accelerator and loader
  RNG, data/config compatibility checks, metrics and elapsed training time for resume.
- Save dedicated best checkpoints and expose their paths separately from final weights.
- Generation benchmarks report CUDA peak allocation, MPS current allocation or
  process-lifetime peak RSS explicitly rather than allocation deltas labeled as peaks.
- Legacy checkpoints fail with a clear explanation instead of silently mixing
  corrected MLA math with old weights/RoPE buffers. Historical checkpoints remain
  readable using the original code revision.
- Added regression coverage for exact CPU split-run replay, incompatible resumes,
  best-checkpoint identity, budget stopping and legacy checkpoint rejection.

Remaining physical limitations: MPS current allocation is not a peak measurement;
accelerator execution/replay was not available for validation in this sandbox.
Old model behavior cannot be losslessly converted to corrected MLA behavior.
