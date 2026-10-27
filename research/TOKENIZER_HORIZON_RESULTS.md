# Tokenizer width256 GQA duration and schedule results

The completed GQA round adopted a width256 serving recipe with50% lower KV
allocation and better mean validation CE than its equally tuned MHA control.
Repeated late-validation degradation motivates the next
[finite duration/schedule protocol](TOKENIZER_HORIZON_PROTOCOL.md).

Compare1200/1800/3000 updates with cooldown and EMA start at two-thirds of each
horizon. This is a combined schedule recipe change, not isolated duration.
One screen seed selects at most one candidate; three fresh paired seeds must
confirm it before adoption. General MHA and width64 baselines remain outside
this comparison. All old stop markers stay closed. Screening is complete; fresh confirmation is pending.

## Prerequisites

The initial regression run had102 passes and one failure: the inference-only
EMA resume rejection was shadowed by an architecture validator accessing a
minimal model without a config. Moving the existing rejection before validators
restores the intended error without changing training or inference math. The
65 affected/new/attention regression checks then passed in12.33seconds. The
initial run also passed the full schedule/trainer resume cases. These suites
overlap; counts are not additive. Failure logs are preserved. Lint found import
ordering and dict-literal style issues in new files; corrections passed lint.

New checks exercise exact declared recipe differences, real schedule endpoints,
EMA recurrence and resume across all durations, complete selection panels,
duplicate/incomplete confirmation rejection and failed-candidate retention.
Successful temporary test fixtures were cleared after preserving a manifest
and logs; no research checkpoints were deleted. The60-step serial MPS pilot
completed and passed saved-artifact review, including3 EMA snapshots, exact
training sampler replay, untouched test RNG, finite weights/optimizer state,
parent hashes and full validation arithmetic. It is feasibility evidence only.
Its raw/H200 pair uses560602508bytes; nine pairs with50% headroom project
7568133858bytes. The measured free space passed that projection plus
the8GiB reserve. [Resource review](tokenizer-evidence/horizon-pilot-001/resource-review.json).

## Launcher failure and correction

The first detached launcher resolved the virtual-environment Python symlink to
the system executable, causing an `omegaconf` import failure before controller
startup. The saved state remained prepared with no trials or checkpoints.
Preserved the failure log; corrected only the launcher executable to the absolute
virtual-environment path without resolving its symlink. Source and frozen plan
are unchanged. This is an environment launch failure, not model-quality evidence.

## Screening review

All three seed2801 trajectories passed artifact checks. The frozen rule selected
1200 updates: H200 CE4.745174 versus4.784553 at1800 and5.475618 at3000
(candidate minus control: -0.730444 nats/token). This single seed selects a
candidate; it does not establish a reproducible improvement. The selected recipe
uses60% fewer training targets than3000, with proportionally earlier cooldown
and a shorter EMA window. These components are not isolated by this comparison.
Fresh paired seeds2811/2812/2813 are now running serially; all gates and final
artifact review remain required before adoption. Source/runtime, recipe,
protocol, corpus, resource review and plan hashes remain unchanged.
[Screen review](tokenizer-evidence/horizon-001/screen-review.json).

## Reviewed trajectories

Only completed, artifact-reviewed records appear here. Test remains closed.

| Stage | Seed | Updates | Raw CE | H200 CE | Training seconds |
|---|---:|---:|---:|---:|---:|
| screen | 2801 | 1200 | 4.733485 | 4.745174 | 201.75 |
| screen | 2801 | 1800 | 4.836779 | 4.784553 | 301.44 |
| screen | 2801 | 3000 | 5.560157 | 5.475618 | 507.23 |

Frozen selection: 1200 updates.
