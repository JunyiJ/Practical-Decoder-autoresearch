# Tokenizer width256 GQA duration and schedule results

The completed GQA round adopted a width256 serving recipe with50% lower KV
allocation and better mean validation CE than its equally tuned MHA control.
Repeated late-validation degradation motivates the next
[finite duration/schedule protocol](TOKENIZER_HORIZON_PROTOCOL.md).

Compare1200/1800/3000 updates with cooldown and EMA start at two-thirds of each
horizon. This is a combined schedule recipe change, not isolated duration.
One screen seed selects at most one candidate; three fresh paired seeds must
confirm it before adoption. General MHA and width64 baselines remain outside
this comparison. All old stop markers stay closed. All nine trials and the final artifact review are complete.

## Final decision

Adopt the1200-update recipe for width256 GQA LR0.00125, with linear cooldown
and H200 EMA starting at800 updates. All three fresh pairs favor it: mean full
validation H200 CE4.753151 versus5.464003 at3000, a -0.710852-nat/token delta.
Seed deltas are -0.737451, -0.694650 and -0.700454. Separately recomputed
conditional text-bootstrap95% intervals remain below zero at2048,4096 and8192
blocks; the4096 interval is[-0.775624,-0.655858]. These intervals condition on
the trained seeds and reused validation text; they are not independent-test or
training-seed population uncertainty.

Mean training time was201.73 versus503.91seconds, about60.0% lower; sampled
target exposure fell60%. This adopts the combined horizon/cooldown/EMA-window
recipe, not a causal claim about duration alone or a claim that1200 is optimal.
The general MHA and width64 baselines remain separate. Inference geometry is
unchanged. All checkpoint hashes, nine artifact reviews, source/runtime/recipe/
protocol/corpus/resource identities and the frozen selection were checked.
Test remains closed. [Final review](tokenizer-evidence/horizon-001/final-review.json)
and [adopted recipe](tokenizer-evidence/horizon-001/adopted-width256-gqa-recipe.json).

The next structural priority is MLA: explicitly validate score scaling, latent
cache geometry and optimizer/checkpoint contracts before a bounded comparison.
Use the new schedule as a candidate training regime; do not assume an untested
MLA or MHA configuration inherits this GQA-specific result. Keep the3:1 KDA/MLA
hybrid and broader MoE/activation/norm/residual backlog active. Storage must pass
a new prospective headroom check; no historical checkpoint deletion is allowed.

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

## First fresh pair (incomplete confirmation)

Seed2811 favors the selected1200-update recipe: H200 CE4.745014 versus
5.482465 at3000 (delta -0.737451 nats/token). Measured training time was
202.62 versus499.94seconds, about59.5% less. Both artifacts passed review.
This is one of three required fresh pairs, so there is no adoption decision or
reproducible-improvement claim yet. The effect remains a combined duration,
cooldown and EMA-window comparison. Seed2812 is underway serially.

## Second fresh pair (incomplete confirmation)

Seed2812 also favors1200 updates: H200 CE4.755463 versus5.450113 at3000,
a paired delta of -0.694650 nats/token. Both artifacts passed review. Two of
three required fresh pairs are complete; the third is underway. No adoption
or completed-round conclusion is made before that pair and the final audit.

## Reviewed trajectories

Only completed, artifact-reviewed records appear here. Test remains closed.

| Stage | Seed | Updates | Raw CE | H200 CE | Training seconds |
|---|---:|---:|---:|---:|---:|
| screen | 2801 | 1200 | 4.733485 | 4.745174 | 201.75 |
| screen | 2801 | 1800 | 4.836779 | 4.784553 | 301.44 |
| screen | 2801 | 3000 | 5.560157 | 5.475618 | 507.23 |
| confirm | 2811 | 3000 | 5.566267 | 5.482465 | 499.94 |
| confirm | 2811 | 1200 | 4.736564 | 4.745014 | 202.62 |
| confirm | 2812 | 1200 | 4.737855 | 4.755463 | 200.56 |
| confirm | 2812 | 3000 | 5.532091 | 5.450113 | 506.23 |
| confirm | 2813 | 3000 | 5.543201 | 5.459431 | 505.56 |
| confirm | 2813 | 1200 | 4.751947 | 4.758977 | 202.02 |

Frozen selection: 1200 updates.

- Seed2811, candidate minus3000 H200 CE: -0.737451.
- Seed2812, candidate minus3000 H200 CE: -0.694650.
- Seed2813, candidate minus3000 H200 CE: -0.700454.
