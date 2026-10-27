# Tokenizer MLA results

No quality results yet. The completed horizon round adopted the shorter GQA
recipe; this next round evaluates an MLA architecture using a fresh MHA control.
See [prospective protocol](TOKENIZER_MLA_PROTOCOL.md).

MLA support passed63 CPU checks (22.82seconds), then one source-frozen60-step
width256 MPS feasibility trial (11.29training seconds). Saved-artifact review
passed exact config, finite optimizer/weights,60 optimizer steps,3 EMA snapshots,
checkpoint hashes, training sampler replay, untouched test RNG and validation
arithmetic. This pilot's validation values are sanity data, excluded from quality
selection. [Pilot evidence](tokenizer-evidence/mla-pilot-001/result.json).

Full/cached FP32 MPS logits passed at batches1/8 (maximum absolute differences
3.34e-6/1.67e-6). Actual allocated latent-plus-rotary cache was196608/1572864bytes
at context128. These checks establish implementation feasibility, not comparative
quality or latency. [Cache checks](tokenizer-evidence/mla-pilot-001/cache-check.json).

The raw/H200 checkpoint pair uses559170172bytes. A new ten-trial quality storage
projection includes the larger preserved width256 MHA pair,15% overhead and a
separate8GiB reserve. [Resource review](tokenizer-evidence/mla-pilot-001/resource-review.json).
Completed temporary pytest fixtures were cleared only after saving logs and a
manifest; no research checkpoints were removed. All prior failures remain
preserved. Quality controller and MLA-specific benchmark integration are complete. The
initial41-test suite passed, followed by7 focused controller/guard checks,
including serial ten-trial progression, frozen selection, STOP behavior and
no restart of completed runs. The suites overlap; counts are not additive.
The benchmark retains GQA labels for legacy calls and uses explicit MLA labels
for the new comparison. Arithmetic/cache implementation remains unchanged
from the passed MPS pilot. Source freeze and fresh disk projection precede launch.

## Quality launch prerequisites

The quality protocol contains four width256 screening trials (one seed, two LRs
per family) and six fresh confirmation trials. Quality eligibility is separate
from inference memory/latency measurement and final adoption review. The7GiB
output cap accommodates the measured ten-pair projection plus15% overhead;
the8GiB free-space reserve is retained and checked against all remaining trials.
A global lock prevents a second training or timing worker. Source/runtime,
protocol/recipe/corpus and pilot/resource evidence are hashed into the plan.
Failure logs and all checkpoints are retained. Test remains closed.

## Launch

The ten-trial quality run started successfully on October27,2026. Frozen source
commit5276412, plan recorded in f22616a; plan SHA256
afb45c98f9185f8f659577ac7c1326140aa42349047be956c4ade06edac8ea10.
The first MHA screening worker is active under the single-worker lock. Full
remaining-trial storage projection passed before launch. No completed quality
results or adoption claim yet. The heartbeat now follows this round's helpers.

## Reviewed trajectories

Only completed, artifact-reviewed records appear here. Test remains closed.

| Stage | Family | Seed | LR | Raw CE | H200 CE | Training seconds |
|---|---|---:|---:|---:|---:|---:|
| screen | mha | 2901 | 0.00125 | 4.742776 | 4.746541 | 204.96 |

Screening incomplete; family LRs not yet selected.
