# Output-anchored tying: first complete bundle, panel still running

The [fixed protocol](OUTPUT_ANCHORED_TYING_PROTOCOL.md) remains unchanged.
Three of nine trajectories are complete. This is a partial result, not a
complete-panel inference or replacement for the byte incumbent.

| First fresh bundle62101800000 | H200 validation CE | Parameters | Training update seconds |
|---|---:|---:|---:|
| Native separate tables | 4.6745766243 | 28,492,416 | 272.116 |
| Matched separate tables | 4.6487449195 | 28,492,416 | 272.678 |
| Shared table | 4.6666817786 | 15,626,624 | 214.831 |

Allthree completed1200updates and41averaging snapshots. Each saved score covers
all34278validation targets; scalar arithmetic and final paired training-sampler
hashes agree. Their resource guards finished successfully. Training times exclude
full-validation diagnostics and export; these are descriptive single-run costs.

The shared-minus-native delta is−0.007894846nat/token. Shared-minus-matched is
+0.017936859: sharing is worse than the matched control on this seed, despite
using45.16% fewer parameters. The initialization/coordinate control itself improves
native CE by0.025831705. This does not establish the cause of historical GQA losses,
because the new architecture and recipe differ as declared in the protocol.

The prospective every-seed-negative gate against matched is already not met by
this observed seed. Preserve it; no replacement seed, rate reselection, tolerance
change or early termination. Complete allnine planned trajectories to quantify
the effect across the full panel, then run the independent saved-result reviewer.
No aggregate confidence interval or population claim is justified by this prefix.

Evidence: `tokenizer-evidence/output-anchored-tying-001/{00,01,02}/` includes
immutable result/validation/history/guard/acceptance records and weight receipts.
Final RAW weights and H200 weights remain under `research_runs/` as specified.
The CPU/MPS preflight passed24updates, exact native outputs/gradients, exact
matched/shared initial predictions and summed shared gradients, and exact weight
roundtrip. It did not select a recipe or score validation quality.

At this update, slot03 (second-bundle matched control) is running in the existing
controller session99348. Check current process state rather than relaunching it.
