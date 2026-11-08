# Output-anchored tying: two complete bundles, panel still running

The [fixed protocol](OUTPUT_ANCHORED_TYING_PROTOCOL.md) remains unchanged.
Six of nine trajectories are complete. This is a partial result, not a
complete-panel inference or replacement for the byte incumbent.

| Fresh bundle | Native CE | Matched CE | Shared CE | Shared−native | Shared−matched |
|---|---:|---:|---:|---:|---:|
| 62101800000 | 4.6745766243 | 4.6487449195 | 4.6666817786 | -0.0078948457 | +0.0179368591 |
| 62101900000 | 4.6956314700 | 4.6874285448 | 4.6965106771 | +0.0008792071 | +0.0090821323 |

Allsix completed1200updates and41averaging snapshots. Each score covers all34278
validation targets. Scalar arithmetic and paired training-sampler hashes agree
within each bundle. Resource guards finished successfully. The final independent
saved-artifact and statistical review still awaits allnine trajectories.

Sharing is worse than the matched control on both completed seeds. It is slightly
better than native on the first and slightly worse on the second. Thus neither
prospective every-seed-negative gate is met by this prefix. Preserve all outcomes;
no replacement seed, rate reselection, tolerance change or early termination.
Complete the final bundle to quantify the full-panel effect.

Shared parameters15,626,624 versus28,492,416 for either separate-table control
save45.16%. In the first bundle, training-update time was214.831s shared versus
272.116s native and272.678s matched. These are descriptive single-run costs,
excluding diagnostics/export; no cost-only adoption is permitted by this protocol.
The changed initialization/coordinates alone improve native quality on both
completed seeds. Neither that observation nor this new MLA comparison identifies
the cause of the historical closed GQA tying loss.

Evidence: `tokenizer-evidence/output-anchored-tying-001/{00,01,02,03,04,05}/`
contains immutable result/validation/history/guard/acceptance records and weight
receipts. RAW and H200 weights remain in the declared `research_runs/` directory.
The CPU/MPS preflight passed24updates, exact native outputs/gradients, exact
matched/shared initial predictions and summed shared gradients, and exact weight
roundtrip. It did not select a recipe or score validation quality.

At this update, slot06 (third-bundle shared model) is running in the existing
controller session99348. Check live state rather than relaunching it. The byte
bulk-hash-adoption001 recipe remains incumbent. No holdout or historical-score
pooling is used.
