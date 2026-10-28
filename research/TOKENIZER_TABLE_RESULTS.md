# Width256 table-sharing study status

October28,2026,02:19UTC. The15-trial quality panel is launched under source89309b3,
plan4de77f0 and digest6abcfdfe160acaa34d214bb456720cd8ec16591a3647640201eb2628d16ba353.
This is a launch record, not a completed quality result. Consult the actual
research_runs/tokenizer-table-001 state and progress file for current status.

All three20-update feasibility pilots passed. Actual unique FP32 parameter
storage was114,352,128 bytes for each untied arm and62,888,960 for tied:45.004%
smaller. GQA cache storage stayed unchanged. Raw/H200 pairs fit original reserved
allowances; no pilot validation score selected a width, arm or rate. See
[tokenizer-evidence/table-pilot-001/review.json](tokenizer-evidence/table-pilot-001/review.json).

The model, pilot controller, quality controller and benchmark checks passed20
CPU tests, including exact resume, independent LR choice, complete paired gates,
arm-specific disk accounting, no automatic restart, lock inheritance and rotating
three-arm timings with storage deduplication. The MPS pilots additionally checked
raw/H200 loading and cached equivalence at batches1/8.

The frozen [protocol](TOKENIZER_TABLE_PROTOCOL.md) requires six independent LR
screens followed by nine fresh confirmation trajectories, retaining both untied
controls. Quality and measured inference review are still outstanding. No table
recipe is adopted. The closed test holdout remains unused.

## First completed screens (October28,02:28UTC review)

The two legacy_untied screens completed at seed3301. H200 CE is4.763301 at
LR0.00125 and4.812186 at0.0025. These are single-seed screening observations,
not fresh confirmation or a table-sharing comparison. All three independent
rate choices will be persisted only after the six screens finish.

Both raw/H200 pairs passed checkpoint hashes, exact request/config/tokenizer
lineage,1200-step/41-snapshot checks and saved full-validation arithmetic checks.
Each pair is560,603,788 bytes, below its659,621,888-byte per-trial allowance.
The entire remaining panel fits above the8GiB reserve; one MPS worker continues.

The initial metadata reviewer incorrectly counted persistent RoPE sine/cosine
buffers as trainable parameters and asserted before publishing a review. The
16,384-element discrepancy was diagnosed and the reviewer now counts saved
canonical EMA parameter names while checking the eight expected buffers separately.
`reviewer-failure-001.json` retains this review failure. The corrected review
passed; no training source, result, checkpoint, selection rule or worker changed.
This metadata/hash review does not recompute model losses or tensor finiteness.

## Reviewed screening progress

All six screens at seed3301 completed and passed hash/metadata/EMA/score review.
The frozen rule independently selected LR0.00125 for legacy_untied and0.0025
for matched_untied and scaled_tied. The saved selection preceded the first fresh
confirmation request; screen-review.json records that check and an immutable
snapshot of all six reviewed records. Nine fresh confirmation trials are underway.

| Arm | LR | Raw CE | H200 CE |
|---|---:|---:|---:|
| legacy_untied | 0.00125 | 4.731212 | 4.763301 |
| legacy_untied | 0.0025 | 4.879233 | 4.812186 |
| matched_untied | 0.0025 | 4.840032 | 4.853498 |
| matched_untied | 0.00125 | 5.077150 | 5.135245 |
| scaled_tied | 0.00125 | 5.111563 | 5.168311 |
| scaled_tied | 0.0025 | 4.880606 | 4.884608 |

The matched and tied arms preferred the higher screened LR, unlike legacy.
This is a single-seed optimization observation supporting independent tuning;
it does not establish a relative quality gain, neutrality or adoption. Both
controls remain in all three fresh confirmation seeds. No additional LR is added.

Tied raw/H200 pairs were303,285,837/303,285,773 bytes, below the363,708,672-byte
allowance. Untied pairs remained560,603,788 bytes, below659,621,888. The full
remaining allocation fits above8GiB reserve. Source and test holdout stay frozen.

## Fresh confirmation progress

6 of nine confirmation trajectories have completed and passed the saved-artifact
review. This is an incomplete panel; no final paired quality gate or adoption is declared.

| Seed | Arm | Selected LR | Raw CE | H200 CE |
|---:|---|---:|---:|---:|
| 3311 | legacy_untied | 0.00125 | 4.738458 | 4.742336 |
| 3311 | matched_untied | 0.0025 | 4.879054 | 4.855979 |
| 3311 | scaled_tied | 0.0025 | 4.908078 | 4.881648 |
| 3312 | matched_untied | 0.0025 | 4.845808 | 4.844221 |
| 3312 | scaled_tied | 0.0025 | 4.885470 | 4.876284 |
| 3312 | legacy_untied | 0.00125 | 4.736880 | 4.749471 |

Two fresh seed triples are complete. Tied H200 CE is higher than both controls
in each of those seeds. The third seed triple and measured inference review
remain outstanding; no final panel or adoption conclusion is made. Continue
the exact frozen panel without changing rates, gates or seeds.

All independent LR selections remain frozen. Source/runtime/data provenance and
remaining disk allocations pass; the serial worker continues. Test loss is closed.
