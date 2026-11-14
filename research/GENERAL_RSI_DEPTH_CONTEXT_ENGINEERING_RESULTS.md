# Two-depth context qualification and admission — 2026-11-14 Pacific

The isolated native D256 combined depth/context wrapper passed CPU parity and
all four MPS qualifications. The full12-run study is admitted. This is engineering
evidence; no new scientific quality result is available.

## Measured checks

| Depth | Context | Median synchronized update (s) | Peak RSS (GiB) | Peak MPS (GiB) | RAW endpoint bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 128 | 0.355906 | 1.440 | 2.330 | 113998683 |
| 4 | 256 | 0.366146 | 1.440 | 2.330 | 113998683 |
| 16 | 128 | 0.643098 | 1.459 | 2.549 | 147203671 |
| 16 | 256 | 0.660904 | 1.635 | 2.549 | 147203671 |

CPU checks establish exact initialization, logits, loss, gradients, clipping,
optimizer update and optimizer-state parity with both qualified reference paths.
The two contexts share parameter draws, depth4 is the same initialized prefix,
and extra four-block groups have independent private draws.

Each MPS shape passed full1024-target finite updates, active physical-time EMA
serialization/readback and exact next-update replay using the recorded duration.
RAW and EMA endpoint loading matched exactly. Wrong depth/context loads/restores
were rejected before mutation. Corrected full-context CPU/MPS logits met3e-5
absolute plus3e-4 relative tolerance. Training-document target coverage and FP64
CE oracle met1e-5. Scoring/generation preserved the complete training state.
All96 first engineering generations are retained. No validation or test content
was read by qualification. Eight accepted transient restart/reference files were
retired only after exact receipt, consumer and durable-review checks.

## Preserved harness failure

`general-rsi-depth-context-integration-002/attempt-01/failure.json` records an
end-of-fixture state-preservation assertion after the first24 generations. The
CPU reference constructor called torch.manual_seed inside a CPU-only fork_rng,
which restores CPU RNG but also changes MPS RNG. An isolated zero-update diagnosis
reproduced that difference. The separate v3 fixture explicitly preserves MPS RNG
around CPU reference construction and passes the original invariant. All original
failed payloads, source, receipts and first outputs remain. No numerical tolerance,
scientific seed or quality criterion was changed. The earlier v1 plan was never
executed; its static target-ID/CPU-digest corrections are retained separately.

## Complete admission

The measured forecast is52.121 hours for scientific execution
plus the reserved1-hour engineering envelope. The total ceiling remains63.4hours,
including48training hours and failure/variability allowances. All12runs,24final
endpoints, restart atomic overlap, metadata, engineering and failures are charged.
Admission measured21724610560 free bytes against20674353861 required
bytes, leaving1050256699 bytes beyond the full requirement, including the
10GiB reserve PLUS2GiB extra headroom. Existing failed payloads are already charged
through actual free space. No cleanup or partial-panel reduction was needed.

Across recorded phases, minimum free+inactive+speculative memory was2326298624 bytes.
This is a reclaimable-page proxy, not guaranteed free RAM. Pressure levels1/2
occurred during qualification; pressure was normal at admission. Reserve256MiB
for full evaluation data and longer history beyond the short fixture, plus1GiB
observed memory headroom. RSS/MPS limits remain enforced throughout scientific
execution. RSS and MPS driver counts overlap and are not added.

The frozen protocol uses native-context primary CE, common128 secondary CE,
fresh three-pair controls at each of depths4/16, and a direct depth interaction.
Three primary contrasts use Bonferroni family alpha.01 and100000 document
bootstrap draws. Fixed288 first scientific outputs remain exploratory. Old scores
are planning anchors only; two depths do not establish a scaling law.

Scientific plan SHA256: `b99578af06d57549cee6f9ffad19acc0cdeeb89d5f9c5c3a03fcf8ee26dafe1c`.

Evidence: CPU `general-rsi-depth-context-fixture-002/attempt-01/result.json`;
MPS `general-rsi-depth-context-integration-003/completion-review.json`;
scientific `general-rsi-depth-context-quality-001/plan.json` and `admission.json`.
All evidence directories are under `research/tokenizer-evidence/`.
