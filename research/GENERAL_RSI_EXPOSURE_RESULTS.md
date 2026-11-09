# General RSI exposure results

## First short-horizon control: slot00

Completed10,240updates,342H200 snapshots. H200 validation CE5.142032462383304;
RAW5.146652643827362. H200 FineWeb-Edu5.132911775467566 and
Wikipedia5.17856340029713, over100,416targets in108documents. Successful
attempt02 guard time6842.509seconds; peakRSS935,428,096bytes. Attempt01's missing
temporary-tokenizer-asset failure remains preserved separately; exact restored
bytes passed original hashes before attempt02. No scientific criterion changed.

`tokenizer-evidence/general-rsi-exposure-001/slot-00/review.json` binds the
accepted result and guard, all5094 source hashes, finite gradients/clipping,
full document coverage and retained endpoint hashes. RAW SHA256
80a761742581b5cefb002c9185522e745360d61f794a650131812864fd484c90;
H200 bcc5eca6c534788a640632cc96ec852b742409ee2dd838df34c678525638641b.
The obsolete final optimizer snapshot was retired only after those independent
endpoint weights and metrics were verified; original deletion journal retained.

All48first generations (16initial,16RAW,16H200) were read and preserved unchanged.
Initial outputs are untrained fragments/repetition. Final outputs have sentence
structure but fail relevance/factual consistency and often loop. All16RAW and
15/16H200 samples hit the64-token limit. The single H200 EOS is an off-topic
water→sun passage, not a successful answer. Examples, without rerolls:

- H200 greedy, water heated: “the water is not the water,” followed by repeated
  “The water is the water of the water.”
- H200 greedy, computer storage: “a computer,” then repeated claims that the
  Internet is software used to create software.
- H200 sampled, plants: “cut water during the winter,” followed by water advice.
- RAW sampled, library: initially relevant “to create a variety of important
  resources and services in all areas,” then vague program/project discussion.
- H200 sampled, Maya/cup: shifts into track releases and football relegation.

These diagnostic samples establish continued weakness; they do not provide a
blinded quality score or justify changing frozen decoding. This fresh short run
is a control for its paired longer horizon, not additional evidence pooled into
the closed architecture comparison.

## Next action

Run paired slot01 at20,480updates with the same initializer and sampler roles,
and its own horizon-matched cooldown. Check that its first10,240batch hashes
match slot00 and that initial generated output matches exactly. Complete all
three fresh pairs before applying frozen mean/source/bootstrap gates. No horizon
has been selected. Test remains unopened. The actual active worker pointer is in
`research_state/general-rsi-state.json`.

## Capacity recovery during slot01 — 2026-11-09 17:07 UTC

Free storage fell below the full remaining-panel forecast while training remained within its 10 GiB guard. Exact external-cache recovery003 removed 7,775 unopened Chromium payloads (64,864,256 allocated bytes), verified 16 protected files, and retained its manifest/journal. Receipt free space was 11,566,948,352 bytes versus 12,000,855,844 required: a 433,907,492-byte deficit. This is partial recovery, not a scientific failure or permission to launch slot02.

Other inventoried large cache classes include updater payloads, shared runtime/application assets and OS speech assets; no deletion was inferred from size. Research payload retirement cannot run while PID24324 holds the shared lock. Eight obsolete, accepted LR-screen endpoint weights from the concluded original panel were audited against original receipts: 717,766,656 allocated bytes. All screen source/log/config/metric evidence and all fresh confirmation weights remain protected. The active exposure panel binds no screen weight path; it initializes fresh. The current human obsolete-checkpoint authorization supplies a narrow exception to the old retain-all wording.

Next action: finish/review slot01, then remeasure capacity under a free research lock. If still deficient, execute the exact pending screen-retirement audit with process/consumer checks and durable deletion journal before slot02. Do not repeat the cache scan absent new eligible candidates or changed capacity. Evidence: `research/tokenizer-evidence/general-rsi-capacity-003/`.
