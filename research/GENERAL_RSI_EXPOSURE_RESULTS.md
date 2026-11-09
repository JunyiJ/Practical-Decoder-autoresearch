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
