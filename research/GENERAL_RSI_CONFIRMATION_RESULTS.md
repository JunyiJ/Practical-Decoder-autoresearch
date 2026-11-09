# General-corpus confirmation results

The frozen first paired bundle is incomplete: native slot04 is complete and
byte slot05 is next. No architecture is adopted from one arm or one pair.
All three fresh pairs and the original gates remain required; test stays closed.

## First native confirmation (slot04)

After10,240 updates /10,485,760 sampled training targets, native LR.0025 achieved
H200 validation CE5.19768092068592 and RAW5.2008541908060835 over100,416 targets
in108 separate validation documents. H200 source CE: FineWeb-Edu5.182307517567118;
Wikipedia5.259255760393237. Guard wall time3790.304 seconds; peakRSS803,897,344 bytes.
All342 averaging snapshots, source bindings, endpoint hashes, target coverage,
finite gradients and clipping passed review. Evidence:
`tokenizer-evidence/general-rsi-panel-001/slot-04/{result,review,guard-01}.json`.

H200 is slightly better than RAW here, unlike the shorter screens. Keep the
frozen H200 endpoint rule. The lower loss than the short screen is descriptive:
the horizon, schedule and fresh seed differ, so it is not an isolated causal
estimate of additional training or a matched architecture comparison.

## First completions: native only

All48 original records (initial, RAW and H200; eight prompts ×two decoders) are
retained in slot04/generation-*.json, without rerolls. Inspection of both final
endpoints shows local grammatical fragments and some topical vocabulary, but
poor sustained meaning, prompt relevance and factual consistency. Greedy text
frequently collapses into repeated phrases; sampling varies vocabulary but
still drifts and invents relationships. H200's16/16 outputs and RAW's15/16 hit
the64-token cap; this is truncation, not a successful completed answer. RAW's
remaining sampled output ends with EOT but is still off topic.

Examples from the first H200 outputs, deliberately including failures:

- “Plants use sunlight to” → sampled: “ control the blood vessels.” The following
  text discusses an immune system rather than photosynthesis.
- “When water is heated,” → greedy: “ it is important to keep the water from the
  water.” The continuation loops on water-related fragments.
- “Question: Why does the sun rise in the morning?\nAnswer:” → greedy: “ The sun
  is the sun, the sun is the sun, and the sun is the sun.”
- The Maya/cup story starts sampled “ put it to the front.” then drifts into
  unrelated fragments and references; it does not maintain the simple scene.

These are qualitative diagnostics, not a blinded benchmark or an instruction
following claim. Lower validation loss has not yet yielded reliable prose.
Finish the matched byte run and remaining frozen pairs before interpreting
relative quality. After the panel, use the complete curves and paired outputs
to decide whether a separately justified exposure or equal-time study is useful;
these examples do not justify modifying the active scientific contract.
