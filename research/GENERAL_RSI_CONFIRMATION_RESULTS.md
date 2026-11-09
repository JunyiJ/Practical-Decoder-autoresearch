# General-corpus confirmation results

The first frozen paired bundle (native slot04 / byte slot05) is complete.
The second bundle has completed native slot06; paired byte slot07 is running. No architecture is adopted from
one pair.
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

## First paired confirmation (slots04–05)

| Final endpoint | Native CE | Byte CE | Byte minus native |
| --- | ---: | ---: | ---: |
| H200 (primary) | 5.197680920685920 | 5.148018397436136 | -0.049662523249784 |
| RAW (secondary) | 5.200854190806084 | 5.153779002086082 | -0.047075188720002 |

H200 FineWeb-Edu CE is5.182307517567118/native and5.136165433624663/byte
(delta-0.046142083942455). Wikipedia is5.259255760393237/native and
5.195492879247873/byte (delta-0.063762881145364). All10,240 paired training
batch hashes and all108 document/source/target inventories match. Both runs
finished within their guards, with342 EMA snapshots and unchanged source hashes.
Byte guard time6975.039 seconds versus3790.304/native (1.840x); byte peakRSS
862,142,464 bytes versus803,897,344/native. The byte model has16,351,859 parameters
versus28,492,416/native. Equal-token quality here does not establish equal-time
superiority. The remaining two paired bundles and fixed three-pair acceptance
criteria are still required; no early adoption or bootstrap decision is made.

The matched curves show why an early score is insufficient: at step2048 the
full-validation RAW byte-minus-native delta was+0.006970, while the final RAW
delta is-0.047075. The fixed8-document probe fluctuates across checkpoints
(e.g. -0.052180 at4096 and+0.043432 at5120). These curves are diagnostic only;
no intermediate checkpoint or training horizon was selected from them.

All48 first byte outputs are retained alongside the48 native outputs. Byte
H200 also reaches the64-token cap on16/16 outputs; RAW reaches it on15/16,
with one off-topic EOT completion. Both final byte endpoints show repeated
phrases, broken grammar, irrelevant continuation and incorrect factual links.
For example, H200 greedy completes the library prompt with “ to be a member
of the House of Representatives.” and loops on that institution. Its sampled
plants continuation begins “ save the body.” and drifts into drug discussion.
The Maya/cup scene turns into music/sports fragments. Lower CE in this pair
therefore does not demonstrate coherent generation or instruction following.

Current inference: byte-spelling sharing may transfer its prediction advantage
to this corpus, but the one-pair result is provisional and its substantial
runtime cost remains. Continue the remaining fresh pairs without changing
seeds, endpoints, scoring, prompts or thresholds. After complete confirmation,
use the full evidence to assess exposure and equal-time hypotheses under a
new prospective contract. Keep the test payloads unopened.

## Second native confirmation (slot06)

The second fresh native seed completed10,240 updates and342 EMA snapshots:
H200 CE5.195349189907856, RAW5.195755096272238, again over100,416 targets in108
validation documents. H200 FineWeb-Edu5.178622907029208 and
Wikipedia5.262342696758161. Guard wall time3637.191 seconds; peakRSS887,570,432
bytes. Source bindings, all updates, finite gradients/clipping, endpoint
hashes and document coverage passed the slot06 review. Generation is not
scheduled for this bundle under the frozen protocol.

Its H200 is0.002331730778064 lower than the first native seed, a descriptive
between-seed difference. No matched second-pair effect exists until byte slot07
finishes. Slot07 now runs with the same bundle2 sampler roles; bundle3 slots08–09
remain next. Preserve the initial one-pair inference and all fixed gates.
