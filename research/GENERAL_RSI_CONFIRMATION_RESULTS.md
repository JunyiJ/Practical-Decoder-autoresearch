# General-corpus confirmation results

All six fresh confirmations completed and passed execution review. Raw-byte input
composition passes every frozen adoption gate for equal-token validation quality.
Mean H200 CE is **5.145572 byte versus 5.198176 native**, delta **−0.052604**
nats/token (about5.12% lower perplexity). All three paired deltas are negative:
−0.049663, −0.049588, −0.058562. The frozen one-sided99% document-bootstrap
upper bound is **−0.038091** (10,000 resamples, seed62110300999).
FineWeb-Edu mean delta is−0.048495 and Wikipedia−0.069061; both source guards pass.
These bounds are conditional on this validation corpus and three seeds.

The byte model has16,351,859 parameters versus28,492,416 (42.61% fewer), but
requires about1.84× guard wall time. This establishes equal-token superiority,
not equal-time superiority. First-bundle completions remain repetitive and
unreliable; this is not a claim of coherent generation or instruction following.
Test remains unopened. All fresh endpoints and original outputs remain retained.

Reproducible analysis: `general_rsi_analysis_v1.py`; immutable numerical decision:
`tokenizer-evidence/general-rsi-panel-001/decision.json`. All5089 source bindings,
six execution guards,12 endpoint hashes,108-document inventories and30,720 paired
batch hashes passed. An independent scalar resampling implementation agrees with
the vectorized bootstrap upper bound within1e-12. The frozen panel explicitly
sets upper quantile.99; no confidence threshold was changed after results.
The sections below preserve chronological observations and their then-pending status.

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

## Second paired confirmation (slots06–07)

| Final endpoint | Native CE | Byte CE | Byte minus native |
| --- | ---: | ---: | ---: |
| H200 (primary) | 5.195349189907856 | 5.145761530085512 | -0.049587659822344 |
| RAW (secondary) | 5.195755096272238 | 5.149644709992763 | -0.046110386279476 |

H200 source deltas (byte minus native): FineWeb-Edu
-0.044134298156548, Wikipedia
-0.071429919980520.
All10,240 paired training batch hashes match, as do108 document target inventories.
Both runs passed source, finite-gradient/clipping, checkpoint-hash and guard
checks. Byte guard time6677.601 seconds versus
3637.191/native (1.836x); byte peakRSS
969,621,504 bytes. Generation was not scheduled for
this pair; original first-bundle samples remain the qualitative evidence.

The first two H200 paired deltas are-0.049662523249784 and
-0.049587659822344. Their agreement strengthens the provisional transfer hypothesis;
it does not replace the third fresh pair or the frozen three-pair/document
bootstrap decision. No acceptance calculation is shortened or threshold changed.
Third native slot08 is now running, followed by byte slot09. Final evaluation
will report all three pairs, both source means, the fixed bootstrap bound and
runtime/parameter tradeoffs, with test still unopened.

## Third native confirmation (slot08)

The third native seed completed10,240 updates and342 EMA snapshots:
H200 CE5.201497910604506 and RAW5.204643604136522 over100,416 targets in108
validation documents. H200 FineWeb-Edu5.184515231236275;
Wikipedia5.269518358114966. Guard wall time3636.738 seconds; peakRSS891,994,112
bytes. All source, update, finite-gradient/clipping, endpoint-hash and coverage
checks passed slot08 review. Generation is not scheduled for this bundle.

The three native H200 scores are5.197680920685920,5.195349189907856 and
5.201497910604506. Final byte slot09 is now running with the third paired roles.
Once complete, verify its paired batches and endpoint evidence, then compute
the original three-pair mean/source gates and fixed10,000-resample document
bootstrap. Do not substitute earlier pairs or screens for the last result.

## Third pair and next evidence

Final byte slot09: H2005.142936364187565, RAW5.14877086204826;
guard6736.453seconds, peakRSS985,153,536bytes,342 averaging snapshots.
All10,240 batch hashes match native slot08. Paired H200 delta−0.0585615464;
guard-time ratio1.852334. `slot-09/review.json` records the acceptance audit.

Analysis of the same eight probe documents at every saved point shows continuing
improvement. Three-seed mean RAW probe CE at steps8192/9216/10240:
native5.208948/5.075619/4.995001; byte5.172979/5.033638/4.937302.
Full RAW validation CE from step2048 to10240 falls6.046970→5.200418 for native
and6.049977→5.150732 for byte. This curve combines exposure and scheduled LR
decay; it cannot establish the isolated effect of either. Training batch means
also fall, but differ in target distribution/context from validation and cannot
be interpreted as an exact generalization gap.

Next: prospectively test a doubled horizon on the adopted byte recipe with fresh
paired controls and horizon-matched cooldowns. Existing results motivate the
hypothesis but will not be pooled into the next confirmation. See
`GENERAL_RSI_EXPOSURE_DESIGN.md` for evidence, contrary findings and full cost.
No training worker is active after closure of this panel.
