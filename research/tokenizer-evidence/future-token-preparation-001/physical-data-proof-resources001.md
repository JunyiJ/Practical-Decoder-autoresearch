# NEW physical data-view proof: finite scope and complete prospective resources

Applies ONLY to new physical-data-proof001.py and physical-data-seed-census001.py.
No corpus/RNG/model call has occurred during preparation. Existing data/sampler
implementations and all closed fixtures remain unchanged. This test qualifies
only the own physical CPU data-view consumer; no model/CE/scorer/adoption claim.

The complete runtime source declares TWO genuine encoder calls (only train/val),
zero generator construction during metadata/decode, two CPU View destinations,
prefix train,val then continuation train,val for uninterrupted/restored views:
6 actual view calls/30 sampled anchors, final private counts train2/val2 each.
24 explicit transaction negatives. Exact all1040 sequence/position entries,
1024 offset0/1/2 values, prefix/private state/counter/input/globalCPU RNG and
identity checks; no floating tolerance. Unique monotonically numbered reports.

## Allocation lifetimes

- Corpus reads capped8MiB per buffer; actual pinned source corpus is about1.1MB.
  The independent witness frees its raw bytes/Unicode text after retaining only
  train/val character counts and UTF8 digests. Actual decoder then holds raw,
  Unicode text, two BPE-ID lists and tuple conversion. At most two8MiB raw buffers
  coexist during its post-read comparison. No holdout substring is BPE encoded,
  stored in the Decoded result, sampled or scored.
-267688+34279 physical int64 IDs occupy2,415,736 tensor bytes per sampler. Two
  View destinations retain4,831,472 bytes; constructor temporaries add at most
  2,415,736, peak7,247,208. Immutable Python ID tuples plus integer objects and
  temporary BPE lists are conservatively budgeted32MiB. Both Views share the
  same immutable Decoded object; no copy of corpus/IDs is written to disk.
- Full canonical JSON ID/hash checks, tokenizer ranks/cache/native runtime,
  data-list conversions and copied physical metadata are budgeted128MiB working.
  One raw transaction snapshot and transient comparison snapshot each clone
  about2.5MiB of owned CPU data; mutated state payloads have only small private
  generator tensors. Budget16MiB for these overlaps and1MiB private-state staging.
- Six sequence/position outputs are8x130 CPU int64, each8,320 bytes; only current
  original/restored outputs need be live. Independent physical-block oracle is
  at most8x130 Python indices/IDs. Complete actual/expected values are metadata,
  not serialized tensors. No logits/activations/autograd/optimizer/model/EMA.
- Installed Torch/tiktoken/import/allocator/library margin is conservatively
  768MiB. Forecast peak RSS1GiB (1,073,741,824B), within unchanged4GiB hard guard.
  No MPS allocation/API or CPU-MPS numerical claim.

Expected runtime30seconds including imports inside the120second CPU guard,
codec initialization, full physical encoding/digests, repeated physical identity
validation,6 calls, in-memory restore and24 structural transactions. The guard
begins BEFORE Torch/tiktoken/component imports. No trial allocation determines
fit. A failure preserves the original attempt and cannot trigger source/role/
case/backend rescue or retry.

## Codec access, retained output and census

The installed GPT2 constructor specifies two required cache resources:
vocab.bpe456,318B/SHA1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5,
encoder.json1,042,301B/SHA196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783.
Both existing local files were read-only verified before freezing sources/plans.
Plan pins paths/sizes/hashes and relevant installed codec source/native artifact;
runtime repeats the required entry/exit binding checks. The fixture blocks
read_file network fallback; the genuine encoder spy always delegates to the
original installed Encoding.encode. No tokenizer fetch/cache write is intended
or credited; a cache miss stops the original proof.

The NEW two-role recorded-text census uses the accepted bounded scanner, only
fresh private train/eval proposals in its own plan. Forecast90s/RSS192MiB from
latest65,525 files/5,741,797,485B/60.786s receipt; hard120sCPU/4GiB/8GiB reserve/
32MiB metadata. No Torch/NumPy/model/corpus-loader/RNG calls. No old census replay.
Census completion and independent admission precede corpus/RNG runtime.

Retained caps32MiB census+4MiB proof+2MiB static gate+128KiB root. Add8GiB reserve
and2GiB extra headroom:10,777,395,200B complete entry target. Actual current free
space is in plan.json and covers it. No storage deficit/cleanup/deletion credit.
Zero tensor files/checkpoints/serialization overlap. No quality budget/freeze or
historical18-trajectory output forecast is imported into this finite data test.
