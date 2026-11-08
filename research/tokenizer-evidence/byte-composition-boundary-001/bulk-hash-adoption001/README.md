# Complete-buffer hashing adopted

The active [recipe](recipe.json) uses model-v6 (including the already adopted
optimizer-state batching), state-v6, training-v6, inference-v5 and cache-v4.
The public selected-checkpoint loader is `research.byte_input_checkpoint_consumer_v4.load`.
It accepts the bound implementation, original plan and acceptance receipts plus
CPU/MPS backend. Callers retain the worker lock and resource guards.

The same first-fresh H200 weights, 16,351,859 trainable parameters, model math,
optimizer updates, artifact formats and scientific quality evidence remain.
Every buffer byte is still checked against its original SHA256 digest.

## Change and evidence

The old digest copied each buffer to CPU in 131,072-element pieces and created
Python bytes for each piece. The new method transfers each complete buffer once
and passes its contiguous NumPy memory through the buffer protocol to SHA256.
The largest temporary CPU tensor is 52,267,280 bytes. No codebook row, padding
byte, final tail, mutation guard or finite check is removed. The fixed registered
buffer inventory uses bool, FP32 and int64; arbitrary tensor dtypes are outside
this qualification.

The isolated comparison covered all 14 real buffers on CPU and MPS. Digests were
identical. Both implementations rejected and restored a center mutation in every
buffer, plus codebook mutations at positions 0, 131071, 131072 and the final
position. All 72 rejection checks passed, including old chunk boundaries and tail.
Model weights and RNG remained unchanged. No training was done in that check.

Complete model-validation medians:

| Backend | Previous | Candidate | Candidate paired wins |
|---|---:|---:|---:|
| CPU | 27.043 ms | 26.397 ms | 6/7 |
| MPS | 46.693 ms | 33.494 ms | 7/7 |

The MPS result passed the prospective 10% median improvement and five-of-seven
wins gate. Seven additive source files then integrated the single hash-method
change and exact typed imports. Their reverse maps reconstruct every original
file exactly. The previous adopted optimizer arithmetic is retained.

## Actual training and consumer checks

Fresh CPU sampler roles 62101300001/2 and MPS roles 62101400001/2 are now closed.
Cached B1/B8 logits for a 96-token prefix plus 32-token continuation matched exactly
on both backends, and cache reset passed. CPU complete training states matched
after updates one and two. Restoring the candidate at one produced the same
update-two state, including the first averaging snapshot.

Twenty MPS paired updates used alternating order. The first four were declared
warmups and the next 16 measured. All histories and complete final states matched
exactly, including eight averages. The prospective gate required a median ratio
at most 0.95, at least 12/16 wins and neither averaging phase regressing.

- Median candidate/reference latency: **0.913807**, or **8.62% lower**.
- Candidate faster in **16/16** measured pairs.
- Phase medians: **0.909136** before averaging and **0.920734** with averaging.
- Measured totals: **14.211161 s** previous versus **13.138652 s** candidate.

All gates passed. This is one short local paired trajectory, not independent
training seeds, a confidence interval or a full 1,200-step speed claim. Separate
historical speedups must not be multiplied into an unmeasured combined result.

The public v4 loader then loaded the accepted H200. The new stack restored the
original 1,200-update RAW and captured its complete state exactly, including 41
averaging snapshots. Inference-v5 reproduced the original H200 export exactly in
memory and loaded every parameter exactly. No new weight archive was needed.
The immutable implementation manifest records the preparation state before this
artifact check; this recipe and its closure bind the completed check separately.

## Cost and next work

The current-stack profile used three updates and 6.668 guarded seconds. The
buffer comparison used zero updates and 9.103 seconds. Training/cache/resume
checking used 45 updates and 59.851 seconds. Artifact checking used zero updates
and 6.366 seconds. Total: 48 updates, about 82 guarded seconds, no quality panel,
no holdout scoring and no new weight files. Largest peak RSS was 3,631,169,536 bytes,
within the unchanged 4 GiB guard. All workers exited successfully.

Next assess the unchanged deterministic byte-embedding backward, measured at
about 0.195 s in the profile. A concrete candidate is to avoid computing gradient
rows for byte IDs absent from the current input while preserving every original
2048-position chunk and ordered accumulation. First quantify active-row counts
and transfer/selection overhead, then use a new bounded exact-gradient comparison.
This is a different question from padding cropping: do not reopen that failed
fixture, weaken exactness or change reduction order without a new justified study.
No worker is active.
