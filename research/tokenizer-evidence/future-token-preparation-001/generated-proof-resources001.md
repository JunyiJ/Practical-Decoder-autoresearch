# Generated future-token model proof: allocation and time admission forecast

This is a prospective source-derived forecast, not a measurement or quality
budget. It applies only to model-proof001.py and model-seed-census001.py.
The five model implementation sources remain byte-identical. No model has run.

The original generated proof will perform six actual optimizer updates: one
reference and one own native update, and one joint plus one sequential update
for each auxiliary arm. Every update clips the complete gradients once. Three
additional B8T128 sequential backwards perform no optimizer update. All IDs are
new saved literals; one initializer is proposed exclusively for this proof.
No sampler, private auxiliary initializer, bootstrap or scientific roles are drawn.

## Allocation lifetimes

An auxiliary model has116,732,032 FP32 parameter bytes. Its55 Muon matrices have
3,440,640 elements; remaining AdamW elements25,742,368. One complete gradient
copy is116,732,032 bytes; Muon momentum plus both AdamW moments are219,701,504
bytes, plus tiny CPU scalar steps. Native corresponding parameter/gradient bytes
113,969,664 and moments216,929,280. These are scalar algebra, not construction.

- Native exact reference/own phase: two small models and two optimizers coexist.
  Parameters+gradients+moments <=906,331,136 bytes (using the larger auxiliary
  bound). Native child-state comparisons additionally hold two CPU moment
  snapshots, <=439,403,008 bytes, one child at a time. Comparison temporaries
  are bounded by the largest table51,463,168 bytes, up to four live arithmetic/
  boolean equivalents. No full-batch activation exists in this phase.
- Auxiliary joint/sequential phases use B2T11, never full batch joint. Two models,
  two optimizer states and two small head graphs fit the same906,331,136 bound.
  Each B2T11 vocabulary buffer is4,422,616 bytes; allow32 such buffers142MiB
  for graphs, temporary comparisons and backward work. All joint objects are
  deleted before own-state rejection tests. The joint helper itself rejects
  B>2 or T>16 before trunk/model forward.
- In-memory load phase uses one post-update model and optimizer, one CPU weight
  payload116,732,032, supplied same-backend moments219,701,504 and the loader's
  deep clone219,701,504. Adam step scalars are explicitly CPU. No archive or
  serialized tensor file exists. Subsequent transaction tests keep one CPU
  state snapshot (~337MiB including buffers), temporarily two when deliberately
  malformed state is installed, and one transient current snapshot. Budget
  1,080MiB for these CPU copies plus the sole model/moments. Their checker reads
  child state directly without validating the deliberately invalid config.
  Rejections compare complete original values, grads, modes, optimizer moments/
  exposure, identity, cache and caller IDs; NaN fixture preservation is bitwise.
- CPU/MPS body equivalence holds at most two small models and two gradient
  collections. Optimizers from the preceding phase have been destroyed. Each
  primary CE/full-gather comparison holds two CPU gradient collections, one
  unused auxiliary-None entry per training-only parameter. It does not step.
- Full B8 phases have ONE model. Delete CPU/backend reference and all gradient
  snapshots first. Cache proof migrates the sole model between MPS and CPU;
  no optimizer exists. One B8T128 vocabulary buffer is205,852,672 bytes. Two
  raw outputs coexist; log-probability comparison is split61/67, <=107,751,008
  bytes per log-probability buffer. Up to four additional full-output-equivalent
  comparison temporaries are conservatively reserved. Cache is1,572,864 bytes
  at B8,196,608 per item, four layers. Two resets are compared. No full LP copy
  is retained across chunks.
- Full B8 sequential backward holds parameters+complete grads233,464,064 bytes,
  at most one head's logits/log-softmax/backward gradient (3*205,852,672),
  trunk/leaf/attention/FFN activation allowance192MiB and workspace allowance
  256MiB. Primary head graph is released before auxiliary head, then trunk
  backward occurs once. The native ordinary CE path is included in this budget.

Use a conservative combined runtime/allocator/high-water margin of1.75GiB over
these phase-specific estimates, with additional overlap included above. Forecast
RSS3.75GiB (4,026,531,840 bytes), MPS driver4.5GiB (4,831,838,208 bytes), below
unchanged hard RSS4GiB/MPS6GiB. These are conservative admission estimates; the
independent guards can still reject. No trial allocation is used to discover fit.

## Time and retained output

Forecast model proof300seconds including imports, all source checks, construction,
initialization hashing, six small updates,39 rejections per arm, backend/gather/
causality/cache checks, three full backwards, CPU snapshots, MPS synchronization
and final RNG/source assertions. Hard body guard600seconds; external process
watchdog must bound total process including imports to620seconds at admission.
Each actual CPU initialization/backend/gather/cache phase has an independent
120second guard; forecast maximum20seconds, total CPU work70seconds within the
300second serial estimate. Whole-run MPS/RSS/reserve guard remains in force.

The single-pattern census forecasts90seconds from the existing bounded scanner
and earlier25-role census65,288 text files/5.701GB/62.612seconds. This is a
conservative forecast, not another scan or performance guarantee. It scans no
corpus data directory, imports no Torch/NumPy and makes no RNG calls. Hard120sCPU,
RSS4GiB/reserve8GiB/32MiB metadata; projected inventory<24MiB. A guard failure
closes the original attempt, without role/census replacement.

Retained caps:32MiB census +8MiB proof +2MiB source-review =42MiB, zero tensor
files, zero checkpoints/copies/deletions. Entry requirement includes all42MiB,
8GiB reserve and2GiB additional headroom:10,781,458,432 bytes. Current free space
is recorded in plan.json and covers this target. There is no storage deficit or
cleanup need; no old output/deletion credit is counted. This is not the unfrozen
18-trajectory science forecast. Serialization, full EMA/samplers/RNG resume,
physical scorer, real4/20 and full horizon remain separate unqualified gates.
