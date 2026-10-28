# Wider direct MLA/GQA comparison status

Completed and audited October27,2026, Los Angeles (October28 UTC). All12 quality
trajectories,24 checkpoint tensor/hash/EMA audits and three serial inference pairs
passed. Adopt a scoped width512 quarter-latent RoPE MLA option versus equally tuned
fresh GQA. Source607ef9f, planb098d72, digest
`12c60cbf4e01550e09b8317ccbbb90d13ed2390ad01a3d3304db205450f37240`.
No training or inference worker remains. Closed controllers must not be rerun.

Mean fresh H200 CE: MLA4.657553 versus GQA4.705184, delta-0.047630nat.
All three seeds and conditional interval strict/neutrality gates pass. Actual
allocated KV is68.75% smaller in every seed/batch. Decode is19.88% slower at batch1
and22.98% slower at batch8; prefill12.82%/1.73% slower. These are medians across
seed-specific median ratios on FP32 MPS/context128,32prefill/64decode. No speed-win
claim. Both models were resident; allocator readings are not deployment peaks.

Mean training time413.58s MLA versus421.86s GQA, about1.96% less. Actual parameter
storage249,696,768 versus251,510,784bytes. Cached/full parity passed in all pairs;
maximum absolute error0.000017166. Three seeds on reused Shakespeare validation
support this scoped decision, not larger-data/width transfer or a global optimum.
Test holdout remains closed. Keep the width256 recipes separately scoped.

[Final review](tokenizer-evidence/wide-attention-001/final-review.json),
[tensor audit](tokenizer-evidence/wide-attention-001/tensor-audit.json), and
[adopted recipe](tokenizer-evidence/wide-attention-001/adopted-width512-mla-recipe.json).

The [frozen protocol](TOKENIZER_WIDE_ATTENTION_PROTOCOL.md) independently screens
three LRs per family at3401, then compares selected GQA/MLA recipes at fresh
3411/3412/3413. Width512 has62.88M/62.42M parameters. The common1200-update
schedule is a fixed comparison budget, not a demonstrated wider optimum. No
screen/pilot result is a final ranking. Source and selection rules stay frozen.

Both20-update pilots retained raw/H200 pairs and passed tokenizer/checkpoint/EMA,
finite saved tensor and cached/full parity checks at batches1/8. Measured cache
per batch item/context128 is1,048,576 bytes GQA and327,680 MLA (68.75% less).
The initial pilot was geometry/feasibility evidence only. The>=60% memory gate
was subsequently verified on every fresh pair with quality and serving latency.
No speed win was required for the memory adoption route.

Width1024 full-panel accounting currently exceeds available disk, so it is not
in this finite round. Existing8GiB reserve and sampled memory/time guards remain.
All old campaigns stay closed and cleanup records remain authoritative. Test
holdout is closed; adaptive validation and small-data limitations still apply.

Preparation evidence: [CPU and resource review](tokenizer-evidence/wide-attention-preparation-001/review.json),
[pilot review](tokenizer-evidence/wide-pilot-001/review.json), and
[quality/timing controller review](tokenizer-evidence/wide-attention-preparation-001/quality-timing-review.json).
The initial invalid two-update CPU fixture failure is retained and corrected;
no MPS pilot failure or retry occurred. In-memory CPU resume checks do not stand
in for disk export checks; both were completed separately.

## Reviewed screening progress

October28,04:30UTC: all six screening trials completed1200updates. Checkpoint
hashes, exact config/tokenizer/EMA lineage, parameter counts and saved validation
arithmetic passed review at the screening transition. The full tensor/EMA audit
and fresh confirmation subsequently completed as summarized above.

| Family | Seed | LR | Raw CE | H200 CE | Training seconds | Raw/H200 bytes |
|---|---:|---:|---:|---:|---:|---:|
| GQA | 3401 | 0.000625 | 4.825032 | 4.857613 | 427.64 | 1,212,186,956 |
| GQA | 3401 | 0.00125 | 4.704309 | 4.709348 | 423.57 | 1,212,186,892 |
| GQA | 3401 | 0.0025 | 4.996874 | 4.883027 | 421.32 | 1,212,186,892 |
| MLA | 3401 | 0.0025 | 4.959769 | 4.842255 | 415.56 | 1,205,038,972 |
| MLA | 3401 | 0.00125 | 4.651238 | 4.649909 | 415.38 | 1,205,039,036 |
| MLA | 3401 | 0.000625 | 4.799393 | 4.834718 | 417.59 | 1,205,039,036 |

Both families independently selected LR0.00125, the interior point of the frozen
three-rate grid. The independent selection audit matches the controller's saved
choice, written before the first fresh confirmation request. This is a one-seed
tuning outcome, not proof of an optimum or an attention-family quality advantage.
The selected rates were used unchanged for all six fresh confirmations,
followed by final tensor/EMA audit and three frozen inference pairs.
[Screening review](tokenizer-evidence/wide-attention-001/screen-review.json) records
the selection checks.
[Completed-record review](tokenizer-evidence/wide-attention-001/completed-review.json)
records the bounded audit scope; source and prospective rules remain frozen.

## Fresh confirmation progress

All six fresh trajectories completed. Paired MLA-minus-GQA H200 deltas are
-0.047069/-0.049238/-0.046584nat. Final tensor/EMA audits and frozen serving
measurements passed. Conditional95% text-bootstrap intervals at block sizes
2048/4096/8192 are[-0.052160,-0.043069],[-0.052322,-0.043790],
[-0.053527,-0.043658]. They condition on these trained seeds and adaptive
validation; they do not measure population uncertainty or an independent test.

| Family | Seed | Selected LR | Raw CE | H200 CE | Training seconds |
|---|---:|---:|---:|---:|---:|
| GQA | 3411 | 0.00125 | 4.708778 | 4.707809 | 420.93 |
| MLA | 3411 | 0.00125 | 4.671780 | 4.660740 | 413.52 |
| MLA | 3412 | 0.00125 | 4.666685 | 4.659804 | 411.48 |
| GQA | 3412 | 0.00125 | 4.701124 | 4.709042 | 418.36 |
| GQA | 3413 | 0.00125 | 4.693315 | 4.698700 | 426.31 |
| MLA | 3413 | 0.00125 | 4.663606 | 4.652116 | 415.75 |
