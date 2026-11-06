# Native versus 8K GPT2 vocabulary: closed, not adopted

All twelve original trajectories independently passed 1,200 updates, 1,228,800
targets, 41 H200 snapshots, complete RAW/H200/optimizer/EMA/RNG/sampler/clipping,
full CPU/MPS common-byte validation and cache/reset checks. Both final inference
and independent statistics completed. The terminal STOP closes every original
training, controller, inference and review command. Do not replay them.

The frozen three-rate screen selected native LR .00125 and prefix8192 LR .0025.
Those are the best rates in this finite screen, not optima.

| Fresh seed | Native H200 BPB | 8K H200 BPB | 8K minus native |
|---|---:|---:|---:|
| 200100011 | 2.078518726 | 2.103840908 | +.025322182 |
| 200100012 | 2.083367033 | 2.111287304 | +.027920271 |
| 200100013 | 2.081433827 | 2.099428423 | +.017994595 |

Mean delta is **+.023745683009 BPB**. All three seeds worsen. The practical
mean-delta gate below -.01, every-seed-negative gate, and simultaneous negative
upper-bound gate all fail. Upper bounds are +.031805216, +.032711990 and
+.030227165 for 4096/8192/16384-byte blocks. The independent arithmetic agrees
within 1e-12. **Keep the native vocabulary and existing incumbent recipe.**

The target is the same 111,475 validation bytes, with the frozen 64-byte prefix
boundary and deterministic token coding. These conditional intervals reuse
validation text and trained seeds; they do not estimate holdout or population-seed
confidence. CE per token cannot be compared across vocabularies.

The existing-data analysis found positive mean deltas in 26/28, 12/14 and 7/7
contiguous byte blocks. This is descriptive localization, not new inference.
The mean fresh training times were 276.67s native and 96.19s 8K, while mean
sampled target-byte exposure was 4,096,835 and 3,478,215 bytes respectively.
These count repeated exposure, not unique coverage. The smaller model uses
6,955,136 versus 28,492,416 parameters. Faster equal-token work did not meet
the quality rule; no cost-only adoption route was declared.

Across the three paired local serving measurements, 8K/native full128 time
ratios were .640-.641 at B1 and .492-.521 at B8; decode32 ratios were .921-.947
and .868-.886. One resident model, two warmups and five repeats; loading,
initialization and cache allocation excluded. Raw work differs with tokenization.
Cache remains 196,608 bytes per item. These are local descriptive timings.

Sparse online curves have only five points. Within native, sampled validation
CE declines across the saved tail points; the 8K curves are mixed/near-flat in
that tail. RAW versus H200 final BPB also differs: mean RAW-H200 is -.006680039
native and +.028154483 for 8K. This does not identify an optimal horizon, EMA,
clipping policy, raw-context benefit or isolated vocabulary mechanism. No
retrospective checkpoint selection, rescue or new bootstrap was performed.

All 24 screen/fresh RAW/H200 files (4,126,696,056 bytes) remain protected in their
original paths. The closure verifies all 1,004 frozen sources and original
acceptance/guard/result references. Sixty-seven newly closed metadata files were
mirrored with zero weight copies. No cleanup or tensor retirement occurred.

The first new post-study analysis mistakenly compared CPU corroboration scores
with the frozen MPS-primary scores at 1e-12, and failed before writing outputs.
Its source/failure receipt remain closed. Separate additive analysis002 uses the
original MPS primary. This was an analysis-schema error, not a training failure
or change to scientific results; no original command was rerun.

Evidence: [final review](tokenizer-evidence/tokenizer-capacity-boundary-001/quality001-final-review.json),
[independent review](tokenizer-evidence/tokenizer-capacity-boundary-001/quality001-independent-final-review.json),
[closure](tokenizer-evidence/tokenizer-capacity-boundary-001/quality001-closure002.json),
[existing-data analysis](tokenizer-evidence/tokenizer-capacity-boundary-001/quality001-existing-analysis002.json).

Continue RSI through the [next evidence boundary](tokenizer-evidence/post-prefix-boundary-001/evidence-review.md).
There is no active numerical worker or newly admitted quality panel.
