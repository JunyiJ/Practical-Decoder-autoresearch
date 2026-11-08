# Full-horizon training and saved inference costs

This analysis uses the already accepted selected-rate screens (native05,
real-byte08, code06) and the first complete fresh bundle (09,10,11). Selection is
reconstructed from all nine screening reviews; every analyzed run's eight
acceptance evidence hashes are verified. The frozen driver source is verified
against the original plan. `result.json` binds all inputs and the analyzer source.
It preserves each five-repeat timing minimum, median and maximum. Analysis used
only the standard library and saved JSON, completing in 0.0081 seconds; no model,
GPU, checkpoint load, new validation loss or holdout execution occurred.

## Observed first-fresh-bundle tradeoff

| Measurement | Native | Real bytes | Nonlexical codes |
|---|---:|---:|---:|
| Full-validation averaged-model CE, lower is better | 4.673415 | 4.573299 | 4.641407 |
| Mean training seconds/update, all 1,200 updates | 0.7515 | 3.1217 | 3.1349 |
| Whole guarded run seconds | 927.67 | 3,822.64 | 3,840.06 |
| Trainable parameters | 28,492,416 | 16,351,859 | 16,351,859 |
| Maximum sampled step-end MPS driver bytes | 2,510,077,952 | 3,623,665,664 | 3,632,054,272 |
| Averaged-weight archive bytes | 114,002,715 | 65,445,293 | 65,445,357 |
| Batch-1 full 128-token forward median, ms | 13.47 | 54.88 | 31.32 |
| Batch-1 cached 96-token prefill median, ms | 13.00 | 30.17 | 27.49 |
| Batch-1 cached next-32-token decode median, ms | 281.09 | 534.35 | 529.95 |
| Batch-8 full 128-token forward median, ms | 53.27 | 152.76 | 150.42 |
| Batch-8 cached 96-token prefill median, ms | 34.88 | 109.41 | 109.50 |
| Batch-8 cached next-32-token decode median, ms | 279.47 | 556.72 | 580.54 |

The fresh real-byte model has a -0.100116 nat/token paired CE difference but
4.1538 times the training-update cost. Cached next-32-token decoding takes
1.9010 times native at batch1 and 1.9921 times at batch8. Batch8 prefill takes
3.1369 times native. These operations have different work and checking paths;
their ratios must not be combined into a universal inference slowdown.

Selected screens independently show the same broad cost direction: real-byte
update time is 4.1648 times native, cached decoding 1.9318 times at batch1 and
1.9394 times at batch8, and batch8 prefill 2.9791 times. Keep these tuning results
separate from fresh confirmation; do not pool them as additional fresh seeds.

There is substantial local timing variation. Fresh real-byte batch1 full-forward
times range 31.51–64.29 ms (median54.88), whereas its selected-screen range is
29.07–32.79 ms (median29.82). Fresh native batch1 decoding ranges273.47–351.33 ms,
and fresh real-byte decoding524.44–541.19 ms. The five repeated measurements
are not independent trained models or a formal latency confidence interval.
The anomalous full-forward timing is retained, not discarded as a harness error.

## What this supports and what remains unmeasured

The frozen driver times one resident averaged model with two warmups and five
measured repeats at each batch size. Model loading and cache construction/reset
are excluded. The timed consumers retain their existing validation/safety work.
Inputs are the same fixed training prefix repeated across a batch, not generated
text or a representative serving workload. Other machine activity and order
effects were not randomized by this protocol. Driver allocation is sampled at
step ends, not an allocator peak or whole-machine memory measurement. Archive
size is not the complete deployment footprint, which also requires sources and
codebook assets. This analysis checks saved acceptance bindings, not checkpoint
payload bytes again; the original accepted model/inference checks remain authority.

The measured quality difference is at equal training exposure. It does not prove
an advantage at equal wall time: intermediate diagnostics use raw weights and
small changing validation samples, whereas final scores use averaged weights
and the full validation sequence. Interpolating those into an equal-time winner
would mix incompatible measurements. No equal-time quality claim is made.

This extends the short-pilot cost observation to complete 1,200-update training
and the already collected inference timings. It strengthens the existing reason
to investigate implementation overhead after the frozen panel, without identifying
which component causes the slowdown. The prior direct-byte hashing experiment
and padding work count remain narrow leads, not demonstrated integrated gains.
No new architecture hypothesis, literature performance claim, experiment,
selection rule, scientific gate or adoption decision is introduced here. The
existing literature basis is `../quality-evidence001/evidence.md`.

Complete the current three-bundle panel and independent final review under the
original gates. Any eventual quality adoption must report the measured training,
memory and inference costs alongside parameter savings. After the active worker
releases the shared lock, the already identified direct-hash integration can be
tested for exact buffer/digest behavior and measured runtime without changing
the completed quality comparison or presuming its outcome.
