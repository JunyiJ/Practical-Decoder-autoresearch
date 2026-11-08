# Fixed-training-window feature diagnostic

Local evidence: the closed nine-run transform panel favors byte on every seed.
Tanh fits training more slowly and its H200 endpoint is worse than its final
online endpoint. The completed weight-dynamics analysis rules against uniformly
larger relative displacement or stronger global clipping as simple explanations.
Tanh's output table has about20% larger RAW RMS than byte; prior two-window H200
diagnostics observed mean elementwise tanh derivatives near.15-.16. Those windows
were selected for a numerical failure and are not representative training samples.

New question: does bounded feature amplitude and attenuated local sensitivity
appear consistently across the three seeds and both endpoints on the same fixed
training inputs? If tanh features have lower RMS and lower local transform gain
than byte/linear while their output weights are larger, that supports an
amplitude/sensitivity explanation worth distinguishing from orientation and
optimizer effects. If the difference is absent or inconsistent, do not promote
the original two-window observation into a general mechanism. Even a consistent
result is not proof that saturation caused the quality loss or that removing it
would improve quality.

Use every existing RAW and H200 checkpoint from slots00..08. Reconstruct the
original models with their recorded roles solely to load complete saved weights;
no new seed search, training or initialization comparison occurs. Use exactly the
first two training windows, starts0 and128, length128, B1. The same token IDs must
be verified for all36 endpoint/window cases. No labels, vocabulary logits, CE,
validation targets or holdout are evaluated. This is a genuinely new activation
analysis; it does not rerun a closed quality fixture or select a new endpoint.

Compute the final normalized Transformer feature z, preactivation u=Az (u=z for
byte), and mapped feature f=u or tanh(u). Use the qualified ordinary no-grad
FP32 RMSNorm scope; require eight corrected normalization calls per window.
Report FP64-aggregated RMS for z/u/f, f-to-z RMS ratio, and the output-weight RMS.
For tanh report the fraction |f|>=.95, mean d=1-f² and mean d². These thresholds
describe saturation, not an acceptance gate. Report the corresponding identity
or linear derivative values for controls.

For J=diag(d)A, report `sqrt(mean_tokens ||J||_F²/256)`, computed from squared
row norms of A and d². Byte has J=I and gain1. Also report the linear A gain and
their ratio. This is RMS local gain for isotropic infinitesimal input variation;
it is not a full-model Jacobian, a singular-value spectrum, a measured training
gradient, or a prediction of the direction of quality change. No random
perturbations or backward passes are required.

Primary basis: [Glorot and Bengio,AISTATS2010](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf),
§3.2 studies tanh saturation and §4.3 discusses changing activation/gradient
dynamics during learning. Its deep feedforward image models, SGD and separately
tuned long training differ from our single added readout map and Muon/AdamW.
Figure8 also warns that smaller propagated gradients need not imply smaller
weight gradients. This motivates measuring the map, not prescribing a new
activation or initialization. The existing SWA applicability limits remain.

Freeze source/dependency and input receipts before execution. CPU only, one model
at a time, no disk weight copies, shared nonblocking lock,180seconds,4GiB RSS,
4MiB output,8GiB reserve plus2GiB headroom. Expected evidence is under2MiB including
the historical source inventory. Verify checkpoint hashes, exact loaded and
unchanged weights, model semantics, input identities, absent gradients and
unchanged CPU/MPS RNG. Read-only reconstruction may initialize the MPS RNG
interface but performs no MPS model computation. No model adoption or numerical
exploration credit follows; all measures are descriptive and sample-limited.
