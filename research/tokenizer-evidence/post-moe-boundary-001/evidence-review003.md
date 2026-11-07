# Four independent gradient cohorts: complete, with ratio instability

The new three-file `tokenizer_gradient_cohorts_v1` runtime and generated proof
preserve all 1043 predecessors, giving 1047 immutable source bindings. New private
fixture roles 600501–600504 supplied four independent cohorts, 600505 the CPU-only
initializer, and 600506 the generated proof. Their nonuse audit preceded source
creation. These roles are now used; none is a scientific quality seed.

The generated proof passed six FP32/FP64 ordinary, constant and zero-mean cases
against independently stacked vectors (maximum error 1.78e-15). It rejected 76
malformed, reordered, foreign-seed, partial-count, inventory, nonfinite or output
budget cases before destination mutation. Negative signals remained negative and
ratios undefined. It did not replay the closed CPU/MPS gradient proof.

The actual diagnostic completed four cohorts × 32 B8 draws × three fixed RAW
controls (slots 09/14/16), 393216 target evaluations. The first probe's 600492
draws are excluded. All 88 gradients were streamed through separate cohort and
combined CPU FP64 moments. No optimizer update, clipping application, EMA,
validation/holdout encoding, bootstrap or tensor output occurred. Every cohort
was retained. Identical cohort streams across fixed models and private sampler
replay passed; source, original RAW files, complete model state and global
CPU/MPS RNG were unchanged. The copied initializer construction uses the default
CPU generator inside a CPU RNG fork, without changing the old initializer policy.

The original 600 s / RSS 4 GiB / MPS driver 6 GiB / reserve 8 GiB guard finished
in 122.215 s, peak RSS 2398978048 B and MPS driver 2477211648 B. Full remaining
metadata/reserve/external/entry floor 9876074496 B and target 12023558144 B fit
80622993408 B free at entry. No storage blocker, cleanup or retirement arose.
The new root's terminal STOP closes every original command; no worker remains.

The independent scalar review loaded no model or gradients. Per-parameter norm
identities, aggregates, group sums and within/between-cohort decomposition agreed
to 6.94e-17. Seven closed metadata files totaling 3043949 B were mirrored without
weights. `cohort-review001.json` binds original evidence; all prior closures and
weights remain protected. Neither training nor bootstrap was replayed.

| RAW slot | Four cohort ratios, B8 draw units | Combined 128-draw ratio | Combined corrected signal |
|---|---|---:|---:|
|09|203.37, 368.32, 99.29, 109.94|118.2461|.00612434|
|14|72.69, 41.60, 45.91, 61.30|49.0417|.01508729|
|16|83.37, 299.76, 117.62, 81.34|95.3104|.00815193|

All twelve cohort signals and all three combined signals are positive; this does
not make the ratios unbiased. The combined correction still subtracts 48.0%,
27.7% and 42.7% of the measured mean-gradient squared norm. The first probe's
69.38/45.34/72.51 ratios remain a separate result, not pooled evidence.

## What the existing records distinguish

`existing-analysis002` used only closed scalar metadata. Across four cohorts,
trace covariance max/min is 1.067/1.062/1.031, while corrected signal max/min is
3.475/1.818/3.674. The broad ratio ranges therefore primarily reflect denominator
variation in these records. This is descriptive, not a population confidence
claim or a causal bottleneck finding. The largest single draw contributes
1.2–1.5% of total squared-gradient norm; the largest eight contribute 8.6–9.5%.
Shared-input norm correlations between models are .629–.724, which are not three
independent replications of the probe inputs or evidence of semantic difficulty.

The raw covariance remains large relative to corrected signal, but this does not
establish a stable critical batch, optimal batch, preconditioned update noise or
quality gain. Do not add cohorts or select favorable cohorts to rescue the ratio.
No B16/B32, LR, beta, decay, clipping, horizon or architecture quality panel is
admitted. Historical character/MHA larger-batch results remain contrary context:
both larger recipes failed the all-width neutrality rule, and each worsened one
D256 seed despite lower D256 mean loss. Their LR/update-count/momentum-time/decay
changes differ from native BPE/MLA and are not pooled here.

## Next prerequisite, based on the actual optimizer

Using the evidence-grounded-research skill, the next enabled work is a small
NEW generated optimizer-response component/proof, not more raw sampling or batch
training. Source inspection confirms global clipping precedes the 44 Muon and
44 AdamW groups. Muon uses momentum .95, Nesterov, five Newton-Schulz steps with
internal bfloat16 arithmetic and shape-adjusted RMS LR. AdamW uses beta .9/.999,
bias correction and epsilon 1e-6. Independent decay .00006 is applied once to
active parameters. Thus raw-gradient Euclidean geometry alone does not describe
the stateful nonlinear response, and averaging gradients cannot simply be
identified with averaging optimizer responses. Existing optimizer state was not
restored or advanced in either completed gradient diagnostic.

[McCandlish et al. 1812.06162v1](https://arxiv.org/html/1812.06162v1) was retrieved
again: §2.4 describes conditioning, local-step and LR assumptions; Appendix A.1
warns about ratio bias and cancellation in the signal estimate; E.3 limits
extrapolation to generalization. These support caution, not a local batch gain.
The paper does not validate our Muon/Adam setting or correlated tokens as IID.

See `optimizer-response-design001.json` for source hashes, generated controls,
falsifiable oracle-agreement prediction and a 120 s CPU / 2 MiB metadata budget
with the unchanged 4 GiB RSS and 8 GiB reserve. Use tiny deterministic synthetic
states, genuine installed optimizers and independent arithmetic; no corpus,
model, RAW payload, scientific seed or actual-model update. Agreement enables
only a separate response-diagnostic design; disagreement closes the proof and
requires a separate additive correction. No actual endpoint response diagnostic
or quality campaign is frozen. All 1047 current sources and closed commands stay
immutable. Preparation is not numerical quality exploration; the 70/20/10 split
remains proposed. Keep the native dense incumbent.
