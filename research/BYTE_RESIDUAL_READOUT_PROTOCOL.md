# Zero-gated residual readout: implementation pilot and research decision

## Evidence and hypothesis

Measured local evidence: the closed plain-transform panel lost to the byte
baseline on every seed; tanh minus byte H200 mean CE was +.064992.
See `BYTE_CONTEXT_TRANSFORM_RESULTS.md` and the original
`tokenizer-evidence/byte-context-transform-001/final-analysis001/result.json`.
Its `weight-dynamics001/result.json` rules against uniformly greater relative
weight motion or stronger clipping as simple explanations. Its prospectively
fixed `feature-diagnostic001/result.json` shows tanh feature RMS near .91 versus
byte 1.90–1.98, and 66–68% saturation. These first256 training tokens do not
represent the full distribution or establish causality. H200 actually increases
tanh feature RMS, contrary to a simple amplitude-shrinkage account of its penalty.

Primary literature: Bachlechner et al., *ReZero*, UAI2021 final PMLR161,
[paper](https://proceedings.mlr.press/v161/bachlechner21a/bachlechner21a.pdf),
Eq1 initializes a residual scalar at zero, preserving an identity path while
branch parameters initially receive zero gradients. Section5.3/Fig7 reports
faster convergence; Table3 gives the same final1.17BPB for its three12-layer
Transformer variants. The stronger depth results concern previously unstable
64+layer models. Their normalization removal and deep character models differ
from our preserved normalization, four-layer byte-composed token model and
Muon/AdamW. This supports checking initialization, not predicting a local gain.

Our inference: adding a nonlinear correction while preserving z may avoid the
replacement bottleneck and let the model learn whether it needs that correction.
Use byte `Wz/4`, residual-linear `W(z+alpha Az)/4`, and residual-tanh
`W(z+alpha tanh(Az))/4`. A is bias-free256x256 with private U(-1/16,1/16)
initialization; alpha is one unrestricted scalar initialized exactly zero.
There is no additional normalization or shared input/output table. At zero,
features and the input Jacobian equal the baseline. This is a local identity,
not full-model dynamical isometry or an increased softmax rank. Global clipping
includes the gate gradient, so initial optimizer updates need not equal baseline.
Muon can normalize small A gradients; a small gate does not imply uniformly
small branch-weight updates. H200 averaging of A and alpha may also interact.

Contrary local families remain closed: `RESIDUAL_GAIN_RESULTS.md` scaled existing
attention/FFN branches (learned tenth worsened width256 H200 by .020310), and
`tokenizer-evidence/branch-scale-preparation-001/quality001-closure.json`
fixed half block scaling worsened mean CE by .024273. Neither added an
identity-preserving output correction to this byte incumbent. The closed
parallel topology and value-reuse studies changed body connections, and the
plain-context panel replaced z. This new output path is a distinct intervention,
not a retry, changed endpoint or rescue of those trials. The matched linear
residual controls added parameterization, gate learning and identity initialization.

Falsifiable quality prediction: at fixed budget, residual-tanh improves H200 CE
against both byte and residual-linear by more than .02 on average, with every
fresh seed improving and all declared uncertainty bounds negative. Failure
rejects this specific recipe; a linear-only advantage cannot adopt a fallback.
If neither branch helps, adding output capacity remains unsupported. If both
help similarly, an activation-specific explanation remains unsupported. A win
would support this recipe, not prove saturation caused the earlier loss.

## Smallest next test: fixed implementation pilot

New source `byte_residual_readout_v1.py`; no frozen namespace edits. Reserve
fresh pilot roles62102500001–62102500006 after a repository collision check;
never reuse pilot roles for quality. Three arms, CPU and MPS, four updates each
(24 total), existing B8T128 paired_blocks512 sampler, LR.0025 and the inherited
four-step cooldown/H200 fixture. All common weights and both A initializations
must match within each backend. Compare initial logits and all common gradients
exactly with the actual ByteInputModel training path. Analytically verify
input/A/gate gradients on fixed synthetic values at alpha0 and .25 on both
backends (atol2e-6/rtol2e-4); zero gate must give exactly zero A gradient and
identity input gradient. Real first-step A gradient must be zero and gate must
receive a nonzero gradient; later A gradients and gate values must become
nonzero. Include A in Muon at half LR and scalar in AdamW at base LR, preserving
betas(.9,.999),eps1e-6, independent decay.00006 and clip1.

Verify all optimizer states and H200 names include the gate and A, three pilot
snapshots and four completed updates. Serialize RAW and H200 weight-only payloads
in memory, load every parameter exactly, and compare CPU/MPS token losses on
the same first128 training positions using corrected no-grad RMSNorm and stable
CPU CE. Limits remain mean3e-5/max3e-4; compare the CPU reduction with FP64 at
max3e-5. Persist numerical errors before assertions. No validation or holdout
scoring, no tuning, no quality ranking, no pilot checkpoint retained. Preserve
source/input/runtime receipts, sampler/global RNG identities and all failures.
Use the shared nonblocking research lock. Stop after600seconds,4GiB RSS,
6GiB MPS memory or8MiB evidence. Require8GiB disk reserve plus2GiB headroom and
pilot output allowance before entry. Timing is diagnostic, not a speedup claim.

## Proposed complete quality cost, not yet admitted

If the implementation passes, the next decision is one fresh three-seed panel
of these three arms, nine1200-update trajectories, LR.0025, unchanged data and
H200 snapshots800..1200 every10 (41). Primary comparison uses every34278
validation target; RAW remains diagnostic and holdout closed. Preserve mean
improvement>.02, all-seed improvement and negative joint one-sided conditional
text-bootstrap bounds at block lengths2048/4096/8192,2000 shared PCG64 draws,
quantile1-.05/6 for the two contrasts. No LR sweep, seed replacement or pooling
with prior scores. All actual scientific roles and order must be frozen fresh.

Expected training/evaluation cost is about105minutes based on the previous
nine-run103.5minute panel, plus this pilot and source integration. Prospective
hard allowance is10hours total including pilot/recovery and independent result
review, at most1hour per trajectory. Models have16,351,859 versus16,417,396
parameters. Eighteen FP32 RAW/H200 weights require about1.18GB, plus one largest
serialization overlap, source/config/log/report/per-target-loss receipts and
independent review. Measure both endpoint payload sizes now; before freezing a
panel, derive its complete retained high-water allowance from those measurements
and comparable report sizes, plus8GiB reserve and2GiB extra headroom. No panel
launch before that complete requirement fits; execute authorized cleanup if it
does not. A full panel is separate admission; passing this pilot is not evidence
of better quality. This is architecture exploration, not validation tuning.
