# Saved-weight dynamics: bounded descriptive analysis

Local basis: the independently reviewed nine-run byte-context panel is closed
negative. `final-analysis001/result.json` records tanh H200-minus-final-online
loss gaps of+.024026,+.039427,+.019843. In the last two comparable pairs, final
online byte/tanh scores are within.004 but H200 gaps favor byte by.052-.077.
Tanh's last100 training losses remain.672-.724 higher. All nine arms clip every
update in that late window, so clipping occurrence alone does not distinguish
them. The first byte online curve retains its old-scorer limitation.

Question: does tanh's repeated averaging penalty coincide with unusually large
RAW/H200 separation, and which parameter group contributes it? Uniformly larger
common-group movement would be consistent with general late nonstationarity.
Movement concentrated in the output table or transform would instead localize a
follow-up representation question. If the linear control has equal or greater
movement while averaging helps it, displacement magnitude alone is inadequate.
These are descriptive alternatives, not causal tests or adoption criteria.

Analyze every parameter in all18 existing RAW/H200 files, paired within the nine
runs; no checkpoints or seeds are selected by result. Fixed groups are output
table (`lm_head.*`), composer (`composer.*`), Transformer (`transformer.*`) and
added transform (`context_weight`, absent for byte). Report FP64 accumulated
RAW/H200 squared norms, dot product and squared displacement; derive RMS values,
relative displacement `||H-R||/||R||`, norm ratio, cosine, radial displacement
along R and orthogonal relative displacement. Zero denominators produce null,
not invented ratios. Preserve per-parameter rows and group totals, including the
share of squared displacement. Do not compare raw totals across differently sized
groups as if they measured functional importance.

Use the already saved training logs to report mean pre-clip norm and clip
coefficient in the100-update windows ending300,600,900,1200 for every run. This
distinguishes clipping strength from the nondiscriminating occurrence count.
No gradients are recomputed. Link the existing losses, with the original caveat;
do not run a forward pass, rescore quality, access holdout or change endpoints.

External basis: [Izmailov et al., arXiv1803.05407v3,25Feb2019](https://arxiv.org/html/1803.05407v3),
§§2,3.1-3.2 distinguish EMA under decaying learning rates from SWA's constant or
cyclical SGD. Their geometric observations start from pretrained image networks
and explore a high-performing region. Our short Muon/AdamW recipe and nonlinear
readout do not match those conditions. The paper motivates examining weight
space but supplies neither a local averaging guarantee nor a displacement-only
generalization criterion. Norms are parameterization-dependent, and two endpoints
cannot recover a training trajectory or establish saturation/curvature causality.
The prior first-bundle Glorot/SWA review remains relevant and is not overridden.

Cost: CPU only, no model construction/training, no new scientific roles, no
weight copies on disk. Process one checkpoint pair at a time. Shared nonblocking
lock;120seconds,4GiB RSS,8GiB reserve plus2GiB headroom,4MiB output including plan,
all parameter metrics and report. Bind original acceptances/weight hashes,
final review, curves/logs and this source before execution. Either outcome informs
the next measurement; no initialization/activation/averaging sweep is admitted.
