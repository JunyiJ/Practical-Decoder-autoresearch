# Body allocation geometry and optimizer applicability

The bounded [CPU inventory](geometry001.json) passed with the actual incumbent
constructor and optimizer; [source](geometry001.py) and resource guard are saved.
It verifies28,492,416 parameters,88 tensors,44 Muon and44 AdamW groups, unique
optimizer coverage and196,608 cache bytes per item. All918 frozen sources remain
unchanged. No forward pass, optimizer update, MPS work, checkpoint loading or
writing, holdout access or scientific seed assignment occurred. The temporary
counting model used real CPU storage inside a restored private RNG scope.

Full-model meta construction was rejected by the existing normalizer anti-alias
check because meta storage pointers are zero. The audit used real CPU storage
instead of weakening that check. Candidate leaf-shape arithmetic on meta is
explicitly separate from validated full-model construction.

## A controlled structural question

A deeper, narrower body can closely match incumbent non-embedding capacity while
changing the dominant vocabulary tables. That alone would confound a depth
claim. The next prototype should therefore include a width-matched shallow
control, not just compare a smaller deeper model with the incumbent.

| Proposed role | Layers and width | FFN hidden width | Body parameters including final norm | Total parameters | Projected cache bytes per item | Forward matmul FLOPs per token |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Incumbent reference | 4 by256 | 672 | 2,760,832 | 28,492,416 | 196,608, verified actual | 32,023,040 |
| Width-matched allocation control | 4 by192 | 1024 | 2,758,784 | 22,057,472 | 163,840 | 25,393,536 |
| Deeper allocation candidate | 7 by192 | 504 | 2,730,944 | 22,029,632 | 286,720 | 25,771,392 |

The latter two have identical vocabulary-table width, head count6 and latent
rank48. Their body parameter counts differ by about1.01%, and the forward
matmul estimate differs by about1.49%. They allocate similar capacity to deeper
serial computation versus wider FFNs. They do not isolate an abstract effect
of depth independent of all other structural choices.

Against the incumbent, the seven-layer candidate has1.08% fewer body parameters
and22.68% fewer total parameters. Its cache estimate is45.83% larger, and75%
larger than the width-matched shallow control's. Lower weight counts therefore
do not imply lower serving-state memory or latency. Its forward matmul estimate
is19.52% lower than the incumbent's, principally because the vocabulary projection
is narrower; this is not measured training FLOPs, speed, peak memory or quality.

The deterministic feasibility enumeration considered depths2 through16 and
widths128 through384 in multiples of64, retaining body matches within2%. It
also found16-by128, but that row needs2.67 times the incumbent cache and a much
greater serial depth. Keep it as a feasibility observation rather than enlarge
the first numerical study. This enumeration is not an LR or quality screen.

## Reproducible arithmetic and limits

Let width be D, depth L, vocabulary V=50,257, context T=128 and latent rank r=D/4.
Current head and rotary widths are32. The observed eight MLA projections have
6Dr+32D+D² weights and2r+5D+32 biases per block. With bias-free SwiGLU width H,
two RMSNorm vectors per block and affine final LayerNorm, body parameters are

`L * (6*D*r + 32*D + D*D + 2*r + 5*D + 32 + 3*D*H + 2*D) + 2*D`.

Tables contribute2VD parameters. Cache storage is `4*L*T*(r+32)` bytes per item.
The FLOP estimate counts multiply/add as two, includes the dense output head
and full-square QK/AV attention, and excludes softmax, norms, activations,
embedding reduction, backward, optimizer, dispatch and I/O. It cannot predict
MPS wall time.

For the four-by192 control, H=1024 is the multiple of8 nearest the incumbent's
body count, searching H=8 through2048 with lower-width ties. The actual SwiGLU
leaf constructor independently confirms589,824 FFN parameters per layer.
The saved [control projection](matched-width-control001.json) contains the
resulting counts. An explicit new FFN-width contract is required; this is not
an override of the old21D/8 policy.

FP32 raw-plus-H200 tensor payload estimates are558,969,520B for the incumbent,
430,270,640B for the shallow192 control and429,941,288B for deeper192. They charge
model/EMA/H200, Muon momentum, all Adam moments/step scalars and persistent
buffers. They exclude headers, RNG, clipping history, metadata, publication
overlap and retained earlier stages. They are not a full-panel forecast or a
launch-capacity gate. No conditional retirement credit or storage wait is created.

## Dense and MoE alternative

The refreshed component projection on the current incumbent gives:

| FFN proposal at four-by256 | Total parameters | Muon groups | AdamW groups | Scope |
| --- | ---: | ---: | ---: | --- |
| Dense672 | 28,492,416 | 44 | 44 | Constructed incumbent verified |
| Dense1344 | 30,556,800 | 44 | 44 | Hypothetical wider FFN replacement |
| Four experts336, top two | 30,560,896 | 80 | 48 | Hypothetical expert replacement and router |

The expert proposal matches dense672's active FFN matrix parameters plus router
work, and approximately dense1344's stored FFN capacity. It still needs dense
controls, cumulative exposure/routing diagnostics and actual MPS measurements.
Its local character width reversal, added stored state and implementation cost
do not justify choosing it merely because the family is different. The old
storage projection was not rerun and is not a current dispatch blocker.

## What the existing contracts permit

The current contract correctly rejects layer7, width192 and MoE. Width changes
also conflict with the width256 grouped embedding backward and final-norm
contracts; extra depth conflicts with exactly eight block normalizers and the
fixed44/43 AdamW grouping check. None was patched or disabled.

Applying the old hidden-matrix width expression to D192 would give peak LR
0.0007216878 for those groups and a readout multiplier1/3, versus0.000625 and1/4
at D256 with the same nominal peak0.00125. That is algebraic interpolation of
the existing rule, not evidence of depth transfer or an optimal LR. Changing
FFN aspect ratio is another applicability gap. The projected deeper model has
77 Muon and74 AdamW groups,151 tensors; those require actual independent routing,
coverage, decay, EMA and resume tests in a new implementation.

## Primary literature and contrary local evidence

[Bordelon et al., arXiv2309.16620v2](https://arxiv.org/pdf/2309.16620v2),
section2/equation1/table1, combines residual-depth scaling with its specified
parameterization to study transfer; its experiments include vision models.
It supplies a mechanism to investigate, not a rule that can be copied alone
into our Muon/AdamW MLA model.

[Yang et al., arXiv2310.02244v1](https://arxiv.org/pdf/2310.02244v1), table1 and
sections9.1/9.2, distinguishes shallow residual branches from deeper branches
and demonstrates hyperparameter shifts in the latter setting. This cautions
against equating a depth multiplier with guaranteed transfer for our multi-matrix
attention/SwiGLU branches. These scoped results do not prove a universal failure
or success for seven local layers.

The local [fixed-half branch study](../../TOKENIZER_BRANCH_SCALE_PREPARATION.md)
lost by mean +0.024273102 at the incumbent depth. That argues against blindly
shrinking current branches, but it is not a test of a depth-dependent scheme at
new geometry. Historical extra-depth gains also weakened at character width256.
Retain those negative findings; do not reopen their closed fixtures or infer
that a local recipe label establishes a general theoretical parameterization.

## Next executable work

Implement only the additive model/component contracts needed to test the
incumbent and two192-wide allocations, outside all918 frozen sources. Preserve
the incumbent's actual equations, initialization and groups exactly. Specify
new shape signatures, grouped embedding backward, block/final normalization,
MLA and cache geometry, FFN widths, optimizer routing and EMA/checkpoint identity;
reject foreign geometry rather than falling back to old policies.
The initial prototype keeps unit branch gains and the adopted optimizer recipe;
do not add a depth multiplier from a paper by itself. Any depth-scaling variant
would be a separate evidence-qualified question and prospective design.

The working hypothesis is that allocating nearly equal body/vocabulary capacity
to deeper composition can improve the fresh validation comparison relative
to the width-matched shallow control and incumbent. Failure against either
predeclared quality contrast counts against that candidate. A shallow-control
win is informative, not a posthoc adoption fallback. Shared-seed initialization
and equal bounded LR allowances need prospective definition; paired seeds do
not imply equal shared weights across changed shapes.

No candidate is yet numerically admitted. First complete actual forward/gradient,
causality, cache/reset, grouping, resume and H200 prerequisites, then calculate
complete retained-output/resource high-water for all controls, tuning and three
fresh confirmations per contrast. Do not launch a partial panel or relax guards.
Any new capacity blocker at that stage triggers immediate authorized recovery.
