# Full-horizon matched-matrix FFN comparison: source qualification

This is design and SOURCE preparation, not a scientific freeze or launch.
Apply evidence-grounded-research. All earlier studies, fixtures, commands and
1201 predecessor source bindings remain closed/immutable. No old score pooling,
trajectory or missing-report reconstruction, role reuse, or tolerance rescue.

## Measured local basis and contrary evidence

Incumbent source recipe: final-ln-scale-preparation-001/
quality002-slot03-evidence001/config.json and own relu_squared_model_v1/source.
D256/four layers/V50257/context128/native MLA; unit residuals/native block norms,
untied readout/4, Muon eta/2 RMS matching and AdamW(.9,.999),eps1e-6,
independent decay.00006; batch8 paired_blocks512,clip1,1200updates,
linear cooldown from800 to final.1,spaced H200 start800 interval10 =>41 snapshots.
The proposed comparison changes only its prescribed FFN recipes/finite LR choice.

Original ffn-structure-preparation-001/quality001-final-review.json and its
selection/accepted anchors were read. UngatedSiLU1008 minus SwiGLU672 fresh
deltas were+.013767702604872,-.004855530602348,+.013253719097900;
original mean+.007388630366808, two worse, closed not adopted. Both selected
.00125 from their old finite screen. This is contrary evidence against assuming
removing a gate improves quality, not a test of ungated ReLU-squared. Never pool
those seeds/scores with a new panel. activation-preparation-001/
inference-recovery002-closed-review.json was read: exactGELUgate minus SiLUgate
mean+.001550755925793319, two worse, also not adopted. Its original training
succeeded and later inference recovery closed; do not label it a training failure.

New original relu-squared-worker001/pilot001 plans/reviews/accepted worker metrics
and relu-squared-prerequisite-chain001-closure001 were read, without replaying
closure or weight audits. All3four accepted BEFORE any20 construction;72updates,
73728targets, all88/84 exposures and complete RAW/H200/EMA/sampler/global RNG;
physical GPT2 FULL34278 CPUANDMPS validation and durable pairs/cache/reset passed.
All four/twenty clipped-update counts were zero. Five twenty-update online points
are early, sparse and one cohort; they establish neither final quality nor a
clipping/conditioning mechanism. They do not select an LR or optimizer recipe.
Native twenty training5.851748s,ungatedSiLU5.528434s,squared5.672137s; first-step
cold start and serial order matter. Max observed update after first .314903792s,
max worker47.854000s/parent17.961159s; RSS2792112128B/MPS2553167872B. These are
cost priors and prerequisites, not full-horizon time or speedup guarantees.

The failed projection QUALITY001 remains CLOSED_FAILED_INCOMPLETE; its two
missing reports stay unavailable. No normalizer/backend/precision change is
inferred or made. The successful new paired publisher preserves future failure
evidence, and does not explain the old failure.

## Primary literature actually retrieved and read at this boundary

[Primer2109.08668v1](https://arxiv.org/html/2109.08668v1), section3 SquaredReLU/
Figure6 discussion, section4 comparison, A.7 insertion/ablation and A.8 setup:
two-matrix squared ReLU shares a projection in its ReGLU interpretation. Its
activation comparison motivates a quadratic positive branch, while the combined
Primer/convolution results cannot isolate this activation. Adafactor, long
LM1B/C4/PG19 runs, tokenizers, attention and hardware differ from this local
Muon/AdamW ShakespeareBPE/1200-update setting; published quality/compute gains
are not local gains. No convolution, published speed multiplier or tuned scale
is imported here.

[GLUVariants2002.05202v1](https://arxiv.org/html/2002.05202v1), section2 equations4-6,
3.1/Table1 and3.2: the three-matrix GLU comparison uses two-thirds hidden width
against two-matrix FFNs; SwiGLU/GEGLU are strong comparators in T5 span filling.
That task and Adafactor setup differ locally. This challenges an assumption that
simpler ungated FFNs must dominate, and supports retaining genuine SwiGLU.

These paragraphs summarize the retrieved versions, not a new literature search
result or an inference that their operating points are optimal here.

## Mechanistic inference, falsifiable prediction and learning

Candidate: Wdown[max(Wup x,0)^2], H1008. Matched ungated control:
Wdown SiLU(Wup x),H1008. Native control:
Wdown[SiLU(Wgate x)*Wup x],H672. No independent squared gate, bias, dropout,
rescale, normalizer, router or convolution. Each has516096 FFN matrix elements
per layer and28492416 learned model elements; native88tensors/44Muon versus
ungated84/40Muon, both44AdamW. Equal elements do not establish equal runtime.

Our inference is that the positive quadratic self-interaction may retain useful
multiplicative behavior missing from asymptotically linear ungatedSiLU, without
an independent gate. The accepted source uses paired ungated up/down weights
and exact common native-first nonFFN initialization. Squared-versus-ungatedSiLU
therefore distinguishes these two recipes under their shared initialization;
native-versus-ungated also changes hidden width/factorization/initialization and
cannot identify an isolated gating cause. No optimal initialization claim.

Falsifiable adoption-level prediction: after equal finite LR allowance and
independent fresh confirmation, squared ReLU improves mean H200 native CE by
strictly more than.02 against EACH control, all3 paired deltas negative for each,
and all6 prespecified simultaneous conditional-text upper bounds negative.
Winning against ungatedSiLU alone is insufficient. A failure narrows this fixed
recipe; a successful result justifies separate adoption/consumer review, not a
universal polynomial/conditioning explanation. Incomplete numerical execution
remains an implementation result, never negative quality evidence.

## Prospective finite design and complete cost

Nine screens:3arms x LR{.000625,.00125,.0025},one NEW paired cohort; no boundary
extension/extra tuning. Select minimum complete H200 MPS CE independently perarm,
lower LR on exact tie. These are finite choices, not optima. Then9 fresh runs:
3new cohorts x3selected arms, rotated arm order. No screen scores in confirmation.
Each run has1200updates/1228800targets and41H200 snapshots. Only squared ReLU is
candidate; no control, RAW, cost-only or partial-panel fallback. Keep all36weights.

Every cohort needs NEW distinct initializer/privateungated/train/eval/globalCPU/
globalMPS roles; screen+3fresh cohorts plus bootstrap =>25roles. No values proposed
or reserved here. Each arm/rate resets its cohort streams deliberately; no used
historical seed is borrowed. New census and proof roles must be separately
reviewed before construction/draws. Full validation remains reused validation
text, holdout closed. Two contrasts x blocks2048/4096/8192,2000PCG64 draws perwidth,
shared indices across seeds/contrasts,linear quantile1-.05/6; intervals condition
on these trained seeds/text, not population seeds or holdout. Independent final
statistics groups128-target windows and computes the linear quantile literally.

Source retained-output forecast:18pairs x640MiB=12079595520B,plus512MiB root
metadata,128MiB sidecars,NEW prerequisite3pairs x640MiB=2013265920B,
128MiB consumer workspace,8GiB reserve and2MiB entry =>floor23490199552B;
extra2GiB headroom =>target25637683200B. Coinciding with an older family's numeric
floor is new arithmetic from these prospective own consumers, never old output
credit. .tmp shares its pair allowance; every failed partial stays and closes
execution. The three prerequisite pairs are prospective, not a permission to
write fixtures or discard prior outputs. Their actual source/cases/retention
must fit before admission; a larger prerequisite expands the forecast first.
Current~35.74GB free covers this SOURCE forecast; no cleanup/storage blocker.

Working forecast3311381008B includes parent RAW CPU read up to512MiB, live/saved
EMA, moment staging and loader deepclone, CPU model and RAW/H200 copies, full
B8 log-probability arrays,16MiB full-history metadata and runtime margin. MPS
forecast3.5GiB; guards remain RSS4GiB/MPS6GiB/reserve8GiB. For a conservative cost
prior use2*1200*.314903792 +.953573917 +2*47.854001 =852.430677s perworker;
parent2*17.961159+30=65.922318s. All18 plus240s whole9fresh serving,120s primary,
120s independent statistics and120s closure margin =>17130.35391s (~4.76h).
Forecasts are unqualified by the new runtime. Limits stay7200s perworker/parent,
600s fresh serving and120s CPU/statistical phases, one localMPSworker/fourthreads,
fallback0fastmath0. Model/load/cache allocation are budgeted even when excluded
from descriptive serving medians. No paid compute or subagents.

Rank this as a LOCAL FFN followup, not broad architecture or interaction credit.
The old table/localconv/depth/gate/MoE/readout outcomes and rejected anchor offer
no newly qualified replay. Carry broad/interaction allocation forward. Reading,
source implementation and correctness preparation do not count as numerical
exploration. No quality budget or scientific freeze exists yet.
