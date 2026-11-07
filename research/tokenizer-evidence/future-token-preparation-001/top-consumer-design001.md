# TOP consumer boundary: six-source trace complete; forwarding owner unresolved

2026-11-07. Applied evidence-grounded-research. READ-ONLY six incumbent consumer
source files,53,825B total/largest18,456B, within6files/256KiB-each/512KiB
source+notes/10min bound. Exact hashes in top-consumer-design001-provenance.json.
No imports or execution of research/Torch/target/objective/oracle/autograd/model/
optimizer/corpus/sampler/RNG/checkpoint/scoring code. No numerical experiments,
old proof/oracle/weight/source-census replay, new executable files or seed roles.

## Source findings that constrain the design

Read final_ln_scale_model_v1/integration.py, runtime/models/{readout_norm,
block_norm}.py, runtime/train/{_grouping,optimizers}.py and
final_ln_scale_worker_v1/training.py. This is the adopted-model wrapper lineage,
not a claim that an early quality protocol's MHA recipe is today's MLA incumbent.

integration.identity requires explicit READOUT_POLICY, D256 MLA, RMS incumbent
blocks/unit branch gains, grad_clip1, independent decay.00006, peakLR.00125,
AdamW(.9,.999) epsilon1e-6/cooldown floor.1 and exact4/20/1200 horizons. It
permits explicit final-scale arms, so that file alone is not a current-arm selector.
block_norm validates learned-scale parameter count28,492,416 (fixed unit removes
256), four causal/dropout-zero MLA blocks, SwiGLU672, final normalized readout,
untied input/output[50257,256] and no output bias. readout_norm constructs an
explicit final LayerNorm, bias learned, epsilon1e-5, with scale-arm validation.
No claim that the normalizer computes a new operation or fixes CE is made.

_grouping assigns two-dimensional transformer.h.* matrices to Muon, RMS matched,
with hidden LR1/sqrt(width_mult); embedding/readout/vectors get AdamW at base LR.
Independent decay is applied once to parameters having gradients before child
steps. optimizers.build_optimizer then enforces exactly44 AdamW groups for learned
scale (43 fixed unit), sets epsilon1e-6 and returns its inherited wrapper. Merely
adding a TOP readout before this constructor fails that exact inventory check.
Wrapping the native model under a prefix and blindly applying the inherited name
rule also loses its transformer.h.* match. Neither changing a fake cfg nor
silently adding a param group after construction is an admitted integration.
An eventual TOP consumer needs its own explicit identity/group registration and
full preflight; no optimizer ledger or probe is justified by this source fact.

training.run_training imports GPT from runtime.models.gpt, creates the model,
requests logits/loss/breakdown, then zero_grad, finite-loss check, backward, one
complete-model clipping call, scheduled LR, optimizer step and EMA update. This
fixes the relevant caller order. It does NOT show GPT's actual CE reduction,
forward multiplier, hidden-state exposure or cache/inference projection path.
Those definitions were not among the six files. Filesystem checks also show no
physical runtime/models/gpt.py or grouped_embedding.py at that import location.
Package path/fallback resolution must be traced. This is an unresolved source
owner, not evidence of a broken runtime or reason to import/execute it.

## Independent prospective contract and real-arithmetic algebra

A coherent distinct TOP source question exists: shared causal hidden states from
ALL four native blocks plus native final normalization, unchanged next-token
projection Wn, and a training-only independent [V,D] projection Wa. No copied
fourth block, no shared unembedding between branched blocks. This differs from
the closed future_two architecture; it cannot complete, reuse, or rescue that
proof or its roles/cases. There is no model implementation admission yet.

Let h be the still-to-be-traced native final hidden state, c its actual readout
forward multiplier (NOT yet source-resolved here), N=B*T, z_n=cWn h and
z_a=cWa h. L_n is genuine mean integer-target CE. L_a is the accepted sparse
weighted mean: -(1/N)sum_i,v w_iv log q_a,iv. Each w is the ONCE represented
Python target mass; m_i=sum_v w_iv need not be exactly1. For explicit weights
a,b, the shared-position derivatives in exact real arithmetic are:

 dL/dz_n = a*(q_n-onehot(y))/N
 dL/dz_a = b*(m*q_a-w)/N
 dL/dh = c*[a*Wn^T(q_n-onehot(y))+b*Wa^T(m*q_a-w)]/N.

Head matrix derivatives are those two residuals outer h times c. The shared
trunk receives its ordinary Jacobian-transposed hidden derivative. No stop-grad
is proposed: stopping the auxiliary-to-h path would remove the hypothesized
representation supervision. Future IDs may enter ONLY detached training targets,
never h. Full T+W split-local raw sequences, all T positions nonempty, direct
future1..W/nearest-type support are required. The previously accepted physical
CPU data view is not automatic acceptance of a new arbitrary-W consumer.

A prospective equal-capacity copied-next control uses the SAME added Wa/head
placement/config/objective weighting as TOP, but its auxiliary target is next
one-hot. Wa is initially copied from Wn, with separate storage/state. This is a
new READOUT copy, distinct from the old copied-block/shared-W control. Both
branches use a=b=1/2 as a candidate source contract, not a tuned optimum or the
authors' summed2*(1-a),2*a convention. An explicit future protocol must freeze
weights/window and equally bounded allowances before any result; none is frozen.

At equal heads and duplicate-next targets, shared gradients equal native in
real arithmetic and the two head gradients are gW/2,gW/2. Thus copied-readout
norm²=||g_shared||²+||gW||²/2, versus native norm²=||g_shared||²+||gW||². This differs from the
old copied-block algebra. Clipping can differ, followed by different epsilon/
momentum histories. Branch symmetry under equal state/policy/determinism does
not prove native updates, CPU-MPS equivalence or runtime correctness. Keep BOTH
fresh native and matched copied-next controls. TOP-minus-copy is an end-to-end
target intervention including gradient norm/clipping, not isolated information.

## Evidence and decision limits

Local original JSON train-fit-diagnostic-001/result: train3.61246929127793,
recorded validation4.69082676267658; different text distributions prevent an
overfit diagnosis. Original input-capacity-boundary-001 records267688 train
IDs/10653 observed types and explicitly warns covariance/frequency counts are
not sampler exposure or a trained benefit. context-information-001 means
32->64 and64->128 gains.025058919884608954/.00977503391898793 are reused fixed
models; context-information-003 local128-at256 improves only.0007308486960410457
on average and not all models. Feature dropout/table changes were previously
contrary. None predicts that an auxiliary head improves generalization.

Existing actually-read primary evidence remains post-data-consumer-review001.md
and top-source-trace001.md: Gloeckle2404.19737v1 shared-trunk supervision with
small-code regressions/scale limits; Zuhri2508.19228v2 method3-4/Algorithm1/
eq9-10/5.1/5.5/Tables7-9/limitations/AppendixB, different340M+/FineWeb52B+/
AdamW4096 and mixed task results. Paninski2003 forbids inferring trained benefit
from sparse empirical entropy. This continuation reused those established
primary readings, made no new paper-retrieval or final-proceedings-PDF claim.
The authors' shifted-window trace is coherent for its pinned current caller;
our explicit target/objective contracts are independently finite-qualified math,
not upstream kernel/paper reproduction. No newly published or local gain claimed.

Hypothesis/inference: distributing supervision over nearby future token types
may favor continuation features while relaxing exact future offset identity.
Falsifiable future prediction: a qualified TOP model must improve next-token
H200 validation versus BOTH fresh native and same-capacity copied-next controls
under unchanged complete quality requirements, equal finite tuning and fresh
confirmation. Native-only improvement cannot distinguish extra readout/scale;
copy-only improvement cannot establish an overall improvement. A loss or failure
against either control rejects adoption. Runtime failure remains scientifically
inconclusive. Auxiliary training loss alone, different target entropy, and short
pilot curves cannot decide the prediction. No panel/roles/budget/selection now.

## Cost accounting: known structural terms, full cost still unadmitted

For learned-scale native P=28,492,416, additional head A=12,865,792, combined
P+A=41,358,208. FP32 weights165,432,832B; an all-parameters upper allowance of
weight+grad+two moment copies+EMA is827,164,160B (Muon need not use two moments).
This does not include body activations/optimizer temporaries/source/runtime.
One B8T128 FP32 vocabulary tensor205,852,672B; simultaneous native/aux logits,
logprobs and backward scratch multiply it. Sequential head backward can reduce
some overlap only with a correctly specified shared-graph lifetime; not proven.
Worst sparse support B*T*W<=131072 pairs gives int64 coordinates2,097,152B and
FP32 masses524,288B before Python objects and staging. FP32 mass underflow for
far offsets is the existing represented-weight contract, not exact real support.

RAW state weights+two-moment allowance+EMA alone661,731,328B, before buffers,
metadata/global RNG/samplers/clipping. Old512MiB RAW caps cannot be inherited as
sufficient. CPU archive staging plus destination moments plus retained model and
EMA can overlap; exact live-state/serialization plan must be costed. Inference
intends to omit Wa and retain one native next-token readout; native inference
parameters113,969,664 FP32 weight bytes under learned scale, native cache196608B/
item is prior accepted scope, NOT newly qualified here. H200/export parent ties
and omission of auxiliary state are unimplemented. Training adds a full vocabulary
projection; it is not free, fused, or timed. No runtime/time/high-water acceptance.

This partial accounting deliberately DOES NOT claim the required complete model
cost. Actual GPT forwarding owner, shared graph operations/scale, serialization
adapter/source lifetime and full timing remain unresolved. Do not write model
source, start a fixture, or borrow old quality forecasts until the remaining
trace and full prospective model/resource plan are complete. No new active
capacity requirement/storage blocker or cleanup trigger was created by arithmetic.
Current useful work is the next bounded package/GPT source-resolution question.

Decision: retain TOP as a coherent conditional source candidate, NOT model-source
admission; close this six-file read only. Next trace package resolution and actual
GPT forward/inference projection first, without rereading these six files. No
successor recapture of future_two/index/CE. Future1255/closedmodelproof1236/
scientificruntime1211 unchanged; ZERO new executable files/scientific roles/
quality budget/numerical exploration/weights/copies/deletions/retirement. All
original evidence/weights protected. Native/holdout/future_two panel unchanged.
