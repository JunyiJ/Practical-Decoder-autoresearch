# Post-data consumer boundary: reject counts; clarify controls; review TOP source

November 7, 2026. Applied evidence-grounded-research. This is a completed
literature/source/record review, with real-arithmetic derivations below. No new
corpus encoding/analysis, research/Torch imports, extracted-function execution, tensors,
model, optimizer, sampler, RNG, scoring, checkpoint or numerical fixture calls.
No old proof/audit replay. Future source bindings remain1248; old scientific
runtime1211 and closed model proof1236 remain unchanged. No quality conclusion.

## Actual local basis and limitations

Read TOKENIZER_QUALITY_PROTOCOL.md and the architecture/attention/exploration
roadmaps, current preparation/progress and post-physical-data handoff; original
physical-data acceptance/completed006 and model-proof001 failure review. The
finite CPU data view is accepted. Model/index failures remain closed and do not
qualify auxiliary construction, backward, cache, state, workers or science.

Read original JSON results, without loading/scoring their models:

| Original record under tokenizer-evidence | Relevant observation |
| --- | --- |
| train-fit-diagnostic-001/result.json | Mean train3.61246929127793; recorded val4.69082676267658; gap1.0783574713986506. Different texts/distributions; not an overfit diagnosis. |
| input-capacity-boundary-001/result.json and evidence-review.md | Frequency-weighted input concentration matches the Gaussian null; does not identify a learned input bottleneck. Complete input dropout comparison lost all three fresh pairs, mean+.014701832892. |
| generalization-boundary-001/evidence-review.md | Train/val frequency shift and unseen validation IDs complicate the gap. Rare-token fit correlation is not a causal representation diagnosis. |
| context-information-001/result.json | Fixed-weight32/64/128 CE4.725660716480177/4.700601796595568/4.69082676267658. |
| context-information-002/result.json | Global256 CE4.7070645257861505; all-model improvement false. |
| context-information-003/result.json | Local128-at256 CE4.69009591398054; gain versus128 only.0007308486960410457, inconsistent across models. |
| exploration-boundary-002/analysis001.json | Earlier train-only speaker-marker census already measured label availability: .6649143129786171 at128 versus.8136753941573158 at256; not model usage. Target exposure4.590418696392815 is not literal epochs. |

These records constrain applicability. They are not fresh confirmation, pooled
scores, a reason to reopen context/dropout/table families, or evidence that
future-two is a winner or loser.

## Decision: do not admit TRAIN-ONLY offset entropy/ambiguity counts

The proposed count study would add a new descriptive offset statistic to prior
frequency/context availability analyses, but has no specified outcome that
changes the model decision. High low-order dependence is compatible with both
positive and negative transfer through a limited shared trunk. Low dependence
at a short counted context does not rule out useful structure at neural128
context. Unique long contexts can make empirical conditional entropy vanish
without demonstrating generalization. Counts never identify gradient alignment
or learned feature use. No new corpus analysis/source/roles/census admitted.

Actually retrieved/read [Paninski, Neural Computation15,2003](https://www.cns.nyu.edu/pub/lcv/paninski-infoEst-2003.pdf),
introduction and§3.3/eq3.5/Figure1: plug-in entropy has negative bias; small
variance does not ensure accuracy; mutual-information bias has no universal
positive sign. The i.i.d. results are not an error certificate for overlapping
Shakespeare windows. Smoothing or a text bootstrap alone would not close this
applicability gap. We did not run an estimator or estimate this corpus's bias.

Re-read [Gloeckle2404.19737v1](https://arxiv.org/html/2404.19737v1),§2/eq2–3,
§3.1/3.5/3.7/4.1/5.1–5.2 and AppendixL.2. Shared-trunk future supervision and
sequential backward motivate the objective. Small-code regressions, mixed NLP,
and an induction epoch oracle limit transfer. AppendixL.2 separates ground-truth
entropy from model-relative cross-entropy; a measured corpus MI is not measured
representation benefit. Their parameter matching and scale differ locally.
No published generation improvement is a local next-token CE improvement.

## Completed independent source question: duplicate-next control interpretation

Read own frozen future_token_model_v1/model.py:55–67, backward.py and
optimizer.py:127–150, inherited final_ln_scale runtime/train/_grouping.py:25–49,
and installed torch/optim/_muon.py:31–70,318–350 and nn/utils/clip_grad.py:164–181.
This is read-only source reasoning, not execution or model proof acceptance.

Question: can the copied duplicate-next branch be treated as a replicated native
trajectory or independent ensemble? No. Let S denote input table+first three
blocks, H denote block4+final norm, and W the single shared readout. On the same
batch at equal head values, let gS,gH,gW be native real-arithmetic CE gradients.
Both duplicate heads get the same input/target, with weights1/2 each. Its
gradient vector is (gS,gH/2,gH/2,gW), whereas native is (gS,gH,gW).

Thus ND²=||gS||²+||gW||²+||gH||²/2 and NN²=ND²+||gH||²/2. For the source clip
coefficient c(N)=min(1,1/(N+1e-6)), c(ND)>=c(NN). Equality of forward loss does
not imply equality of clipped gradients or updates. This identity is conditional
on symmetric heads and same parameters/batch; it says nothing measured about
clipping counts, trained states, or differing subsequent native trajectories.

Under exact equal corresponding branch state, deterministic computation, no
branch-specific randomness and identical corresponding optimizer policies,
the two duplicate branches remain equal by induction: equal gradients, moments,
decay and updates. This symmetry does not imply equality with native, bitwise
CPU/MPS equality, or runtime acceptance. It is a redundant-task control, not two
independently learned native heads. Sequential finite arithmetic remains unproved.

Do not invoke optimizer scale invariance as an equivalence proof. AdamW's fixed
epsilon changes m/(sqrt(v)+epsilon) under constant gradient rescaling; varying
clip factors additionally change moment history. Installed Muon casts to BF16,
normalizes with a clamped norm and uses momentum; no blanket native-update
equivalence follows from an ideal scale-invariant normalization. Independent
decay is once per parameter, not multiplied by loss weight. No optimizer change
or gradient/norm probe is proposed to eliminate these differences.

Decision: retain BOTH native and duplicate-next as prospective controls. The
future-two/duplicate comparison varies the target at matched architecture,
objective weights and optimizer policy. Its end-to-end effect includes changes
in gradient direction, norm and clipping; it is not a causal isolation of
information content. A native-only win would remain insufficient. This refines
interpretation, not frozen source or gates; no new experiment is needed to
establish this conditional algebra.

## Ranked new consumer: token-order target semantics and local cost

A primary-source search found Zuhri/Fuadi/Aji's TOP work. Read
[arXiv2508.19228v2](https://arxiv.org/html/2508.19228v2),§3–4/Algorithm1/eq9–10,
§5.1/5.5/Tables7–9/§7/AppendixB; also compared v1's method. PMLR306 landing page
confirms publication; its octet-stream PDF fetch failed, so no claim to have read
the final proceedings PDF. v2 reports an auxiliary proximity-ranking head and
mixed per-task results; some Lambada perplexities improve. Larger-model/data,
AdamW/4096 context, window and loss-weight ablations do not establish our CE gain.
Lower raw auxiliary loss is not evidence of an easier task unless target entropy
and normalization are accounted for.

New mechanism to evaluate: a distribution over nearby future token types could
retain continuation supervision while relaxing exact offset identity. This is
distinct from fixing/replaying the failed future-token model. It is currently
a SOURCE/LITERATURE question only, not an adopted direction or admitted model.

Retrieved the authors' landing-repository README, naive/Triton target builders
and ListNet loss via GitHub connector, without importing/executing them. Head
resolved to b347f4850887ada141231c6d1dd8492e0f20e28f. Original Git blob IDs:
README e22aaeb1c7091f448686485a54355bffb13fe7a9;
seq_to_top_naive.py e05eaa42f21fdd8353687d6a6320d5c8490707a5;
seq_to_top_triton.py f51bbc5e271a0c89033b23eb69bdc71c8157914e;
listnet_loss.py 6d36c6902c64f746b4a4cf26749a02c15e95b6e3.
These are Git blob IDs, not invented local SHA256/executed-source receipts.

There is a concrete unresolved target contract: Algorithm1 updates the current
token before requiring0<distance<=W; the naive builder's validity allows zero
through W; Triton permits distance<W. Builders document an already left-shifted
input. Consequently one cannot infer training targets from the pseudocode alone.
Current-token repeats, unseen-token sentinel values, right endpoints, padding,
and caller shifts need reconciliation. The naive ListNet helper also turns
all-masked softmax NaNs into zero; silently empty supervision is not acceptable
as a new local contract. These are static observations, not measured kernel
failures or a claim that the authors' training used incorrect targets.

Local cost inference: D256*V50257 gives12,865,792 additional head parameters,
51,463,168 FP32 weight bytes, versus the prepared MTP copy's690,592 parameters.
Native plus that head would have41,358,208 parameters. Five FP32 copies for
weights/gradients/two Adam moments/EMA alone add257,315,840B. One B8T128 full
FP32 vocabulary tensor is205,852,672B, before its gradient or any dense targets.
Triton/CUDA fusion in the paper is not an available MPS cost proof. No free
compute, lower-memory, RAW archive cap or complete quality budget is claimed.
Large output-table cost is especially relevant given the local input-rank/null
and vocabulary exposure findings; an extra projection is not automatically cheap.

Ranking: (1) resolve this new published target/caller contract and decide if a
bounded independent target-only component is meaningful; (2) evidence review
of other auxiliary prediction tasks only if that mechanism fails applicability;
(3) offset counts rejected, longer-context/table/regularizer repeats lower
priority without new evidence. No quota-driven probes or automatic TOP promotion.

Falsifier/decision: if the training caller cannot be reconciled with an explicit
causal future-only, nonempty, split-local target contract, do not implement or
test that recipe as a reproduction. If it can, distinguish the published method
from any local change and then prospectively specify literal cases/oracle/cost
for a target-only consumer. Either outcome resolves an actual implementation
assumption; it does not decide quality. No model proof successor is authorized.

## Cost, preserved scope and next action

This review used text/JSON/source reads, primary web retrieval and written
derivations only; no numerical exploration time. No source hashes/bindings were
added and no old weights, binary kernels or closed receipts were re-audited.
Retained new notes/receipt cap2MiB, no tensor/weight outputs;8GiB reserve plus
2GiB headroom gives10,739,515,392B target. Previous observed free35,114,741,760B
was ample; no material storage blocker/cleanup trigger was discovered.

Next executable action is the bounded read-only authors' training-caller trace
in top-target-source-handoff001.md. No new corpus, sampler, RNG, oracle, model,
source execution, scientific roles, quality budget/freeze or launch. Closed
index/model/CE fixtures and all sources/evidence remain immutable. Main active
five-minute continuation advances that concrete question; old research paused.
