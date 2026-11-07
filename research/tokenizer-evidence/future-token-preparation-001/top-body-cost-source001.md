# TOP actual body source/cost001: graph specified; attention namespace still open

2026-11-07. Read6 NEW sources26,592B/largest10,621B within the6-file/256KiB-each/
512KiB source+notes/10min/60s-tool bound. No prior18 source repeats, research or
Torch imports/functions, tensor/autograd/model/optimizer/RNG/corpus/sampler/
checkpoint/scoring calls, closed proof/oracle/weight/kernel/census/audit replay.
Exact read identities in top-body-cost-source001-provenance.json. The existing
primary/local evidence boundary remains top-consumer-design001.md; this is
implementation applicability/cost, not a new scientific result.

## Body and normalizer operations actually read

Selected relative .blocks definition is local_mix_v1/runtime/models/blocks.py.
For incumbent serial unit-branch MLA/SwiGLU/no mixer policy, it normalizes input
for attention, applies attention, adds the complete attention branch to residual,
normalizes that new residual for FFN, then adds the complete FFN branch. The
parallel and causal-three-tap mixer paths are explicit alternatives; they are
not silently included in TOP. RoPE buffers are registered per block. The block
returns x and optional aux_loss; no new hidden output or changed operation.

FFN-structure mlp.SwiGLU creates gate/up/down bias-free Linear modules, SiLU(gate),
product with up(x), optional product norm, then down/dropout. Incumbent hidden
width672/dropout0/no product norm/SiLU were established by prior contract reads.
No independent gate-square, replacement activation or extra normalizer proposed.

block_norm_v1/spec.py calls EXACT nn.RMSNorm(256,eps=None) for rms_incumbent.
checked_forward validates input and module then delegates module(x), without new
normalizer math. FinalLayerNorm uses F.layer_norm(width,weight,bias,eps1e-5),
learned bias and explicit learned/fixed-unit scale. Both remain ordinary native
backend operations; no anchor, precision override, epsilon or CE change.

Read physical block_norm_v1/runtime/attention/mla.py, the only non-evidence
runtime/attention/mla.py found by filename inventory. Its MLA compresses q/kv to
64, expands content/head projections, adds separate rotary q/key coordinates,
concatenates per head, materializes q@k^T, divides by sqrt(32+32), applies explicit
score multiplier, causal masked_fill, torch.softmax/dropout, att@v and output
projection/dropout. It caches shared compressed KV64 and rotary key32. This is
not fused attention. At B8/T128/H8 the attention matrix is1,048,576 FP32 elements
(4,194,304B) per layer; q/k each524,288 elements(2,097,152B), FFN gate/up/product
each688,128 elements(2,752,512B), hidden262,144 elements(1,048,576B). These are
shape arithmetic, not allocation profiling or autograd workspace measurements.

## Exact remaining source identity boundary

The final-scale attention/__init__.py extends adamw_epsilon_model_v2.runtime.attention.
That next initializer and the remaining attention package path chain were NOT
read at this6-file bound. A unique physical filename is strong inventory evidence,
but does not by itself prove the complete package import path or exclude an
unread initializer's behavior. Therefore MLA operator graph accounting is
explicitly CONDITIONAL on that body being selected. GPT and model .blocks/.mlp
owners remain resolved from prior model-path trace, not re-opened. Do not claim
attention namespace resolution, a runtime profile, or selected-kernel precision.
The next bounded source task follows only NEW attention-path edges.

## Prospective graph schedule resolved without execution

Choose the simplest prospective schedule: ONE native body/final-normalizer h,
two ordinary cW*h readout losses in one joint autograd graph, weighted.5/.5,
one backward accumulation before full parameter clipping/scheduled optimizer/EMA.
No head recomputation, chunking, sequential detached-hidden gradients or fusion.
This preserves a direct auxiliary-to-shared-body path in the intended equations;
its numerical correctness is unproved. Model SOURCE, proof and quality are not
admitted by the schedule. Both fresh native and NEW copied-readout-next controls
remain necessary; old copied-block/shared-W future_two proof remains closed.

## Complete category forecast, conditional source envelope only

Full machine/output/time categories are now explicit in
 top-body-cost-source001-forecast.json. They are forecasts/allowances, not measured
TOP maxima, free-compute claims, a frozen runtime/panel budget or guard guarantees.

Training: augmented P41,358,208; five full FP32 parameter copies827,164,160B
conservatively cover weights/grads/two moments/EMA (Muon may need fewer). Six
full-vocab tensors1,235,116,032B cover two logits/two logprob paths/backward scratch;
body graph512MiB, runtime allocator256MiB, Python sparse/staging64MiB and small
buffers/IDs/RNG8MiB make2,943,084,032B. An extra512MiB host reporting/clone
allowance plus residual allocator headroom gives planned RSS3.75GiB/MPS3.5GiB,
below unchanged hard4GiB/6GiB. This is not an analytical upper bound on PyTorch.
Exact implemented graph/negative fixtures may invalidate it BEFORE runtime.

State/export: no live activation graphs during archive staging/restore. No
simultaneous native/reference/augmented triple model residency. Prospective CPU
archive/staging allowance2GiB plus graph-free model/destination/moments/EMA1.5GiB
plus256MiB runtime gives planned RSS3.75GiB; planned MPS2.5GiB. Concrete implemented
clone/load lifetimes must later be reviewed against this, not presumed. RAW
all-parameters two-moment+EMA upper4P=661,731,328B plus8MiB other-data allowance
fits a PROSPECTIVE NEW augmented768MiB archive cap. Native512MiB/H200primary128MiB
are prospective caps, not inherited acceptance. No cap was relaxed after a run.
An upper bound cannot prove exact archive size/actual failure of an older cap.

Inference export intends to omit Wa, retain own complete RAW parent provenance
and use one native primary head. Primary weights113,969,664B/cache196608B per
item or1,572,864B atB8;1GiB serving workspace plus256MiB CPU transfer allowance.
Auxiliary omission/export/load is UNIMPLEMENTED and requires its own future
state/serving qualification, not native relabeling or old proof acceptance.

Output envelope (NOT archive creation authorization): at most2 native RAW512MiB,
4 augmented RAW768MiB and3 primary H200128MiB gives4,697,620,480B retained;
RAW768MiB+H200128MiB temporary overlap+2MiB whole metadata gives5,639,241,728B.
With unchanged8GiB reserve plus2GiB extra entry headroom the prospective target is
16,376,659,968B. Current free space covers it; no new storage block/cleanup trigger.
All current protected weights/evidence remain untouched. No old25.6GB quality
forecast is used as requirement or credit. A future quality panel needs its OWN
complete retained-output/time forecast; this small model-consumer envelope is not
that panel and creates no retirement eligibility.

Timing forecast420s, apportioned startup30/forward-backward180/graph-free state120/
inference-reporting90. No TOP timing measured; hard120s each CPU phase and600s
whole MPS prerequisite remain. Four threads/one worker/fallback0/fastmath0 and
hard memory/output/reserve gates unchanged. Exact future harness call counts,
new unique cases/roles if applicable/source pins/source gate/once-only admission
remain unimplemented; these numbers DO NOT admit any numerical run.

Decision: conditional architecture/category-cost design is coherent enough to
continue the final attention namespace SOURCE question. Model implementation
remains UNADMITTED until that identity plus a minimal source plan/review are
complete. This is not rejection of TOP quality, nor qualification of model
causality/state/serving/science. Future1255/closedmodelproof1236/scientificruntime1211
unchanged. Zero new executable files/numerical exploration/roles/weights/copies/
deletions/retirement; no worker launched. Native/holdout/future_two panel unchanged.
