# Native namespace trace003: GPT physical source owner resolved statically

2026-11-07. Six NEW package source versions,1,198B total/largest246B; original
6-file/256KiB-each/512KiB source+notes/10min/60s-tool bound preserved. No imports,
research functions, Torch/target/objective/oracle/autograd/tensor/model/optimizer/
RNG/corpus/sampler/checkpoint/scoring calls. Prior12 source files were not reread;
no closed numerical proof, metadata/weight/kernel/census audit or oracle replay.

## Exact new path edges and owner

Read models/__init__.py in readout_centering_model_v1, branch_scale_model_v1,
local_mix_v1, ffn_norm_v1, mla_score_v1 and ffn_structure_v1. Their explicit
extensions are readout_centering -> branch_scale -> local_mix -> ffn_norm ->
mla_score -> ffn_structure. The final FFN package APPENDS block_norm models
as fallback after its own local directory. It does not prepend the fallback.
Together with the prior trace002 scale->epsilon->finalbias->beta2 edges and
physical inventories, the first matching gpt.py is therefore
research/ffn_structure_v1/runtime/models/gpt.py under ordinary Python path search.
That candidate was actually read in trace002; its exact recorded SHA is reused
from that provenance, not rehashed/reread or imported here. The separate physical
block_norm GPT has lower precedence and was NOT read. We make a STATIC owner
claim for the inspected overlay path order, not a runtime sys.modules/profile
or fresh-import execution receipt. No broken-runtime inference remains.

Relative imports in this GPT execute within the final-scale models namespace,
so they search the ENTIRE overlay path list. The GPT's physical containing
folder alone does not identify its .blocks/.mlp/.readout_norm owners. Directory
inventories already read locate .blocks first at local_mix_v1/runtime/models/
blocks.py and .mlp first at ffn_structure_v1/runtime/models/mlp.py; their bodies
were not read in the current bound. Do not assume block_norm.blocks or the GPT's
neighbor blocks.py supplies all actual operations. This is the next useful
body/cost question, not another unresolved GPT-owner claim.

## Forwarding facts whose ownership is now resolved

The source identified in trace002 is the selected GPT definition: input IDs
feed embeddings, a block loop, and one final normalization. Ordinary untied
projection is lm_head(h)/width_mult. For the adopted D256/base64/muP cfg this
scales by1/4; actual config/policy validation still belongs to its original
contracts, not an arbitrary override. Integer targets enter the default mean
cross_entropy only after logits. Cached and uncached calls use the same loop,
final normalization and primary projection; the method returns logits/loss or
breakdown, never final h. Constructor initializes the native head before any
prospective auxiliary copy. No native source modification is necessary to state
these facts, and no model numerical acceptance follows from them.

A TOP training adapter would need one own explicit h-producing body path through
those unchanged child modules and a training-only copied readout, with native
primary inference kept on its original API. Hooking unseen hidden tensors or
running the body twice is not the resolved contract. No adapter/source written.
The previous shared-body/head represented-mass algebra and both fresh controls
remain prospective; they do not prove native updates, branch symmetry under
floating-point optimizers, CPU-MPS equality or causality of new consumer code.

## What remains before model source: actual body and complete resource design

TOP full head state/logit arithmetic alone is not full cost. Before any model
source, read the actual .blocks/.mlp bodies and sufficient selected attention/
normalizer implementation to inventory shared-body autograd storage, graph
lifetime, buffers/cache, output logits/logprobs and both head gradients. Reuse
accepted native facts only in their stated scopes. No re-proving old model cases
or loading old states. Select a SINGLE explicit prospective head schedule:
one shared graph + jointly accumulated gradients, or a fully specified sequential
scheme; detach alone does not preserve representation supervision.

Then complete the whole prospective category budget before source admission:
parameters/grads/Muon+Adam states/EMA, two vocabulary paths and backward scratch,
body activations/reductions, CPU reporting/RNG snapshots, runtime allocator;
one-arm versus simultaneous destination/resume/reference state staging;
RAW/H200/temporary serialization/publication overlap and retained-output highwater;
CPU startup/functional/state/inference timing allowances versus hard120/600s;
4GiB RSS/6GiB MPS/reserve8GiB/four threads/no fallback/fastmath/2MiB metadata for
preparation, with NEW complete cost for any later runtime. No old25.6GB quality
forecast credit. Upper moment bounds are not exact serialized sizes or measured
memory. Until that full prospective design is coherent, no model source, fixture,
roles, census, scientific panel budget/freeze/launch/adoption is admitted.

This is not a new active storage requirement or blocker. Current source notes fit
the existing2MiB notes+8GiB reserve+2GiB entry-headroom target10,739,515,392B;
no cleanup trigger. All new notes and prior selected/fresh/screen/prerequisite/
failed/partial weights and evidence protected; zero weights/copies/deletions/
retirement. Future1255/closedmodelproof1236/scientificruntime1211 unchanged,
zero new executables/numerical exploration. Native/holdout/future_two panel
unchanged. Next bounded read starts with actual local_mix blocks and FFN mlp,
not the now-resolved initializer chain or old proofs.
