# Native namespace trace002: extension chain advanced; GPT candidate only

2026-11-07. Completed the next bounded read of6 NEW source versions,11,260B,
largest10,493B. No research/Torch import, module resolution by execution,
extracted-function call, numerical oracle, tensor/autograd/model/optimizer/RNG/
corpus/sampler/checkpoint/scoring call. Prior six sources were not reread; no
closed proof, census, kernel, weight or metadata audit replay. Provenance records
only these six freshly read source identities. Source+new notes <512KiB,
reading <10min, every tool call <60s. No new executable file or source binding.

## What the package source actually establishes

final_ln_scale_model_v1/runtime/__init__.py extends its __path__ with
adamw_epsilon_model_v2.runtime. Its models/__init__.py separately extends model
paths with adamw_epsilon_model_v2.runtime.models. The latter extends
final_ln_bias_model_v1.runtime.models, whose initializer extends
adamw_beta2_model_v1.runtime.models, whose initializer extends
readout_centering_model_v1.runtime.models. This last initializer was NOT read:
the6-file cap was reached after a GPT candidate read. This is a partial static
path extension chain, not Python import/module-resolution execution.

Filesystem inventory found physical GPT definitions in both
research/ffn_structure_v1/runtime/models/gpt.py and
research/block_norm_v1/runtime/models/gpt.py. Read the former ONLY. Do not silently
choose it because it exists or reuse a historical source binding as proof of
current package precedence. The older side of the namespace chain and potential
module side effects remain unread. There is no evidence of a broken runtime.

## Candidate-specific forwarding facts

The READ candidate constructs grouped or ordinary embeddings according to the
explicit block policy, then a ModuleList of n_layers TransformerBlock instances,
then a final normalizer. It constructs an untied bias-free nn.Linear readout;
under muP it multiplies initial readout weights by sqrt(width_mult), where
width_mult=dim/base_width. It allocates output bias afterward, with explicit table
tying branches. An eventual copied auxiliary readout should copy the resulting
native readout, not insert another nn.Linear before the native construction or
silently consume additional common initialization draws. No model was built.

The candidate forward obtains embeddings from input idx, iterates the block
list, applies final normalization once, and produces all-position vocabulary
logits. In the ordinary untied branch logits=lm_head(h)/width_mult; the other
explicit table-tying branches add table_readout_scale. At D256/base64/muP this
would be c=1/4. This is CONDITIONAL on actual resolved owner and full cfg;
no current source-resolution or initial-logit equivalence claim is made.

Integer targets first enter cross_entropy after the readout, flattened over all
B*T positions, with the default mean reduction. Training-objective and MoE aux
paths are explicit conditionals, not TOP paths. A cache is checked for list/tuple,
length, per-policy validity, and passed per layer through the same block loop
and final projection. Both cached/noncached calls use the same next-token head.
Return is logits/loss or logits/loss/breakdown; final h is not returned.

Thus this candidate provides no direct hidden-state API for the new training-only
head. Frozen-source edits or hooks that quietly capture a different h are not
admitted. An eventual own training forward would need explicit native operation
parity while leaving native inference on its qualified API; that requires first
resolving the actual owner. The called block/attention definitions were not read
here. Input-only embeddings and target placement do not substitute for a full
causality/cache/numerical model proof. No claim that h is newly qualified causal.

## Graph lifetime/cost consequence, not resource admission

Even if this is the owner, calling native.forward for logits and then running a
second body forward just to obtain h would duplicate shared graph/work and could
change initialization or state semantics. The intended consumer must use ONE
shared h from one training-body forward. A direct two-loss autograd graph retains
both readout paths until backward; a sequential-head scheme needs an explicit
head/hidden-gradient accumulation and shared-graph lifetime contract, not an
assumption that detach preserves supervision. No schedule is implemented/chosen.

Earlier structural state/table arithmetic in top-consumer-design001.md remains
partial: no actual model graph, optimizer temporary, export staging, serialization
or timing measurements occurred. An all-parameters two-moment bound cannot alone
prove an exact serialized size or an actual failure of the old512MiB cap; it
only prevents certifying that cap from the bound. Complete model cost remains
unadmitted, and therefore no model SOURCE or runtime follows this partial trace.
No new storage requirement/blocker, cleanup trigger, scientific budget, roles or
quality panel is created. No native CE/backend/normalizer/precision changes.

Decision: the source question has advanced but native GPT ownership remains
UNRESOLVED. Next read starts at readout_centering_model_v1/runtime/models/__init__.py
and follows only NEW package extensions, reserving a slot for the actual GPT owner
if it differs from this candidate. Do not reread the6 sources from trace001 or
these6 trace002 files merely to refresh state. If another bounded read cannot
finish the chain, record the precise unread next edge; never claim unseen code.
Only after exact ownership/body/loss/inference and complete prospective costs
are resolved may an independent TOP model source plan be considered. All old
future_two/index/CE failures remain closed, no successor/missing-output recapture.
Future1255/closedmodelproof1236/scientificruntime1211 unchanged. Native/holdout/
future_two prospective panel unchanged; zero numerical exploration, weights,
copies, deletions, retirement or new executables. All evidence remains protected.
