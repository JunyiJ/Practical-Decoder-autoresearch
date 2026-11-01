# Matched-budget FFN structure runtime

This narrow runtime overlay supports D256 MLA with either SwiGLU hidden672 or
ungated SiLU hidden1008, under `tokenizer_mla_ffn_structure_v1`. Both retain
RMSNorm, grouped-embedding v2, SiLU, final affine LayerNorm and untied GPT2 tables.
Only the exact two FFN/width-policy combinations are allowed. The candidate has
bias-free `linear_up`, `activation`, `linear_down`, `dropout` modules, with ordinary
Linear fan-in initialization. No gate or hidden normalization is present.

The six changed runtime modules are local copies. Package search paths resolve
unchanged dependencies from `research/block_norm_v1/runtime`, under the new module
namespace so relative imports use this overlay. The normalizer specification is
imported directly from the frozen implementation. No monkeypatching, replacement
of historical modules, or edits to the399 inherited frozen source files occur.
Future frozen inventories must bind both those399 files and every overlay source.
The first tested source snapshot is in ffn-structure-preparation-001/
model-integration001-source; subsequent fixes require new attempt evidence.

Full candidate parameters:28,492,416; FFN matrix parameters:516,096/layer.
Muon covers2,752,512 parameters in40 tensors (control44); AdamW covers25,739,904
in44 tensors. Existing width-LR scale0.5 and shape-dependent match_rms_adamw
updates are retained. Equal matrix parameter counts do not imply equal optimizer
geometry, latency or activation memory. The control's initialization/logits match
the legacy runtime bitwise. Candidate random draw shapes differ; identical seeds
do not promise identical attention/readout initial tensors between arms.

CPU full-model forward/backward/one synthetic optimizer step and MPS forward/
all-parameter gradients/cache/causality/reset/interleaving passed. Signatures bind
FFN structure and reject cross-arm checkpoints/caches. CPU checkpoint validation
was in-memory only. No scientific training or corpus/test access occurred.
Serialization, H200 export, resume, worker/parent validation and inference remain
prerequisites, followed by measured storage projection, seed census and finite
pilot/quality freezes. Do not launch a scientific campaign from these fixtures.

Full file-backed MPS repeat/resume and H200 export now passed bitwise on generated
text. The serial actual-trainer/independent-parent and trained inference paths
also passed. validation.py, validation_data.py and inference.py bind the new
policy while retaining the prior scoring/timing arithmetic. A historical-validator
import failure is preserved separately; recovery accepted the existing control
without retraining. These are prerequisites only; scientific pilots/quality
remain unfrozen. See transport-worker-inference001-review.json for exact evidence.
