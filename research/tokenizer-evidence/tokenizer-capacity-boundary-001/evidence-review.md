# Tokenizer capacity: qualified component work, no quality panel yet

## Local basis and contrary evidence

The latest dropout study is CLOSED_NOT_ADOPTED; see
`../embedding-input-dropout-preparation-001/quality001-closure001.json`.
Input/readout geometry and exact-initialization counterfactuals in
`../input-capacity-boundary-001/` do not justify a low-rank or frozen-input quality
panel. Earlier table-sharing/matched-initialization recipes also lost; no old
scores or retired checkpoint payloads are replayed.

The original267,688-token training split observes10,653 of50,257 GPT2IDs. The
new fixed-prefix codec census uses the same external GPT2 merge order, regex and
byte primitives, with no normalization or tokenizer fitting. It changes only the
allowed rank prefix. It is not SentencePiece or a corpus-trained BPE.

| Vocabulary | Train tokens | Inflation | Train-observed IDs | Val targets with train-unseen IDs | Parameters at current body |
|---:|---:|---:|---:|---:|---:|
|4,096|338,596|26.49%|2,949|0.1495%|4,857,984|
|8,192|315,220|17.76%|4,828|0.4202%|6,955,136|
|16,384|291,834|9.02%|7,103|1.2490%|11,149,440|
|50,257|267,688|0|10,653|2.7919%|28,492,416|

These intrinsic counts are not language-model losses or proof of generalization.
A128-token train window averages362.33bytes at8K versus426.67 at50K; smaller
vocabulary loses raw-text context at a fixed token limit. Equal tokens/steps do
not equal raw exposure or computation. The8K component is a single feasibility
anchor (largest power-of-two below the10,653 observed native types), not an
optimal vocabulary selected from model scores. Other sizes were census controls.

## Primary papers actually retrieved and read

[SentencePiece,EMNLP2018](https://aclanthology.org/D18-2012.pdf),§3.1/3.3–3.5:
reproducible segmentation needs explicit vocabulary/normalization and reversible
encoding. Its reversibility is for normalized text; defaultNFKC would not satisfy
our raw-byte identity. We therefore preserve raw UTF8 and use existing byte-BPE,
not its normalizing defaults. Its NMT results do not establish a local LM gain.

[Tokenizer Choice2310.08754v4](https://arxiv.org/pdf/2310.08754v4),§3.1–3.4,
§5.2.3/Table5,§5.3/equation1,§6/Figure4 and§8: vocabulary and sequence length
jointly affect cost; intrinsic metrics are not sufficient quality predictors.
The English/multilingual preferred sizes differ. Their2.6B models,large corpora,
33K–100K vocabularies and downstream tasks do not validate8K on our corpus. They
did not tune hyperparameters per tokenizer or test multiple seeds. Our future
comparison must retain equal finite tuning and complete fresh confirmation.

[Rust etal.,ACL2021](https://aclanthology.org/2021.acl-long.243.pdf),§4.2/4.3
andAppendixA.5/B.2: controlled tokenizer/data comparisons distinguish effects that
aggregate pretraining results confound. Their analyses include fertility, unknown
units and corpus-size correlations; these are not causal proof of an English
causal-LM gain. Their BERT/MLM multilingual setting differs substantially.

## Mechanism, falsifier and control commitments

Hypothesis for feasibility: reducing the long-tail vocabulary can change capacity
allocation and output-class sample efficiency while keeping the current body.
Prediction for a later properly frozen quality study would concern common-byte
loss and measured resource cost, not per-token CE or compression alone. It can
fail through lost lexical units, shorter raw context, changed softmax normalization
or reduced raw training exposure. A loss would not prove every smaller tokenizer
fails. No automatic horizon/context rescue or vocabulary search is authorized by
this note. Either result informs whether table allocation merits more work.

The native control and8K component share exact initial body tensors and overlap-ID
embedding/readout rows, with native readout scale4 and the exact post-construction
CPU RNG. This avoids the uncontrolled initialization/scaling change found in the
old table-sharing control. Both use44Muon/44AdamW groups with inherited width
rates,epsilon1e-6 and decay.00006. Changed vocabulary normalization remains part
of the recipe effect; this is not a claim of isolated parameter-count causality.

A future panel must declare its resource axis and exposure tradeoffs prospectively.
Fixed1200updates/1024targets is a possible fixed-token-axis comparison, not equal
bytes/FLOPs/time. Equal raw-exposure alternatives change schedule/EMA accounting
and need their own controls. No numerical panel, new quality seed or adoption
threshold is frozen by these component/metric proofs. Do not reuse old scores as
a new common-byte baseline or treat a smaller checkpoint as an adopted model.

## Actual engineering gates and next action

`census001` passed exact Unicode/literal-marker/raw split round trips and40foreign
identity/ID cases. `metric-contracts001` passed56negative cases and independent
window/grouping/BPB oracles. Every codec covers the SAME111,475 validation bytes,
SHAad7da8cdcacc24b5312461c23f2d00aead59d544fcc2fad4fb2de40da4964fe8,
after a fixed64-byte Unicode prefix, encoded separately from the suffix. Native
has34,257 targets;8K has39,963. This is a new evaluation boundary. Byte-block
loss allocation is an explicit uniform-within-token bookkeeping convention, not
per-byte entropy. Raw context still differs. Holdout remains unencoded/unscored.

`component001` is CLOSED_FAILED at the inherited50257-row MPS embedding guard;
no real-data optimization or quality trial occurred. Keep all v1sources/logs/guard.
Separate additive `gpt2_prefix_model_v2` supplies an explicit8192x256 grouped MPS
embedding reduction. It does not relax or pass fake shapes through the old guard.
`component002` passed exact native CPU/MPS initialization,RNG,logits,gradients and
two optimizer steps;8K independent sliced-readout/gradient oracles;cache/reset/
causalityB1/8;own checkpoint/foreign optimizer/cache/ID rejection. Cache remains
196,608B/item. Driver/RSS peaks2,210,201,600/1,695,121,408B fit original guards.

`data001` passed actual native/8K corpus IDs, native sampler equivalence,20draw
fresh repeats per split, private sampler restore and invalid-state rejection
before destination mutation. The genuine new config uses `prefix_model` and
`backbone_template`, not a fake old-vocabulary model config. All984 current source
bindings preserve the original972 and the closed failed v1. No model/worker/pilot
or numerical quality root is active; no next scientific seed is assigned.

NEXT executable work: implement additive raw/H200 transport with the actual
codec/model/data identity, optimizer/EMA/complete clipping/CPU+MPS RNG and private
samplers; execute finite real4update fresh-repeat and serialized2+2resume proof.
Then independent real worker/full common-byte validation acceptance,resource pilots,
finite design/statistics and COMPLETE retained-output budget before any quality
freeze. Keep one local worker/four threads and all original time/RSS/MPS/disk/output
guards. Space is ample, so do the enabled work without asking redundant permission.

### Actual real-data transport gate

`resume001.json` now PASSED both native and8K four-update fresh repeats and
serialized2+2 resumes. Exact dtype/shape/SHA256 identities cover RAW model,
optimizer,88EMA tensors,all4clipping records,private train/val samplers,CPU/MPS
RNG,H200 and logits. Each trajectory has4096targets and3H200 snapshots.
Thirteen malformed architecture/kind/config/RNG/sampler/clipping/optimizer/RoPE/
data-identity cases per arm were rejected before destination or live RNG change.
Fixture600424 is not a scientific quality seed. In-memory split payloads were
445,082,599B/native and100,486,119B/8K; no tensor file was written. This is a
transport observation,not a measured full-run comparative memory benefit.
Guard finished22.29s,peakRSS1,686,913,024B/MPS2,510,766,080B. All987 current source
bindings preserve984 predecessors and972 original frozen bindings.

NEXT: implement two new four-update generated workers and independent parent
acceptance using these genuine configs,own RAW/H200 transport and the complete
111,475-byte validation suffix. Score RAW/H200 on CPU/MPS using exact per-codec
windows; check all target counts/byte conservation and cache/resetB1/8. The old
34,278-target validation loop cannot substitute for the new common-byte metric.
Before creating a new worker root, bind source/proof/data identities and forecast
the complete serial pair (native560MiB+8K160MiB payload caps,16MiB metadata,
original8GiB reserve and measured1,267,904,512B external+1,458,176B entry headroom).
These are conservative prerequisite caps to verify against actual serialization,
not a future scientific budget or measured output. Do not pre-credit retirement.
Then run resource pilots,admit a finite fair-comparison design and full retained
output budget before quality training. No generated-worker root,quality panel,
new scientific seed or adoption currently exists; all completed proofs are closed.
