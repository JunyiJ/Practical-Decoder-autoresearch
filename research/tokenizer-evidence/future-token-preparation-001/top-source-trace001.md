# TOP caller trace complete within its bound; own target SOURCE prepared

November7,2026. Followed top-target-source-handoff001.md: exactly12 fetched
source/config file versions,132603bytes total,largest40442bytes. No imports,
authors' function/kernel execution, installs, clones, data/tokenizers/tensors,
models/samplers/RNG/checkpoints/scoring or numerical tests. Provenance lists
exact commits, Git blob IDs and source URLs in top-source-trace001-provenance.json.
The preceding local evidence/applicability review remains post-data-consumer-review001.md.

## Branch and dependency identity

Resolved zaydzuhri/flame token-order-prediction to
5bcd6b6423606e07b92dd2644ecc24d908d2c7a4 and
zaydzuhri/flash-linear-attention token-order-prediction to
9ae8c4cb092268592eb2f7034875b9bbbe8e626a. Flame's .gitmodules names that branch,
but its actual FLA gitlink is9fa447e86b8227daf0c13dafbedf8ea4a62d345c. A branch
label is not a pinned dependency or evidence of the paper's original run.
The launcher copies the checked-out third-party FLA into its run directory;
no historical run manifest was retrieved. No reproduction claim is established.

At gitlink9fa447, read the different model blob9908c26b5ea7d5deaa5fd438f0a059d5a4de220a.
It has the same once-shifted-label call and summed losses, but calls seq_to_top
without ctx_len; current9ae8c4 supplies ctx_len. Gitlink target-builder blob
c03d48cb66e6f4c20e9620efc431895ae766c85a differs from current b698ea66b570dc89d86bb0d403cffddf3c949a13.
That older builder was not read: the12-file-version bound was respected.
Do not attribute current-builder semantics or configurable loss ratios to the
older gitlink or infer which dependency produced published results. This is an
exact provenance limit, not a reason to run or reconstruct anyone's old study.

## Current-branch pretraining caller and target contract

Read [Flame train.py](https://github.com/zaydzuhri/flame/blob/5bcd6b6423606e07b92dd2644ecc24d908d2c7a4/flame/train.py),
lines378–398,627–630,704–718: pretraining requests2T tokens, feeds firstT as
inputs and all2T as labels. The nonpipeline call passes both to the model.
The default fla.toml is pretrain/varlenfalse; this is not a captured paper request.
Data.py:155–182 packs tokenized documents into fixed chunks;359–387 clones
unshifted input IDs into labels and masks padding by attention mask. It is not
our raw80/10/10 split contract, and document boundaries can be crossed by packing.

Read [current TOP model](https://github.com/zaydzuhri/flash-linear-attention/blob/9ae8c4cb092268592eb2f7034875b9bbbe8e626a/fla/models/transformer_top/modeling_transformer.py),
lines394–420: the model shifts labels once to s[t]=x[t+1], appends ignore_index,
uses the firstT for primary loss, and passes s plus contextT to seq_to_top.
The separate TOP projection receives the final hidden states; primary and TOP
losses are summed at ratio.5. At ratio a the current implementation uses
2*(1-a)*NTP+2*a*TOP, not the normalized weighted mean. This factor matters for
global clipping/optimizer epsilon; no loss-weight change is proposed locally.

Read [current target kernel](https://github.com/zaydzuhri/flash-linear-attention/blob/9ae8c4cb092268592eb2f7034875b9bbbe8e626a/fla/modules/seq_to_top.py),
lines42–65,78–109. Reverse update then distance<W includes shifted offsets
0..W-1, hence original physical future offsets1..W. Repeats of a type collapse
to its nearest future occurrence; a token equal to current input remains eligible
if it occurs in the future. The initial sentinel is beyond every output window
when labels have sufficient tail. At W=T and2T raw IDs, the last output's Wth
target is x[2T-1], so the appended ignore entry is beyond the required support.
This is index reasoning under those conditions, not executed proof or a claim
that every configured caller supplies sufficient tail when W>T.

For a type whose first future offset is k, the raw score is W+1-k. Softmax
removes the common W, giving target mass proportional to exp(1-k). The current
target dtype is FP16, so this exact-real equivalence is not a guarantee of
identical rounded large-window scores. Padding/OOV are skipped by the kernel;
finite support is not guaranteed on a padded suffix. We will reject such input
in our own target contract. No upstream GPU/backend execution occurred.

This reconciles the current caller's zero distance. The paper Algorithm1 uses
different written indexing/inequalities; the landing naive builder permits one
more shifted endpoint. Neither is silently substituted for the traced caller.

## Loss and interpretation limits

[Current fused ListNet source](https://github.com/zaydzuhri/flash-linear-attention/blob/9ae8c4cb092268592eb2f7034875b9bbbe8e626a/fla/modules/fused_linear_listnet_loss.py)
normalizes target scores across vocabulary and divides its mean by allN rows,
not a count of nonempty rows. Its target logsumexp applies NaN-to-positive-infinity
and the kernel zeros gradients only for positive-infinity target logsumexp.
This does not establish well-defined all-negative-infinity rows; do not import
the landing helper's empty-row behavior as a fused-kernel guarantee.

Static risk also visible: tail vocabulary loads are masked but their loss sum
has no explicit vocabulary mask after exponentiation. Whether extra lanes vanish
through rounding/underflow is unmeasured and depends on scores. This source is
not a validated oracle and will not be ported or tested here. This observation
does not identify any local CE failure, measure an upstream counterexample, or
admit a new kernel diagnostic. Native CE/backend/precision remain unchanged.

For an explicit valid sparse target p and predicted distribution q, ordinary
soft-target CE equals H(p)+KL(p||q); softmax outputs are distributions even when
used for ranking. Different raw auxiliary losses cannot establish learning
difficulty without accounting for the different target entropy/normalization.
No new loss values or trained comparison were measured. Duplicate-next still
needs its own capacity/objective control; the original copied-head symmetry
analysis cannot be transferred to separately initialized unembeddings.

## Applicability decision and independent next consumer

Do not reproduce or integrate the authors' training/kernel recipe. Historical
dependency identity, padding/general tail behavior and numerical kernels are
not qualified. The current branch nevertheless provides a coherent mathematical
future-proximity target on valid, fully supplied sequences. This supports a
small independent TARGET-ONLY component, explicitly named a local contract,
without any model/head choice, published-gain or MPS-fusion claim.

New research/top_proximity_targets_v1.py implements direct future lookahead,
immutable sparse first-occurrence rows and stable Python-float target masses.
It has no Torch, model, sampler, tokenizer, scorer or checkpoint dependency.
It rejects padding/OOV/bool/partial tails rather than adopting skip/empty-row
policies. Removing a common score before exponentiation is its own target-math
specification, not a native model CE/normalizer/precision change. SOURCE ONLY:
no import, extracted function, oracle or test execution, including no AST exec.

Consumer: before considering an ordinal future-supervision model, establish
that exact first-future occurrence, type deduplication, endpoints and target
mass have an unambiguous contract. Discriminators are W1 one-hot, repeated
future types, current-token repeats, last-offset inclusion and unequal distances.
If the new component fails any prospectively fixed predicate, close that new
attempt; never patch it into acceptance. A pass would accept target math only,
not learned transfer, head parameters, causality, gradients, state or science.

Read top-target-component-source-plan001.json for complete literal cases,
resources and remaining harness/reviewer work. No execution is admitted yet.
No new seed roles or census: deterministic literal target arithmetic needs none.
Source bindings1248 predecessors plusone new implementation=>1249 for future
binding; old modelproof1236/scientificruntime1211 unchanged. No old-source audit.

The D256/V50257 TOP head would still add12,865,792 parameters/51,463,168 FP32
weight bytes; five copies alone257,315,840B, before full-vocabulary intermediates.
No head implementation/quality budget/panel/roles are frozen. Do not attach this
source to the failed future-two fixture. Future-two's quality remains unknown;
all original failed/successful sources, cases, roles, STOPs and evidence preserved.
