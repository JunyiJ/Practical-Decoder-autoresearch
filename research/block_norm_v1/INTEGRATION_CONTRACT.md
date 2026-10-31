# Block-normalization runtime integration contract v1

October 31, 2026. Prospective implementation contract, not a frozen training
plan. The original baseline was an unmodified control copy. A named candidate
construction policy has now passed41 CPU tests; no training trajectory, serialized
resume test or MPS result exists. Historical baseline applicability is preserved
by snapshots and the original manifest.

## Source applicability

The original `src/**/*.py` and requirements fingerprint stays unchanged. The
seven v1 and six v2 activation-recovery runtime files stay unchanged. Closed
reviews continue to resolve their original imports and hashes; new research
must not reinterpret those reviews as tests of changed code.

An AST dependency map identifies 43 local modules (213,313 bytes) reachable
conservatively from GPT, cache, optimizer, checkpoint, EMA and training modules.
The copy is under `research.block_norm_v1.runtime`. Relative imports resolve
within that namespace. The map found no absolute local or dynamic import hazards.
The baseline-copy manifest records origin paths/hashes, exact copied hashes and
two empty package initializers. Never rewrite that baseline manifest after edits:
record a new delta manifest against it, and keep the baseline snapshots.

Do not redirect `sys.modules` or `sys.path`, monkeypatch original validators,
remove signatures, temporarily substitute RMS modules to pass production checks,
or run old research controllers on the new policy. The unchanged baseline test
contains a disposable negative replacement test and restores the module afterward;
it does not provide a production replacement path.

All future producers, loaders, optimizers, cache builders and H200 exporters must
import the new namespace explicitly. The copied Hydra CLI has an unresolved
relative `../config` path and is not supported: do not invoke it. A new controller
must supply an explicit configuration and bind its tokenizer/data/training
contract. Importability does not demonstrate an executable copied training CLI.

For every new prerequisite, pin all relevant new source files, verify frozen
original/recovery hashes before and after, preserve exact tested sources and
record actual import locations. At campaign freeze, bind the complete new source
inventory and environment plus original immutable dependencies. A source edit
invalidates earlier applicability to that edited code; preserve the earlier
result and perform targeted new verification. An added file must be included in
the inventory, not omitted by a fixed historical list. Inference/final reviews
must verify this same binding; no implicit latest-code acceptance.

## Named model and configuration policy

Add a new explicit versioned policy for this D256, four-layer RoPE MLA/SiLU
backbone. Old configurations retain their exact behavior and RMS-only rejection.
The new policy supports exactly `rms_incumbent` and `layer_matched_epsilon` from
`spec.py`. It must validate the full field set, not silently project away unknown
or conflicting fields. Bind kind, FP32, width256, epsilon2^-23, affine/bias,
eight pre-attention/pre-FFN placements, final LayerNorm epsilon1e-5, serial unit
residuals, SiLU/hidden672, attention geometry/score0.5 and untied tables.

Construction validates the actual eight native modules and parameter identities,
not config alone. RMS uses native eps=None with an FP32 contract; LayerNorm uses
explicit2^-23 and bias. Constant initialization must preserve RNG and exact shared
weights. Model execution must reject non-FP32/autocast violations. Model metadata
must describe the actual constructed policy; same-shape opposite-arm signatures,
wrong epsilon, missing bias, mixed modules and mutated final norms must fail.

The implementation must adapt copied architecture field validation and construction
explicitly. Do not feed a new configuration to an RMS-only helper and bypass its
failure. Reuse the old geometry checks through a reviewed policy-aware branch
that preserves old rejections, with negative tests for each newly allowed field.

## Integration surfaces and required evidence

| Surface in the new runtime | Required change or invariant | Required validation |
| --- | --- | --- |
| `models/tokenizer_architecture.py`, new block-norm policy module | Explicit named-policy config and actual-module validation; retain old RMS contract | Unknown/missing/conflicting fields, wrong module types/epsilon/dtype/placement rejected; old control unchanged |
| `models/blocks.py`, `models/gpt.py` | Construct eight norms through the specification; record new signature; retain all other operations and initialization | Parameter counts28,492,416/28,494,464, shared state/RNG equality, formulas/gradients, causality and old-control parity |
| `train/optimizers.py` | Validate new policy before grouping; new bias vectors in AdamW at base LR; preserve Muon scaling and independent decay | Unique full name/identity coverage, group order, eight new vectors, gradient/update/decay and optimizer-state resume |
| `utils/checkpoint.py` | Require saved/current new-policy signatures before strict loading; retain raw/H200 purpose distinction | Actual serialized round trips; opposite arm and same-shape signature/epsilon mutations rejected; H200 optimizer-resume rejection |
| `train/train.py` | Propagate full signature/config through training contract and raw checkpoints; future explicit controller only | Interrupted/resumed CPU numerical/RNG/optimizer agreement, batch/data/tokenizer identity and raw payload completeness |
| `train/weight_averaging.py` | H200 reconstruction uses new GPT and loader; average all gains/biases without folding them | Exact controlled EMA oracle, candidate gains/biases included, export/load round trip, resume preserves accumulator state |
| `attention/cache.py` and attention modules | Preserve state layout/math; consistently validate the new model policy | Full vs cached tolerances, offsets/reset/interleaving/causality, sizes and CPU/MPS agreement |

Initial prospective edit allowlist is the seven named existing files above
(two model construction files, architecture, optimizer, checkpoint, training,
weight averaging), plus a new policy module and separate prerequisite/controller
files. Cache/attention code should remain byte-identical unless a demonstrated
policy-validation dependency requires a narrowly documented extension. Any such
extension gets an explicit delta review and targeted tests before acceptance.
The dependency closure includes older policies to preserve their import graph;
the new experiment does not authorize changing their semantics or testing new arms.

## Completed baseline and remaining gates

`runtime-baseline-prerequisite-002-review.json` passed five CPU checks on the
unmodified copy: all43 import locations, exact initial weights/RNG/logits/CE,
old-versus-new chunked cache outputs/state, optimizer group metadata/order/counts,
and old policy's rejection of LayerNorm configs and silent module replacement.
This compares old versus new cached execution; it does not establish full versus
cached numerical agreement. No gradients/optimizer steps were run.

Attempt001 stopped before tests on sandbox PermissionError during the safety
entry; no guard or test log was created. It remains preserved. Attempt002 used
the same sources with authorized process access:0.851s, peakRSS543,424,512 bytes,
four threads,120s cap,4GiB RSS cap and8GiB reserve. Original source, all13 frozen
recovery modules and three closed campaign inventories/nonweight hashes were
unchanged. No checkpoint or corpus loaded, no training, MPS, cleanup or holdout.

Next implement the explicit full-model policy, then validate construction,
optimizer, strict checkpoint/EMA integration and caches with new evidence. Full
serialized resume, numerical/MPS and controller/retention/statistical prerequisites
remain. Only after all prerequisites and sufficient full-panel capacity may the
finite seeds/order/plan be frozen. Recorded storage shortfall remains2,186,024,144
bytes; a baseline CPU guard observation is not a new training capacity approval.


## Construction implementation status

`model-policy-delta-001.json` records three changed model files plus the new
`models/block_norm.py`; all other copied files retain their baseline bytes.
The dedicated `tokenizer_mla_block_norm_v1` policy validates the full field set
directly using fixed backbone constants; it does not disguise LayerNorm as RMS
for validation. `block_normalization` carries the complete standalone signature.
Construction and execution validate actual modules; optimizer entry inherits
these checks from the architecture validator, with no grouping-code change.

Attempt002 passed41 CPU construction, parameter/group, rejection and old-control
regression tests, including direct original versus named-RMS equality. No update,
serialized checkpoint, EMA/resume or candidate cache-numerical claim follows.
Checkpoint save/load and H200 exporter still require dedicated signature
transport and actual serialized tests. Do not treat their config-only checks as
that completed prerequisite. Keep both policy attempts and original baseline
snapshots; subsequent code changes require a new reviewed delta and test scope.
