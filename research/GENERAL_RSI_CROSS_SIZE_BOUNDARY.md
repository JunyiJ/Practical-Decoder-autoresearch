# Cross-size boundary — 2026-11-13

Select **byte/native equal-token transfer across transformer depth** first.
The context panel is complete, including all six reviews, the frozen decision,
common128 scores and144 first-generation ratings. This boundary selects a
question and a candidate size path. It does not admit scientific training.

## Evidence and ranking

1. Byte/native addresses the unresolved quality/cost contradiction. The original
   equal-token decision reports−0.052603910nats/token, all three pairs favorable,
   with42.61% fewer parameters and guard ratios1.836–1.852. Equal7,200-second
   training reverses the quality ranking: byte-minus-native+0.122101660.
   Original evidence: `tokenizer-evidence/general-rsi-panel-001/decision.json`
   and `tokenizer-evidence/general-rsi-equal-time-001/decision.json`.
   The original confirmation curves also change sign between early and final
   scores, so shortened timing pilots cannot answer the quality question.
2. Context is a strong second candidate. The complete14,400-second panel accepts
   context256 with mean−0.095711917 and99% upper−0.083908521; common128 mean is
   −0.022220934. All144 fixed generation ratings are mixed and remain weak.
   Evidence: `tokenizer-evidence/general-rsi-context-length-quality-001/decision.json`,
   `generation-evaluation/summary.json`, `completion-review.json` in that panel,
   and `GENERAL_RSI_CONTEXT_LENGTH_QUALITY_RESULTS.md`.
   A three-size repetition of the full time contract alone requires72 training
   hours before evaluation or failures. A shorter contract would change the
   finding being transferred. Its primary contrast also combines training and
   native-context scoring. Byte/native is therefore the more direct first test
   of the human's parameter-allocation and runtime question. This judgment does
   not compare effect magnitudes as if the studies had equal horizons.

The closed composer-FP32 result is inconclusive. Retain the existing byte math
rather than selecting the FP32 variant from a near miss. Books/Python transfer
was inconclusive and formatting-sensitive. Neither supplies broad-domain success.
No old scores become fresh confirmation pairs. No batch-reuse, precision sweep,
output-lexical pilot or same-size continuation is added.

## External evidence and limits

Primary versions retrieved and sections read for this boundary:

- Yang et al., [Tensor Programs V,2203.03466v1](https://arxiv.org/html/2203.03466v1),
  §§3–4 and6.1: conventional optimal learning rates can move with width.
  Their transfer method needs a specific parameterization; depth and
  regularization have limits. Our mixed Muon/AdamW path has no demonstrated
  theoretical muP transfer guarantee, despite a legacy configuration label.
  We keep width fixed and describe the proposed result as fixed-recipe transfer.
- Pagnoni et al., [BLT,2412.09871v1](https://arxiv.org/html/2412.09871v1),
  §§5.1,5.3 and Table2: relative performance changes with scale and training
  budget; architectural and dynamic-patching choices matter. Their parameter
  and fixed-inference comparisons use much larger models and byte patches.
  Our composer maps GPT-2 token IDs to embeddings, retains a vocabulary output
  head and changes no token sequence length. BLT gains are not local predictions.
- Hoffmann et al., [2203.15556v1](https://arxiv.org/html/2203.15556v1), AppendixB
  and AppendixC: endpoint-aligned LR decay matters; their cross-dataset scaling
  discussion limits conclusions with repeated data. Our finite10M-token stream
  and modest repeated exposure do not establish compute-optimal scaling.

## Mechanism and candidate path

Measured parameter inventories come from hash-verified retained native and byte
confirmation weights. Each D256 MLA block has690,080 parameters. The input table
has12,865,792; the composer has725,235. The output table remains12,865,792.
Use depths4,8,16 at fixed D256, eight heads and identical per-block shapes.
This geometrically doubles body parameters while holding vocabulary and composer
cost fixed. It requires new depth execution, not new composer widths.

| Depth | Body parameters | Native total | Byte total | Native body share | Parameter saving |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 2,760,320 | 28,492,416 | 16,351,859 | 9.69% | 42.61% |
| 8 | 5,520,640 | 31,252,736 | 19,112,179 | 17.66% | 38.85% |
| 16 | 11,041,280 | 36,773,376 | 24,632,819 | 30.03% | 33.01% |

These are arithmetic predictions from retained tensor shapes, not measured new
models. The largest native body is4× the reference, although its vocabulary
still accounts for69.97% of total parameters. This narrow depth path cannot
establish a scaling law or billion-parameter transfer.

Inference: a fixed input-composer overhead should become a smaller fraction of
runtime as body work grows. Prediction: byte/native synchronized update-time
ratio falls at depth16 relative to4. A flat or increasing ratio falsifies that
simple overhead explanation on this Mac. Quality persistence is a separate test:
byte's paired CE benefit may persist, disappear or reverse as body capacity grows.
A diminishing benefit would narrow the case for composer adoption; a persistent
benefit with lower relative runtime cost would justify a later larger-range test.
Parameter-saving dilution is deterministic accounting, not a statistical discovery.

## Prospective controls and required work

- Plan18 fresh runs: three paired bundles at each of three sizes. Use equal
  10,240-update endpoints,1,024 targets/update,10,485,760 targets/run. The corpus
  has10,001,163 distinct stored training tokens; random sampling allows revisits
  and omissions. Same exposure across sizes does not imply compute optimality.
- Keep context128, existing byte composer, FP32, independent output head,
  two-random512-block sampler reshaped8×128, optimizer family, LR.0025, clipping,
  decay, final-third cooldown and H200 averaging controlled. Context256 is now
  selected for the separate native time recipe; context128 here deliberately
  preserves the finding being transferred. There is no context contrast.
- Use a fixed recipe without an LR screen. This tests transfer of that recipe,
  not each size's optimal attainable performance. A size-specific failure must
  be reported, not tuned away inside a launched comparison.
- New initialization must preserve independent fresh block draws. Within each
  size/bundle, arms must share identical body and output weights and sampler
  streams. Specify and test cross-size initialization mapping explicitly before
  freezing roles; never clone one block repeatedly with identical weights.
- Before scientific launch, freeze within-size quality gates, largest-minus-
  reference interaction and practical threshold, all seed roles, uncertainty
  and multiplicity handling. Report paired seed effects as well as conditional
  document uncertainty. A nonsignificant interaction is not equivalence.
- Freeze a small broader-domain evaluation selected without model scores and
  deduplicate it against training and itself. Existing Books/Python scores
  cannot serve as untouched confirmation. Keep the original test unopened.
- Record operator-based FLOP estimates separately from synchronized Mac timing.
  Include body, output and actual composer work; disclose optimizer, memory,
  lookup and other uncounted operations. Do not use6NT blindly for embeddings.
- Qualify only newly required depth construction, initialization, finite
  forward/backward/update behavior, complete restart with optimizer and active
  averaging, scorer coverage, serialization, timing and RSS/MPS memory. A
  depth4 parity check is required for the new wrapper; do not rerun its closed
  scientific experiment as a prerequisite. Use a new namespace. Historical
  depth4 identities reject arbitrary depths and must remain unchanged.

## Whole-comparison planning cost

`tokenizer-evidence/general-rsi-cross-size-boundary-001/allocation.json` binds the
source and original weight inventories. Preliminary caps for all18 confirmations:
36 final endpoints3,834,266,712bytes including2MiB/container; active restart plus
serialization overlap1,342,177,280bytes; metadata512MiB; engineering artifacts
1GiB; failed attempts2GiB; reserve10GiB plus extra2GiB. Total21,819,442,264bytes.
Free space at calculation was22,509,068,288bytes, leaving689,626,024bytes beyond
that complete planning requirement. There is no new storage blocker at this
snapshot. These are caps, not measured new-depth serialization or RAM admission.
The largest restart cap includes parameters, up to two optimizer states, EMA and
64MiB additional space, rounded upward; two copies cover atomic overlap.
All screen costs are zero under the explicit fixed-recipe scope.

The deliberately pessimistic old-wall-time × depth envelope, plus30%, is
293,893.865seconds (81.64hours). It overcharges vocabulary/composer work as if it
scaled with depth and is not a measured runtime prediction. The36-hour initial
tranche is not yet admitted. The next timing pilot must replace this envelope,
include scorer/generation/restart/failure time, and either fit36hours or justify a
prospective revision before any scientific outcomes. Do not shrink a launched
panel, silently omit a size or start only the convenient arms.

Next executable action: implement the isolated fixed-width depth wrapper and
its exact depth4 parity/parameter-inventory check. Then freeze a bounded
engineering qualification for the new8/16-depth shapes under the shared lock,
measure both arms, and complete the scientific/evaluation/resource contract.
No cross-size training worker is active and no scientific seeds are assigned yet.
