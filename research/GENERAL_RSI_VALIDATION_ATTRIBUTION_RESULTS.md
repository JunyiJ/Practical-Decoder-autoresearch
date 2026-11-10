# Original validation: native's advantage persists outside whitespace

**Whitespace does not explain the accepted native equal-time validation win.**
Across the97,816 targets that are not pure ASCII whitespace, native averages
0.125862 nats/token lower loss than byte. Each of the three pairs favors native
outside whitespace on both FineWeb-Edu and Wikipedia. This is descriptive error
analysis of the existing comparison, not a new independent confirmation.

## Exact reconstruction and fixed partition

All six retained time-EMA checkpoints scored all108 original validation parents
and100,416 targets. Every one of the648 parent CEs reproduced its original value
exactly, with maximum absolute difference0.0. The overall native-minus-byte CE
remains−0.122101660243. No checkpoint, seed, document or target was selected or
removed according to the new output. Original quality gates were not rerun.

The four byte-span categories reuse the transfer diagnosis's predicates. Pure
ASCII whitespace takes precedence, followed by whole tokens within grid lines
or Greek-character placeholders, then the remainder. End-of-document tokens
are assigned to the remainder and are not interpreted as literal markup. The
first context token of each parent remains excluded, as in original scoring.
All108 EOT targets remain included.

| Category | Targets | Native CE | Byte CE | Contribution to overall native-minus-byte CE |
|---|---:|---:|---:|---:|
| Pure ASCII whitespace | 2,600 | 1.176717 | 1.157338 | +0.000502 |
| Grid-line non-whitespace | 334 | 5.172527 | 5.244600 | −0.000240 |
| Greek-placeholder non-whitespace | 0 | — | — | 0 |
| Remainder | 97,482 | 5.069414 | 5.195460 | −0.122364 |
| All targets | 100,416 | 4.968966 | 5.091068 | −0.122102 |

Contributions divide category loss differences by the original100,416 targets,
so they sum to the original result. Category CE divides by that category's count.
Whitespace accounts for2.59% of validation targets and slightly favors byte
in the overall aggregate. It therefore offsets, rather than causes, native's win.

| Source | Non-whitespace targets | Native-minus-byte CE on those targets |
|---|---:|---:|
| FineWeb-Edu | 78,827 | −0.122941 |
| Wikipedia | 18,989 | −0.137991 |
| Combined | 97,816 | −0.125862 |

“Non-whitespace” includes mixed whitespace/text tokens, punctuation, markup and
EOT. The remainder is not a prose-only or semantic-content benchmark. There is
no new category significance test, bootstrap, human rating or general-competence
claim. All per-arm/category counts, all paired contributions and original input
bindings are retained in `general-rsi-validation-attribution-001/summary.json`.

## Hypothesis update and next action

The formatting explanation from transfer does not generalize to the original
validation mixture. In transfer, native's mean advantage came from whitespace,
while byte had lower mean loss in every other nonempty category. Here native
has lower mean loss on the dominant remainder in all three pairs and both source
families. This supports a domain-dependent interpretation. It does not explain
whether exposure, spelling sharing, optimization or another factor causes it.
The original native adoption remains scoped to its equal-time validation result;
neither arm passed the transfer gates, and fixed generation quality remains weak.

The next qualified engineering question is computation reuse, documented in
[the composer efficiency boundary](GENERAL_RSI_COMPOSER_EFFICIENCY_BOUNDARY.md).
An exact reconstruction of384 historical training batches found a mean of491.951
distinct IDs per1,024 input positions. The current model recomputes the same
composition for repeated IDs within an update. This is51.96% repeated composer
rows, not a measured speed gain. The batch hashes match both original paired
arms' logs. No model training was performed for this census.

Next, freeze and execute a standalone composer reuse correctness/reproducibility
and timing test. Preserve the mathematical function and explicitly qualify any
floating-point reduction change. Historical MPS repeated-index backward issues
make a naive gather-based implementation unsafe to assume equivalent. If the
component fails its declared correctness or benefit gate, retain that result
and stop the candidate. If it passes, full-model optimizer/restart qualification
and a complete prospective quality-panel forecast are still required. No full
training panel, collision-length repair, new model size or test opening is
admitted by this analysis.

## Execution and evidence

The diagnostic finished in222.695 guarded seconds under its900-second cumulative
limit. Peak RSS was1,157,660,672bytes. Free space at worker completion was
25,416,863,744bytes, above the10GiB reserve. Retained evidence before this report
and review totaled2,212,057bytes within the32MiB full output/retry/overlap cap.
All six first FP32 token-loss arrays are retained. There were no numerical
failures, retries, new model weights or generation outputs. The worker exited.

The new variable-length retention and explicit EOT boundary were qualified;
completed training, restart and checkpoint-loader prerequisites were not repeated.
The protocol and source were frozen in commit475049355 before outputs. Original
source/log/metric evidence, selected models, all first generations and historical
restrictions remain preserved. Test contents remain unopened.

Evidence root: `research/tokenizer-evidence/general-rsi-validation-attribution-001/`.
It contains the plan, label inventory, qualification, full first attempt, guard,
summary and completion review. The separate batch census is in
`general-rsi-composer-efficiency-preparation-001/batch-reuse.json`.
