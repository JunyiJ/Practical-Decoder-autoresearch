# Fixed-weight four-condition context diagnostic

**Older context adds predictive value at both depths.** Avoiding 128-token resets yields the larger gain. Shuffling only the older 128 subword tokens removes the supported gain over rolling128: the remaining difference is unresolved at depth4 and adverse at depth16. This supports sensitivity to the older prefix’s ordering/placement, without isolating semantic structure or proving reasoning.

Six saved context256-trained time-EMA models, three bundles at each depth, scored the same 72,876 validation targets in 108 documents. We performed no training or generation. The original test remained unopened. A reuses accepted reset128 scores on this exact subset. B uses rolling128, C rolling256, and D shuffles only the oldest128 tokens of C. The recent128 tokens remain identical.

| Depth | A reset128 | B rolling128 | C rolling256 | D shuffled older128 |
| --- | ---: | ---: | ---: | ---: |
| 4 | 4.827565 | 4.679464 | 4.654010 | 4.681325 |
| 16 | 4.799533 | 4.646210 | 4.616976 | 4.651995 |

Cross-entropy is in nats per target token. Lower is better. Each mean averages the three fixed models at that depth.

| Comparison | Depth4 delta [adjusted interval] | Depth16 delta [adjusted interval] |
| --- | --- | --- |
| B−A: avoiding resets | -0.148101 [-0.170682, -0.126430] | -0.153323 [-0.176435, -0.131646] |
| C−B: older-token value | -0.025454 [-0.033273, -0.018591] | -0.029234 [-0.037895, -0.021414] |
| C−D: shuffle sensitivity | -0.027315 [-0.036125, -0.019341] | -0.035020 [-0.044584, -0.025888] |
| D−B: benefit remaining after shuffle | +0.001861 [-0.001518, +0.005188] | +0.005786 [+0.001270, +0.011544] |

B−A, C−B and C−D are negative in all three bundles at both depths, and all adjusted intervals exclude zero. D−B has mixed signs at depth4; its interval crossing zero does not establish equivalence. D−B is positive in all three depth16 bundles, with an adjusted interval above zero. Thus the shuffle does not leave a supported predictive benefit at either depth.

| Direct depth16−depth4 comparison | Delta [adjusted interval] |
| --- | --- |
| B−A: avoiding resets | -0.005222 [-0.010583, -0.000362] |
| C−B: older-token value | -0.003780 [-0.006632, -0.001326] |
| C−D: shuffle sensitivity | -0.007705 [-0.010396, -0.004928] |
| D−B: benefit remaining after shuffle | +0.003925 [+0.001117, +0.007418] |

These direct contrasts support larger reset benefit, older-context benefit and shuffle sensitivity in the fixed depth16 models. Two depths provide transfer/replication evidence, not a scaling law.

All intervals use the prospectively frozen joint twelve-contrast family: 100,000 paired document-bootstrap draws, family alpha0.01 and Bonferroni quantiles0.01/24 and1−0.01/24. Bundle loss sums are averaged before document resampling. These intervals condition on reused development documents, these six saved models and one fixed shuffle per target. They do not estimate population-seed or all-shuffle uncertainty. The original B/C three-contrast analysis remains byte-for-byte unchanged in its own evidence directory.

| Depth / bundle | B−A | C−B | C−D | D−B |
| --- | ---: | ---: | ---: | ---: |
| 4 / 1 | -0.148404 | -0.026000 | -0.027603 | +0.001603 |
| 4 / 2 | -0.144015 | -0.023469 | -0.022628 | -0.000841 |
| 4 / 3 | -0.151884 | -0.026894 | -0.031715 | +0.004821 |
| 16 / 1 | -0.153639 | -0.030433 | -0.036139 | +0.005706 |
| 16 / 2 | -0.149451 | -0.026145 | -0.031250 | +0.005105 |
| 16 / 3 | -0.156879 | -0.031124 | -0.037670 | +0.006546 |

Source means are descriptive, without additional significance claims. FineWeb-Edu contributes58,679 targets and Wikipedia14,197.

| Depth / source | B−A | C−B | C−D | D−B |
| --- | ---: | ---: | ---: | ---: |
| 4 / fineweb_edu | -0.142692 | -0.023132 | -0.024295 | +0.001163 |
| 4 / wikipedia | -0.170457 | -0.035052 | -0.039797 | +0.004746 |
| 16 / fineweb_edu | -0.148594 | -0.026702 | -0.031639 | +0.004937 |
| 16 / wikipedia | -0.172869 | -0.039700 | -0.048995 | +0.009295 |

The results weaken a pure reset explanation: C improves over B even though both retain uninterrupted recent history. They also weaken an account based only on an unordered older-token multiset under this intervention. Shuffling subword tokens can break words, syntax, positional relations and the older/recent boundary, while creating inputs unlike training. The test cannot distinguish these mechanisms or establish reasoning. Representation, exposure, optimizer and schedule uncertainty remain across the saved training runs.

Independent completion review verified5,297 source bindings, all six endpoint hashes,1,302 score/saved-A file hashes, all target IDs and positions, and all72,876 permutation/input reconstructions. The3,762 A/B identical-input checks passed the frozen1e−5 tolerance, with maximum difference9.656e−6. A separate frequency-count implementation reproduced all100,000 bootstrap draws’ final intervals within1.12e−16. Qualification and unchanged-parameter records passed.

B/C scoring took4,524.28 guarded seconds and D3,062.42 seconds, about2.11 hours combined. Corrected final analysis took4.04 seconds. Across scoring, peak RSS was0.744GiB and peak MPS driver memory1.073GiB. The minimum recorded host reclaimable-page proxy was3.436GiB and free disk15.534GiB. These quantities are not additive measures of unified-memory use. Qualification also stayed inside its guards. All successful phases preserved the4GiB RSS,6GiB MPS and12GiB disk limits.

Two execution failures remain intact. The first D launch failed the combined pre-model host-memory admission condition before any target was scored; the absent snapshot prevents identifying the failed conjunct. A separate attempt passed the same guards and scored all models. Its analysis then hit the preserved stale result-path error. The corrected analysis changes only D input directories, with AST equivalence verified. No scores, scientific rules or first outputs were replaced.

Evidence: [integrated analysis](tokenizer-evidence/general-rsi-rolling-four-001/analysis.json), [completion review](tokenizer-evidence/general-rsi-rolling-four-001/completion-review.json), [frozen design](GENERAL_RSI_ROLLING_FOUR_CONDITION_DESIGN.md), and [original B/C analysis](tokenizer-evidence/general-rsi-rolling-context-001/analysis.json).

**Next action:** this requested diagnostic is complete. Close its monitor and await human direction. No new campaign is scheduled.
