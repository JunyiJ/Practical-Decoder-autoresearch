# Transfer boundary after the equal-time result

Status: evidence analysis and source-metadata audit complete. No transfer text has
been scored and no transfer panel is admitted. This note does not change any
closed campaign. The next scientific question is broader-domain fit, before an
additional model-size training panel.

## Measured local basis

- `tokenizer-evidence/general-rsi-equal-time-001/decision.json`: all three fresh
  equal-time pairs favor native. Mean time-EMA CE is4.968966 versus5.091068.
  Both existing source means improve. The adopted recipe preserves all three
  native references and all three byte comparators.
- `tokenizer-evidence/general-rsi-transfer-preparation-001/local-basis.json`
  binds this analysis to original results, reviews, update logs and document
  scores. Seed-averaged native loss is lower on80/85 FineWeb-Edu documents and
  22/23 Wikipedia documents. Six documents favor byte. This is an unweighted
  diagnostic; it does not replace the frozen token-weighted acceptance rule.
- The same eight-document RAW probes improve from first probe through the
  second probe to the final endpoint in every run. Every logged loss and gradient
  norm is finite. Clipping occurs on3.6–4.1% of native updates and41.9–64.0% of
  byte updates. The clipping difference is a measured optimization difference,
  not proof that clipping causes byte's loss disadvantage or an invalid run.
- `general-rsi-equal-time-001/generation-evaluation/summary.json`: both arms
  show weak generation. Greedy repeated-trigram means are0.508 and0.542;
  sampled means are0.020 and0.024, with continued semantic drift. Lower CE does
  not establish reliable generation. Agent ratings are exploratory.
- Contrary evidence: the closed equal-update panel favored byte, and longer byte
  training passed its own frozen gates. Equal-time superiority depends on the
  operating constraint. Those findings are preserved, not pooled as fresh seeds.

## Primary literature and applicability

Penedo et al., [FineWeb, arXiv:2406.17557v2](https://arxiv.org/html/2406.17557v2),
sections3.1,4.1–4.2 and appendixF.4/Table3: educational filtering changes topic
coverage. FineWeb fits broad web and several social sources better, while
FineWeb-Edu favors several academic/Wikipedia and programming sources. Their
Paloma comparison deliberately omits decontamination. Thus it motivates domain
checks but supplies no uncontaminated local effect estimate. The1.71B models,
2048-token context and much larger training corpus differ from this study.

Magnusson et al., [Paloma, arXiv:2312.10523v2](https://arxiv.org/html/2312.10523v2),
section3, appendixC.2.3 and appendixD.1.1: vocabulary and inference format must
be controlled; they score separate documents using disjoint chunks. Domain
improvement varies, and some C4/mC4 domains worsen with more training. Most
domains nevertheless improve. AppendixA cautions that downstream correlations
are unstable at their scale. Our six small checkpoints cannot establish their
large-model scaling claims. Their decontamination removes training documents;
we must instead exclude overlapping evaluation documents because our checkpoints
are already frozen. This produces a different sampling population.

## Mechanism, prediction and decision value

Inference: extra native updates may improve broadly useful token predictions,
or may chiefly improve fit to our educational/encyclopedic mixture. The present
validation sources cannot distinguish those alternatives. Byte's spelling-based
sharing may retain advantages on unfamiliar lexical distributions, but no local
transfer gain is established.

Prospective prediction: native retains lower mean loss on both historical-book
prose and technical documentation, across all three paired checkpoints. A reversal
on either domain falsifies the strongest two-domain generalization claim. A mixed
or uncertain result narrows the claim to the domains supported by evidence. A
consistent result justifies retaining native for this local time budget on those
sources; it does not establish general language competence or all-domain gains.
Neither outcome changes the already accepted in-distribution comparison.

Ranked next steps:

1. Evaluate the existing six time-EMA checkpoints on prospectively selected,
   deduplicated book prose and technical documentation. This directly tests a
   missing generalization dimension with no new training.
2. Consider one additional model size only after transfer. Current evidence
   does not identify a size interaction that warrants another full training
   panel first. More training would also confound the immediate domain question.
3. Preserve the clipping observation for a later optimization hypothesis. Do
   not change clipping or reopen closed studies during transfer evaluation.

## Source audit, before scientific text selection

- Historical books: [DeepMind PG-19](https://github.com/google-deepmind/pg19),
  repository commit `b0e1f2925359688af19f8784c23c3deeda52d3ea`. The official
  README identifies `deepmind-gutenberg` as its public data bucket. The captured
  `validation/` object listing contains50 books totaling17,733,002bytes, with
  object generations, sizes and checksums. Its validation split is new to these
  models and separate from the original RSI test, which stays unopened.
  The publisher declares Apache2.0; the books predate1919 and include historical
  style/bias and publisher-applied text substitutions. See the official README
  and [dataset card](https://huggingface.co/datasets/deepmind/pg19). This proposed
  short-context sample is not the full PG-19 long-context benchmark.
- Technical documentation: CPython tagv3.13.0 resolves through annotated tag
  `3f27099d916c7b885e3daf1fabedcc119462014d` to commit
  `60403a5409ff2c3f3b07dd2ca91a7a3e096839c7`. Its293 `Doc/library/*.rst` blobs
  total6,476,534bytes. The captured tree records every blob SHA and size.
  [Python's license page](https://docs.python.org/3.13/license.html) states PSF
  License2 for documentation and dual PSF/0BSD for examples. Pin the license
  text at the selected commit during acquisition. Raw reStructuredText, prose
  extraction and code-block inclusion are different tasks; freeze one format
  before model scoring. Programming is not guaranteed to oppose educational
  filtering, so this is a deliberately contrasting domain, not a predicted win.
- Considered but not selected: annual CFR legal prose. The official
  [2025 Title1 directory](https://www.govinfo.gov/bulkdata/CFR/2025/title-1)
  and an HTTP HEAD identify a795.6KB XML volume last modified19November2025.
  Its sections have strong within-volume dependence, requiring a different
  sampling unit. [GPO policy](https://www.govinfo.gov/about/policies#copyright)
  also distinguishes government text from embedded third-party material.
  Do not add this domain opportunistically after seeing the two-domain result.
- Paloma is relevant methodological evidence, but its hosted files currently
  require accepting an access agreement. AG News's card lists an unknown
  license. Neither source is needed for this bounded comparison. No access
  agreement was accepted, credentials supplied, or gated data fetched.

Exact metadata files are in `general-rsi-transfer-preparation-001/`. The initial
system-Python metadata request failed certificate verification. Curl with normal
certificate validation recovered it. An initially guessed `pg19` bucket returned
401; the official README supplied the correct public bucket, whose listing
succeeded. These are acquisition errors, not quality results. No TLS checks were
disabled and no gate was bypassed.

## Prospective controls and complete cost boundary

Before admission, write a separate immutable protocol and data manifest with:

- Both arms, all three paired time-EMA checkpoint hashes from the adopted recipe.
  No best-seed selection, no new training, no reuse as fresh initialization seeds.
- Pinned source objects and licenses, domain definitions, document/excerpt units,
  deterministic selection seed/order, target token budget, minimum coverage,
  exclusions, text extraction and tokenizer hashes. Exclude original training
  and validation overlap before scoring. Do not open the original test to dedup.
  Retain exclusion receipts and state residual contamination limitations.
- Same GPT-2 output vocabulary, context128, document reset and nonoverlapping
  target scoring as the qualified scorer. Only qualify the new acquisition,
  deduplication and loader boundaries with synthetic inputs or already authorized
  validation data. Do not repeat the completed training/restart qualifications.
- Token-weighted CE within domain, equal domain weighting for the macro score,
  all paired seed results, and a prospectively fixed document/parent-cluster
  bootstrap. Domain samples are not independent new training seeds. Freeze
  direction, uncertainty, multiplicity treatment and any practical threshold
  before scores. Generation remains the already completed fixed evaluation;
  no post-result decoder tuning or composite quality score.

Measured primary evaluation time is18.82–19.68s per100,416targets for native and
44.90–45.84s for byte. Using each arm's observed maximum gives an estimated
400.91s for204,800targets per model across six models. This is a linear scoring
estimate, not total runtime or a guarantee. The source audit is24,209,536bytes
before licenses, extraction and tokens. A provisional1GiB incremental envelope
must cover retained source copies, exact-overlap index, extracted text/tokens,
all six per-document score files, metadata and atomic-write overlap, plus the
existing10GiB reserve. Existing weights remain retained and are not copied.
No checkpoint serialization or new generation output is needed. Use a prospective
2-hour total preparation/scoring/retry guard, then replace the envelope with an
exact manifest-based forecast before admission. Current free space was25.6GB
at equal-time completion. No new capacity blocker is known.

Next executable action: freeze the two-domain acquisition/selection/deduplication
protocol, acquire the pinned public source objects within the byte envelope,
qualify only the new loader and overlap checks, then bind the complete dataset,
analysis and resource contracts before scoring all six models serially under the
shared nonblocking research lock. No numerical transfer evaluation is active.
