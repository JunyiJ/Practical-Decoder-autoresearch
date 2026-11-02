# Token-based LLM RSI: major learnings

Four figures summarize completed token-based research through November 1, 2026
(Los Angeles). This is a descriptive synthesis of saved evidence, not a new
experiment, training launch, checkpoint evaluation, or adoption decision.

Open [the four-page PDF](token-rsi-major-learnings.pdf). Each figure is also
available as a high-resolution PNG and editable SVG.

| Figure | Learning | Main evidence |
|---|---|---|
| [1. Training recipe](01-training-recipe.png) | Lowering the initial learning rate produced large gains at both tested widths. The shorter GQA training/cooldown/EMA recipe improved CE by 0.711 nats/token while cutting training time about 60%. | [LR review](../../tokenizer-evidence/quality-001/final-review.json), [horizon review](../../tokenizer-evidence/horizon-001/final-review.json) |
| [2. Attention tradeoffs](02-attention-tradeoffs.png) | GQA's quality result depended on width. MLA achieved better validation quality and substantial cache savings in its scoped comparisons, alongside slower local decoding. | [GQA review](../../tokenizer-evidence/gqa-001/final-review.json), [MLA review](../../tokenizer-evidence/mla-001/final-review.json), [D512 direct comparison](../../tokenizer-evidence/wide-attention-001/final-review.json) |
| [3. Later refinements](03-later-refinements.png) | LayerNorm, H100 EMA, and attention multiplier 1.0 improved every fresh seed, but gains were below the prospective 0.02-nat adoption threshold. Other follow-ups were mixed or worse. All nine retained their controls. | [Quality results](../../TOKENIZER_QUALITY_RESULTS.md), [closed activation review](../../tokenizer-evidence/activation-preparation-001/inference-recovery002-closed-review.json); exact nine-study values and source hashes in [figure-data.json](figure-data.json) |
| [4. Architecture lessons](04-architecture-lessons.png) | Tying saved parameter memory but failed quality neutrality; its matched control separates part of the loss from other recipe changes. The tested small and wider NoPE hybrids failed their adoption routes. | [Table review](../../tokenizer-evidence/table-001/final-review.json), [small hybrid review](../../tokenizer-evidence/hybrid-001/final-review.json), [wider hybrid closure](../../TOKENIZER_WIDE_HYBRID_PREPARATION.md) |

## Interpretation

- The scope begins with the GPT-2 BPE transition, not the earlier character-level
  campaigns. Character CE and subword CE are not comparable.
- Data are from Tiny Shakespeare, with a 50,257-entry vocabulary and 267,688
  training tokens. The validation split is reused adaptively. Test loss remains
  unscored in these studies.
- All displayed scientific contrasts use three fresh seeds relative to their
  selection stages. This does not make the reused validation text a fresh holdout.
- CE denotes cross entropy in nats/token. H200 denotes an exponential moving
  average with a half-life of 200 updates, not GPU hardware. D denotes model width.
- These are separate controlled comparisons, with different seeds, widths,
  controls and sometimes budgets. Their deltas must not be added into a cumulative
  improvement curve or interpreted as a shared absolute-loss leaderboard.
- The shortened-horizon result changes duration, cooldown and EMA start jointly.
  It does not isolate duration or establish an optimal number of updates.
- Plotted whiskers are descriptive conditional 95% text-bootstrap intervals at
  4,096-token block size. They condition on the trained models and are not
  population-seed confidence intervals. Adoption decisions used each study's own
  frozen rules, including other block sizes and simultaneous bounds where required.
- In Figure 3, open dots show the three seed differences separately from the text
  intervals. Small consistent gains are distinguished from practical adoption.
- Attention latency is batch 1, FP32 MPS, context 128, 32-token prefill and 64-token
  teacher-forced decode with two models resident. The figure preserves reported
  aggregation: ratio of aggregate medians for GQA/D256 MLA; median of seed-paired
  median ratios for D512 MLA. Cache savings do not imply equal total-memory or
  production-cost savings.
- The table-sharing loss versus legacy includes readout scaling/initialization
  differences. Its matched contrast still jointly changes sharing, gradients,
  optimizer histories, expressivity and selected LR.
- MoE has no completed tokenizer quality result in this synthesis; capacity
  deferral is not negative quality evidence. Parallel residual work is in
  preparation and is excluded from the scientific outcomes.

## Reproduction and verification

`build_figures.py` reads existing JSON reviews and the wider-hybrid closure
section without loading checkpoints. It writes the exact plotted values and
SHA-256 digests of its evidence sources to `figure-data.json`. The wider-hybrid
closure excerpt is included because that figure uses the recorded prose summary.
Later-review mean deltas are checked against their three seed deltas.

From any directory, with Python, NumPy and Matplotlib available:

```sh
python /path/to/LLM-pretraining-RSI/research/figures/token-rsi-learnings-2026-11-01/build_figures.py
```

The original render used Python 3.12.14, NumPy 2.5.4 and Matplotlib 3.11.2 in
temporary tooling; project training dependencies were not modified. All four
rendered PNGs were visually inspected. The PDF contains the same four figures.
Existing source, checkpoints, research state and concurrent uncommitted work
are outside this artifact change. The next summary update should follow a new
completed quality comparison; no training action follows from these figures.
