# MLA versus GQA: historical training and validation curves

The selected MLA recipe shows **no logged validation reversal through 1,200
updates**, at widths 256 and 512. Every MLA fresh seed's sampled validation loss
decreases at every saved observation. The same is true of the short GQA runs.
The previously observed reversal belongs to the longer, 3,000-update GQA recipe.
These data do not show whether MLA would overfit if trained to 3,000 updates.

Open [the comparison](mla-gqa-loss-comparison.png), [complete histories with
individual runs](mla-gqa-full-histories.png), or [both pages as PDF](mla-gqa-training-validation.pdf).
Editable SVG versions, [93 observations as CSV](loss-history.csv), and
[derived values with source hashes](analysis.json) are included.

## Comparisons and results

| Width | Model | Updates | Fresh seeds | Mean sampled validation at update 901 | Final sampled validation | Final H200 full validation |
|---|---|---:|---|---:|---:|---:|
| 256 | GQA | 1,200 | 2811–2813 | 4.645504 | 4.585290 | 4.753151 |
| 256 | GQA | 3,000 | 2811–2813 | 4.654480 | 5.358063 | 5.464003 |
| 256 | MLA | 1,200 | 2911–2913 | 4.600953 | 4.541677 | 4.707100 |
| 512 | GQA | 1,200 | 3411–3413 | 4.607174 | 4.550383 | 4.705184 |
| 512 | MLA | 1,200 | 3411–3413 | 4.554096 | 4.517259 | 4.657553 |

All losses are cross entropy in nats/token. The plotted curves are the sampled
online-model measurements, not the final full-validation H200 scores.

The width512 comparison is direct: both families use the same three seeds and
common training configuration, including peak LR0.00125, 1,200 updates and
cooldown/EMA start800. Source requests were checked to have identical training
settings except output paths, and identical data configuration. MLA has lower
sampled validation CE at each post-startup logged point in each of these three
paired seeds. Both continue improving through their last observation.

The width256 GQA and MLA runs come from different campaigns and different seed
sets. They share width, peak LR, evaluation sample, context/batch settings and
the short schedule, but are **not a fresh paired architecture comparison**.
Their curves are descriptive context for the preceding GQA overfitting plot.
The initial width256 MLA campaign's actual paired control was MHA, not GQA.
No paired-seed confidence or causal architecture claim is made from the D256 overlay.

The long GQA curve turns upward in validation while its training loss falls,
which supports overfitting in that recipe. Short-run MLA/GQA validation keeps
falling despite a training–validation gap. A gap alone does not establish the
late-training reversal observed in long GQA. The MLA lines end at 1,200; there
is no extrapolation or invented 3,000-update MLA curve.

## Metric definitions and presentation

- The figures use 15 fresh confirmation runs from `horizon-001`, `mla-001`, and
  `wide-attention-001`. Pilot and LR-screen runs are excluded. The conclusion
  concerns these selected recipes, not every tested learning rate or MLA variant.
- Original trainer source was inspected at each campaign's frozen Git commit
  and verified against its source hash. `train_ce_loss` is a single minibatch
  before its optimizer update; `val_ce_loss` averages ten fixed sampled validation
  batches using online weights after the update. The validation RNG resets to
  seed2026 each time. It is not a fresh random sample at each point.
- All runs use GPT-2-tokenized Tiny Shakespeare, width256 or512, context128,
  batch8 and peak LR0.00125. Validation sampling totals 10,240 target positions
  per log, with possible overlap. It is not full-split coverage.
- `step` is zero-based; the x-axis is `step + 1`, completed updates. Thus the
  commonly cited “900-step” log is plotted at901. Logging is about every300
  updates, with the exact final update included. A minimum between logs cannot
  be identified from these histories.
- Top panels are validation; bottom panels are noisy training minibatches.
  Colors distinguish recipes. The main figure begins at301 and shows means
  with three-seed min–max bands, not confidence intervals. The second page
  includes startup and thin individual-seed traces. Straight segments connect
  observations without smoothing. Width columns have different x-axis ranges;
  corresponding metrics use the same y-axis limits.
- Dotted lines mark configured cooldown/EMA starts800 or2,000. The start-step
  versus completed-update indexing differs by at most one update; these markers
  are schedule context, not extra measured loss observations.
- H200 means an EMA half-life of200 updates. Its final full-validation results
  are listed separately above; no intermediate H200 history is invented.
- The same validation text has been used adaptively in research. Test loss
  remains unscored. Improving for1,200 updates does not establish immunity to
  overfitting, an optimal stopping time, or behavior on larger corpora/models.

## Provenance and reproduction

`build_curves.py` reads preserved metrics, requests and results under
`research/tokenizer-evidence/`, plus the three original run plans. It checks
portable metric bytes against original logs, exact observation steps, finite
values, configurations, and source hashes. It records per-seed minima and
monotonic-validation diagnostics. These checks do not rerun any historical
campaign, evaluate a checkpoint or launch training.

Run with Python, NumPy and Matplotlib:

```sh
python /path/to/LLM-pretraining-RSI/research/figures/mla-gqa-loss-curves-2026-11-02/build_curves.py
```

Rendered using Python3.12.14, NumPy2.5.4 and Matplotlib3.11.2 in temporary plotting
tooling. Both figures were visually inspected; PDF page count, text bounds and
the93-row export were checked. Changes are confined to this artifact directory;
research source, checkpoints, progress state and concurrent work are untouched.
The requested historical comparison is complete. Answering the unmeasured
3,000-update MLA question would require a prospective matched-duration study.
