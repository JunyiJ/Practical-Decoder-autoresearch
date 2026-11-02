# GQA loss curves: evidence of overfitting

The saved histories provide strong evidence of overfitting in the 3,000-update
GQA recipe. Every fresh seed's fixed-sample validation loss reaches its lowest
logged value around 900–1,200 updates, then rises while training minibatch loss
falls overall. All three 1,200-update runs finish with their sampled validation
loss still improving. This is a retrospective diagnostic, not a new adoption or
optimal-stopping decision.

- [Mean loss curves and LR schedules](gqa-loss-curves.png)
- [Individual seeds, including complete startup histories](gqa-loss-curves-by-seed.png)
- [Both figures as PDF](gqa-training-validation-curves.pdf)
- [Exact logged observations as CSV](loss-history.csv)
- [Derived values, diagnostics and source hashes](analysis.json)

Both plots also have editable SVG versions. They use the six fresh confirmation
runs, seeds 2811/2812/2813, from `tokenizer-horizon-001`. Screening seed 2801 and
the 1,800-update screening arm are excluded.

## What the long-run curves show

| Measurement, averaged over three seeds | Update 1,201 | Update 3,000 |
|---|---:|---:|
| Training minibatch CE | 3.241778 | 1.612543 |
| Sampled validation CE | 4.637112 | 5.358063 |

The validation rise starts before the 2,000-update cooldown/EMA boundary and
continues through the cooldown. It is visible in the online-model validation
curves, so it cannot be explained solely as a final EMA averaging artifact.
The respective training schedules still differ; this does not isolate duration
from cooldown or establish that 1,200 is optimal.

| Seed | Lowest logged validation update in long run | Minimum sampled CE | Final sampled CE |
|---|---:|---:|---:|
| 2811 | 901 | 4.618543 | 5.385853 |
| 2812 | 1,201 | 4.614378 | 5.336611 |
| 2813 | 1,201 | 4.649289 | 5.351724 |

Each long run's sampled validation CE increases at every logged observation after
update 1,201. Individual training minibatches fluctuate, so their losses are not
claimed to fall monotonically. No smoothing or additional evaluations were used.

## Measurement semantics

The producer was inspected at the campaign's original Git commit
`de0da6894da5392c4925695507f88ee13a602157`, and its trainer and schedule source hashes
were checked against the frozen plan.

- `train_ce_loss` is one training minibatch's CE measured **before** its optimizer
  update. It is not a full-training-set loss or a moving average.
- `val_ce_loss` evaluates the current online model **after** that update, averaging
  ten sampled validation batches. The validation generator resets to seed 2026
  at each evaluation, so the sample is fixed across steps and runs. Each batch
  has eight contexts of length 128; 10,240 target positions are sampled, with
  possible overlap, rather than exhaustive validation coverage.
- Logged `step` is zero-based. The plots show `step + 1` completed updates. Thus
  the common log labels 300/600/900/1200 correspond to 301/601/901/1201 completed
  updates. The last points are exactly 1,200 or 3,000. The training value is
  associated with its update's row despite being measured just before it.
- Dotted lines show the configured start steps 800/2,000 for cooldown and EMA.
  The small LR panel reconstructs the exact schedule; its values are checked
  against all logged scheduled learning rates. The one-update indexing convention
  does not affect interpretation at this plot's resolution.
- Main-figure lines are means across three seeds; shaded regions are seed minima
  and maxima, **not confidence intervals**. Straight segments connect sparse
  observations recorded about every 300 updates. The detail starts at update 301;
  the companion figure includes all startup observations at update 1 (CE ~10.8).
- The sample-based online validation curves are different measurements from the
  prior figure's full-validation H200 endpoints. Final full-validation means:

| Recipe | Raw online checkpoint | H200 checkpoint |
|---|---:|---:|
| 1,200 updates | 4.742122 | 4.753151 |
| 3,000 updates | 5.547186 | 5.464003 |

The different sampling and weight averaging explain why the plotted terminal
validation values are not 4.753/5.464. There are no intermediate full-validation
or H200 curves in these saved logs; none are invented or inferred from endpoints.
Training and validation samples differ, so their instantaneous numerical gap
should not be treated as an accurately estimated full-dataset generalization gap.

## Scope and reproduction

Width256 GQA, four layers, GPT-2-tokenized Tiny Shakespeare, context128, batch8,
peak LR0.00125, separate 1,200/3,000-update recipes. These are locally trained
models evaluated on adaptively reused validation text, not independent test
evidence or a statement about larger-scale GQA. The test holdout remains unscored.

`build_curves.py` reads the preserved metrics, requests and final results under
`research/tokenizer-evidence/horizon-001/confirm-gqa-w256-s{seed}-n{horizon}/`.
It also verifies portable log bytes against the original run logs when present,
checks all expected log steps, frozen configuration settings and LR arithmetic,
and exports all 48 logged observations. It does not load model weights, restart a
controller, modify research state or launch training. Both rendered figures were
visually inspected, and the PDF's page count and text bounds were checked.

Run with Python, NumPy and Matplotlib:

```sh
python /path/to/LLM-pretraining-RSI/research/figures/gqa-overfitting-2026-11-01/build_curves.py
```

The environment used Python3.12.14, NumPy2.5.4 and Matplotlib3.11.2 from temporary
plotting tooling. Training dependencies and existing concurrent edits were left
untouched. This requested diagnostic is complete. Any future experiment intended
to isolate duration/cooldown or select a stopping rule would require its own
prospective comparison; these plots authorize no additional training.
