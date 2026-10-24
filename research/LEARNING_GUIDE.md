# Learning LLM pretraining through our optimizer transfer project

**A project walkthrough and lab notebook · October 23, 2026**

This project asks a practical question: **can we find a useful training recipe cheaply at width 64, then carry it to widths 128 and 256 without retuning?** You will understand it best by following the chain from a character prediction, through a parameter update, to a controlled comparison. The aim is to be able to explain why an experiment was designed, reconstruct its result from the files, and identify what the result does not establish.

The completed experiments already teach an important lesson: the recipe with the lowest small-model loss can fail at a larger width. Our independent-decay, μP and Muon study has now completed all 99 trials. Muon with μP and independent decay improved every paired seed against the fresh reference at widths 64, 128 and 256. μP's additional benefit over standard Muon is less certain. The bounded Adam learning-rate follow-up is also complete (39 trials). The next MLE campaign uses the measured Muon recipe as its fixed baseline and begins with a zero-decay ablation.

**Suggested route:** read sections 1–4 to understand pretraining, sections 5–8 to understand the optimizer changes, then sections 9–12 to practice reading the evidence. The exercises include answers so you can test your explanation before checking it.

## 1 What is actually being pretrained

A language model assigns a probability distribution to the next token given the preceding tokens. Our tokens are individual characters from Tiny Shakespeare. This makes the vocabulary small and permits fast experiments on an Apple M4 Mac. Production LLMs generally use subword tokenizers and much larger corpora, but the prediction, gradient and optimizer loop is the same basic operation.

For a piece of text `To be`, the input and target sequences can be imagined as:

```text
input:  ["T", "o", " ", "b"]
target: ["o", " ", "b", "e"]
```

Each position predicts the next character. A causal attention mask prevents a position from looking at later input positions. Otherwise, the model could read the answer and produce a misleadingly low loss.

One batch has 8 sequences of 128 positions. It therefore contributes 1,024 next-character targets. A 3,000-step run processes 3,072,000 target positions. These are **training exposures**, not 3,072,000 distinct characters: windows sampled from the small training corpus can overlap and repeat.

The frozen protocol divides the corpus contiguously into 80% training, 10% validation and 10% held-out test. Training supplies gradients. Validation compares configurations. The held-out test remains closed during continuous exploration. Repeated use of validation can adapt our choices to that validation set even though no validation gradient is computed.

Our model has two transformer blocks. Width is the length of the hidden representation at each position. With fixed head dimension 32, widths 64, 128 and 256 have 2, 4 and 8 attention heads. The dense feed-forward hidden dimension is four times the model width. Increasing width therefore changes much more than a single vector length: many matrix dimensions and the parameter count grow, and matrix work grows roughly quadratically in width for fixed depth and sequence length.

The embedding table first turns each character ID into a learned vector. Attention lets a position combine information from earlier positions; the feed-forward network transforms each position’s features. The final output projection maps those features back to one score per vocabulary item.

The blocks use RMSNorm and rotary positional embeddings. RMSNorm rescales a position’s features using their root-mean-square magnitude and a learned scale. Rotary embeddings rotate query and key features according to position, giving attention positional information. The current final normalization is LayerNorm. This small implementation detail matters when reproducing the experiment: “a transformer with RMSNorm” is not a complete configuration.

**Read the implementation:** [model construction](../src/models/gpt.py), [transformer block](../src/models/blocks.py), [attention](../src/attention/mha.py), and [data loader](../src/data/loaders.py).

## 2 One optimizer step from beginning to end

A training step has six conceptual parts:

1. Sample matching input and next-character target windows from the training split.
2. Run the model forward to produce logits, one score per vocabulary item at every position.
3. Compute cross-entropy between the predicted distributions and the actual next characters.
4. Backpropagate to obtain a gradient for each trainable parameter.
5. Clip the global gradient norm at 1.0 in our current recipe.
6. Let the optimizer turn those gradients and its saved state into a weight update.

For logits z and correct character y, the loss at one position is `−log softmax(z)[y]`. The run averages over batch positions. If the model assigns the correct character probability 0.5, that position contributes about 0.693 nats; probability 0.1 contributes about 2.303 nats. Lower loss means the model assigns more probability to the observed continuation.

“Backward” and “optimizer step” are different operations. Backward computes derivatives of the current loss. The optimizer decides how far and in what direction to move using those derivatives, momentum or other accumulated information, and the learning rate.

Validation temporarily disables training-mode behavior and computes losses without changing weights. We evaluate 30 fixed-seed validation batches at a checkpoint. A reported **mean validation loss** first averages those batches for a run. A configuration's mean then averages the final losses of three separately initialized runs. It is not the average training loss over the entire learning curve.

A checkpoint contains more than model weights. Exact continuation also requires optimizer state, completed-step count, random-generator state and data-loader generator state. Restoring only weights can restart Adam's moments or change the sampled batches, creating a different training trajectory. Our tests compare interrupted-and-resumed training with uninterrupted training parameter by parameter.

**Read the implementation:** [training and validation](../src/train/train.py), [checkpoint state](../src/utils/checkpoint.py), and [continuation tests](../tests/test_research_training.py).

## 3 What the loss curves can tell you

Three different questions often get called “is training done?”:

- Did the process complete its requested budget? Here, that means 3,000 updates.
- Has the training objective stopped improving materially? This requires inspecting the trajectory under a stated criterion.
- Has generalization stopped improving, or started to worsen? Validation can behave differently from training loss.

Our runs answer the first question by construction. Several measured validation curves were still falling near the end. We therefore report **fixed-budget performance**, not convergence. The earlier 300-step runs were rapid screens; extending to 3,000 steps changed which apparent improvements transferred.

Perplexity is `exp(cross-entropy)` when the loss is in natural-log units. A loss of 1.70 corresponds to about 5.47 perplexity. This is an uncertainty measure, not an accuracy percentage or a statement that the model literally has 5.47 candidate characters in mind. Character perplexity cannot be compared directly with subword perplexity from a different tokenizer.

Compare curves at the same training exposure when asking about data efficiency. Compare elapsed training time when asking about hardware efficiency. A method may need fewer steps but spend longer on every step, especially if its optimizer performs extra matrix operations.

Our timing separates accumulated training-step time from evaluation and checkpoint time. Peak process RSS is a host-process memory statistic. MPS current allocation is a snapshot; neither should be renamed “peak GPU memory.”

## 4 Why hyperparameters can fail to transfer

A learning rate is a coefficient in an update rule. Its numerical value does not by itself specify the size of the resulting change in a model's function.

Imagine a hidden output `h = W x`. A small matrix change gives `Δh ≈ ΔW x`. If the input dimension doubles, the products contributing to that output also change. Their correlations, initialization scales, normalization and optimizer transformation all affect the size of `Δh`. Holding the number called “learning rate” fixed can therefore produce a different feature-learning regime at a wider model.

This is the distinction between **parameter space** and **function space**. A small change in individual parameters need not imply a small change in predictions. Conversely, different parameter magnitudes can sometimes represent similar functions, especially around normalization layers.

Our fifth long-budget iteration demonstrates the practical problem. LR 0.003 with zero decay improved the width-64 mean validation CE by 0.123375 nats/character against the fixed baseline. At width 256, it worsened CE by 0.015435, with all three seeds worse. The small-width choice was kept frozen; we did not retroactively select another recipe and call the transfer successful.

That reversal is a measurement. “The larger model took updates that were too large” is a plausible explanation, but remains a hypothesis until diagnostics or controlled interventions distinguish it from alternatives.

## 5 AdamW and the two meanings of decoupling

Adam tracks an exponential average of gradients and an exponential average of their squares. Ignoring bias-correction notation for a moment, an update direction resembles:

```text
first moment   ← β1 × first moment + (1 − β1) × gradient
second moment  ← β2 × second moment + (1 − β2) × gradient²
direction      ← corrected first moment / (sqrt(corrected second moment) + ε)
```

This is coordinate-wise adaptation: different scalar entries of a matrix can receive different effective rescaling. Our default AdamW settings retain PyTorch's betas 0.9 and 0.999 and epsilon 1e-8. Those settings are part of the experiment even when they are not included in the search.

Adding an L2 penalty to the loss puts a term proportional to the weights into the gradient, where Adam's adaptation acts on it. AdamW instead applies a separate shrinkage operation. That distinction is the subject of [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).

However, **separate from the adaptive gradient does not necessarily mean independent of learning rate**. In the conventional PyTorch-style update:

```text
W_next = (1 − η λ) W − η × adaptive_direction
```

The shrinkage per step is `η λ`. In the independent-decay variant used by our new implementation:

```text
W_next = (1 − δ) W − η × adaptive_direction
```

Here δ is the per-step shrinkage coefficient. Its units differ from λ. The two formulas can represent the same single update when `δ = η λ`, but behave differently when η changes across widths, parameter groups or time. This distinction is central to the [recent weight-decay transfer paper](https://arxiv.org/html/2510.19093v2).

For our actual recipes:

| Recipe | LR η | Conventional λ | Per-step shrinkage ηλ |
| --- | ---: | ---: | ---: |
| Historical baseline | 0.0006 | 0.1 | 0.00006 |
| Higher LR with same decay setting | 0.002 | 0.1 | 0.00020 |
| Higher LR with lower decay setting | 0.002 | 0.03 | 0.00006 |

Thus the WD=0.03 experiment restored the baseline shrinkage while changing LR. It did not prove that matching shrinkage caused the observed improvement; other effects remain.

As a thought experiment, ignore gradient updates entirely. A weight under constant independent decay is multiplied by `(1−δ)^T` after T steps. At δ=0.00006 and T=3,000, this is about 0.835. At δ=0.00020 it is about 0.549. These are **decay-only calculations**, not predictions of actual final weight norms, because training also changes the weights.

Copying a conventional WD value of 0.1 into independent mode would mean 10% shrinkage every step. This is why we test units and exact shrinkage, including zero LR, before interpreting model quality.

## 6 What μP changes in this project

μP means maximal update parameterization; μTransfer is the strategy of tuning at a small size and transferring the resulting hyperparameters to a larger size. The intended benefit is a more stable relationship between a base learning rate and feature learning as width grows. It is not transferring pretrained weights, and it is not a guarantee that every hyperparameter transfers across arbitrary depth, data or training-budget changes. See [Tensor Programs V](https://arxiv.org/abs/2203.03466).

A parameterization includes initialization scales, forward-pass multipliers and optimizer learning-rate rules. We use base width 64 and define `m = width / 64`.

Our implementation keeps the hidden matrices' fan-in initialization. For Adam, a hidden matrix uses nominal LR `base_lr / m`. Embeddings and vector parameters retain their base LR. The output head uses a coordinated initialization and forward scaling: multiply the default initialized output weights by `sqrt(m)`, then divide the output logits by m. Its optimizer LR stays at the base value because the explicit forward multiplier supplies its effective scaling. Linear biases in hidden layers are rescaled at initialization so their initial distribution does not silently shrink with width.

These are the exact choices in our code; a statement such as “we enabled μP” would omit necessary details. At the base width m=1, our SP and μP implementations have identical initial weights and pass a bit-exact one-update equivalence test on CPU. At larger widths, they intentionally differ.

The attention head dimension remains 32. Its normalization factor is therefore constant across this width sweep, while the number of heads changes. We retain that factor so the base model is not also receiving an unrelated attention-temperature change. This protocol does not test a head-dimension scaling rule.

A useful correctness diagnostic records hidden activation RMS and the size of feature or logit changes at several widths early in training. Nonfinite values or enormous unexplained drift are warning signs. Finite five-step synthetic checks establish that the implemented path runs; they do not prove asymptotic theory or guarantee that the best LR transfers after thousands of updates. Global gradient clipping and finite-width effects can still affect dynamics.

**Read our implementation:** [model scaling](../src/models/gpt.py), [optimizer groups](../src/train/optimizers.py), and [scaling tests](../tests/test_optimizer_transfer.py). The [reference μP library](https://github.com/microsoft/mup) provides a useful independent comparison of parameterization conventions.

## 7 What Muon does differently

Muon transforms an entire matrix-shaped momentum update. An idealized way to understand the transformation is to write a momentum matrix as `M = U Σ Vᵀ`, its singular value decomposition. The columns of U and V describe output and input directions; the diagonal values in Σ describe how strongly the matrix stretches those directions. Replacing the singular values with a normalized scale gives a direction related to `U Vᵀ`. The singular-vector directions are retained while the singular-value magnitudes are reshaped. This is different from Adam's entry-by-entry scaling.

Practical Muon approximates this transformation with a small number of Newton–Schulz matrix iterations rather than a full SVD. The approximation is not an exact orthogonal projection. The operation acts on the **update**, not a constraint forcing the learned weight matrix itself to stay orthogonal. See [Muon’s original explanation](https://kellerjordan.github.io/posts/muon/).

Our implementation uses native PyTorch Muon for hidden 2D matrices. Embeddings, the output head, normalization parameters and biases use AdamW. This is a **Muon–AdamW hybrid**, a meaningful part of the recipe. It uses momentum 0.95, Nesterov enabled and five Newton–Schulz steps. We record the installed PyTorch version because implementation details affect reproducibility. See the [PyTorch Muon API](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html).

Why might matrix normalization help? A gradient can have strong correlations across rows and columns. An optimizer that uses matrix structure can change the balance between update directions in a way a coordinate-wise method does not. Whether that gives better language-model quality per second is an empirical question; the extra matrix operations also cost time.

This distinction suggests two plots worth reading together: validation loss against processed tokens, and validation loss against training time. A lower loss after the same 3,000 steps establishes a fixed-token advantage in our setup. It does not automatically establish a wall-clock speedup.

## 8 Why μP for Muon is not Adam μP copied twice

The update magnitude produced by an optimizer matters to its scaling rule. Adam's normalized scalar entries and Muon's matrix-normalized directions have different width dependence. The [matrix-optimizer transfer paper](https://arxiv.org/html/2512.05620v1) derives optimizer-specific rules and documents finite-width deviations; its results do not justify applying Adam's inverse-width rule indiscriminately to Muon.

Our Muon uses PyTorch's RMS-matched update factor `0.2 × sqrt(max(rows, columns))`. For hidden matrices whose aspect ratio stays fixed, multiplying model width by m multiplies this factor by `sqrt(m)`.

We therefore use the following nominal hidden-matrix LRs:

| Width | m | Adam SP | Adam μP | Muon SP | Muon μP |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | η | η | η | η |
| 128 | 2 | η | η/2 | η | η/√2 |
| 256 | 4 | η | η/4 | η | η/2 |

For Muon μP, the inverse square root cancels the RMS-matching factor's square-root growth. The resulting hidden-matrix update scale is constant with width for our fixed-aspect-ratio matrices, with layer-specific constants inherited from the base model. Output and embedding handling still matters; the hybrid retains the coordinated Adam/readout rules described above.

A subtle implementation point: PyTorch Muon applies conventional weight decay using its **nominal group LR**, before the shape-based update adjustment. Independent decay makes the shrinkage explicit and avoids having its width behavior silently follow that nominal LR. Our wrapper sets inner decay to zero and applies the independent shrinkage once.

The cited paper also discusses inverse-width independent decay under compute-optimal scaling. Our test keeps training tokens fixed as width changes, so it would be a different experiment to import that additional rule. We first isolate the parameterization and decay semantics under our existing protocol.

## 9 Reconstructing the completed experiments

An original MLE iteration has 18 trials: 2 arms × 3 seeds × 3 widths. An arm is a complete recipe, not a single training step. Each trial starts from scratch. Training weights are not carried from width 64 into width 256.

Within an iteration, the candidate is selected using only width-64 validation. The support rule requires a mean paired candidate-minus-baseline CE below −0.005 and a negative difference for every seed. This is a practical consistency rule, not a statistical significance test.

The following are completed 3,000-step measurements. All candidates use the historical model parameterization and ordinary AdamW. The fixed baseline is LR 0.0006, WD 0.1. Negative deltas favor the candidate.

| Iteration | Candidate LR | Candidate WD | Δ CE width 64 | Δ CE width 128 | Δ CE width 256 | What it shows |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.002 | 0.1 | −0.103923 | −0.053096 | −0.005923 | Mean improves at all widths; one width-256 seed worsens |
| 2 | 0.002 | 0.03 | −0.115496 | −0.062459 | −0.011935 | Strong small-width result; one width-256 seed still worsens |
| 3 | 0.002 | 0 | −0.112533 | −0.057727 | −0.013534 | Passes the rule at all three widths |
| 4 | 0.0006 | 0 | −0.016891 | −0.010498 | −0.001944 | Removing decay alone gives smaller gains; fails width-256 rule |
| 5 | 0.003 | 0 | −0.123375 | −0.060501 | +0.015435 | Best observed width-64 mean here; width 256 worsens on every seed |

The best observed recipe satisfying the rule at all three widths remains LR 0.002 with zero decay. Its candidate mean CEs are approximately 1.693263, 1.614932 and 1.579424 at widths 64, 128 and 256. This is an adaptive development ranking among the recipes tried, not a universal optimum.

You should be able to articulate three distinctions from this table:

**A favorable mean is not uniform seed consistency.** Iterations 1 and 2 have negative width-256 means but fail the every-seed condition.

**The best small-width recipe is not the most reliable transfer recipe.** Iteration 5 wins the current small-width ranking but reverses at width 256.

**A recipe comparison does not isolate every component.** Relative to the fixed baseline, iteration 3 changes both LR and decay. Iteration 4 helps study the decay component, but historical comparisons across iterations are not extra randomized controls and do not prove synergy.

The earlier 300-step work also found width-dependent reversals. Moving to 3,000 steps was therefore a test of whether early rankings persisted, not a certificate of convergence. Keep the two horizons separate when reading the trajectory.

The controller preserves reports, failed attempts and evaluator recoveries. A previous interpretation failure was recovered from saved evidence rather than rerunning completed training. That is an operational repair, not a new independent experimental replication.

## 10 What the completed optimizer round teaches

The completed round tested six experimental families plus a fresh reference:

| Family | Optimizer | Parameterization | Decay |
| --- | --- | --- | --- |
| A | AdamW | SP | Conventional 0.1 |
| B | AdamW | SP | Independent 0.00006 |
| C | AdamW | μP | Conventional 0.1 |
| D | AdamW | μP | Independent 0.00006 |
| E | Muon–AdamW | SP | Independent 0.00006 |
| F | Muon–AdamW | μP | Independent 0.00006 |
| Reference | AdamW | SP | Zero, fixed LR 0.002 |

A–F each screened base LRs 0.001, 0.002 and 0.003 at width 64 with the same three seeds and 3,000-step horizon. That is 54 local trials; the reference adds 3. After those 57 trials, all six LR choices were frozen before any transfer result was observed. Each selected family and the reference then received six larger-width runs, adding 42 trials. Total completed: **99 serial training trials**, with all checkpoint hashes, frozen selections, source/corpus hashes and report aggregates verified.

This is deliberately different from the agent loop's 18-trial iteration. The controlled round compares predefined families and tunes them with the same small-width LR budget. It is not an evolutionary search and it does not launch research agents to change recipes mid-round.

The design can test how these selected recipes perform and whether the μP/decay combination helps within the measured setup. It does not map the full large-width LR optimum because it does not sweep LR at larger widths. It also does not optimize the decay coefficient independently for each optimizer. A selected boundary LR is a reason to extend a future frozen grid, not evidence that the optimizer's best LR has been found.


The final measured validation cross-entropy is below. Each cell averages three seeds at 3,000 updates; lower is better. These are **final-budget endpoints, not converged losses**.

| Recipe | Selected base LR | Width 64 | Width 128 | Width 256 |
| --- | ---: | ---: | ---: | ---: |
| Adam SP, conventional decay | .003 | 1.697775 | 1.622245 | 1.591768 |
| Adam μP, conventional decay | .003 | 1.697776 | 1.606847 | 1.569218 |
| Adam SP, independent decay | .003 | 1.685782 | 1.610236 | 1.599959 |
| Adam μP, independent decay | .003 | 1.685784 | 1.605934 | 1.572956 |
| Muon SP, independent decay | .002 | 1.646208 | 1.562756 | 1.533617 |
| Muon μP, independent decay | .002 | **1.646054** | **1.559346** | **1.523978** |
| Fresh Adam SP, zero decay reference | .002 fixed | 1.693263 | 1.620936 | 1.584907 |

**First learning: the Muon recipe is a useful measured starting point.** Its μP version improves mean CE relative to the fresh reference by 0.047208, 0.061590 and 0.060928 at widths 64, 128 and 256. Every paired seed improves at every width. This passes the declared practical support rule; it is not a statistical significance test. The family LR was chosen from width 64 before transfer. The tiny base-width difference between SP and μP is not evidence of a meaningful μP advantage: they are theoretically equivalent there, with small numerical differences in execution.

**Second learning: separate “Muon works here” from “μP adds a robust gain to Muon.”** At width 256, Muon μP minus Muon SP is −0.009638 mean CE, but the three paired differences are +0.000074, −0.011491 and −0.017498. At width 128 the mean is −0.003410, with two positive pairs. Thus the average favors μP, while a consistent incremental gain is not established. For Adam at width 256, μP improves every pair relative to SP under both tested decay modes: means −0.022550 with conventional decay and −0.027003 with independent decay. These are package comparisons involving initialization, output scaling and parameter-group learning rates. Conventional decay also changes with group LR, so that contrast does not isolate a pure parameterization mechanism.

**Third learning: independent decay is a control mechanism, not an automatic quality improvement.** It improves Adam's small-width mean under this coefficient choice. At width 256, however, conventional Adam μP scores 1.569218 versus independent Adam μP's 1.572956. Fixing shrinkage across LR changes makes an intervention easier to interpret; it does not prove that the chosen shrinkage is optimal.

**Resource tradeoff:** Muon μP takes mean training times of 33.91 / 51.91 / 66.11 seconds across the three widths; the fresh Adam reference takes 31.92 / 48.44 / 66.17 seconds. Those observed changes are approximately +6.3%, +7.2%, and −0.1%. Mean process peak RSS is about 473 / 474 / 477 MiB versus 471 / 471 / 473 MiB. These are this Mac's measurements, not isolated causal throughput estimates, total unified-memory use, or compute-matched comparisons.

**Why we ran one more bounded round:** Every Adam family selected .003, the upper edge of the original grid. Before attributing the remaining gap to optimizer choice, a separately frozen follow-up tested Adam μP with conventional and independent decay at .002, .004 and .006, plus the fixed .002 zero-decay reference. This completed 21 width-64 trials followed by 18 frozen-choice transfer trials: **39 total**, again with seeds 11/22/33 and 3,000 updates. It directly probes above the original boundary while retaining a lower anchor. The original .003 measurements and selection remain unchanged. This is adaptive development, not fresh-seed confirmation; it is one finite follow-up, not an endless tuning loop. The optimizer-grid study is now closed; MLE carries both successes and failures forward.

**Follow-up measurements:** independent-decay Adam μP selected LR .004 from its new small-width grid. Its mean CE is **1.678748 / 1.598929 / 1.564250** at widths 64/128/256. Every paired seed improves against the follow-up's fresh zero-decay reference; mean deltas are −.014515 / −.018191 / −.019504. The higher LR improves the Adam control, but its means remain above the earlier Muon μP values by .032694 / .039583 / .040272. This is a cross-round adaptive comparison, not a newly randomized head-to-head.

Conventional-decay Adam μP selected .002 in the follow-up because .004 and .006 worsened width-64 loss. This does **not** replace the first round's frozen .003 selection: the earlier .003 result remains better in the combined historical observations. We did not test .003 again in the follow-up. The repeated fresh-reference means also differed slightly between rounds (about .003816 at width 128 and .001153 at width 256), so reuse of a seed should not be confused with exact reproducibility on this runtime.

**Decision for continued research:** use Muon/AdamW hybrid, μP, LR .002 and independent decay .00006 as the fixed baseline for a new campaign. Begin by changing only decay to zero, then use the existing bounded mutation/crossover rules. This asks whether the remaining decay helps this optimizer recipe. It does not assume the ablation will win. The baseline is explicitly frozen at campaign initialization; old campaigns and selections stay intact. Read the [follow-up report](../research_runs/optimizer-adam-boundary-002/report.json) and [review](../research_runs/optimizer-adam-boundary-002/review.json).

Read the [full measured matrix](../research_runs/optimizer-transfer-001/report.json), [paired-contrast review](../research_runs/optimizer-transfer-001/review.json), and [best-observed recipe](../research_state/best-observed-config.json). The strongest recipe so far remains a development finding; there is no universal optimum claim. No held-out test data was consumed.

The [synthetic MPS diagnostic record](../research_state/optimizer-checks-20261023.json) preserves the early-update checks.

Read the [exact protocol](OPTIMIZER_TRANSFER.md) and [study driver](../src/research/optimizer_study.py). Correctness validation currently includes 438 passing repository tests and finite synthetic MPS checks for both optimizers and parameterizations at all three widths. Synthetic checks are separate from the quality measurements in this round.

**First MLE ablation after the optimizer study:** removing independent decay from the Muon μP recipe did not improve quality. All 18 paired training trials completed; the only changed setting was decay .00006 → 0.

| Width | Decay .00006 mean CE | Zero-decay mean CE | Zero minus baseline |
| --- | ---: | ---: | ---: |
| 64 | 1.646262 | 1.649774 | +.003513 |
| 128 | 1.559252 | 1.562944 | +.003692 |
| 256 | 1.523766 | 1.530401 | +.006635 |

Positive differences mean worse loss. All three seeds worsened at widths 64 and 256; two worsened at width 128. The frozen width-64 choice retained the baseline before the larger results arrived. Because the candidate never won locally, the controller labels overall transfer **not applicable**; larger-width comparisons are diagnostic. Width-256 throughput improved about .96% while quality worsened, and process peak RSS stayed approximately unchanged. This is a conditional result for this recipe and budget, not proof that decay always helps or that .00006 is optimal. Four evaluator learnings entered the shared bank, and the next iteration continued automatically. [Evidence](../research_runs/continuous-20261023-005/iteration-001/evidence.json) · [Interpretation and limitations](../research_runs/continuous-20261023-005/iteration-001/learnings.json).

**Second Muon iteration: doubling independent decay** from .00006 to .00012 also failed the frozen width-64 improvement rule. All 18 trials completed. Candidate-minus-baseline mean CE was **+.005433 / +.005246 / −.000802** at widths 64/128/256. Every seed worsened at64 and128; width256 had two improvements and one regression, with mean improvement smaller than the .005 threshold. The baseline was already selected from width64, so larger-width results cannot promote this candidate. Throughput and process RSS changed by less than .31% and .08% respectively in the paired means; these small observations do not establish a resource advantage. Compared with the first zero-decay ablation, we now have evidence against both removing and doubling the current decay in this fixed-budget local search. This retains .00006 as the working control, not as a proven optimum. [Iteration2 evidence](../research_runs/continuous-20261023-005/iteration-002/evidence.json).

**Third Muon iteration: 5% attention-probability dropout.** We changed only `model.attn_dropout` from 0 to .05 and completed all 18 trials at 3,000 updates. This drops attention probabilities during training; it is not a change to every dropout setting in the model.

| Width | Baseline mean CE | Dropout mean CE | Candidate minus baseline | Paired-seed result |
|---|---:|---:|---:|---|
| 64 | 1.645867 | 1.657164 | +.011296 | All three worsened |
| 128 | 1.559394 | 1.558715 | −.000678 | Two improved, one worsened |
| 256 | 1.524287 | 1.521629 | −.002658 | All three improved, but below the .005 improvement threshold |

The frozen width-64 rule retained the baseline before transfer. Larger-width measurements cannot retroactively promote the candidate. The width-256 result is a small measured benefit at this setting, but does not establish a general width interaction or a failure of μP. Throughput fell on every paired seed, with mean changes of −2.80%, −3.79%, and −2.82% at widths 64, 128, and 256. Process RSS changes were small and inconsistent across seeds, so we do not infer a memory advantage. The working recipe remains Muon/AdamW + μP, LR .002, independent decay .00006, and attention dropout 0. These are adaptive development results on reused validation data, not held-out confirmation. [Iteration 3 evidence](../research_runs/continuous-20261023-005/iteration-003/evidence.json) and [evaluator learnings](../research_runs/continuous-20261023-005/iteration-003/learnings.json).

**Fourth Muon suite: tighter gradient clipping.** We lowered only `training.grad_clip` from 1.0 to 0.5. All 18 trials completed at 3,000 updates; checkpoint hashes were verified and the test holdout stayed closed. Candidate-minus-baseline mean CE was **+.007581 / +.002553 / +.005212** at widths 64/128/256. Every paired seed worsened at widths 64 and 256; at width 128, two worsened and one improved. The width-64 rule therefore retained the baseline. Mean throughput changes were +.11%, −.65%, and −.04%, with process RSS changes below .34% in magnitude; these observations establish no useful resource advantage.

The prediction was that tighter clipping might reduce harmful minibatch contributions to optimizer state. These quality results contradict the predicted improvement for this setting and budget, but do not diagnose its mechanism: we did not measure gradient-norm distributions or clipping frequency. A lower clipping threshold is also not equivalent to halving the learning rate in this hybrid optimizer. The working control retains clipping at 1.0; this single comparison does not identify an optimal threshold. [Iteration 4 evidence](../research_runs/continuous-20261023-005/iteration-004/evidence.json). The completed evaluator recorded three learnings covering failed quality improvement, lack of supported transfer, and approximately neutral resources. [Evaluator learnings](../research_runs/continuous-20261023-005/iteration-004/learnings.json).

**Fifth Muon iteration: block LayerNorm versus RMSNorm.** Changing only `model.norm_type` to `layernorm` completed all 18 trials at 3,000 updates; all checkpoint hashes were verified. Candidate-minus-baseline CE was **+.004065 / +.005220 / −.000315** at widths 64/128/256. All three seeds worsened at width 64; two worsened at each larger width. The tiny negative width-256 mean did not pass the improvement criterion. The frozen local selection therefore retains RMSNorm.

This experiment did reveal a resource tradeoff: LayerNorm increased recorded training throughput on **every paired seed**, with mean gains of **10.38%, 14.38%, and 9.29%** at widths 64, 128, and 256. That contradicts the tentative slowdown expectation, but does not establish better quality for equal wall time: both arms received the same token budget, and throughput excludes other trial overhead. The result is specific to this PyTorch/MPS execution; different kernels or dispatch costs are possible explanations, not measured causes. Process RSS changes were below .39% in magnitude in the paired means and establish no memory advantage. The normalization setting changes centering, learned bias, epsilon, and implementation together, so the quality difference cannot be attributed to centering alone. This is adaptive development on reused validation data, with the test holdout closed. [Iteration 5 evidence](../research_runs/continuous-20261023-005/iteration-005/evidence.json) and [four evaluator learnings](../research_runs/continuous-20261023-005/iteration-005/learnings.json).

**Sixth Muon iteration: relaxing gradient clipping to 2.0.** All 18 trials completed at 3,000 updates, with verified checkpoint hashes and the test holdout closed. Candidate-minus-baseline mean CE was **+.000614 / −.000162 / +.000381** at widths 64/128/256. All width-64 seeds worsened slightly. All width-128 seeds improved slightly, but far less than the registered .005 threshold; at width 256, two seeds worsened and one improved. The frozen rule retains clipping at 1.0. Mean throughput and process RSS changes were below .22% in magnitude and do not establish an efficiency advantage.

Together with iteration 4, both tested alternatives, 0.5 and 2.0, failed to replace the working clipping threshold of 1.0. This supports retaining the control in this search, not calling it optimal. The small effects at 2.0 do not prove equivalence or that clipping rarely activates: gradient norms and clipping frequencies were not measured. These remain adaptive comparisons using three reused development seeds. [Iteration 6 evidence](../research_runs/continuous-20261023-005/iteration-006/evidence.json) and [three evaluator learnings](../research_runs/continuous-20261023-005/iteration-006/learnings.json).

**Seventh Muon suite: a modest learning-rate increase.** We changed only the base learning rate from .002 to .0025, retaining μP and the hybrid optimizer's existing group-scaling rules. All 18 trials completed at 3,000 updates and their checkpoint hashes were verified. Candidate-minus-baseline mean CE was **+.000195 / +.001365 / +.001277** at widths 64/128/256. At each width, two paired seeds worsened and one improved. The frozen width-64 rule retains LR .002; neither larger width passed the improvement criterion.

Mean throughput changes were +.04%, +.20%, and −.23%, while mean process RSS changes were below .11% in magnitude. These small changes establish no resource advantage. This comparison tests one nearby learning rate under the existing fixed-budget protocol; it neither proves .002 optimal nor establishes equivalence between the rates. Reused seeds and validation data make this adaptive development, with the test holdout closed. [Iteration 7 evidence](../research_runs/continuous-20261023-005/iteration-007/evidence.json).

**Eighth Muon iteration: removing attention-projection biases.** We set `model.attn_bias=false`, removing both QKV and attention-output biases while retaining MLP biases and the final LayerNorm. All 18 trials completed at 3,000 updates, with verified checkpoint hashes and the test holdout closed. Candidate-minus-baseline mean CE was **+.042014 / +.037814 / +.031412** at widths 64/128/256. Every one of the nine paired seed/width comparisons worsened, so the frozen local rule retains the baseline with attention biases.

Mean training-throughput gains were **3.78%, .20%, and .65%** at widths 64, 128, and 256; the width-128 timing gain was inconsistent across seeds. Mean process RSS reductions were below .42%. These resource observations cannot replace the failed quality criterion, and equal-time quality remains untested. Removing biases also changes the sequence of random initialization draws, so this experiment does not isolate the causal contribution of learned offsets. The smaller mean penalty at greater width is descriptive, not proof that increasing width compensates for bias removal. This result applies to the tested Shakespeare workload and budget, not attention biases universally. [Iteration 8 evidence](../research_runs/continuous-20261023-005/iteration-008/evidence.json) and [four evaluator learnings](../research_runs/continuous-20261023-005/iteration-008/learnings.json).

**Ninth Muon iteration: lowering the base learning rate to .00175.** All 18 trials completed at 3,000 updates, with verified checkpoint hashes and the test holdout closed. Candidate-minus-baseline mean CE was **+.004263 / +.002450 / +.003051** at widths 64/128/256. Two seeds worsened at each smaller width; all three worsened at width 256. The frozen local selection retains LR .002. Both the Muon and auxiliary AdamW rates changed through their existing scaling rules, so this is a shared-rate intervention.

Together with iteration 7, neither nearby rate, .00175 nor .0025, displaced .002 in this fixed-budget search. That supports retaining the working control, not declaring an optimum. Slower optimization is one possible explanation for the lower-rate outcome, but final validation loss alone cannot distinguish it from generalization or trajectory effects. Mean throughput changes were below .24% and process RSS changes below .35% in magnitude, establishing no resource advantage. These remain adaptive comparisons on reused seeds and validation data. [Iteration 9 evidence](../research_runs/continuous-20261023-005/iteration-009/evidence.json) and [three evaluator learnings](../research_runs/continuous-20261023-005/iteration-009/learnings.json).

**Tenth Muon iteration: 5% residual-output dropout.** We set `model.dropout=.05` and explicitly kept `model.attn_dropout=0`. This applies dropout jointly to the attention-output and MLP-output residual branches while leaving attention probabilities undropped; it differs from iteration 3. All 18 trials completed at 3,000 updates with verified checkpoint hashes and the test holdout closed. Candidate-minus-baseline mean CE was **+.014125 / +.011068 / +.005575** at widths 64/128/256. Every paired seed worsened at widths 64 and 128; two worsened at width 256. The frozen local rule retains zero dropout.

Every paired run was slower, with mean throughput changes of **−1.26%, −1.04%, and −1.32%** at widths 64, 128, and 256. Mean process RSS increases were below .24%, with mixed pair signs. The measured tradeoff was worse mean quality plus slower training at this budget. Added optimization noise and mask-processing overhead are plausible explanations, not measured mechanisms. Because both residual branches changed together, this does not identify either branch's independent contribution or their interaction. The smaller mean quality penalty at width 256 does not establish a scaling trend. These are adaptive development results on reused validation data. [Iteration 10 evidence](../research_runs/continuous-20261023-005/iteration-010/evidence.json) and [five evaluator learnings](../research_runs/continuous-20261023-005/iteration-010/learnings.json).

**Eleventh Muon iteration: learned absolute positions instead of RoPE.** Setting `model.rope=0` activates the existing learned absolute-position embedding path. It does not remove all positional information. All 18 trials completed at 3,000 updates, with verified checkpoint hashes and the test holdout closed. Candidate-minus-baseline mean CE was **+.097136 / +.069502 / +.048862** at widths 64/128/256. All nine paired seed/width comparisons worsened, so the frozen rule retains RoPE.

Recorded training throughput increased on every pair, with mean gains of **12.49%, 13.76%, and 6.96%** at widths 64, 128, and 256. The measured tradeoff is faster training with worse fixed-token quality; equal-time quality remains untested. Mean process RSS increased by .28%, .61%, and .23%, providing no memory-saving evidence. This setting simultaneously removes query/key rotations, adds positional input vectors, and activates optimization of the positional table, so the experiment cannot isolate which component caused the loss increase or speed gain. The smaller mean loss penalty at greater width does not justify extrapolation to larger models. These are adaptive development results within this fixed-depth family, not held-out confirmation or a universal positional-encoding ranking. [Iteration 11 evidence](../research_runs/continuous-20261023-005/iteration-011/evidence.json) and [four evaluator learnings](../research_runs/continuous-20261023-005/iteration-011/learnings.json).

**Twelfth Muon iteration: halving independent weight decay.** We changed only `training.weight_decay` from .00006 to .00003. All 18 trials completed at 3,000 updates, with verified checkpoint hashes and the test holdout closed. Candidate-minus-baseline mean CE was **+.001515 / +.003311 / −.000120** at widths 64/128/256. Two seeds worsened at width 64, all three worsened at width 128, and width 256 had two improvements and one regression. The frozen width-64 rule retains the baseline; the tiny negative mean at width 256 cannot override that selection.

Mean throughput changes were −.11%, −.32%, and +.06%; mean process RSS changes were below .10% in magnitude. These measurements establish no resource advantage or equivalence. Together with the earlier zero-decay and doubled-decay trials, this supports retaining .00006 in the current search, without establishing an optimal decay value. The result is conditional on the fixed Muon/AdamW hybrid, μP, corpus and training budget; three reused seeds and adaptive validation comparisons do not establish a general scaling trend or a mechanism. [Iteration 12 evidence](../research_runs/continuous-20261023-005/iteration-012/evidence.json) and [four evaluator learnings](../research_runs/continuous-20261023-005/iteration-012/learnings.json).

**Thirteenth Muon iteration: disabling global gradient clipping.** We changed only `training.grad_clip` from 1.0 to 0.0. All 18 trials completed at 3,000 updates with verified checkpoint hashes and the test holdout closed. Candidate-minus-baseline mean CE was **+.000634 / −.000067 / +.000924** at widths 64/128/256. Two seeds worsened at widths 64 and 256; width 128 had two improvements and one regression. None passed the quality criterion, and the frozen local rule retains clipping at 1.0. Small observed differences do not establish quality equivalence or statistically established harm.

Every paired run was faster without clipping: mean training-throughput gains were **48.04%, 37.14%, and 17.10%** at widths 64, 128, and 256. Mean process RSS changes were −.32%, +.31%, and +.05%. The implementation skips `clip_grad_norm_` when the threshold is zero, removing norm computation, rescaling, and its nonfinite-gradient-norm check together; the separate loss-finiteness check remains. Component timings were not measured, so the gain cannot be assigned to any one operation. All nine unclipped runs completed, but that does not establish stability for other configurations. The faster candidate received the same number of updates: **equal-time quality remains untested**. This is a useful measured speed/quality tradeoff on this MPS host, while the best-quality recipe remains unchanged. These are adaptive results on reused development seeds. [Iteration 13 evidence](../research_runs/continuous-20261023-005/iteration-013/evidence.json) and [four evaluator learnings](../research_runs/continuous-20261023-005/iteration-013/learnings.json).

**Fourteenth Muon iteration: 1% attention-probability dropout.** We changed only `model.attn_dropout` from 0 to .01, retaining zero residual-output dropout. All 18 trials completed at 3,000 updates, with verified checkpoint hashes and the test holdout closed. Candidate-minus-baseline mean CE was **+.000029 / −.005591 / −.000481** at widths 64/128/256. Width 64 had one regression and two improvements, failing the frozen local rule. Width 128 improved on all three seeds and met its target-specific criterion. Width 256 was mixed: paired deltas were −.008000, −.001614, and +.008170. The baseline remains selected; the width-128 diagnostic cannot retroactively select the candidate or establish transfer of a small-model improvement.

Every paired run slowed, with mean throughput changes of **−3.14%, −3.78%, and −2.94%** at widths 64, 128, and 256. Mean process RSS changes were below .12% in magnitude. The speed penalties fell within the frozen subjective forecasts; quality forecasts were unquantified. This illustrates why a larger-width benefit and near-zero small-width mean are not enough for the declared selection rule, nor proof of quality equivalence. Mask-processing overhead and width-dependent regularization are plausible explanations, not measured mechanisms. The width-128 quality benefit is at equal tokens; equal-time quality remains untested. These are adaptive comparisons on three reused development seeds, with no established monotonic width trend. [Iteration 14 evidence](../research_runs/continuous-20261023-005/iteration-014/evidence.json) and [four evaluator learnings](../research_runs/continuous-20261023-005/iteration-014/learnings.json).

**Fifteenth Muon iteration: increasing independent decay by 50%.** We changed only `training.weight_decay` from .00006 to .00009. All 18 trials completed at 3,000 updates, with verified checkpoint hashes and the test holdout closed. Candidate-minus-baseline mean CE was **+.003409 / +.002175 / +.000004** at widths 64/128/256. Every seed worsened at widths 64 and 128. Width 256 had two improvements and one regression, with a tiny positive mean; this is not evidence of equivalence. No width met the quality criterion, and the frozen local selection retains .00006.

Mean throughput changes were +.24%, +.52%, and +.002%; mean process RSS increases were below .10%. These observations matched the frozen approximate-neutrality forecasts but do not establish resource equivalence or a causal speedup. Together with the other tested decay alternatives, this supports retaining the working coefficient within the current search, not declaring it optimal. Greater contraction or altered update-to-weight dynamics are possible explanations for the quality result, but neither mechanism was measured. Historical trials informed this adaptive proposal; they are not additional randomized controls or evidence of component synergy. [Iteration 15 evidence](../research_runs/continuous-20261023-005/iteration-015/evidence.json) and [four evaluator learnings](../research_runs/continuous-20261023-005/iteration-015/learnings.json).

**Sixteenth Muon suite: base learning rate .00225.** We increased the shared base learning rate from .002 to .00225 while retaining the existing Muon and auxiliary AdamW group-scaling rules. All 18 trials completed at 3,000 updates with verified checkpoint hashes and the test holdout closed. Candidate-minus-baseline mean CE was **−.002077 / +.003597 / +.004380** at widths 64/128/256. Although the width-64 mean improved, its seed-ordered deltas were +.008422, −.008290, and −.006363. The mean missed the required −.005 threshold and one seed regressed, so the frozen selection retains .002. Every paired seed worsened at both larger widths.

This is a concrete example of why the selection rule checks both mean improvement and seed consistency. It does not demonstrate transfer of a selected small-model improvement or refute μP generally. Mean throughput changes were +.55%, +.53%, and −.05%; mean process RSS changes were below .12% in magnitude, establishing no material resource advantage. These are adaptive development measurements on reused seeds, not evidence that .002 is globally optimal. [Iteration 16 evidence](../research_runs/continuous-20261023-005/iteration-016/evidence.json).

**Seventeenth Muon iteration: 1% residual-output dropout.** We set `model.dropout=.01` and explicitly retained `model.attn_dropout=0`, applying dropout jointly to the attention-output and MLP-output residual branches. All 18 trials completed at 3,000 updates, with verified checkpoint hashes and the test holdout closed. Candidate-minus-baseline mean CE was **+.007772 / +.003013 / +.004705** at widths 64/128/256. Every seed worsened at width 64; two worsened at each larger width. No width met the quality criterion, so the frozen selection retains zero dropout.

Every paired run slowed, with mean throughput changes of **−1.45%, −1.76%, and −1.36%** at widths 64, 128, and 256. Mean process RSS changes were below .09% in magnitude, with mixed pair signs. Resource measurements fell within the frozen subjective forecast ranges, while numerical quality forecasts were unquantified. Together with the 5% residual-dropout trial, this supports retaining the working zero-dropout control at this budget; it does not establish universal harm or isolate either residual branch. Impaired fitting and mask-processing overhead remain plausible, unmeasured mechanisms. Equal-time quality and independent confirmation on fresh seeds remain untested. [Iteration 17 evidence](../research_runs/continuous-20261023-005/iteration-017/evidence.json) and [four evaluator learnings](../research_runs/continuous-20261023-005/iteration-017/learnings.json).

**Eighteenth Muon suite: clipping threshold 1.5.** We changed only `training.grad_clip` from 1.0 to 1.5. All 18 trials completed at 3,000 updates with verified checkpoint hashes and the test holdout closed. Candidate-minus-baseline mean CE was **−.000076 / −.000069 / +.000017** at widths 64/128/256. At each width, two seeds improved and one regressed. None met the improvement threshold or the all-negative pair requirement, so the frozen local selection retains 1.0. These near-zero averages do not establish quality equivalence or how often clipping activated.

Mean throughput changes were +.26%, −.45%, and −.16%; mean process RSS changes were below .16% in magnitude. Raising a positive clipping threshold retains the norm computation and clipping call, unlike disabling clipping in iteration 13. The large speed gain observed when clipping was disabled did not recur here, but no component profiling identifies the source of that earlier gain. This comparison supports retaining the current control, without identifying an optimal threshold, a causal training mechanism, or a general width trend. These remain adaptive development results using three reused seeds. [Iteration 18 evidence](../research_runs/continuous-20261023-005/iteration-018/evidence.json).

**Nineteenth Muon iteration: base learning rate .002125.** We increased the shared base rate from .002 to .002125, retaining the existing optimizer-group scaling rules. All 18 trials completed at 3,000 updates, with verified checkpoint hashes and the test holdout closed. Candidate-minus-baseline mean CE was **−.002294 / +.003377 / +.004196** at widths 64/128/256. The width-64 seed-ordered deltas were +.004094, −.006640, and −.004337. The mean improvement missed the −.005 threshold and one seed regressed, so the frozen selection retains .002. Every paired seed worsened at both larger widths.

Mean throughput changes were +.11%, +.52%, and +.15%; mean process RSS changes were below .02% in magnitude. All six resource observations fell within the frozen approximate-neutrality forecasts, establishing no meaningful efficiency advantage. The pattern resembles iteration 16, but the proposals and reused development seeds make this adaptive evidence, not independent replication or an identified width interaction. The experiment neither demonstrates transfer of an established small-model improvement nor identifies an optimal learning rate. [Iteration 19 evidence](../research_runs/continuous-20261023-005/iteration-019/evidence.json) and [three evaluator learnings](../research_runs/continuous-20261023-005/iteration-019/learnings.json).

**Twentieth Muon iteration: base learning rate .001875.** We reduced the shared base rate from .002 to .001875, changing both Muon and auxiliary AdamW updates through their existing scaling rules. All 18 trials completed at 3,000 updates, with verified checkpoint hashes and the test holdout closed. Candidate-minus-baseline mean CE was **+.003749 / +.003697 / +.002203** at widths 64/128/256. Two seeds worsened at each smaller width, and all three worsened at width 256. No width met the improvement criterion, so the frozen local selection retains .002.

Mean throughput changes were −.29%, −.37%, and −.07%; mean process RSS reductions were below .15%. All six resource observations fell within the frozen subjective forecast ranges, with no established meaningful resource advantage. Slower fitting at the fixed endpoint is a possible explanation for the quality penalty, but endpoint validation CE alone does not establish it. None of the 20 candidates so far has displaced the fixed baseline under the declared local rule; this describes an adaptive search on reused seeds, not proof that the baseline is globally optimal. [Iteration 20 evidence](../research_runs/continuous-20261023-005/iteration-020/evidence.json) and [three evaluator learnings](../research_runs/continuous-20261023-005/iteration-020/learnings.json).

**Twenty-first Muon iteration: reducing independent decay by 25%.** We changed only `training.weight_decay` from .00006 to .000045. All 18 trials completed at 3,000 updates, with verified checkpoint hashes and the test holdout closed. Candidate-minus-baseline mean CE was **+.000770 / +.002229 / +.000438** at widths 64/128/256. Every seed worsened at widths 64 and 128; two worsened at width 256. No target met the quality criterion, and the frozen local selection retains .00006.

Mean throughput changes were +.33%, +.07%, and −.11%; mean process RSS reductions were below .06%. All six resource observations fell within the frozen subjective forecast ranges, establishing no meaningful efficiency advantage. The single favorable width-256 seed cannot rescue selection. Historical decay trials informed this adaptive proposal, but are not additional randomized controls; this contrast does not identify a dose-response curve, an optimal coefficient, or a mechanism. [Iteration 21 evidence](../research_runs/continuous-20261023-005/iteration-021/evidence.json) and [four evaluator learnings](../research_runs/continuous-20261023-005/iteration-021/learnings.json).

The twenty-second candidate reduced global gradient clipping from **1.0 to 0.75**, with all other settings fixed. All 18 trials completed 3,000 updates and their checkpoint hashes were verified; held-out test data stayed closed. Mean candidate-minus-baseline CE was **+.003023 / +.002101 / +.003350** at widths 64/128/256. Width-64 seed deltas were +.012195, −.003694, and +.000567: the candidate failed both the mean-improvement threshold and the requirement that every pair improve. All six larger-width pairs worsened. The frozen local winner remains the baseline.

Mean paired throughput changes were +.12%, −.28%, and −.01%; mean process RSS increases were +.02%, +.12%, and +.20%. These small differences fell within the frozen subjective forecast ranges and establish no efficiency advantage or resource equivalence. Unlike disabling clipping, a positive threshold retains the norm calculation and clipping operation. The observed quality changes do not reveal how often clipping occurred or which gradient contributions changed. Reused development seeds and adaptive proposal selection limit broader inference. [Iteration 22 evidence](../research_runs/continuous-20261023-005/iteration-022/evidence.json) and [three evaluator learnings](../research_runs/continuous-20261023-005/iteration-022/learnings.json).

Operational interruption after iteration 22: while this report was being updated, candidate creation for iteration 23 encountered the harness’s clean-committed-checkout guard. The guard stopped work before candidate training; all 22 completed suites and the failed phase, prediction, counters, and failure learning remain preserved in campaign005. There is no general failed-phase retry command, so campaign006 continues from the same fixed Muon baseline and complete shared bank. Its first proposed contrast is the unmeasured decay coefficient .000075, not a rerun of a completed suite. Future report edits must be prepared and committed in a separate checkout, with the campaign stopped at a clean boundary before updating its checkout. This is an operational recovery, not new model-quality evidence.

Campaign006’s first completed candidate increased independent per-step decay from **.00006 to .000075**. This is the previously untrained proposal from campaign005’s interruption. All 18 trials completed 3,000 updates, checkpoint hashes were verified, and held-out test data stayed closed. Mean candidate-minus-baseline CE was **+.001734 / +.000130 / +.000404** at widths 64/128/256. All three width-64 pairs worsened; each larger width had one favorable and two unfavorable pairs. The baseline remains the frozen local winner. Small larger-width means and mixed signs establish neither equivalence nor a reliable population-level effect.

Mean paired throughput changes were −.07%, −.27%, and −.04%; mean process RSS changes were −.003%, −.12%, and +.04%. All fell within the subjective forecast ranges, with no demonstrated compensating resource advantage. Changing the coefficient isolates this conditional comparison, but does not measure the mechanism or locate an optimum. Previous decay experiments informed the proposal; they are not additional randomized controls. This remains adaptive development on reused seeds. [Campaign006 iteration 1 evidence](../research_runs/continuous-20261024-006/iteration-001/evidence.json) and [four evaluator learnings](../research_runs/continuous-20261024-006/iteration-001/learnings.json).

Campaign006’s second candidate relaxed global gradient clipping from **1.0 to 1.25**. All 18 trials completed 3,000 updates and passed checkpoint-hash verification; held-out test data stayed closed. Mean candidate-minus-baseline CE was **+.001024 / +.000216 / +.000293** at widths 64/128/256. Every width had one favorable and two unfavorable seed pairs. Width-64 deltas were −.000125, +.002486, and +.000709, failing both the mean-improvement threshold and the all-pairs requirement. The baseline remains the frozen local winner; larger-width diagnostics cannot revise that choice.

Mean paired throughput changes were −.41%, −.44%, and −.02%; mean process RSS changes were −.09%, +.07%, and +.03%. These observations fell within the subjective forecast ranges and establish no compensating efficiency advantage or resource equivalence. A positive clipping threshold retains the norm calculation; changing it does not test removal of that work. Neither clipping frequency nor optimizer-history changes were measured, so a mechanism cannot be inferred from the endpoint losses. These reused-seed adaptive comparisons are not independent confirmation. [Campaign006 iteration 2 evidence](../research_runs/continuous-20261024-006/iteration-002/evidence.json) and [three evaluator learnings](../research_runs/continuous-20261024-006/iteration-002/learnings.json).

Campaign006’s third candidate jointly changed **base learning rate .002 → .002125 and global clipping 1.0 → 1.5**. All 18 trials completed 3,000 updates and passed checkpoint-hash verification; held-out test data stayed closed. Mean candidate-minus-baseline CE was **−.002883 / +.003099 / +.003505** at widths 64/128/256. The lower width-64 mean did not meet the required improvement of more than .005, and seed 11 worsened: paired deltas were +.004338, −.007779, and −.005208. The baseline therefore remains selected. Every pair at widths 128 and 256 worsened; those diagnostics cannot rescue the failed local criterion.

Mean paired throughput changes were +.40%, −.12%, and −.09%; mean process RSS changes were +.07%, −.10%, and +.12%. These small measurements establish no compensating resource advantage or equivalence. This experiment tests a two-setting combination against the fixed baseline. Earlier single-setting trials informed the adaptive proposal, but are not contemporaneous component controls: differences across those rounds cannot identify an interaction or show synergy. A better small-width average is insufficient for promotion under the frozen rule, and it does not imply beneficial transfer. [Campaign006 iteration 3 evidence](../research_runs/continuous-20261024-006/iteration-003/evidence.json) and [three evaluator learnings](../research_runs/continuous-20261024-006/iteration-003/learnings.json).

Campaign006’s fourth candidate increased the shared base learning rate by **3.125%, from .002 to .0020625**. All 18 trials completed 3,000 updates and passed checkpoint-hash verification; held-out test data stayed closed. Mean candidate-minus-baseline CE was **+.000565 / +.002255 / +.001568** at widths 64/128/256. Two width-64 seeds improved, but seed 11 worsened enough to leave the mean positive: paired deltas were +.007513, −.004947, and −.000870. The candidate failed both frozen selection requirements. Every larger-width pair worsened, and the baseline remains selected.

Mean paired throughput changes were −.04%, +.17%, and −.01%; mean process RSS changes were +.11%, −.07%, and −.009%. All fell within the subjective forecast ranges, establishing no meaningful efficiency advantage or equivalence. This shared scalar changes both the Muon and auxiliary AdamW updates, so the experiment does not isolate either optimizer branch. Repeatedly narrowing the learning-rate interval on reused seeds remains adaptive development; these results neither locate an optimum nor broadly refute muP. [Campaign006 iteration 4 evidence](../research_runs/continuous-20261024-006/iteration-004/evidence.json) and [three evaluator learnings](../research_runs/continuous-20261024-006/iteration-004/learnings.json).

Campaign006’s fifth candidate increased the shared base learning rate by **9.375%, from .002 to .0021875**. All 18 trials completed 3,000 updates and passed checkpoint-hash verification; held-out test data stayed closed. Mean candidate-minus-baseline CE was **−.004619 / +.004290 / +.002739** at widths 64/128/256. The width-64 mean improvement fell short of the frozen .005 threshold, and seed 11 worsened: paired deltas were +.000450, −.007837, and −.006471. Both selection requirements therefore failed. At width 128, one seed improved and two worsened; all three width-256 pairs worsened. The baseline remains selected.

Mean paired throughput changes were +.21%, −.25%, and −.003%; mean process RSS changes were −.006%, +.03%, and +.06%. These small differences establish no compensating resource advantage or equivalence. This result illustrates why a near-threshold average does not justify relaxing a preregistered decision rule after seeing the data. Reused seeds and repeated adaptive learning-rate proposals also limit independent confirmation; these comparisons do not identify an optimal learning rate or a mechanism for the width differences. The shared rate changes both Muon and auxiliary AdamW updates. [Campaign006 iteration 5 evidence](../research_runs/continuous-20261024-006/iteration-005/evidence.json) and [three evaluator learnings](../research_runs/continuous-20261024-006/iteration-005/learnings.json).

Campaign006’s sixth candidate added **attention dropout .005 (0.5%)**, leaving residual dropout at zero. All 18 trials completed 3,000 updates and passed checkpoint-hash verification; held-out test data stayed closed. Mean candidate-minus-baseline CE was **−.000084 / +.003130 / −.002309** at widths 64/128/256. Width-64 deltas were +.007105, −.007948, and +.000590, failing both frozen selection requirements. Width 128 had two worse pairs and one better pair. All three width-256 pairs improved, but their mean gain was below .005. These diagnostics cannot revise the baseline selection or establish a reproducible width interaction.

Every paired run slowed: mean throughput changes were **−2.63%, −3.99%, and −2.83%**, within the frozen forecast ranges. A low dropout probability therefore did not remove the observed runtime penalty. Masking and elementwise work are plausible contributors, but no component profiling established the cause. Mean process RSS changes were +.08%, −.07%, and −.17%; these are not peak accelerator-memory measurements or an equivalence test. No target met its registered quality criterion, and equal-token comparisons do not establish equal-time quality. Reused-seed adaptive exploration limits confirmation. [Campaign006 iteration 6 evidence](../research_runs/continuous-20261024-006/iteration-006/evidence.json) and [four evaluator learnings](../research_runs/continuous-20261024-006/iteration-006/learnings.json).

Campaign006’s seventh candidate added **residual dropout .005 (0.5%)**, with attention dropout explicitly kept at its baseline value of zero. Although the saved genotype lists both keys, only residual dropout differs from the fixed baseline. All 18 trials completed 3,000 updates and passed checkpoint-hash verification; held-out test data stayed closed. Mean candidate-minus-baseline CE was **+.005315 / +.002216 / +.004083** at widths 64/128/256. Every width-64 and width-256 pair worsened. At width 128, two seeds improved, but the third worsened enough to leave the mean positive. The baseline remains selected.

Every paired run slowed: mean throughput changes were **−1.58%, −1.94%, and −1.40%**. Mean process RSS changes were +.08%, −.05%, and +.06%; these do not establish equivalence or accelerator peak-memory demand. Residual dropout affects both residual branches in this implementation, so this experiment does not isolate either branch’s contribution. The comparison supplies no compensating quality gain and no equal-time quality evidence. Historical attention-dropout runs informed the adaptive proposal but are not contemporaneous controls for a placement effect. [Campaign006 iteration 7 evidence](../research_runs/continuous-20261024-006/iteration-007/evidence.json) and [four evaluator learnings](../research_runs/continuous-20261024-006/iteration-007/learnings.json).

Campaign006’s eighth candidate raised the shared hybrid **base learning rate from .002 to .0022 (+10%)**. All 18 trials completed 3,000 updates and passed checkpoint-hash verification; held-out test data stayed closed. Mean candidate-minus-baseline CE was **−.003873 / +.004788 / +.001913** at widths 64/128/256. At width 64, seed deltas were +.001120, −.007342, and −.005397: the mean improvement fell short of .005, and one seed worsened. Both frozen acceptance requirements therefore failed. Each larger width had one improving seed and two worsening seeds. The baseline remains selected.

Mean throughput changes were −.27%, −.37%, and −.04%, with process RSS changes of +.02%, −.09%, and +.06%. These small differences do not establish performance equivalence. This shared learning-rate change affects both the Muon and AdamW parameter groups; it does not isolate an optimizer-specific cause. Nearby learning-rate results guided the adaptive proposal, so this is further development evidence, not independent confirmation of a response curve or an optimum. [Campaign006 iteration 8 evidence](../research_runs/continuous-20261024-006/iteration-008/evidence.json) and [four evaluator learnings](../research_runs/continuous-20261024-006/iteration-008/learnings.json).

Campaign006’s ninth candidate reduced **global gradient clipping from 1 to .875**. All 18 trials completed 3,000 updates and passed checkpoint-hash verification; held-out test data stayed closed. Mean candidate-minus-baseline CE was **+.004041 / −.000271 / +.000233** at widths 64/128/256. Width-64 seed deltas were +.008520, −.000460, and +.004063, so the frozen local improvement rule failed. Width 128 had two improving seeds and one worsening seed, while width 256 had one improving seed and two worsening seeds; neither diagnostic met the improvement criterion. The baseline remains selected.

Mean throughput changes were −.03%, +.06%, and −.08%; process RSS changes were −.01%, −.10%, and −.01%. These small differences establish neither savings nor equivalence. A positive clipping threshold retains the norm calculation and rescaling path, unlike the earlier zero-clipping experiment; changing the threshold does not remove that work. Clipping frequency and optimizer-history effects were not measured, so the quality changes do not identify a mechanism. This remains adaptive development evidence on reused seeds. [Campaign006 iteration 9 evidence](../research_runs/continuous-20261024-006/iteration-009/evidence.json) and [three evaluator learnings](../research_runs/continuous-20261024-006/iteration-009/learnings.json).

Campaign006’s tenth candidate reduced **independent per-step decay from .00006 to .0000525 (−12.5%)**. All 18 trials completed 3,000 updates and passed checkpoint-hash verification; held-out test data stayed closed. Mean candidate-minus-baseline CE was **−.000486 / −.000099 / +.000126** at widths 64/128/256. Width-64 seed deltas were −.001452, −.000249, and +.000245: the mean improvement was too small and one seed worsened, failing both frozen acceptance requirements. Width 128 also had two improving seeds and one worsening seed; width 256 had one improving seed and two worsening seeds. No width met the improvement criterion, and the baseline remains selected.

Mean throughput changes were +.31%, +.24%, and +.04%; process RSS changes were +.02%, +.03%, and −.07%. These small differences do not establish causal speedups, memory savings, or equivalence. Changing a positive independent-decay coefficient preserves the operation and tensor shapes while reducing the per-step contraction. Parameter norms and training-validation gaps were not measured for a mechanism test. The adaptively chosen interior setting therefore establishes neither a decay optimum nor reliable width dependence. [Campaign006 iteration 10 evidence](../research_runs/continuous-20261024-006/iteration-010/evidence.json) and [four evaluator learnings](../research_runs/continuous-20261024-006/iteration-010/learnings.json).

Campaign006’s eleventh candidate raised the shared hybrid **base learning rate from .002 to .002375 (+18.75%)**. All 18 trials completed 3,000 updates and passed checkpoint-hash verification; held-out test data stayed closed. Mean candidate-minus-baseline CE was **+.000391 / +.002017 / +.003151** at widths 64/128/256. At width 64, seed deltas were +.008829, −.005650, and −.002006: two seeds improved, but the first seed’s worsening left the mean positive. Width 128 had one improving seed and two worsening seeds; every width-256 pair worsened. The frozen local rule failed, and the baseline remains selected.

Mean throughput changes were −.05%, −.44%, and −.09%; process RSS changes were −.02%, −.03%, and +.05%. These small resource differences do not establish equivalence or a causal cost. Raising the shared base rate changes both the Muon and auxiliary AdamW updates, so this comparison does not separate their contributions. Neighboring-rate evidence informed the adaptive proposal; the observed means do not establish population-wide harm or a learning-rate optimum. [Campaign006 iteration 11 evidence](../research_runs/continuous-20261024-006/iteration-011/evidence.json) and [three evaluator learnings](../research_runs/continuous-20261024-006/iteration-011/learnings.json).

Campaign006’s twelfth candidate raised the shared hybrid **base learning rate from .002 to .00215625 (+7.8125%)**. All 18 trials completed 3,000 updates and passed checkpoint-hash verification; held-out test data stayed closed. Mean candidate-minus-baseline CE was **−.003619 / +.002575 / +.003291** at widths 64/128/256. Width-64 seed deltas were +.001611, −.008212, and −.004256: the mean improvement fell short of .005, and seed 11 worsened, failing both frozen acceptance requirements. Width 128 had one improving seed and two worsening seeds; all three width-256 pairs worsened. The baseline remains selected. Without a supported local gain, these larger-width results remain diagnostics rather than a test of transferring an established improvement.

Mean throughput changes were +.07%, −.16%, and +.03%; process RSS changes were −.03%, −.13%, and −.14%. These small differences establish neither equivalence nor resource savings. The shared rate affects both optimizer components, and endpoint validation does not identify their separate contributions or prove faster useful learning. This additional interpolation uses the same development seeds; it is not independent replication or evidence of an optimum. [Campaign006 iteration 12 evidence](../research_runs/continuous-20261024-006/iteration-012/evidence.json) and [three evaluator learnings](../research_runs/continuous-20261024-006/iteration-012/learnings.json).

Campaign006’s thirteenth candidate added **attention dropout .0025 (0.25%)**. All 18 trials completed 3,000 updates and passed checkpoint-hash verification; held-out test data stayed closed. Mean candidate-minus-baseline CE was **+.001055 / +.002033 / +.000038** at widths 64/128/256. Width-64 seed deltas were +.003913, −.000549, and −.000200, failing both frozen acceptance requirements. Each larger width had one improving seed and two worsening seeds. The almost-zero width-256 mean does not establish equivalence. No width met the improvement criterion, and the baseline remains selected.

Every paired run slowed: mean throughput changes were **−2.78%, −3.77%, and −2.81%**. Mean process RSS changes were +.04%, −.01%, and +.04%. Even this low dropout probability incurred an observed throughput penalty without a supported quality gain. Mask generation and elementwise work are plausible causes, but no profiling isolated them. Equal-time quality remains unmeasured, and RSS does not establish peak accelerator demand. Historical dropout settings informed this adaptive proposal; they are not independent confirmation or a basis for selecting from the larger-width scores. [Campaign006 iteration 13 evidence](../research_runs/continuous-20261024-006/iteration-013/evidence.json) and [four evaluator learnings](../research_runs/continuous-20261024-006/iteration-013/learnings.json).

Campaign006’s fourteenth candidate increased **independent per-step decay from .00006 to .0000675 (+12.5%)**. All 18 trials completed 3,000 updates and passed checkpoint-hash verification; held-out test data stayed closed. Mean candidate-minus-baseline CE was **+.000174 / +.000294 / +.000156** at widths 64/128/256. Every width had one improving seed and two worsening seeds. Width-64 deltas were −.000536, +.000813, and +.000244, so both frozen acceptance requirements failed. The baseline remains selected. These small positive means establish neither equivalence nor general harm; the width-256 mean also masks opposing seed responses.

Mean throughput changes were −.13%, −.52%, and −.06%; process RSS changes were −.11%, −.33%, and −.20%. The lower recorded RSS does not establish causal memory savings or accelerator peak demand. Increasing this positive decay coefficient strengthens per-step contraction while preserving tensor shapes and decay operations. Neither reduced overfitting nor altered update-to-weight dynamics was directly measured. This is another adaptive development comparison, not evidence of a decay optimum or an independent replication. [Campaign006 iteration 14 evidence](../research_runs/continuous-20261024-006/iteration-014/evidence.json) and [three evaluator learnings](../research_runs/continuous-20261024-006/iteration-014/learnings.json).

Campaign006’s fifteenth candidate reduced **global gradient clipping from 1 to .25**. All 18 trials completed 3,000 updates and passed checkpoint-hash verification; held-out test data stayed closed. Mean candidate-minus-baseline CE was **+.005825 / −.000029 / +.000825** at widths 64/128/256. Every width-64 seed worsened, with deltas +.010072, +.002099, and +.005304, contradicting the frozen improvement rule. Each larger width had one improving seed and two worsening seeds. The tiny negative width-128 mean does not establish equivalence or consistent benefit. The baseline remains selected.

Mean throughput changes were −.09%, +.67%, and −.18%; process RSS changes were −.07%, +.18%, and +.14%. These observations do not establish causal savings or a quality-resource advantage. The positive clipping threshold retains norm calculation and rescaling work; it does not reproduce the earlier zero-clipping speedup. Stronger clipping could change optimizer histories or suppress useful gradient contributions, but those mechanisms were not measured. Historical clipping trials supplied adaptive context, not independent controls. [Campaign006 iteration 15 evidence](../research_runs/continuous-20261024-006/iteration-015/evidence.json) and [three evaluator learnings](../research_runs/continuous-20261024-006/iteration-015/learnings.json).

Campaign006’s sixteenth candidate enabled **residual-output dropout at .0025**, retaining attention-probability dropout at zero (an explicit unchanged setting). All 18 trials completed 3,000 updates and passed checkpoint-hash verification; held-out test data stayed closed. Mean candidate-minus-baseline CE was **+.000808 / +.001884 / −.003222** at widths 64/128/256. Width-64 seed deltas were +.001704, +.001431, and −.000711, so the frozen local rule retained the baseline. Width 128 had two worsening seeds and one improving seed. Every width-256 seed improved (−.005003, −.003816, −.000846), but its mean gain remained below the registered .005 threshold. This diagnostic cannot change the earlier selection or establish a reproducible width interaction.

Every paired run was slower: mean throughput changes were **−1.55%, −1.02%, and −1.39%**. Mean process RSS changes were +.13%, +.27%, and +.08%; these lifetime high-water marks do not measure peak accelerator demand. A small nonzero dropout probability still executes masking work, but profiling did not isolate the cause of the observed slowdown. Perturbed fitting and capacity-dependent regularization are hypotheses, not measured mechanisms. This adaptive development comparison does not establish significance, an optimum, or equal-time quality. [Campaign006 iteration 16 evidence](../research_runs/continuous-20261024-006/iteration-016/evidence.json) and [four evaluator learnings](../research_runs/continuous-20261024-006/iteration-016/learnings.json).

Campaign006’s seventeenth candidate reduced **independent per-step decay from .00006 to .00005625**. All 18 trials completed 3,000 updates and passed checkpoint-hash verification; held-out test data stayed closed. Mean candidate-minus-baseline CE was **+.000896 / +.000267 / −.000389** at widths 64/128/256. Every width-64 seed worsened (+.000874, +.000530, +.001284), so the frozen local rule retained the baseline. Every width-128 seed also worsened. All three width-256 seeds improved, but the mean gain was far below the registered .005 threshold and cannot change local selection. Small differences do not establish equivalence or a reproducible width-dependent effect.

Mean throughput changes were +.10%, −.35%, and −.03%; process RSS changes were +.09%, −.28%, and approximately 0.00%. These measurements do not establish causal resource savings. A lower positive independent-decay coefficient weakens per-step parameter contraction while preserving the decay operations and tensor shapes; its effect on fitting or generalization was not isolated. This is adaptive development on reused validation data, not an independent replication or a decay optimum. [Campaign006 iteration 17 evidence](../research_runs/continuous-20261024-006/iteration-017/evidence.json) and [evaluator learnings](../research_runs/continuous-20261024-006/iteration-017/learnings.json).

Campaign006’s eighteenth candidate raised the **shared base learning rate from .002 to .002175**. All 18 trials completed 3,000 updates and passed checkpoint-hash verification; held-out test data stayed closed. Mean candidate-minus-baseline CE was **−.003907 / +.003104 / +.003672** at widths 64/128/256. Width-64 seed deltas were +.002066, −.008057, and −.005729: the negative mean missed the .005 improvement threshold, and one seed worsened. Both requirements of the frozen local rule therefore failed, retaining the baseline. Width 128 had one improving pair and two worsening pairs; all width-256 pairs worsened. These are larger-width diagnostics, not failed transfer of an established small-model improvement.

Mean throughput changes were +.10%, +.48%, and −.04%; process RSS changes were +.05%, +.14%, and −.22%. All lay within the frozen resource forecast ranges, which are subjective forecasts rather than equivalence tests. The scalar changes both Muon and auxiliary AdamW rates under their existing width rules. This single-setting comparison cannot attribute an effect to either optimizer component, greater optimization progress, or synergy. Neighboring-rate history informed adaptive exploration; it does not establish an optimum or independent replication. [Campaign006 iteration 18 evidence](../research_runs/continuous-20261024-006/iteration-018/evidence.json) and [three evaluator learnings](../research_runs/continuous-20261024-006/iteration-018/learnings.json).

Campaign006’s nineteenth candidate raised **global gradient clipping from 1 to 1.75**. All 18 trials completed 3,000 updates and passed checkpoint-hash verification; held-out test data stayed closed. Mean candidate-minus-baseline CE was **+.000943 / −.000308 / +.000311** at widths 64/128/256. Width-64 seed deltas were +.000737, +.002254, and −.000161, so the frozen local rule retained the baseline. Each larger width had two improving pairs and one worsening pair, but neither met the improvement criterion. The tiny negative width-128 mean does not establish equivalence or beneficial transfer.

Mean throughput changes were +.11%, +.14%, and +.09%; process RSS changes were −.11%, +.17%, and +.06%. All six observations fell within their frozen forecast ranges, without establishing causal savings. A positive clipping threshold retains norm calculation and rescaling operations. Clipping frequency, gradient norms, and optimizer-history effects were not measured, so useful-gradient preservation and outlier suppression remain conjectures. Historical threshold comparisons are adaptive context, not independent controls. [Campaign006 iteration 19 evidence](../research_runs/continuous-20261024-006/iteration-019/evidence.json) and [three evaluator learnings](../research_runs/continuous-20261024-006/iteration-019/learnings.json).

### Stage 1 conclusion and the architecture transition

The human concluded stage 1 on October 24, 2026. The strongest observed arm remains **Muon/AdamW hybrid + muP, base LR .002, independent decay .00006, clipping 1, zero dropout, dense GELU MHA with RoPE**, two layers and head dimension32. Its measured width64/128/256 CE is **1.646054 / 1.559346 / 1.523978** from the controlled optimizer study. This is the frozen control for architecture experiments, not a universal optimum. Across the two completed optimizer studies (138 trials) and the subsequent 42 paired candidate suites (756 trials), the later small mutations did not displace that control under the local rule. These counts do not include all earlier project experiments.

The last suite tested independent decay .00005: CE deltas were +.000185 / +.001021 / +.000098. All 18 trials completed and checkpoint hashes verified; local selection retained the baseline. Campaign006 stopped before its evaluator agent ran, so the coordinator recorded a separate [stage-close review](../research_runs/continuous-20261024-006/iteration-020/stage-close-review.json) from the completed evidence without resuming training or clearing the human stop. The [stage-1 closure record](../research_state/stage1-conclusion.json) preserves the exact control and provenance.

**Stage 2A isolates GELU versus SiLU** at identical parameter counts and matrix shapes. Each activation receives the same three-LR screen (.0015/.002/.0025) on fresh development seeds44/55/66. A separate fresh fixed-LR control preserves the stage-1 recipe. Freeze each family's LR and the architecture winner using width64 only, before widths128/256. SiLU must beat both the fixed control and equally tuned GELU by mean CE greater than .005 with every paired seed improving. There are 21 local and 18 larger-width runs, 3000 updates each, no test access. Tests and short MPS smoke runs are correctness checks, not quality evidence. Fresh seeds on the same adaptively used validation split are not independent holdout confirmation. See the [stage-2 protocol](ARCHITECTURE_STAGE2.md) for limits and the later gated-FFN, attention, depth and MoE roadmap.

### Completed stage 2A: activation and learning-rate transfer

All **39 trials** completed and were reviewed against checkpoint hashes, worker results,
frozen selection, source/corpus/runtime contracts and recomputed report aggregates.
Mean final validation CE at 3000 updates (three fresh development seeds44/55/66):

| Arm | Frozen LR | Width64 | Width128 | Width256 |
| --- | --- | --- | --- | --- |
| Fixed GELU control | .002 | 1.659749 | 1.573942 | **1.520457** |
| Equally tuned GELU | .0025 | **1.647783** | **1.571541** | 1.525777 |
| Tuned SiLU | .0025 | 1.659710 | 1.576024 | 1.521624 |

**Keep the exact stage-1 control.** SiLU failed the predeclared width64 architecture
rule: its mean delta was −.000039 against fixed GELU (one seed worse), and +.011928
against tuned GELU (all three worse). At widths128/256 its deltas against fixed
control were +.002082 / +.001168, both mixed by seed. SiLU was better than tuned
GELU on every width256 seed, but the mean −.004153 missed the .005 threshold;
transfer diagnostics cannot change the frozen architecture decision.

The GELU LR change is a useful transfer counterexample: .0025 improved over .002
by .011966 at width64 and .002401 at width128, then worsened by .005320 at width256.
All three paired seeds followed each direction. Width scaling does not guarantee
that a locally preferred LR preserves its advantage at larger widths. This does
not invalidate muP in general or identify the optimal LR at any width.

Mean training seconds for control / tuned GELU / SiLU were 32.23 / 32.48 / 32.35
at width64, 51.26 / 51.00 / 51.84 at width128, and 66.40 / 66.41 / 66.36 at width256.
Process peak RSS means ranged from 472.1 to 475.6 MiB. These small observed time
and RSS differences do not establish a reliable speed advantage; RSS is not peak
accelerator memory. Equal token budgets and shapes are not equal measured compute.

Both tuned families selected the upper grid boundary; no grid extension is planned.
The validation split remains adaptively reused, despite fresh training seeds; test
was never opened. Historical control scores use different seeds and must not be
pooled as independent replication. Correctness tests and short smoke runs remain
separate from these measured quality results.

Evidence: `research_runs/architecture-stage2a-001/{plan,selection,report,review}.json`.
Next is a separately preregistered gated-FFN round, after parameter-budget,
bias/initialization, optimizer-aspect-ratio and execution checks. Stage1 stays stopped.

### Stage 2B: gated FFNs, with explicit budget and scaling controls

The next round compares bias-free dense SiLU with bias-free SwiGLU, retaining the
original GELU control. Dense hidden widths are256/512/1024; SwiGLU widths are
168/336/672. Its three projection matrices contain1.5625% fewer FFN weights than
the dense comparator's two. Total model counts are115712/428032/1642496 for
bias-free dense SiLU and114688/423936/1626112 for SwiGLU, versus the fixed GELU
control's116352/429312/1645056. The unused learned-position table is included.

Both new families receive LR.0015/.002/.0025 and fresh development seeds77/88/99.
There are21 local and18 transfer trials; a SwiGLU replacement must beat BOTH
comparators at width64 under the frozen mean/all-seed rule. Equal tokens and
near-matched parameters do not mean equal compute. Removing biases and changing
matrix shapes changes initialization draws; paired seeds do not give identical
initial weights. Six three-update MPS/checkpoint checks passed, but establish only
correct execution, not quality. The [round2B protocol](ARCHITECTURE_STAGE2.md)
explains the fixed21/8 aspect ratio and its narrowly supported width contract.

### Completed stage 2B: SwiGLU wins locally, with mixed largest-width transfer

All39 trials completed and were reviewed against final checkpoint hashes, worker
result contracts, frozen local selection, report aggregates, source/corpus/runtime
and the full shared-knowledge snapshot. Mean final validation CE at3000 updates:

| Arm | Frozen LR | Width64 | Width128 | Width256 |
| --- | --- | --- | --- | --- |
| Fixed GELU control | .002 | 1.656103 | 1.575256 | 1.528558 |
| Bias-free dense SiLU | .0025 | 1.658701 | 1.571751 | 1.536284 |
| **Bias-free SwiGLU** | **.0025** | **1.637745** | **1.553588** | **1.523561** |

SwiGLU passed the frozen width64 selection rule against both comparators.
Its mean deltas versus fixed GELU were−.018358 / −.021668 / −.004997245 at
widths64/128/256. Every seed improved at64/128. At256, seed77 worsened by+.001391,
seed88 improved by−.001545, and seed99 improved by−.014838. Thus **width256 does
not pass the support rule**: the unrounded mean also narrowly misses−.005.
Rounding it to−.005 cannot turn this into a success, and the positive seed would
still fail. The local winner stays frozen; transfer does not retroactively alter it.

Against the equally tuned bias-free dense SiLU comparator, SwiGLU improved every
seed at all three widths, with mean deltas−.020955 / −.018163 / −.012722.
That makes SwiGLU the most promising observed architecture in this development
round. It does not establish uniformly supported superiority over the original
GELU control at all widths. The original recipe remains an explicit reference;
its historical best-config record is preserved separately.

SwiGLU mean training time was33.41 / 54.72 / 67.10 seconds, about2.96% / 3.97% /
1.70% longer than the fixed control's32.45 / 52.63 / 65.98 seconds. This is a
fixed-token quality/time tradeoff, not evidence from equal-time training. Peak
process RSS means were471.31 / 473.40 / 475.38 MiB versus471.58 / 472.74 / 474.17.
Process RSS is not peak accelerator allocation. Parameter-count reductions did
not produce a measured training-speed win in this round.

Both tuned arms selected the upper LR grid boundary; no extension of this completed
grid is planned. Fresh seeds77/88/99 remain development seeds on the reused
validation split. No held-out test was consumed. The change combines gating,
hidden ratio and different initialization draws; comparison to fixed GELU also
changes FFN bias and activation. Do not attribute the result to gating alone.
Evidence: `research_runs/architecture-stage2b-001/{plan,selection,report,review}.json`.
The next attention/head round requires its own scaling and resource review and
preregistration. Neither previous stage nor completed grid will be restarted.

### Stage 2C: change attention head size while keeping the FFN fixed

The next bounded round compares SwiGLU with MHA head dimensions16 and32. QKV/output
matrix shapes, parameter counts and initial parameter tensors are identical at
paired seeds, as are optimizer groups. Head16 has twice as many heads at each
width. Its attention scores divide by4 instead of sqrt32, and its RoPE frequency
basis changes. These are parts of the head-size change; we cannot attribute a
result to one of them alone. Attention-score tensors double in element count,
which does not imply twice the total memory or time.

Keep both the original GELU .002 reference and fixed SwiGLU .0025 incumbent.
Give head16 and head32 equal LR screens .0015/.002/.0025 on development seeds
111/222/333. The48-trial plan freezes choices after24 width64 runs, before24
transfer runs. A head16 replacement must pass the mean/all-seed rule against all
three comparators. See the [round2C protocol](ARCHITECTURE_STAGE2.md). This tests
head partition within MHA; GQA/MLA, depth and MoE are separate future rounds.

### Completed stage 2C: retain head32; head16 gains are inconsistent and slower

All 48 trials, checkpoint hashes, per-family worker head counts, frozen selection,
report aggregates and source/corpus/runtime/knowledge contracts were verified.
Mean final validation CE at 3000 updates on development seeds 111/222/333:

| Arm | LR | Width 64 | Width 128 | Width 256 |
| --- | --- | --- | --- | --- |
| Original GELU head32 | .002 | 1.649678 | 1.567840 | 1.534493 |
| Fixed SwiGLU head32 incumbent | .0025 | 1.639925 | 1.560148 | 1.533336 |
| Tuned SwiGLU head32 | .0025 | 1.639156 | 1.559536 | 1.529393 |
| Tuned SwiGLU head16 | .0025 | 1.633562 | 1.559925 | 1.527879 |

The frozen local winner remains the head32 incumbent. Head16's width 64 mean
improved by −.006363 versus the incumbent and−.005594 versus tuned head32, but
seed 333 worsened by +.002028 / +.001835. It failed the all-negative-pair condition.
Both tuned families chose the upper LR boundary; no grid extension is planned.

At 128, head16 versus tuned head32 changed mean CE by +.000389 with mixed seeds.
At 256 it improved by −.001514, again with mixed seeds. Neither comparison passes.
Head16 versus the fixed incumbent at 256 did pass (mean−.005456, every seed better),
but **the separately trained tuned head32 arm uses the same .0025 recipe** and
produces a different comparison. The two head32 repeat means differ by .000768 /
.000612 / .003943 at 64/128/256; width 256 seed 111 differs by .007560. Preserve both
runs. The threshold-sensitive result is not evidence of an independently robust
head16 advantage, and larger-width diagnostics cannot revise local selection.
This round does not establish the cause of repeated-run variation.

Head16 took 36.01 / 60.80 / 74.10 mean training seconds versus 33.45 / 54.73 / 67.10
for tuned head32: **7.65% / 11.10% / 10.43% longer**. Parameter counts are identical;
more heads still change intermediate tensors and execution cost. No equal-time
quality experiment was run. Process RSS differences were small and do not measure
peak accelerator demand. The repeated SwiGLU incumbent beat GELU on all 64/128
seeds, but width 256 remained mixed; this is additional adaptive development
evidence, not independent held-out confirmation.

Evidence: `research_runs/architecture-stage2c-001/{plan,selection,report,review}.json`.
The most promising recipe remains bias-free SwiGLU 21/8, MHA head32, Muon/AdamW+muP,
LR .0025 and independent decay .00006, with the original GELU recipe retained as a
reference. Next is a separately reviewed grouped-query attention comparison;
no GQA/MLA, depth or MoE claim follows from this head-partition experiment.

### Stage 2D design: grouped-query attention

Stage 2D keeps the SwiGLU FFN and query head dimension 32 fixed while sharing
one key/value head between two query heads. At widths 64/128/256 that means
2/4/8 query heads and 1/2/4 KV heads. The fused QKV matrix shrinks from 3d × d
to 2d × d, saving 2(d²+d) parameters across two layers. It also changes Muon's
matrix-shape multiplier; this is explicitly included in the implementation
contract. Equal LR tuning for MHA and GQA and the original fixed references
help assess the resulting recipe, but do not isolate KV sharing from every
parameter/initialization effect. No quality or speed improvement is assumed.

The [stage 2D protocol](ARCHITECTURE_STAGE2.md) freezes 48 serial trials, fresh
development seeds 444/555/666, and selection at width 64 before larger-width
results. Synthetic tests and short MPS checkpoint checks establish correctness
only. Attention scores still use all query heads; halving KV heads does not
promise halving total memory or elapsed training time.

### Completed stage 2D: GQA is smaller and modestly faster, but does not replace MHA

All 48 checkpoint hashes, saved QKV shapes, worker requests/results, frozen
selection, report aggregates and source/corpus/runtime/knowledge contracts were
verified. Mean final validation CE at 3000 updates on development seeds 444/555/666:

| Arm | Selected LR | Width 64 | Width 128 | Width 256 |
| --- | --- | --- | --- | --- |
| Original GELU/MHA reference | .002 | 1.661810 | 1.569754 | 1.522443 |
| Fixed SwiGLU/MHA incumbent | .0025 | 1.647115 | 1.558650 | 1.521123 |
| Tuned SwiGLU/MHA | .002 | 1.646205 | 1.559586 | 1.521749 |
| Tuned SwiGLU/GQA | .0025 | 1.643881 | 1.560941 | 1.522255 |

GQA's frozen width-64 mean deltas were −.003234 versus the incumbent and
−.002325 versus tuned MHA; seed 666 worsened by +.003501 and +.001727 respectively.
It failed both the mean threshold and all-seed requirement. GQA chose the upper
LR boundary; MHA chose the interior .002. The grid remains closed, and the
prior SwiGLU/MHA .0025 incumbent is retained. This does not declare its LR
universally optimal or revise the historical stage-2B selection.

At widths 128/256, GQA's mean CE was worse than the incumbent by +.002291/+.001133
and worse than tuned MHA by +.001356/+.000506; every comparison had mixed seed
signs. At width 256, GQA versus tuned MHA ranged from −.009231 to +.011184 across
seeds. GQA beat the original GELU reference under the rule at widths 64/128,
but not at 256. The original reference alone would therefore overstate the case
for changing attention: the stronger SwiGLU/MHA comparator matters.

GQA mean training times were 32.24/53.57/63.94 seconds versus tuned MHA's
33.47/54.16/67.07: **3.68%/1.09%/4.65% faster**. GQA had about 7.3–8.1% fewer
total parameters under this contract. RSS did not fall consistently; process
RSS is not accelerator memory. These measurements support a modest training
speed/capacity tradeoff, not a quality win, statistical equivalence, an inference
speed claim or superiority at an equal wall-clock budget. Shape-dependent Muon
updates, initialization draws and parameter counts differ as preregistered.

The repeated fixed SwiGLU incumbent again beat GELU on every seed at 64/128,
with mixed results at 256. This adds adaptive development evidence while leaving
the largest-width uncertainty intact. Tuned MHA selected .002 here, so its
transfer results and the fixed .0025 reference are different recipes; do not
label them identical-recipe repeats or pool their measurements.

Evidence: `research_runs/architecture-stage2d-001/{plan,selection,report,review}.json`.
The most promising incumbent remains bias-free SwiGLU 21/8, MHA head32,
Muon/AdamW+muP, LR .0025 and independent decay .00006. Next is a separately
reviewed depth comparison; width scaling alone does not validate depth scaling.

### Stage 2E design: adding transformer layers

Stage 2E compares two and four SwiGLU/MHA layers at widths 64/128/256. Each block
keeps the same attention, FFN, normalization and residual coefficients. The
four-layer model has about 85–96% more total parameters; equal training tokens
will not mean equal elapsed compute. Both depths receive the same three-LR
screen, with fixed original GELU and SwiGLU references retained.

Width and depth are separate scaling axes. Our existing muP/Muon width factors
apply within each fixed-depth family. We do not introduce or claim a validated
depth-normalization rule: four layers use the existing unscaled residual sums
and fan-in initialization. Adding layers changes residual accumulation, readout
RNG draws and the parameter set subject to clipping. Short correctness checks
cannot predict full-horizon quality. See the [depth protocol](ARCHITECTURE_STAGE2.md)
for the frozen selection rule, parameter calculation and resource limits.

### Stage 2E measured conclusion: four layers win, with a resource cost

All 48 preregistered trials completed and passed checkpoint-hash, saved layer/shape,
parameter-count, request/result, source/corpus/runtime, full-bank snapshot,
frozen-selection and aggregate verification. No training was rerun for review.
Both tuned depths chose LR .0025 at the grid boundary; the grid is closed.

| Recipe | LR | CE64 | CE128 | CE256 |
|---|---:|---:|---:|---:|
| Original two-layer GELU/MHA | .002 | 1.661662 | 1.580095 | 1.535058 |
| Fixed two-layer SwiGLU/MHA | .0025 | 1.646487 | 1.559472 | 1.534609 |
| Tuned two-layer SwiGLU/MHA | .0025 | 1.645470 | 1.561114 | 1.529471 |
| Four-layer SwiGLU/MHA | .0025 | **1.583606** | **1.516719** | **1.506713** |

Four layers passed the frozen width-64 rule against all three references.
Its mean CE deltas versus the fixed SwiGLU incumbent were
−.062881/−.042753/−.027896 at widths 64/128/256; versus tuned two layers they were
−.061863/−.044395/−.022758. Every paired seed improved against every comparator
at every tested width, and all mean differences passed −.005. Larger-width
results support this frozen selection; they did not revise it.

The gain uses more resources. Four-layer parameter counts are
212736/814592/3185664, about 85.5%/92.2%/95.9% above two layers. Mean training
seconds were 55.040/63.150/125.819 versus 33.421/54.142/67.025 for tuned two layers:
**+64.69%/+16.64%/+87.72%**. These are measured timings, not a smooth scaling law
or equal-compute comparison. Mean lifetime peak process RSS was
472.93/475.14/481.46 MiB versus 471.74/472.38/475.28 MiB; RSS is not peak accelerator
memory. No inference-speed or efficiency-optimum claim follows.

The fixed and tuned two-layer arms selected the same recipe but differ on repeat:
tuned-minus-fixed mean CE was −.001018/+.001642/−.005138. At width 256, seed 777
alone differed by−.013150. Both references remain recorded; no exact numerical
reproducibility or identified cause is claimed. The four-layer gain survives
comparison with both. The fixed two-layer arm still had mixed width 256 support
against original GELU, while its tuned repeat passed the rule there.

Four-layer bias-free SwiGLU 21/8, MHA head32, Muon/AdamW with muP base64,
LR .0025, independent decay .00006, clipping 1 and unscaled residual branches is
now the most promising observed **quality** recipe in this development setting.
The two-layer SwiGLU recipe remains a cheaper reference and original GELU is
preserved. This is adaptive Shakespeare validation with seeds 777/888/999 and
3000 updates, not independent held-out confirmation, a universal optimum,
matched capacity, pure depth causality or validated infinite-depth muP.
Stage 2E is closed. Future MoE/MLA work requires its own implementation,
initialization, optimizer/scaling and finite parameter/runtime budget review.

Evidence: [report](../research_runs/architecture-stage2e-001/report.json),
[selection](../research_runs/architecture-stage2e-001/selection.json),
[review](../research_runs/architecture-stage2e-001/review.json).

### Next controlled comparison: MLA at the winning depth

Stage 2F holds four-layer SwiGLU fixed and compares MHA with the repository's
MLA: query/KV latent dimensions d/4, content and rotary head widths 32. Original
GELU and cheaper two-layer SwiGLU remain references alongside the fixed four-layer
incumbent. Both four-layer attention families get equal bounded LR tuning.

MLA has fewer parameters here, but adds factorized projections and a shared
rotary key. That key matrix has fixed 32 rows while model width grows, so the
existing fixed-aspect-ratio argument is insufficient. The explicit optimizer
factor cancels width algebraically, but feature/update statistics and optimal
LR transfer remain empirical questions. Different initialization draws, score
scaling and projection shapes prevent pure compression attribution.

The [stage2F protocol](ARCHITECTURE_STAGE2.md) declares exactly 57 trials and
separates parameter/cache calculations, short correctness checks and eventual
quality/runtime measurements. It does not predict an inference speedup or reuse
the paper's performance claims. Held-out test remains closed.

### Completed stage 2F: MLA does not replace four-layer MHA

All 57 serial trials completed on historical source
`5dd790d51f7ee30c44f3633c00648d227ab0f1e8`. Checkpoint hashes, saved layer/head/latent
and projection shapes, parameter counts, requests/results, runtime, frozen local
selection, report aggregates, corpus and full-bank snapshot were verified before
updating the checkout. The held-out test remains closed.

Both tuned families selected base LR .0025 at the frozen grid boundary; the grid
is closed. At width64, MLA is worse on every paired seed than both fixed and tuned
four-layer MHA: mean CE differences +.018238 and +.013924. It fails the replacement
rule. Four-layer SwiGLU/MHA remains the quality incumbent; cheaper two-layer
SwiGLU and original GELU remain references.

| Family | Selected LR | CE width64 | CE width128 | CE width256 |
| --- | ---: | ---: | ---: | ---: |
| Original two-layer GELU | .002 | 1.658560 | 1.573045 | 1.538094 |
| Cheaper two-layer SwiGLU | .0025 | 1.641669 | 1.562098 | 1.519259 |
| Fixed four-layer MHA | .0025 | 1.579393 | 1.518302 | 1.506017 |
| Tuned four-layer MHA | .0025 | 1.583707 | 1.517095 | 1.504022 |
| Tuned four-layer MLA | .0025 | 1.597631 | 1.521783 | 1.498427 |

MLA minus tuned MHA is +.013924 / +.004688 / -.005595 at widths64/128/256.
At128, seeds1022/1033 worsen; at256, seed1033 worsens by+.001803 despite the lower
mean. Against fixed MHA, width256 mean is-.007589 with seed1033+.002252.
Neither larger width passes the all-seed rule against either four-layer reference.
The lower width256 mean is a useful diagnostic, not permission to revise local
selection or an established architecture-by-width mechanism.

Fixed and tuned MHA use identical selected recipes but are separate runs.
Tuned-minus-fixed means are+.004314 / -.001207 / -.001994; keep both references.
There is no identified cause for these repeat differences or claim of exact
training reproducibility. Both four-layer MHA arms beat the original GELU and
cheaper two-layer SwiGLU on every paired seed at every tested width.

MLA's mean training times are86.62 / 101.64 / 131.04 seconds, versus tuned MHA
54.52 / 60.81 / 127.77: increases of58.88% / 67.14% / 2.55%. Historical total
parameter counts are196864 / 733568 / 2826880, about7.46% / 9.95% / 11.26% fewer
than four-layer MHA. Fewer parameters did not yield faster measured training.
These timings are observations on this Mac, not inference benchmarks or equal
compute comparisons. Projection shapes, initialization draws and attention score
decomposition change together; this is not a pure latent-compression effect or
DeepSeek replication. Finite width-scaling algebra does not validate MLA muP.

An agent maintenance stop after trial50 allowed the human-requested RoPE table
repair to be tested separately. Only the seven unfinished trials resumed, on the
unchanged original source and plan. No completed trial was rerun. The repaired
source `7577129` was adopted only after this full study audit. Historical counts
include the dormant position table; new runs omit it and use fresh initialization.
The fix and its 552 tests/three short MPS checks establish correctness, not new
quality evidence.

Evidence: [report](../research_runs/architecture-stage2f-001/report.json),
[selection](../research_runs/architecture-stage2f-001/selection.json), and
[review](../research_runs/architecture-stage2f-001/review.json).

Next: the authorized saved-checkpoint full-validation audit, with paired text-block
uncertainty, no retraining and no held-out test. Context/batch robustness is the
next research priority. A compact interaction design was recommended in discussion
but has not yet been frozen or launched; no MoE launch is planned.

### Full-validation audit: depth persists; MLA's largest-width seed sign changes

The frozen `validation-coverage-001` audit completed81 saved-checkpoint evaluations
in125.86seconds active controller time, with zero training runs. All checkpoint
and result hashes, worker contracts,111538 targets per model, source/corpus/full
bank, independently calculated paired means and reconstructed bootstrap/report
were verified on evaluator source `c4ff0a9` before documentation changes. The panel
contains36 selected stage2E and45 selected stage2F checkpoints, preserving repeated
references. The held-out test remains closed; checkpoint bytes are unchanged.

Each validation target except the first character was scored exactly once in
consecutive context128 windows, batch8, with a short unpadded final window and
token-weighted aggregation. This differs from the historical30-batch randomly
sampled fixed panel in coverage, overlap and window boundaries. It is not a new
independent dataset or a sliding maximum-context evaluation. Scores must not be
spliced into the old training study reports.

| Comparison (candidate minus reference) | Full-val delta width64 | Width128 | Width256 |
| --- | ---: | ---: | ---: |
| E: four-layer minus tuned two-layer SwiGLU | −.061794 | −.043728 | −.022303 |
| F: MLA minus fixed four-layer MHA | +.017336 | +.007729 | −.007308 |
| F: MLA minus tuned four-layer MHA | +.013453 | +.006146 | −.007100 |

**Depth:** every seed still improves versus all three E references at every width.
All conditional95% text-block bootstrap intervals are below zero at block sizes
2048/4096/8192. The depth improvement survives this evaluation-coverage change;
its previously measured extra training/parameter cost remains unchanged.

**MLA:** every seed still worsens versus both four-layer MHA references at64;
128 has a worse mean and mixed seeds. At256, all three seeds now improve against
both references. MLA minus tuned MHA seed deltas are−.001353/−.014357/−.005590 for
1011/1022/1033; versus fixed MHA they are−.005353/−.015992/−.000579. Seed1033's sign
therefore changes from the historical sampled panel's positive difference.
The4096-character conditional interval is[−.011499,−.003231] versus tuned MHA
and[−.011526,−.003243] versus fixed MHA; all three block-size intervals remain
negative. This strengthens the largest-width diagnostic under broader coverage,
but neither establishes a causal width interaction nor reverses the frozen
width64 selection. Four-layer MHA remains the historical selected incumbent.

Intervals resample paired contiguous text blocks jointly across the three fixed
trained seeds. They are conditional text uncertainty, not training-seed confidence
intervals, independent holdout confirmation or multiplicity-adjusted tests.
There are55/28/14 blocks at the three sizes, and residual text dependence may
remain. Report the full sensitivity set and per-seed differences; do not select
a favorable block size or infer universal significance. Shared seeds do not make
this adaptively reused text an independent validation source.

Full-validation mean CE, widths64/128/256:
E originalGELU1.672886/1.592635/1.551230; fixed2LSwi1.659265/1.575020/1.547150;
tuned2LSwi1.658437/1.576396/1.544050; four-layer1.596643/1.532668/1.521747.
F originalGELU1.670300/1.585966/1.547213; cheaper2LSwi1.654783/1.574416/1.531199;
fixed4LMHA1.592719/1.528306/1.519030; tuned4LMHA1.596602/1.529889/1.518822;
MLA1.610056/1.536035/1.511722. Fixed/tuned repetitions remain separate.

This audit shows why broader evaluation should precede stronger architecture
claims: a small seed-level contrast can depend on the scoring panel even without
retraining. It cannot isolate whether coverage, boundary placement, or overlap
caused the sign change. Future context comparisons need a common evaluation panel
as well as native-context scores, and fresh controls under repaired RoPE
initialization. Context/batch training is authorized; an interaction-aware redesign
is being discussed and no new training grid has been frozen or launched.

Evidence: [plan](../research_runs/validation-coverage-001/plan.json),
[report](../research_runs/validation-coverage-001/report.json),
[review](../research_runs/validation-coverage-001/review.json).

### Prospective efficiency selection after human steering

The human now permits adopting GQA or MLA with a small quality loss when measured
inference savings justify it. Future selection therefore includes a quality
non-inferiority objective alongside the existing quality incumbent. Historical
stage 2D/2F decisions remain frozen; neither is retroactively declared a winner.

The human accepted a maximum loss of +0.01 validation CE nats/token (exp(.01)-1,
about 1.005% higher perplexity). This is a decision preference, not an established
universal threshold. The matched comparator, uncertainty rule and minimum useful
inference benefit must still be frozen before new selection.
Report paired seeds separately from conditional text-block intervals; the latter
do not quantify training-seed variation. Assess each intended width/context rather
than averaging away a harmful setting.

Measure prefill and cached decode separately on matching prompts, lengths,
batches, precision and hardware. Record warmed-up synchronized latency/throughput,
actual KV-cache bytes and peak device memory. Training speed, theoretical cache
sizes and parameter counts do not establish deployment savings. A reduction in
memory can be useful even without a latency gain; report the actual tradeoff.

The completed full-validation audit makes width256 MLA an efficiency candidate:
its mean CE is .007100 below tuned four-layer MHA, with all three paired seeds
better. Width64 is .013453 worse and width128 is .006146 worse on average. These
are adaptive diagnostics, not approval to deploy or proof of non-inferiority.
Fresh context controls still require the repaired RoPE initialization. The next
training grid remains unfrozen; this policy does not authorize extending closed
grids or launching MoE.

## 11 How to audit a claim yourself

For a concrete starting point, open the [fifth iteration’s evidence](../research_runs/continuous-20261023-004/iteration-005/evidence.json) and [saved learnings](../research_runs/continuous-20261023-004/iteration-005/learnings.json).

Start from one `evidence.json` under `research_runs/continuous-20261023-004/iteration-NNN/`. Read `comparisons`, then inspect its three `pairs` for a width. Subtract baseline CE from candidate CE for each seed. Average those differences. Check the all-negative condition separately.

Next open the corresponding `suite/trials/.../request.json`. This freezes the actual configuration, seed and output locations. `metrics.jsonl` gives the learning curve; `result.json` gives the endpoint and timing; `worker.log` shows the process output. A final checkpoint is hashed so later edits or substitution can be detected.

For the optimizer round, `research_state/optimizer-study-plan.json` freezes source commit, source hash, corpus hash, grid, seeds and budgets. `research_runs/optimizer-transfer-001/selection.json` records the frozen small-width choices. Its completed `report.json` contains the full measured matrix. The follow-up has its own plan and output directory; it cannot revise the first round. During a running round, the final report appears only after all trials finish.

The durable knowledge bank is under `research_state/knowledge/entries/`. Numerical results, literature claims and evaluator interpretations are separate records. An evaluator can propose a mechanism; it cannot override the measured selection criterion. When a claim changes, append a correction linked to the prior evidence rather than rewriting history.

A practical notebook entry has five sentences: **hypothesis; frozen intervention; controls; measurement; scoped conclusion**. For example: “I hypothesized that the width-64 recipe would improve wider models. I transferred LR 0.003 and zero decay unchanged. I retained the corpus, seeds and 3,000-step budget. Width 256 worsened by 0.015435 mean CE with all pairs positive. This intervention did not transfer beneficially to width 256 in this setup.” Use that wording only after you have personally reconstructed the cited result; the walkthrough is meant to support understanding, not to invent work you performed.

## 12 Exercises with answers

**1. How many targets does one run process?** Calculate `8 × 128 × 3000`. Answer: 3,072,000 target positions, including repeated exposures. Why is this not the number of unique corpus characters?

**2. A candidate has paired deltas −0.02, −0.02 and +0.001. Did it pass?** The mean is −0.013, better than −0.005, but the third seed worsened. Answer: it fails our all-negative support rule. That does not prove the candidate is useless; it describes the declared criterion.

**3. Match the baseline decay at LR 0.003.** Baseline shrinkage is `0.0006 × 0.1 = 0.00006`. Conventional λ must be `0.00006 / 0.003 = 0.02`. In independent mode use δ=0.00006. Do not put 0.02 into the independent coefficient.

**4. Scale Adam from width 64 to 256.** Here m=4. Hidden-matrix nominal LR 0.002 becomes 0.0005 under our μP implementation. The output-head optimizer LR remains 0.002 while its explicit forward multiplier is 1/4. Explain why applying both an additional output LR division and the existing multiplier would change the intended recipe.

**5. Scale our Muon hybrid over the same widths.** The hidden nominal LR becomes `0.002 / sqrt(4) = 0.001`. PyTorch's RMS-matching shape factor doubles. Answer: their product stays constant relative to the same matrix type at the base width. Embeddings and readout still use AdamW.

**6. Does a finite synthetic smoke prove good transfer?** No. It verifies that the path executes and supplies bounded early numerical diagnostics. Transfer quality requires the frozen multi-seed corpus experiment.

**7. Is Muon faster if it reaches a loss in 20% fewer steps but each step costs 30% more?** Under that simplified assumption, total time is `0.8 × 1.3 = 1.04` times baseline, or 4% slower. Measure evaluation and other overhead separately when discussing total job time.

**8. Why keep the large-width results hidden until selection?** Otherwise, a configuration might be chosen partly because it worked at the target width. That is target-informed tuning, which cannot support a zero-shot small-to-large selection claim for that same comparison.

**9. What would convince you the conclusion is robust?** Specify it before another experiment: new paired seeds, a broader LR grid if the winner lies at its boundary, a separately frozen longer horizon if curves still decline, and ultimately a different scale or corpus. Each answers a different uncertainty; none is supplied merely by running more adaptive iterations on the same validation set.

## 13 A reading order that connects papers to code

1. Read [AdamW](https://arxiv.org/abs/1711.05101) for the distinction between L2 regularization and direct weight shrinkage. Reproduce exercise 3 using our optimizer code.
2. Read [Tensor Programs V](https://arxiv.org/abs/2203.03466) for the μTransfer objective. Inspect which dimensions change in our model before interpreting a scaling rule.
3. Compare the [μP implementation guide](https://github.com/microsoft/mup) with our explicit output multiplier and optimizer groups. Write down the parameterization convention before comparing LR numbers.
4. Read [Muon’s design](https://kellerjordan.github.io/posts/muon/) and distinguish an ideal matrix transformation from its finite Newton–Schulz approximation. Then inspect the native optimizer settings we use.
5. Read [the weight-decay transfer study](https://arxiv.org/html/2510.19093v2) and [the matrix-optimizer transfer study](https://arxiv.org/html/2512.05620v1) as empirical arguments with specific setups. Ask which conditions match our fixed-token character-model experiment and which do not.

You are ready to explain this project when you can trace one number in the results table back to its three final validation scores, derive the decay units, identify which parameter groups receive each optimizer, and state why a small-model winner can still fail the transfer test.
