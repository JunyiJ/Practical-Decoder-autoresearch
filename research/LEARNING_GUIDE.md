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
