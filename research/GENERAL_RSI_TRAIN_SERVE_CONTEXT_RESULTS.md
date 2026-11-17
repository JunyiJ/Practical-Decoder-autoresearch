# Training context versus serving context: completed diagnostic

**Longer training with shorter serving wins these comparisons.** At both depths,
context256-trained models served at rolling128 beat context128-trained models
served at rolling256 or rolling512. All three paired bundles agree, and all six
priority comparisons exclude zero under the frozen 36-contrast correction.
Longer serving alone does not recover longer-training performance here.

## Matched six-cell comparison

CE in nats per target token. Lower is better. Each entry averages three saved
time-EMA models. Every cell scores exactly the same 51,457 validation targets
from 65 documents, with at least 512 preceding tokens. No new training occurred.

| Depth | Training context | Rolling128 | Rolling256 | Rolling512 |
| --- | --- | ---: | ---: | ---: |
| 4 | 128 | 4.731438 | 5.030463 | 5.423952 |
| 4 | 256 | 4.674651 | 4.650411 | 4.980751 |
| 16 | 128 | 4.685386 | 4.849568 | 5.235217 |
| 16 | 256 | 4.641043 | 4.612930 | 4.841767 |

- Training at 256 improves rolling128 CE by 0.056787 at depth4 and 0.044343 at depth16.
- For trained256 models, serving256 instead of128 improves CE by 0.024240 and 0.028114.
- For trained128 models, serving256 instead of128 worsens CE by 0.299025 and 0.164181.
- Extending serving256 to512 worsens CE for both training lengths at both depths.
  These serving effects have the same sign in all three bundles and adjusted intervals excluding zero.

## Priority policy comparisons

Delta is the first policy minus the second. Negative favors longer training with
shorter serving. Intervals are from the frozen family, not unadjusted comparisons.

| First policy minus second | Depth4 delta [interval] | Depth16 delta [interval] |
| --- | --- | --- |
| Train256/serve128 − train128/serve256 | -0.355812 [-0.394554, -0.316169] | -0.208524 [-0.257745, -0.169172] |
| Train256/serve128 − train128/serve512 | -0.749300 [-0.808896, -0.673904] | -0.594173 [-0.663730, -0.521691] |
| Train256/serve256 − train128/serve512 | -0.773541 [-0.835281, -0.697593] | -0.622287 [-0.692975, -0.549067] |

## Previous training results alongside this diagnostic

These rows preserve their original target sets and uncertainty rules. Compare
directions and designs, not the numerical magnitudes across different subsets.

| Comparison | Target set | Depth4 CE delta | Depth16 CE delta |
| --- | --- | ---: | ---: |
| Previous native evaluation: train256/reset256 − train128/reset128 | Original 100,416 targets | -0.096031 | -0.089993 |
| Previous common evaluation: train256/reset128 − train128/reset128 | Original 100,416 targets | -0.021068 | -0.012125 |
| Current common evaluation: train256/rolling128 − train128/rolling128 | Matched 51,457 targets | -0.056787 | -0.044343 |
| Current fixed train256: rolling256 − rolling128 | Matched 51,457 targets | -0.024240 | -0.028114 |

The original native primary intervals remain [-0.111309, -0.081081] and
[-0.106336, -0.074394]. Its depth interaction remains unresolved. The common128
rows were descriptive secondary analyses. See [original training report](GENERAL_RSI_DEPTH_CONTEXT_QUALITY_RESULTS.md).
Original B/C and four-condition intervals remain unchanged in their own families.
See [four-condition report](GENERAL_RSI_ROLLING_FOUR_RESULTS.md).

## Interpretation and limits

Rolling history h supplies the immediately preceding h tokens for every target.
Reset evaluation divides a document into segments. At the start of each segment,
available history drops and then grows. This diagnostic uses rolling histories
only. It does not introduce reset256 or reset512 arms.

Inference at 512 is executable through the original unscaled RoPE formula and
extended deterministic buffers. It is extrapolation beyond both training lengths.
The larger errors are measured quality results after numerical qualification,
not implementation failures. The results are compatible with a failure to
generalize to longer histories and unseen relative positions. This experiment
does not separately identify positional extrapolation, attention dilution, or
other mechanisms. It does not test interpolation, fine-tuning, or retraining at512.

The training comparison inherits equal four-hour training budgets. Token exposure,
optimizer trajectory, schedule and clipping differ. Thus the gain belongs to the
completed training recipe; it is not an isolated causal estimate of context length.
These are reused development documents and fixed models. Document bootstrap
intervals are conditional on the saved bundles, not population seed uncertainty.
Two depths provide replication evidence, not a scaling law or a reasoning claim.

Updated hypothesis: longer-context training retains an advantage under short
serving, but increasing serving context beyond the training range does not
substitute for that training under this direct positional policy. No new campaign
is implied. Next action: await human direction.

## Verification and cost

Independent CPU review checked all 12 endpoint hashes, 5,309 bound source/evidence
hashes, 648 original reuse hashes, and 780 document score files. Every position,
target ID and reused loss matched. An independent contrast matrix and frequency
bootstrap reproduced all 100,000 draws and 36 interval pairs to within
4.45e-16. All pair-sign decisions and per-source means matched.

Four qualification shapes passed native, batching, full/scalar logits, float64 CE
and causal-prefix controls. Original buffer prefixes, restoration and unchanged
weights passed. The review confirmed identical inference/window source ASTs
across the resource repair. The first qualification failure and its reference-
lifetime repair remain separately preserved. No scientific retry was performed.

Scoring completed in 15,328.30 guard seconds (4.258 hours), below the admitted
25,004.72-second forecast. Analysis took 2.66 guard seconds. New inference covered
24 cells and 434,708,736 input-token presentations; 12 cells reused exact scores.
These are uncached stride-one measurements, not optimized serving latency.

Peak scoring RSS was 641,613,824 bytes; MPS driver memory was 1,152,073,728 bytes.
Minimum observed host headroom proxy was 2,699,264,000 bytes. Pressure stayed at
or below2 and every successful pre-load admission had pressure1 and at least
1.25GiB proxy. Minimum observed scoring free space was 17,520,197,632 bytes.
All RSS4GiB/MPS6GiB and disk12GiB guards passed. Evidence used about62MB of the
512MiB allowance. No cleanup was required. No original-test access, generation,
weight changes or training updates occurred.

Evidence: [analysis](tokenizer-evidence/general-rsi-train-serve-context-001/analysis.json),
[independent review](tokenizer-evidence/general-rsi-train-serve-context-001/completion-review.json),
[frozen design](GENERAL_RSI_TRAIN_SERVE_CONTEXT_DESIGN.md).

## Full frozen contrast family

100,000 shared paired document draws over65 documents, PCG64 seed621117310001.
Average bundle loss sums before resampling; divide by actual sampled token counts.
Family alpha0.01 over36 contrasts gives quantiles0.01/72 and1−0.01/72.
“Consistent” requires all three bundle signs plus interval exclusion of zero.
An interval alone does not override a mixed pair pattern.

| Group | Contrast | Mean | Adjusted interval | Three paired deltas | Consistent direction |
| --- | --- | ---: | --- | --- | --- |
| depth4 | train256_minus128_at_h128 | -0.056787 | [-0.081669, -0.036676] | -0.066759, -0.043569, -0.060033 | negative |
| depth4 | train256_minus128_at_h256 | -0.380052 | [-0.421173, -0.339860] | -0.425906, -0.281971, -0.432279 | negative |
| depth4 | train256_minus128_at_h512 | -0.443200 | [-0.485955, -0.402225] | -0.355843, -0.444029, -0.529730 | negative |
| depth4 | serve256_minus128_t128 | +0.299025 | [+0.269952, +0.324837] | +0.334745, +0.216056, +0.346274 | positive |
| depth4 | serve512_minus256_t128 | +0.393489 | [+0.351099, +0.422636] | +0.332338, +0.480880, +0.367247 | positive |
| depth4 | serve256_minus128_t256 | -0.024240 | [-0.033663, -0.015254] | -0.024402, -0.022347, -0.025972 | negative |
| depth4 | serve512_minus256_t256 | +0.330340 | [+0.286502, +0.378204] | +0.402401, +0.318823, +0.269797 | positive |
| depth4 | train_by_serve_128_to_256 | -0.323265 | [-0.350809, -0.295728] | -0.359147, -0.238403, -0.372246 | negative |
| depth4 | train_by_serve_256_to_512 | -0.063148 | [-0.091900, -0.026738] | +0.070063, -0.162057, -0.097451 | unresolved |
| depth4 | policy_t256_h128_minus_t128_h256 | -0.355812 | [-0.394554, -0.316169] | -0.401503, -0.259625, -0.406307 | negative |
| depth4 | policy_t256_h128_minus_t128_h512 | -0.749300 | [-0.808896, -0.673904] | -0.733842, -0.740505, -0.773555 | negative |
| depth4 | policy_t256_h256_minus_t128_h512 | -0.773541 | [-0.835281, -0.697593] | -0.758244, -0.762851, -0.799527 | negative |
| depth16 | train256_minus128_at_h128 | -0.044343 | [-0.073105, -0.022222] | -0.041352, -0.039050, -0.052626 | negative |
| depth16 | train256_minus128_at_h256 | -0.236638 | [-0.281644, -0.198164] | -0.260938, -0.215390, -0.233586 | negative |
| depth16 | train256_minus128_at_h512 | -0.393450 | [-0.434536, -0.350415] | -0.429961, -0.387247, -0.363141 | negative |
| depth16 | serve256_minus128_t128 | +0.164181 | [+0.138885, +0.188329] | +0.189963, +0.151393, +0.151188 | positive |
| depth16 | serve512_minus256_t128 | +0.385649 | [+0.347403, +0.414528] | +0.353005, +0.465973, +0.337971 | positive |
| depth16 | serve256_minus128_t256 | -0.028114 | [-0.039008, -0.018150] | -0.029622, -0.024948, -0.029772 | negative |
| depth16 | serve512_minus256_t256 | +0.228838 | [+0.190903, +0.266165] | +0.183982, +0.294117, +0.208415 | positive |
| depth16 | train_by_serve_128_to_256 | -0.192295 | [-0.216512, -0.169170] | -0.219586, -0.176341, -0.180959 | negative |
| depth16 | train_by_serve_256_to_512 | -0.156812 | [-0.177453, -0.131850] | -0.169023, -0.171857, -0.129555 | negative |
| depth16 | policy_t256_h128_minus_t128_h256 | -0.208524 | [-0.257745, -0.169172] | -0.231316, -0.190442, -0.203814 | negative |
| depth16 | policy_t256_h128_minus_t128_h512 | -0.594173 | [-0.663730, -0.521691] | -0.584320, -0.656416, -0.541784 | negative |
| depth16 | policy_t256_h256_minus_t128_h512 | -0.622287 | [-0.692975, -0.549067] | -0.613943, -0.681363, -0.571556 | negative |
| depth16_minus4 | train256_minus128_at_h128 | +0.012444 | [-0.014238, +0.035178] | +0.025406, +0.004519, +0.007407 | unresolved |
| depth16_minus4 | train256_minus128_at_h256 | +0.143414 | [+0.113645, +0.168154] | +0.164968, +0.066581, +0.198694 | positive |
| depth16_minus4 | train256_minus128_at_h512 | +0.049751 | [+0.013636, +0.082825] | -0.074118, +0.056782, +0.166589 | unresolved |
| depth16_minus4 | serve256_minus128_t128 | -0.134844 | [-0.145861, -0.123693] | -0.144781, -0.064663, -0.195087 | negative |
| depth16_minus4 | serve512_minus256_t128 | -0.007839 | [-0.023745, +0.006649] | +0.020666, -0.014907, -0.029277 | unresolved |
| depth16_minus4 | serve256_minus128_t256 | -0.003874 | [-0.007431, -0.000837] | -0.005220, -0.002601, -0.003800 | negative |
| depth16_minus4 | serve512_minus256_t256 | -0.101502 | [-0.117905, -0.085448] | -0.218419, -0.024706, -0.061382 | negative |
| depth16_minus4 | train_by_serve_128_to_256 | +0.130970 | [+0.120191, +0.141996] | +0.139561, +0.062062, +0.191287 | positive |
| depth16_minus4 | train_by_serve_256_to_512 | -0.093663 | [-0.116174, -0.071428] | -0.239086, -0.009799, -0.032105 | negative |
| depth16_minus4 | policy_t256_h128_minus_t128_h256 | +0.147288 | [+0.115288, +0.173269] | +0.170188, +0.069182, +0.202493 | positive |
| depth16_minus4 | policy_t256_h128_minus_t128_h512 | +0.155127 | [+0.121175, +0.184810] | +0.149521, +0.084089, +0.231770 | positive |
| depth16_minus4 | policy_t256_h256_minus_t128_h512 | +0.151253 | [+0.119326, +0.179315] | +0.144301, +0.081488, +0.227971 | positive |

The depth4 training-by-serving 256→512 interaction has a negative adjusted
interval but mixed pair signs. The depth difference in the training effect at512
also has mixed signs. Neither meets the frozen consistency rule. The depth
difference in the training effect at128 and in the trained128 256→512 serving
increment has an interval crossing zero. These unresolved results are retained.

## Per-bundle and source means

| Depth | Bundle | Training | All h128 / h256 / h512 | FineWeb h128 / h256 / h512 | Wikipedia h128 / h256 / h512 | New scoring seconds h128 / h256 / h512 |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 1 | 128 | 4.735015 / 5.069760 / 5.402098 | 4.686362 / 5.026449 / 5.350728 | 4.940118 / 5.252343 / 5.618655 | 166.65 / 266.95 / 490.69 |
| 4 | 1 | 256 | 4.668257 / 4.643854 / 5.046256 | 4.626669 / 4.604202 / 5.006557 | 4.843575 / 4.811013 / 5.213607 | 0.00 / 0.00 / 494.25 |
| 4 | 2 | 128 | 4.742188 / 4.958243 / 5.439123 | 4.697446 / 4.910701 / 5.386145 | 4.930800 / 5.158665 / 5.662457 | 164.32 / 263.25 / 492.03 |
| 4 | 2 | 256 | 4.698619 / 4.676272 / 4.995095 | 4.653406 / 4.632596 / 4.950828 | 4.889216 / 4.860392 / 5.181705 | 0.00 / 0.00 / 494.76 |
| 4 | 3 | 128 | 4.717111 / 5.063385 / 5.430633 | 4.669462 / 5.020265 / 5.379244 | 4.917979 / 5.245162 / 5.647266 | 164.67 / 263.66 / 492.73 |
| 4 | 3 | 256 | 4.657078 / 4.631106 / 4.900903 | 4.613246 / 4.589775 / 4.857746 | 4.841856 / 4.805343 / 5.082833 | 0.00 / 0.00 / 493.54 |
| 16 | 1 | 128 | 4.676834 / 4.866798 / 5.219802 | 4.632322 / 4.823970 / 5.171252 | 4.864478 / 5.047342 / 5.424467 | 385.48 / 523.30 / 1386.75 |
| 16 | 1 | 256 | 4.635482 / 4.605859 / 4.789841 | 4.598606 / 4.571475 / 4.756122 | 4.790933 / 4.750810 / 4.931986 | 0.00 / 0.00 / 1383.12 |
| 16 | 2 | 128 | 4.696512 / 4.847905 / 5.313878 | 4.656195 / 4.807115 / 5.272407 | 4.866470 / 5.019860 / 5.488704 | 385.67 / 524.06 / 1382.09 |
| 16 | 2 | 256 | 4.657462 / 4.632515 / 4.926631 | 4.612700 / 4.590356 / 4.877721 | 4.846162 / 4.810239 / 5.132818 | 0.00 / 0.00 / 1386.56 |
| 16 | 3 | 128 | 4.682813 / 4.834000 / 5.171971 | 4.643358 / 4.799175 / 5.133154 | 4.849138 / 4.980811 / 5.335607 | 382.10 / 525.22 / 1386.59 |
| 16 | 3 | 256 | 4.630186 / 4.600415 / 4.808830 | 4.588376 / 4.560256 / 4.760112 | 4.806440 / 4.769709 / 5.014202 | 0.00 / 0.00 / 1388.18 |

Source counts are41,591 FineWeb and9,866 Wikipedia targets in every cell.
Zero new scoring seconds denotes exact reuse, not free production inference.
