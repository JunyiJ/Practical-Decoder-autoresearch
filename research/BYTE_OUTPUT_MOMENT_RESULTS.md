# Epsilon already matters strongly for unseen output rows

The preselected incumbent RAW checkpoint passed all identity, parameter-binding,
finite-value, step and arithmetic checks. At step1200, epsilon1e-6 contributes
most of the output Adam denominator for tokens absent from training. Its share
is much smaller for frequently observed tokens. This is an endpoint numerical
measurement, not a reason to increase or decrease epsilon.

Here s=sqrt(v/(1−beta2^1200)), with beta2=.999 and bias correction
.6989865709066009. Epsilon share means epsilon/(s+epsilon), averaged over
coordinates. A share above50% means epsilon exceeds s.

| Training occurrences | Types | Median s/epsilon | Mean epsilon share | Coordinates with s<=epsilon | Validation-weighted mean share |
|---|---:|---:|---:|---:|---:|
| 0 | 39604 | .283245 | 77.8974% | 99.9915% | 78.0888% |
| 1–9 | 8329 | 15.820648 | 8.2081% | .8810% | 5.8188% |
| 10–99 | 1982 | 58.537282 | 1.8215% | 0% | 1.5101% |
| 100–999 | 309 | 183.616869 | .5823% | 0% | .4793% |
| 1000+ | 33 | 610.191802 | .1672% | 0% | .1069% |

Validation weighting weights each token's coordinate-mean share by its number
of validation targets; it does not use predictions or validation losses. All
12865792 coordinates and34278 validation targets are accounted for. The selected
RAW file is unchanged, as are CPU RNG and all model parameters.

The negligible-epsilon approximation is poor for the unseen group in this
checkpoint. For the1–9 group epsilon usually contributes less than the root;
both patterns coexist, so treating all rare/unseen rows as one optimizer regime
would be misleading. This does not measure the entire trajectory, other seeds,
the H200 average, or rejected residual models. Nor does it establish the
direction of a quality-optimal change. The earlier native epsilon adoption and
all closed quality decisions remain intact. No optimizer intervention is admitted.

Evidence: `tokenizer-evidence/byte-output-moment-001/{attempt,guard,result}.json`;
design and paper scope: [protocol](BYTE_OUTPUT_MOMENT_PROTOCOL.md).
Execution used2.396644seconds, peak RSS1104019456bytes, with11107213312 free
bytes at completion. No training, model construction, new loss, holdout access,
checkpoint output or parameter update occurred.

Next direction: [output lexical-sharing evidence review](BYTE_OUTPUT_LEXICAL_REVIEW.md).
The next useful question is whether a literature-supported output composition
can be made feasible under full-vocabulary training, while distinguishing lexical
sharing from capacity and optimizer effects. This requires a cost and comparison
design before source integration or new numerical training.
