# Schedule quality results

Schedule quality001 completed and reviewed 48/48. Frozen peak LRs: constant 0.0025 (interior), cosine 0.005 (boundary), linear_cooldown 0.005 (boundary). The frozen screen winner was linear_cooldown; its tuned-constant CE deltas at widths 64/256 are -0.071693/-0.075062. It passes the complete mean gain >.005 and every-seed improvement rule versus tuned constant and fixed deep incumbent at both widths, and becomes the most promising observed quality recipe. All models used the same four-layer dense MHA SwiGLU architecture, 3,000 updates and 3,072,000 targets; LR integrals and update/decay balance differ. Held-out test remains closed.

Historical source `e01f7920fc9b3efe3c32bd50a38412ca34f9c1b4`. Result `kb_c6a54b5a4fd0d7ef57510cdad7fb7d5072d26235a795961177913ca3bea3bcbd`. Every distinct checkpoint/raw request/result and saved configuration, LR schedule, sampler/private RNG, optimizer group/rate/state, layer/head/FFN and coverage contract was verified. The complete report and every conditional bootstrap interval were independently reconstructed. No training or quality evaluation rerun.

18 screening trials plus 30 distinct confirmation/reference trials. Prospective reference aliases: `{"deep_reference": "constant"}`. Active controller time 3472.50 seconds (0.96 hours); output before review 348.75 MiB. [Frozen protocol](SCHEDULE_QUALITY_PROTOCOL.md).

## Scores and frozen choices

Peak LRs were selected on width 64 seeds 2101/2102 before any confirmation. Fresh seeds 2111/2112/2113 confirmation used both widths 64/256. The final full validation score covers 111,538 targets exactly once at context 128. No historical pooling.

| Recipe | Peak LR | Schedule | CE64 | CE256 | Train seconds at 64 | Train seconds at 256 |
|---|---:|---|---:|---:|---:|---:|
| constant | 0.0025 | constant | 1.600278 | 1.521010 | 53.79 | 132.92 |
| cosine | 0.005 | cosine | 1.534071 | 1.461622 | 53.84 | 129.68 |
| linear_cooldown | 0.005 | linear_cooldown | 1.528585 | 1.445948 | 53.62 | 131.57 |
| gelu_reference | 0.002 | constant | 1.679492 | 1.549335 | 31.67 | 69.38 |
| shallow_reference | 0.0025 | constant | 1.666460 | 1.541795 | 32.82 | 70.56 |
| deep_reference (alias of constant) | 0.0025 | constant | 1.600278 | 1.521010 | 53.79 | 132.92 |

Cosine and cooldown finish at 10% of their selected peak; constant remains at peak. No warmup or restarts. Independent decay stays at .00006 each step; LR integral and relative gradient/shrinkage balance differ. No boundary grid is extended.

Calculated base-LR sums across the 3,000 updates are 7.50 for selected constant LR, 8.25 for cosine, and 12.75 for cooldown. Thus the selected cooldown has 1.7 times the constant control’s LR sum despite its lower final LR. Group width factors apply separately. These calculations are not effective update norms, measured compute, or an attribution of the quality gain to LR integral alone.

## Paired confirmation contrasts

Positive CE delta is worse. Values are fresh seeds 2111/2112/2113. The table shows 4096-token conditional text intervals; all 2048/4096/8192 intervals are retained and verified in the report. These are conditional on these trained models and reused text, not seed-population uncertainty.

| Width | Candidate minus reference | Mean delta | Paired seeds | Conditional 95% text interval |
|---:|---|---:|---|---|
| 64 | cosine − constant | -0.066207 | -0.060770 / -0.068823 / -0.069027 | [-0.073244, -0.058380] |
| 64 | linear_cooldown − constant | -0.071693 | -0.066185 / -0.070235 / -0.078659 | [-0.078541, -0.064379] |
| 64 | linear_cooldown − cosine | -0.005486 | -0.005415 / -0.001412 / -0.009632 | [-0.008115, -0.002978] |
| 64 | constant − gelu_reference | -0.079214 | -0.084740 / -0.068136 / -0.084766 | [-0.084174, -0.074273] |
| 64 | constant − shallow_reference | -0.066182 | -0.066704 / -0.065052 / -0.066791 | [-0.070402, -0.061862] |
| 64 | constant − deep_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] |
| 64 | cosine − gelu_reference | -0.145421 | -0.145510 / -0.136959 / -0.153793 | [-0.155008, -0.135239] |
| 64 | cosine − shallow_reference | -0.132389 | -0.127474 / -0.133875 / -0.135818 | [-0.140665, -0.123878] |
| 64 | cosine − deep_reference | -0.066207 | -0.060770 / -0.068823 / -0.069027 | [-0.073244, -0.058380] |
| 64 | linear_cooldown − gelu_reference | -0.150907 | -0.150925 / -0.138371 / -0.163425 | [-0.160299, -0.141234] |
| 64 | linear_cooldown − shallow_reference | -0.137875 | -0.132889 / -0.135287 / -0.145450 | [-0.145764, -0.130079] |
| 64 | linear_cooldown − deep_reference | -0.071693 | -0.066185 / -0.070235 / -0.078659 | [-0.078541, -0.064379] |
| 256 | cosine − constant | -0.059388 | -0.064366 / -0.055773 / -0.058025 | [-0.064326, -0.053889] |
| 256 | linear_cooldown − constant | -0.075062 | -0.076132 / -0.075016 / -0.074038 | [-0.079653, -0.069646] |
| 256 | linear_cooldown − cosine | -0.015674 | -0.011766 / -0.019243 / -0.016013 | [-0.017956, -0.013318] |
| 256 | constant − gelu_reference | -0.028325 | -0.026053 / -0.022822 / -0.036100 | [-0.032298, -0.023998] |
| 256 | constant − shallow_reference | -0.020785 | -0.016559 / -0.015825 / -0.029972 | [-0.023066, -0.018217] |
| 256 | constant − deep_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] |
| 256 | cosine − gelu_reference | -0.087713 | -0.090419 / -0.078595 / -0.094125 | [-0.095596, -0.079049] |
| 256 | cosine − shallow_reference | -0.080173 | -0.080925 / -0.071598 / -0.087997 | [-0.085395, -0.073902] |
| 256 | cosine − deep_reference | -0.059388 | -0.064366 / -0.055773 / -0.058025 | [-0.064326, -0.053889] |
| 256 | linear_cooldown − gelu_reference | -0.103387 | -0.102185 / -0.097839 / -0.110138 | [-0.110684, -0.095029] |
| 256 | linear_cooldown − shallow_reference | -0.095847 | -0.092691 / -0.090841 / -0.104010 | [-0.100820, -0.090158] |
| 256 | linear_cooldown − deep_reference | -0.075062 | -0.076132 / -0.075016 / -0.074038 | [-0.079653, -0.069646] |

## Fixed-LR screens and timing

These screening contrasts hold both peak LRs at .0025 and use seeds 2101/2102. They are distinct from tuned confirmation contrasts.

| Candidate minus reference | Mean delta | Seed2101 | Seed2102 |
|---|---:|---:|---:|
| cosine − constant | -0.041341 | -0.044642 | -0.038039 |
| linear_cooldown − constant | -0.062827 | -0.066455 | -0.059200 |
| linear_cooldown − cosine | -0.021487 | -0.021813 | -0.021160 |

| Schedule | Time change at 64 vs tuned constant | Time change at 256 vs tuned constant |
|---|---:|---:|
| cosine | +0.10% | -2.44% |
| linear_cooldown | -0.32% | -1.02% |

## Frozen decision and next research

Only the width 64 screen winner can qualify; confirmation mean gain >.005 and every seed improves at widths 64 and 256 versus tuned constant, fixed incumbent and fixed deep dense reference.

The audited frozen winner qualifies and is recorded as the current observed quality incumbent. The prior constant-LR recipe is archived and retained as a reference.

A separate prospective dense-capacity follow-up is prepared: standard four-layer, wider-FFN four-layer and standard-FFN six-layer MHA, with a common schedule resolved from this final audited decision. It still requires actual model/scaling tests, serial MPS correctness/resource gates, clean reviewed source and unique preregistration. No completed study or human-stopped campaign 006 resumes.

## Limits

- Fresh development seeds on adaptively reused validation; no holdout or population/seed uncertainty claim. Conditional text intervals are descriptive.
- The architecture, token budget, update count and independent per-step decay are fixed; LR integrals and relative update/shrinkage balance differ. This does not identify a pure schedule-shape effect at equal LR integral.
- Each schedule had equal bounded peak-LR tuning at width 64. Boundary selection does not authorize grid extension or demonstrate optimal muP transfer.
- Only the frozen screen winner is eligible for adoption; no larger-width reselection or historical pooling.
- Exact deep-reference aliases, if present, are the same checkpoints, not independent repeats. Other fixed references are fresh.
- Training time is measured separately from controller/full-validation overhead; current MPS allocation is not peak memory or inference efficiency.

Evidence: `research_runs/schedule-quality-001/{plan,selection,state,report,review}.json`, all per-trial artifacts, completion audit and knowledge snapshots. Human kimi source remains untouched.
