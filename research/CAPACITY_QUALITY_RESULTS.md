# Dense capacity quality results

The four-layer cooldown model remains the quality incumbent. Both larger models improve every fresh seed at width 64, but neither reaches the required mean gain at width 256, and each worsens seed 2132. Six layers costs 46.94% more training time at width 256; the wider four-layer FFN costs 40.47% more. Their conditional text intervals versus standard four layers include zero at width 256. These findings do not establish a useful quality replacement at the larger tested width.

Capacity study completed and reviewed 54/54. Frozen peak LRs: dense4 0.005 (boundary), wide4 0.005 (boundary), dense6 0.005 (boundary). The frozen screen winner was dense6; its tuned-dense4 CE deltas at widths 64/256 are -0.028180/-0.000674. It fails the complete adoption rule, so the prior incumbent remains. All selected models use final-third cooldown, 3,000 updates and 3,072,000 targets, with different capacity and compute. Held-out test remains closed.

Historical source `ae566ead098f11aeb72f5aa0901930e6c69ec7e8`. Result `kb_6a735434738e5676e68fb3a6b05d2151848534215a085f395900aec63aee9a68`. Every distinct checkpoint/raw request/result and saved configuration, LR schedule, sampler/private RNG, optimizer group/rate/state, layer/head/FFN shape and parameter count, and validation coverage contract was verified. The complete report and every conditional bootstrap interval were independently reconstructed. No training or quality evaluation rerun.

18 screening trials plus 36 distinct confirmation/reference trials. Prospective reference aliases: `{"incumbent_reference": "dense4"}`. Active controller time 4568.61 seconds (1.27 hours); output before review 526.50 MiB. [Frozen protocol](CAPACITY_QUALITY_PROTOCOL.md).

## Scores and frozen choices

Peak LRs were selected on width 64 seeds 2121/2122 before any confirmation. Fresh 2131/2132/2133 confirmation used widths 64/256. Full validation covers 111,538 targets exactly once at context 128. Historical results are not pooled.

| Recipe | Peak LR | Schedule | CE64 | CE256 | Train seconds 64 | Train seconds 256 | Parameters 64 | Parameters 256 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| dense4 | 0.005 | linear_cooldown | 1.521006 | 1.448704 | 54.59 | 125.54 | 204,544 | 3,152,896 |
| wide4 | 0.005 | linear_cooldown | 1.496944 | 1.446780 | 59.84 | 176.34 | 333,568 | 5,217,280 |
| dense6 | 0.005 | linear_cooldown | 1.492826 | 1.448030 | 76.39 | 184.46 | 302,592 | 4,712,448 |
| gelu_reference | 0.002 | constant | 1.678178 | 1.550864 | 31.83 | 65.85 | 108,160 | 1,612,288 |
| shallow_reference | 0.0025 | constant | 1.658799 | 1.542714 | 33.01 | 67.00 | 106,496 | 1,593,344 |
| deep_reference | 0.0025 | constant | 1.600382 | 1.522545 | 54.69 | 125.73 | 204,544 | 3,152,896 |
| incumbent_reference (alias of dense4) | 0.005 | linear_cooldown | 1.521006 | 1.448704 | 54.59 | 125.54 | 204,544 | 3,152,896 |

The selected models share context 128, batch 8, paired_blocks512 sampling, Muon/AdamW muP base 64, head 32 MHA, bias-free SwiGLU, unscaled residuals and independent per-step decay 0.00006. Wider four-layer FFNs use hidden width 21d/4; standard four/six-layer FFNs use 21d/8. The six-layer policy is explicitly depth6_dense_unscaled; algebraic width factors do not establish optimal depth/LR transfer. Cooldown starts at zero-based update 2,000 and ends at 10% of peak. No boundary grid is extended.

## Paired confirmation contrasts

Positive CE delta is worse. Values are fresh seeds 2131/2132/2133. The table shows 4,096-token conditional text intervals; all 2,048/4,096/8,192 intervals are retained and verified. These intervals condition on these models and reused text, not seed-population uncertainty.

| Width | Candidate minus reference | Mean delta | Paired seeds | Conditional 95% text interval |
|---:|---|---:|---|---|
| 64 | wide4 − dense4 | -0.024062 | -0.016369 / -0.023245 / -0.032571 | [-0.027110, -0.020778] |
| 64 | dense6 − dense4 | -0.028180 | -0.022326 / -0.035963 / -0.026250 | [-0.030940, -0.025589] |
| 64 | dense6 − wide4 | -0.004118 | -0.005957 / -0.012718 / +0.006321 | [-0.006803, -0.001577] |
| 64 | dense4 − gelu_reference | -0.157172 | -0.166251 / -0.154908 / -0.150358 | [-0.165705, -0.147581] |
| 64 | dense4 − shallow_reference | -0.137793 | -0.141701 / -0.137396 / -0.134281 | [-0.144198, -0.130426] |
| 64 | dense4 − deep_reference | -0.079376 | -0.076014 / -0.083671 / -0.078443 | [-0.085556, -0.072529] |
| 64 | dense4 − incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] |
| 64 | wide4 − gelu_reference | -0.181234 | -0.182620 / -0.178153 / -0.182929 | [-0.191175, -0.170712] |
| 64 | wide4 − shallow_reference | -0.161854 | -0.158071 / -0.160641 / -0.166852 | [-0.169358, -0.153551] |
| 64 | wide4 − deep_reference | -0.103438 | -0.092383 / -0.106916 / -0.111014 | [-0.110885, -0.095610] |
| 64 | wide4 − incumbent_reference | -0.024062 | -0.016369 / -0.023245 / -0.032571 | [-0.027110, -0.020778] |
| 64 | dense6 − gelu_reference | -0.185352 | -0.188577 / -0.190871 / -0.176608 | [-0.194207, -0.175675] |
| 64 | dense6 − shallow_reference | -0.165972 | -0.164027 / -0.173359 / -0.160531 | [-0.172707, -0.158634] |
| 64 | dense6 − deep_reference | -0.107556 | -0.098340 / -0.119634 / -0.104693 | [-0.113829, -0.100595] |
| 64 | dense6 − incumbent_reference | -0.028180 | -0.022326 / -0.035963 / -0.026250 | [-0.030940, -0.025589] |
| 256 | wide4 − dense4 | -0.001924 | -0.007394 / +0.003725 / -0.002104 | [-0.004138, +0.000242] |
| 256 | dense6 − dense4 | -0.000674 | -0.004638 / +0.011782 / -0.009166 | [-0.003220, +0.002173] |
| 256 | dense6 − wide4 | +0.001250 | +0.002756 / +0.008057 / -0.007062 | [-0.001166, +0.004036] |
| 256 | dense4 − gelu_reference | -0.102160 | -0.105934 / -0.103672 / -0.096872 | [-0.108818, -0.094537] |
| 256 | dense4 − shallow_reference | -0.094009 | -0.091859 / -0.085959 / -0.104209 | [-0.099338, -0.088344] |
| 256 | dense4 − deep_reference | -0.073840 | -0.078524 / -0.073678 / -0.069319 | [-0.077760, -0.069708] |
| 256 | dense4 − incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] |
| 256 | wide4 − gelu_reference | -0.104084 | -0.113328 / -0.099947 / -0.098976 | [-0.111052, -0.096131] |
| 256 | wide4 − shallow_reference | -0.095934 | -0.099253 / -0.082235 / -0.106313 | [-0.101842, -0.089800] |
| 256 | wide4 − deep_reference | -0.075765 | -0.085918 / -0.069953 / -0.071423 | [-0.079295, -0.071976] |
| 256 | wide4 − incumbent_reference | -0.001924 | -0.007394 / +0.003725 / -0.002104 | [-0.004138, +0.000242] |
| 256 | dense6 − gelu_reference | -0.102833 | -0.110572 / -0.091890 / -0.106038 | [-0.110587, -0.093879] |
| 256 | dense6 − shallow_reference | -0.094683 | -0.096497 / -0.074178 / -0.113375 | [-0.100943, -0.087593] |
| 256 | dense6 − deep_reference | -0.074514 | -0.083162 / -0.061896 / -0.078485 | [-0.079112, -0.069282] |
| 256 | dense6 − incumbent_reference | -0.000674 | -0.004638 / +0.011782 / -0.009166 | [-0.003220, +0.002173] |

## Fixed-LR screening contrasts and costs

The screening contrasts hold peak LR at 0.0025 and use 2121/2122, separately from tuned confirmation.

| Candidate minus reference | Mean delta | Seed2121 | Seed2122 |
|---|---:|---:|---:|
| wide4 − dense4 | -0.033795 | -0.030758 | -0.036832 |
| dense6 − dense4 | -0.033945 | -0.026243 | -0.041646 |
| dense6 − wide4 | -0.000150 | +0.004514 | -0.004814 |

| Capacity | Training time change64/256 vs tuned dense4 | Parameter change64/256 |
|---|---|---|
| wide4 | +9.63% / +40.47% | +63.08% / +65.48% |
| dense6 | +39.94% / +46.94% | +47.93% / +49.46% |

## Frozen decision and next research

Only the width 64 screen winner can qualify; confirmation mean gain>.005 and every seed improves at widths64 and256 versus tuned dense4, fixed reviewed incumbent and original deep dense constant reference.

The frozen rule does not support replacement. All narrower findings remain recorded and the previous incumbent is retained.

A separate prospective weight-averaging follow-up is prepared. It will resolve its fixed backbone only from this final reviewed adoption, train eight fresh trajectories and compare raw endpoints with three registered spaced tail averages from each trajectory. It still requires actual trainer/export/resume tests, full project tests, serial MPS correctness/resource gates, clean reviewed source and unique preregistration. It is not a completed-grid extension or a rerun for interpretation. No human-stopped campaign006 resumes.

## Limits

- Fresh development seeds on adaptively reused validation; no holdout or population/seed uncertainty claim. Conditional text intervals are descriptive.
- Depth, hidden width, parameter counts and initialization draws differ. Equal tokens and updates do not mean equal compute or isolate a pure capacity/depth mechanism.
- A common cooldown schedule shape was fixed before this study. Each capacity had equal bounded peak-LR tuning at width 64; tuned comparisons can include different LR choices. Fixed-LR screening contrasts are separate.
- Grid boundary selection does not authorize extension, prove optimal muP/depth transfer, or justify choosing a different winner from width 256.
- Only the frozen screen winner can be adopted. Exact incumbent-reference aliases are the same checkpoints, not independent repeats; other controls are fresh.
- Training time is measured separately from controller/full-validation overhead. Current MPS allocation is not peak memory or inference efficiency.

Evidence: `research_runs/capacity-quality-001/{plan,selection,state,report,review}.json`, per-trial artifacts, completion audit and knowledge snapshots. Human kimi source remains untouched.
