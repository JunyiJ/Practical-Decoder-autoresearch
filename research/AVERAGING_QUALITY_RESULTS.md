# Spaced-tail averaging quality results

The frozen 200-update half-life average improved every fresh confirmation seed at both widths and passed every prospective quality gate. Mean validation CE changed by −0.005398 at width 64 and −0.016797 at width 256. It becomes the current observed returned-weight incumbent; the online training recipe remains the four-layer cooldown model.

Eight physical training trajectories produced 32 correlated model outputs. All 8 raw and 24 derived checkpoints, requests/results, saved optimizer and sampler state, schedule and snapshot counters, model shapes and full validation coverage were audited. The complete report and every paired conditional bootstrap interval were independently reconstructed. No training or scoring was rerun, and the held-out test remains closed.

Historical source: `0f15366dc43b5e2a2805ca08bb1eeac25f6b1e14`. Result: `kb_297e2dca399a3a438b7205016233e75d94f8a28b702ed6d8012ea24f9d227135`. Active controller time was 717.76 seconds (11.96 minutes); output occupied 324.71 MiB before review. [Frozen protocol](AVERAGING_QUALITY_PROTOCOL.md).

## Fresh confirmation scores

Means use fresh development seeds 2151, 2152 and 2153. Each final score covers 111,538 validation targets once, at context 128 and batch 8. All four outputs from a seed share one online training trajectory.

| Returned weights | CE at width 64 | CE at width 256 |
|---|---:|---:|
| Raw endpoint | 1.523733 | 1.450487 |
| EMA half-life 10 | 1.523188 | 1.449914 |
| EMA half-life 50 | 1.519801 | 1.445310 |
| EMA half-life 200 (frozen choice) | 1.518335 | 1.433690 |

## Paired confirmation contrasts

Negative CE differences are improvements. Seed columns are 2151 / 2152 / 2153. The table shows 4,096-token conditional text intervals; all 2,048 / 4,096 / 8,192-token intervals were checked. These intervals condition on the trained models and reused validation text; they are not population uncertainty across training seeds.

| Width | Half-life | Mean difference vs raw | Three paired seeds | Conditional 95% text interval | Passes quality rule |
|---:|---:|---:|---|---|---|
| 64 | 10 | -0.000545 | -0.000796 / -0.000639 / -0.000198 | [-0.000689, -0.000393] | No |
| 64 | 50 | -0.003931 | -0.006131 / -0.003725 / -0.001937 | [-0.004357, -0.003486] | Yes |
| 64 | 200 | -0.005398 | -0.006432 / -0.005709 / -0.004052 | [-0.006664, -0.004154] | Yes |
| 256 | 10 | -0.000573 | -0.000462 / -0.000591 / -0.000665 | [-0.000694, -0.000456] | No |
| 256 | 50 | -0.005177 | -0.006448 / -0.005347 / -0.003735 | [-0.005504, -0.004869] | Yes |
| 256 | 200 | -0.016797 | -0.018360 / -0.016356 / -0.015675 | [-0.018171, -0.015522] | Yes |

The two width-64 screening seeds, 2141 and 2142, froze half-life 200 before any confirmation results. Both improved, with a mean gain of 0.006191. Adoption required that locally eligible choice to improve mean CE by more than 0.002, improve every confirmation seed, and have every conditional text interval upper bound below zero at both widths. Half-life 50 also passes the descriptive confirmation gate, but it was not the frozen choice and is not promoted. Half-life 10 improves every seed by less than the required mean gain.

All models use 3,000 updates and 3,072,000 training targets: four-layer MHA, head dimension 32, bias-free SwiGLU hidden width 21d/8, Muon/AdamW with muP base width 64, clipping 1, zero dropout, unscaled residuals and paired_blocks512 sampling. Peak LR 0.005 stays constant until the final-third cooldown and finishes at 0.0005. Independent per-step decay remains 0.00006.

Each average initializes from online weights at completed update 2,000, then updates every 10 completed updates through 3,000: 101 snapshots. The recurrence coefficient is `2^(-10 / half_life)`. All learned parameters are averaged; persistent nonlearned buffers are copied from the raw endpoint. Averages never feed back into gradients or the optimizer.

## Shared costs and verification limits

| Width | Shared training seconds | Shared trajectory wall seconds | Mean sampled process RSS (MiB) |
|---:|---:|---:|---:|
| 64 | 54.84 | 60.16 | 474.27 |
| 256 | 125.78 | 138.64 | 482.48 |

These costs belong to a trajectory and all four returned outputs. Tracking, extra evaluation and checkpoint storage have overhead; this study has no independent raw-only production timing baseline. It does not establish a training or inference speedup. Sampled/current allocations do not establish peak accelerator memory.

The first MPS prerequisite failed independent-run weight agreement and remains preserved. Raw-only repeats also differed without averaging, including with deterministic algorithms enabled. Before research registration, the protocol replaced that unsuitable arithmetic oracle with direct checks: shadow updates leave online weights, optimizer state and CPU/MPS RNG unchanged; an independent CPU float64 recurrence checks the MPS averages; and saved-state continuation uses shared realized gradients to isolate averaging from differences in backward computation. Those checks passed under the original numerical tolerance. Six full device pilots then verified training, saving/loading, private RNG, derived payloads, 18 coverage outputs and resource limits. Their independent-run differences remain reported. No specific kernel or historical cause has been identified, and full independent MPS reproducibility is not established.

Every derived tensor exactly matches its registered final shadow state in the raw parent. Final checkpoints alone cannot reconstruct all 101 historical online snapshots; recurrence correctness also relies on synthetic tests and the reviewed frozen trainer. Correctness pilots are excluded from quality findings.

## Decision and continuing research

Use the frozen half-life 200 endpoint as the best supported returned model at the tested widths. Preserve raw parents for optimizer continuation: averaged inference checkpoints carry no online optimizer or resumable training state. The original constant deep model, GELU and shallow references remain preserved. This is an adaptive development result, not a universal optimum or held-out confirmation.

A distinct finite follow-up is being prepared to compare 3,000 updates with 6,000 updates under stretched or fixed-duration cooldown. It will use this final reviewed returned-weight policy, fresh controls and seeds, and its own source, scaling, testing, serial MPS and resource checks before preregistration. Additional tokens, cumulative decay and compute will be explicit. No completed study or permanently stopped campaign 006 resumes.

Evidence: `research_runs/averaging-quality-001/{plan,selection,state,report,review}.json`, per-trajectory raw and derived artifacts, `research_state/averaging-conclusion.json`, prerequisite failure/repair files and full knowledge snapshots.
