# Residual readout: implementation verified, fresh quality panel running

The new zero-gated residual readout passed its implementation pilot. Eight of nine fresh quality runs have completed and passed numerical
acceptance. All three matched nonlinear/byte pairs fail the mandatory candidate-improvement
criterion. The final third-seed linear control is active; the frozen remaining runs and
independent final review continue. No model is adopted. The selected byte model and its corrected public
inference recipe remain unchanged.

The pilot used three arms on both CPU and MPS, four updates each (24 total).
All initial common weights, baseline logits and common-parameter gradients
matched exactly. The zero gate made A gradients exactly zero on the first
update, then both gates and A learned nonzero values. Fixed analytic
forward/input/A/gate checks had zero recorded error. All parameters participated
in the intended optimizer groups and three H200 snapshots. RAW and H200 payloads
round-tripped exactly; maximum CPU/MPS per-target CE error was1.90735e-6 against
a3e-4 limit. No validation or holdout was scored, and no pilot weights were kept.

The check took55.323seconds, with peak RSS3,185,377,280bytes. Recorded final
pilot evidence occupied845,304bytes before publication. These checks establish
implementation behavior, not a quality or deployment-speed gain. Four pilot
updates cannot predict the1,200-update comparison.

The scientific question is whether preserving the original feature z and adding
`alpha*tanh(Az)` avoids the earlier replacement constraint. Controls are original
byte z and the matched linear residual `z+alpha*Az`. Alpha starts0. A is256x256;
the new scalar and matrix add65,537parameters, for16,417,396 total versus the
baseline16,351,859. All three fresh seeds, the.0025 learning rate,1200 updates,
H200 endpoint and >.02 mean improvement plus every-seed/uncertainty gates are
fixed in advance. The paper's deep-model convergence gains are not local gains;
see the protocol for contrary results and applicability limits.

Measured payloads were65,439,277bytes (byte) and65,701,993bytes (either residual).
The complete new retained-output allowance is1,323,504,227bytes, covering18
checkpoints with padded metadata, one overlapping serialization,64MiB reports
and8MiB pilot evidence. It fits with8GiB reserve and2GiB extra headroom. The
freeze rechecked12,149,395,456bytes free against12,060,922,467 required. No
capacity cleanup, smaller confirmation panel or changed criterion was needed.

The panel's expected runtime is roughly105minutes from the previous complete
panel; the conservative forecast is19,837.524seconds including pilot/review,
under a36,000second total ceiling and3600seconds per trajectory. The controller
performs all nine fresh runs then independent saved-result review. A successful
quality result still requires a qualified consumer before adoption.

- Protocol: `BYTE_RESIDUAL_READOUT_PROTOCOL.md`.
- Pilot: `tokenizer-evidence/byte-residual-readout-001/pilot/acceptance.json`.
- Pilot plan SHA256: `b885bef87a933476f241fc57bc2b16444a299a947fde81ddea6649e59c5b126f`.
- Pilot result SHA256: `81ee3e5c1b52778947c9d82fe2103d87cc0f69da5ecfbcb6829756fb93503bfa`.
- Complete cost: `tokenizer-evidence/byte-residual-readout-001/resource-forecast.json`.
- Admission: `tokenizer-evidence/byte-residual-readout-001/quality-admission.md`.
- Quality plan SHA256: `5a5e11ee79335ff63cfe6761378cb6fc46fa7d4eacd3df6bf7bddb82ad3615a6`.
- Quality outputs: `tokenizer-evidence/byte-residual-readout-001/quality001/`.

Observe the existing controller handle; do not restart it or replay completed
slots. If an observation times out, inspect the same handle/process. Preserve
failed evidence and recover only demonstrated execution defects, with the
scientific cases fixed. The controller remains PID1329 (exec session64971). The last observed active
worker was slot08 residual-linear on the third seed, PID33655;
live logs supersede this snapshot. Slots00–07 are terminal and must not be replayed.

## First completed control (one of nine)

Fresh bundle62102600000 byte baseline completed1200 updates and41 H200 snapshots
in681.732seconds of guarded execution. H200 CE is4.585406921383447 across all
34,278 validation targets. Mean CPU/MPS CE difference is3.903656e-8; maximum
per-target difference2.789497e-5, passing3e-5/3e-4 limits. RAW/H200 checkpoints
were exactly read back and their hashes rechecked for publication. Observed
terminal sampler state and unchanged global RNG are retained. Peak RSS was
820,772,864bytes; all guards finished normally.

The last online CE was4.630511; this is diagnostic and does not replace the
predeclared H200 endpoint. This single control cannot establish a candidate
advantage, and it is not pooled with older baseline scores. All nine fresh runs
and the automatic independent panel review remain required.

## First linear-control result (two of nine complete)

On the same fresh bundle62102600000, residual-linear H200 CE is4.720630601738156,
+.13522368035470933 worse than byte. Its1200-update run and41 H200 snapshots
completed in664.423seconds, with mean/max CPU/MPS differences1.13525e-8/2.67029e-5.
Both checkpoints passed exact readback and publication hash checks. The numerical
acceptance is a correctness result; the positive loss difference is unfavorable
quality evidence, not an implementation failure.

Matched late-training analysis (steps1101..1200) finds mean sampled CE3.072804
for residual-linear versus3.243986 for byte, a-.171182 difference. Despite fitting
these paired training batches better, its final online validation CE is4.835059
versus4.630511 (+.204548), and its H200 score is also worse. Thus inadequate late
training fit is not a sufficient explanation for this one-seed loss. This does
not identify a causal mechanism or establish a population-seed effect.

The gate ended at.992853. Both arms clipped every one of their final100 updates;
mean pre-clip norms were2.602183 versus2.146819. Differences in parameterization
and optimizer dynamics remain possible explanations, not demonstrated causes.
Preserving the baseline at initialization did not guarantee a quality gain for
this linear control. There is no retuning, changed endpoint or fallback adoption.
The nonlinear candidate and remaining fresh seeds continue unchanged.

The bound descriptive calculation is
`tokenizer-evidence/byte-residual-readout-001/quality001/first-control-pair-analysis.json`.
It uses completed saved logs and scores only, checks paired initializer hashes
and observed terminal sampler state, and performs no new model execution or
validation scoring. Independent whole-panel review remains pending.

## First complete fresh bundle (three of nine)

Bundle62102600000 has complete1200-update/41-snapshot results for all three arms.
The nonlinear endpoint passed CPU/MPS mean/max differences3.53812e-8/1.71661e-5,
exact checkpoint readback and publication hashes. Its guard finished normally
in663.748seconds. All three common initial weight inventories and observed
terminal sampler identities match, so the comparisons retain the intended pairing.

| Arm | H200 CE (primary) | Final online CE | Last100 sampled training CE |
|---|---:|---:|---:|
| Byte | 4.585407 | 4.630511 | 3.243986 |
| Residual linear | 4.720631 | 4.835059 | 3.072804 |
| Residual tanh | 4.642792 | 4.702486 | 3.156087 |

Tanh minus byte H200 is+.0573847643; tanh minus linear is-.0778389161. The
nonlinear candidate therefore fails this seed's mandatory improvement against
byte. If these accepted scores are upheld by final review, the frozen all-seed
gate cannot pass. The remaining six trajectories still execute as specified;
there is no post-outcome early-stop rule, tuning, seed replacement or fallback.

Both added branches fit the last100 paired training batches better than byte,
but both final-online and H200 validation scores are worse. This does not
support inadequate late training fit as the sole explanation for this seed.
It does not establish overfitting or an optimizer effect as a causal mechanism.
H200 improves all three endpoints: by.045104 for byte,.114429 for linear and
.059695 for tanh. The observed unfavorable ordering holds at both endpoints;
there is no reason to replace H200 with final-online weights. Tanh's learned
scalar ended at1.230024 (an unrestricted residual scale, not a probability).

The first three guarded jobs took2,009.903seconds. Actual training updates used
1,697.136seconds (84.44%); scheduled online evaluation used203.420seconds (10.12%).
The remaining109.347seconds covers endpoint checks, serialization and other job
overhead. These are measured execution times, excluding engineering time and
the earlier55.323second pilot; they are not deployment performance claims.

Reproducible saved-log analysis: `byte_residual_readout_analysis_v1.py`, output
`tokenizer-evidence/byte-residual-readout-001/quality001/analysis-1-bundles001/result.json`.
It reads only completed records, uses the same final100 training window as the
prior pair analysis, records all input hashes and performs no model execution or
new scoring. The first seed is descriptive; full means, uncertainty and independent
review remain pending. The selected byte model is unchanged.

## Second-seed linear control (four of nine complete)

Bundle62102700000 residual-linear completed1200 updates/41 H200 snapshots, with
H200 CE4.682758260745633 and final-online CE4.790987. Its684.692second guard
finished normally; mean/max CPU/MPS differences6.40071e-8/2.45571e-5 pass the fixed
limits. Both checkpoint hashes and complete target-loss coverage were verified
for publication. The learned unrestricted scalar ended at-1.007271. Its sign is
not a quality criterion.

The matched nonlinear and byte controls for this seed are still pending, so no
new paired quality conclusion is drawn from this individual score. The fixed
order is linear, nonlinear, byte for the second seed. The first seed's failed
mandatory candidate-improvement condition remains recorded.

While training continued, `BYTE_READOUT_REGULARIZATION_REVIEW.md` checked prior
closed decay/dropout/bias evidence, primary weight-decay papers and the actual
local shrinkage convention. Removing decay and input dropout had negative fresh
results in their original recipes; stronger output-specific decay has not been
established by those outcomes. Our independent per-update decay stays constant
while LR falls, so an LR-multiplied coefficient needs explicit unit conversion.
The bound arithmetic is separate from this panel and executes no models or new
scores. No coefficient sweep, old-study replay or new scientific budget follows
from that review; the complete current panel still determines the next decision.

## Second-seed nonlinear result (five of nine complete)

Slot04 residual-tanh completed1200 updates and41 H200 snapshots. H200 CE is
4.607495661696009 and final-online CE is4.672503246509761. Its693.624second
guard finished normally. Acceptance/result/guard hashes, both checkpoint
size/hash records, all34278 CPU/MPS target losses, aggregate means and numerical
errors were independently checked from saved files for publication. Mean/max
CPU/MPS CE differences are1.38916e-8/2.59876e-5, within unchanged limits.
Whole-panel checkpoint reconstruction and independent final review remain pending.

The matched second-seed residual-linear H200 is4.682758260745633, so nonlinear
minus linear is-.0752625990; final-online difference is-.1184836368. This agrees
in direction with the first seed's-.0778389161 H200 difference. Common initial
weights, initial branch matrix, initializer/sampler roles and terminal training
sampler hashes match between the second-seed branches. These are descriptive
paired results, not a completed three-seed uncertainty analysis.

Nonlinear last100 sampled training CE is3.1500290442 versus linear3.0800014544
(difference+.0700275898), despite better validation. The unrestricted nonlinear
scalar ends at-1.237150. Its negative sign is allowed by the frozen implementation;
it is not a numerical failure or a reason to alter the experiment.

The second-seed byte baseline is still running. Neither this branch comparison
nor a comparison with another seed's baseline establishes a gain over byte.
The first bundle's mandatory candidate-versus-byte condition remains failed.
No tuning, changed endpoint, fallback adoption or replacement seed follows from
these results. After slot05 completes, execute the existing completed-bundle
analysis for two bundles once, then let slots06–08 and automatic review finish.

## Two complete fresh bundles (six of nine complete)

Slot05 byte completed1200 updates/41 snapshots in686.011seconds of guarded
execution. Its primary H200 CE is4.5507617862864445 and final-online CE is
4.588576756595113. Acceptance/result/guard bindings, RAW/H200 sizes and hashes,
all34278 saved CPU/MPS losses and aggregate errors were checked. Mean/max CE
errors3.64361e-8/1.71661e-5 pass unchanged limits. The existing analysis helper
ran once for two completed bundles, checking all six receipts, training histories,
score coverage and within-seed initialization/terminal-sampler pairing without
model execution, new scoring, bootstrap or holdout access.

| Second-seed arm | H200 CE | Final online CE | Last100 sampled training CE |
|---|---:|---:|---:|
| Byte | 4.550762 | 4.588577 | 3.250633 |
| Residual linear | 4.682758 | 4.790987 | 3.080001 |
| Residual tanh | 4.607496 | 4.672503 | 3.150029 |

Nonlinear minus byte H200 is+.0567338754 on the second seed, versus+.0573847643
on the first. Linear minus byte is+.1319964745 versus+.1352236804. Nonlinear
beats linear on both seeds but loses to byte on both. The descriptive two-seed
mean differences are+.0570593198 nonlinear-versus-byte and+.1336100774
linear-versus-byte; these are not the frozen full-panel estimates or confidence
bounds. The mandatory all-seed candidate-improvement criterion is already failed,
subject to the independent final review, without changing any threshold.

Both branches again achieve lower late sampled training loss but worse validation
at both online and averaged endpoints. This repeats the first seed's pattern;
it does not establish a causal overfitting, clipping or optimizer explanation.
Averaging improves second-seed byte by.037815, linear by.108229 and nonlinear
by.065008, preserving their ordering. Changing the endpoint has no support here.

All six completed jobs used4074.231seconds of guarded execution:3439.360seconds
of training updates and414.047seconds of scheduled evaluation. The balance is
endpoint verification, serialization and other worker overhead; pilot and
engineering costs are excluded. Analysis evidence is
`tokenizer-evidence/byte-residual-readout-001/quality001/analysis-2-bundles001/result.json`.

The controller automatically started third-seed slot06 residual-tanh (PID25154),
then will run slot07 byte and slot08 linear before independent final review.
After all nine and review finish, execute the existing three-bundle analysis
once. Current source, seeds, controls, H200 endpoint and acceptance criteria
remain fixed. The selected byte model and inference recipe are unchanged.

## Prospective lesson about experiment duration

The first completed bundle already violates the all-seed improvement condition,
so remaining seeds cannot turn this recipe into an adoption success if the
accepted scores survive review. This follows from the fixed conjunction of
criteria, not a statistical significance test. Finishing this panel still
measures reproducibility and the full predeclared contrasts; it has no admitted
early-stop rule and remains unchanged.

Before a future panel is frozen, explicitly decide whether its purpose requires
all seeds after a mandatory failure. For an adoption-only question, a prospective
rule could independently verify a completed matched bundle and close the recipe
as scientifically rejected when an all-seed condition fails. Successful adoption
would still require the entire fresh confirmation panel and all original gates.
A stopped negative panel would report only observed bundles, with no full-panel
mean, bootstrap claim, missing-seed imputation or claim that an entire model
family cannot work. Execution/numerical failures require recovery and must not
trigger scientific rejection. No fallback arm or replacement seed is allowed.

For scale only, this panel's first three guarded jobs took2009.903seconds;
six more at that measured average project about67minutes, excluding pilot and
final-review costs. That is potential future avoided execution, not time saved
in this panel, a promised speedup or a reason to stop it now. A future admission
must choose and freeze either that rejection rule or justified full-panel
mechanism/variability measurement before seeing outcomes. This process lesson
admits no new campaign, source change or numerical budget.

## When the validation penalty appears (two completed seeds)

Existing saved curves permit a narrower diagnosis than the final training versus
validation gap alone. Each entry below is arm minus matched byte online validation
CE; positive means worse. All16 comparisons are positive. The last100 sampled
training-loss differences at those same four endpoints are negative for both
branches on both seeds. No model execution or new score was needed.

| Update | Seed1 linear | Seed1 nonlinear | Seed2 linear | Seed2 nonlinear |
|---|---:|---:|---:|---:|
| 300 | +0.002906 | +0.008125 | +0.007843 | +0.009030 |
| 600 | +0.068392 | +0.032036 | +0.054507 | +0.015243 |
| 900 | +0.116962 | +0.068199 | +0.113015 | +0.053060 |
| 1200 | +0.204548 | +0.071975 | +0.202410 | +0.083926 |

The recipe starts cooldown and weight averaging at update800
(`byte_input_training_v6.py:40–43`, called by this panel's `objects`). The penalty
is already present at300 and600, so neither cooldown nor checkpoint averaging
can be its sole origin. The early nonlinear differences are small and have no
new uncertainty analysis; these observations do not establish a causal mechanism.
There are no measurements between the four scheduled validation endpoints.

Absolute nonlinear validation CE continues falling at every measured endpoint
in both seeds, as does byte CE. Thus, the nonlinear model is still improving
while losing ground relative to byte; its penalty is not equivalent to an
observed late increase in its own validation loss. Linear validation CE does
increase from900 to1200 on both seeds. This distinction argues against assigning
one generic explanation to both branches. It supplies no permission to pick a
better-looking checkpoint, change training length or rescue a closed arm.

Derived by subtracting paired `curves[].ce` entries and checking the saved
`train_window_ending_*` values in the two-bundle analysis. Bound input SHA256:
`2a1f57375c4b4376edc5a16483d1f7211321c2170a74287bdc0d5d45e80ba394`.
The third seed and final review remain pending; no full-panel claim follows.

## Third-seed nonlinear endpoint (seven of nine complete)

Slot06 residual-tanh completed1200 updates and41 H200 snapshots, with H200 CE
4.641951823823287 and final-online CE4.709929901022468. Its689.099second guard
finished normally; training updates used582.173seconds. Publication verification
checked acceptance/result/guard hashes, both retained checkpoint sizes and hashes,
1200 sequential training records, full34278-target CPU/MPS coverage and recomputed
numerical errors. Mean/max CPU/MPS CE differences2.05555e-8/2.03848e-5 pass the
unchanged3e-5/3e-4 limits. Whole-panel independent review is still pending.

The last100 sampled training CE is3.0761310244 and the unrestricted final scalar
is-1.284479. No paired third-seed quality conclusion is available until its byte
and linear controls complete. Cross-seed comparisons cannot replace those controls.
The first two bundles' failed mandatory improvement conditions remain recorded;
this accepted individual run changes neither the adoption decision nor the recipe.

The controller automatically started slot07 byte, PID29393; slot08 linear and
the automatic independent review follow. The review reconstructs the fixed
statistics from saved per-target losses and checks saved weights/receipts; it
requires no new training or validation scoring. Do not rerun either completed
bundle analysis. The three-bundle analysis is due once after final review accepts.

## Prepared follow-up using saved frequency groups

`BYTE_RESIDUAL_FREQUENCY_PROTOCOL.md` and `byte_residual_readout_frequency_v1.py`
prepare the next bounded existing-data question: whether the penalty is dominated
by rare targets or also affects common ones. The prior tying diagnostic supplies
fixed training-frequency bins and target/preceding-input views, not pooled scores.
The current panel's complete three-seed losses, accepted final review and
three-bundle analysis are required before execution. This avoids a partial
subgroup conclusion or a repeated diagnostic after every new seed.

The implementation reconstructs and hashes only the original train/validation
token IDs, reads accepted saved losses, checks complete coverage and verifies
that group contributions reconstruct every global contrast. Limits are120seconds
CPU,1GiB RSS and1MiB output, with10GiB free reserve plus the output allowance at
entry. It uses the shared nonblocking research lock, performs no model execution
or checkpoint loading, and keeps holdout tokens closed. Source syntax was checked;
no diagnostic execution, numerical acceptance or subgroup result is claimed yet.
The training controller remains active. Execute this analysis once after all nine
runs, final review and three-bundle analysis finish; retain failures if any.

## Third matched nonlinear/byte pair (eight of nine complete)

Slot07 byte completed1200 updates/41 snapshots in693.248seconds of guarded
execution, including586.072seconds of training updates. Its primary H200 CE is
4.588101483877167; final-online CE is4.626186339399742. Saved acceptance, result
and guard hashes, both checkpoint hashes/sizes, all1200 sequential training
records and all34278 CPU/MPS target losses were checked for publication.
Recomputed mean/max numerical errors2.63550e-8/1.71661e-5 pass fixed limits.
Common initial weights, initializer/sampler roles and terminal training sampler
hashes match the accepted third-seed nonlinear run.

Nonlinear minus byte H200 is+.0538503399461 on this seed, after+.0573847642768
and+.0567338754096 on the first two. Thus all three fresh matched comparisons
lose to byte; the mandatory candidate-improvement condition fails on every seed,
subject to final independent verification. The third-seed final-online penalty
is+.0837435616227, while the last100 sampled training difference is-.0979814196.
The better-training/worse-validation pattern repeats across all three pairs.

The third seed had a small nonlinear advantage at online update300, reversed by
600. This qualifies the earlier first-two-seed curve observation: no universal
claim that every early checkpoint loses is warranted. The primary averaged
endpoint, late online endpoint and training length remain fixed; no favorable
intermediate checkpoint is selected.

Slot08 residual-linear automatically started as PID33655. This final run is
needed to finish the predeclared nonlinear-versus-linear contrast and full
independent review. After controller completion and accepted review, run the
three-bundle analysis once and then the prepared all-seed frequency diagnostic
under the free shared lock. No further training is needed for either analysis.
The selected byte model remains unchanged.
