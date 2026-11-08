# Residual readout: implementation verified, fresh quality panel running

The new zero-gated residual readout passed its implementation pilot. Six of nine fresh quality runs have completed and passed numerical
acceptance. Both complete seeds fail the mandatory candidate-improvement
criterion against byte. The third seed is active; the frozen remaining runs and
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
worker was slot06 residual-tanh on the third seed, PID25154;
live logs supersede this snapshot. Slots00–05 are terminal and must not be replayed.

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
