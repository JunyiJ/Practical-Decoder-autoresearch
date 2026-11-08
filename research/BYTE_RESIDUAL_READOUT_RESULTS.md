# Residual readout: implementation verified, fresh quality panel running

The new zero-gated residual readout passed its implementation pilot. Two of nine fresh quality runs have completed and passed numerical
acceptance. The nonlinear residual run is active. There is no measured
candidate quality improvement or adoption yet. The selected byte model and its corrected public
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
worker was slot02 residual-tanh PID9138 at246 updates; live logs supersede this
snapshot. Slots00–01 are terminal and must not be replayed.

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
