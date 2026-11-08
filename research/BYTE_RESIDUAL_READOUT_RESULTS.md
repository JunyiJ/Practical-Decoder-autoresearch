# Residual readout: implementation verified, fresh quality panel running

The new zero-gated residual readout passed its implementation pilot. A separate
fixed nine-run quality comparison is now executing; there is no measured quality
improvement or adoption yet. The selected byte model and its corrected public
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
scientific cases fixed. The last published launch observation was slot00 at17
updates, controller PID1329/worker PID1330; live logs supersede this snapshot.
