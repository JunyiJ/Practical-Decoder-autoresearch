# Cross-size execution progress

The first fresh depth4 pair, bundle1, completed10,240updates per arm and342EMA
snapshots. Both execution reviews passed. All four final weights and48first
completions are retained. Each endpoint covers100,416development targets and
32,768broader-probe targets. Semantic ratings remain pending.

| Measure | Native slot00 | Byte slot01 |
| --- | ---: | ---: |
| H200 development CE | 5.196423356 | 5.146958298 |
| RAW development CE | 5.200483851 | 5.154076443 |
| Synchronized training seconds | 3464.333 | 6375.625 |
| Complete worker seconds | 3615.742 | 6719.651 |
| Peak RSS bytes | 1021296640 | 1267777536 |
| Peak MPS bytes | 2512175104 | 3640442880 |

The descriptive H200 byte-minus-native difference is-0.049465058, with1.840×
training time and42.61%fewer parameters. This first pair is consistent with the
prior equal-token result. It does not establish fresh confirmation or the
cross-size interaction. The body/composer dilution hypothesis remains unresolved.
Keep all fixed roles, gates, schedules and comparisons unchanged.

Slot02, depth8 byte bundle1, also passed execution review after10,240updates
and342EMA snapshots. H200 development CE is5.102819586 and RAW is5.111235728.
Synchronized training took7,275.498seconds; complete worker time was7,663.806seconds.
Peak RSS was1,407,188,992bytes and peak MPS allocation was3,715,940,352bytes.
Both endpoints cover all frozen development and broader targets. Its24first
outputs are retained, bringing the panel total to72. Semantic ratings remain
pending. The depth8 byte/native comparison awaits its fixed native pair.

Evidence: `tokenizer-evidence/general-rsi-cross-size-quality-001/slot-00/review.json`,
`slot-01/review.json`, `slot-02/review.json` and
`slot-02-completion-and-next-admission.json` in the same panel directory.
Remaining-panel admission passes:20,483,403,348bytes required and21,081,903,104bytes
available. This includes12GiB reserve/headroom and all retained outputs, restart,
serialization, metadata and failure allowances. The conservative total forecast
is48.073hours against the frozen54-hour tranche.

Slot03, depth8 native bundle1, is running after live full-panel admission. Its
initial sampler prefix matches the paired byte run exactly. Review at exit,
then admit fixed slot04, depth16 native bundle1.
Do not repeat prerequisites or reopen the test.
