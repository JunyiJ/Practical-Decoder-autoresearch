# General-corpus LR screening — complete

All four frozen2,048-update screens completed and passed execution review.
All8,192 training updates and all paired batch hashes were verified. Each
endpoint scored the same100,416 validation targets in108 documents. Source
bindings, endpoint weight hashes, guards and clipping checks passed. Test
contents remain unopened. No scientific seed or endpoint rule was changed.

| Arm | Learning rate | Primary H200 CE | Secondary RAW CE | Guarded seconds |
|---|---:|---:|---:|---:|
| Native | .00125 | 6.17784697 | 6.14887034 | 759.394 |
| Byte input | .00125 | 6.12870995 | 6.09834076 | 1,390.029 |
| Native | .0025 | 5.99346365 | 5.96638470 | 748.893 |
| Byte input | .0025 | 5.97061805 | 5.94332556 | 1,398.365 |

The frozen selector chose **.0025 for both arms**, using final H200 CE only.
The selected byte-minus-native difference is-.02284560 nats/token in this
single screen bundle. By source the selected differences are-.01840529 for
FineWeb-Edu and-.04063030 for Wikipedia. Byte took about1.87x the guarded time
of native at the selected rate. Both rates were tested equally for both arms.

What this tells us: a byte-input advantage remains plausible on the new data,
but equal tuning narrowed the observed gap relative to the lower-rate pair.
The higher rate helped both architectures substantially. These correlated
screens select learning rates only; they are not independent confirmation
seeds and cannot establish model superiority or be pooled into the final test.

RAW CE was lower than H200 CE in all four screens. Preserve this observation
for later analysis, but keep H200 as the frozen primary endpoint. This result
does not authorize choosing a favorable endpoint retrospectively or changing
averaging during the confirmation panel.

Next: six fresh10,240-update runs, three paired bundles, native then byte in
each. The first fresh native confirmation is launched under the unchanged
panel manifest. Its first fixed generation outputs are retained, and the
first bundle will also supply final RAW/H200 completions. Final adoption still
requires all three paired directions, mean improvement at least.02 nats/token,
the frozen99% document-bootstrap bound and the source-regression guard.
Historical Shakespeare results remain separate.

Evidence: `research/tokenizer-evidence/general-rsi-panel-001/selection.json`
and `slot-00` through `slot-03` results, guards, reviews and step logs. Total
guarded screen time4,296.681 seconds. All8 scientific screen endpoints remain
retained; only already-obsolete transient restart snapshots were retired under
their predeclared exact deletion journals.
