# Continuation group-decay integration, 2026-11-12

The separate group-decay wrapper passes the required integration checks.
The all-decay policy matches the canonical implementation exactly across three
updates, including model, optimizer, sampler, clock and RNG state. All352 tensor
checks match the prescribed readout/other shrinkage. Every policy passes full
checkpoint restore and continuation replay with original measured clock times.
Corrupt saved policy and config are rejected before mutation.

Evidence: `tokenizer-evidence/general-rsi-continuation-decay-integration-001/`.
Frozen plan hash:
`f079a7e4b54e27ec44f7a3850141681ff7bc12d368ed1d9f51a19d4c0e9eb866`.
This qualifies new integration. It is not a quality result or a rerun of the
closed decay studies. Frozen optimizer sources remain unchanged.

| Policy | Original updates | Replayed updates | Mean update seconds |
|---|---:|---:|---:|
| Both groups decay | 71 | 22 | 0.341495 |
| Other only | 67 | 22 | 0.358814 |
| Readout only | 70 | 22 | 0.344193 |
| Neither group | 71 | 23 | 0.342390 |

Every run reaches cooldown and23–24 EMA snapshots within the24-second training
budget. The shared natural prefix is67 updates. Full equality applies to replay
with recorded durations; it does not claim identical live-clock scheduling.
The baseline comparison adds three original and three replayed updates.

Guarded runtime149.415seconds; peak RSS2,838,003,712bytes, MPS2,501,689,344bytes.
Four retained restart payloads total1,780,087,508bytes. All5,148 source bindings
remain unchanged. No failures, validation scoring or test access occurred.

The fresh four-arm diagnostic is now frozen separately in
`general-rsi-continuation-decay-diagnostic-001/plan.json`. Its complete admission
covers1,200 updates and63 evaluations, about667.17seconds forecast within900,
64MiB retained evidence plus64MiB overlap,10GiB reserve and2GiB extra headroom.
Actual free space includes the four retained qualification payloads. No trained
checkpoint outputs are planned. Run and review the fixed diagnostic next.
