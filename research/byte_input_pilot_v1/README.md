# Closed failed original resource pilot

Original001 completed20 native updates, then the driver called
`training.write_new(path, payload)` instead of its required `(context, path)`.
It failed before publishing any RAW archive. Original source, all20 update rows,
plan, roles, guard and STOP remain immutable. No quality conclusion follows.

`pilot-recovery001/recovery.json` records the exact correction in pilot_v2;
that distinct attempt uses fresh roles and unchanged cases/criteria. Never replay
original001 or reuse its roles. The earlier command-argument failure is separately
preserved in `pilot-launch-correction001.json` and occurred before admission.
