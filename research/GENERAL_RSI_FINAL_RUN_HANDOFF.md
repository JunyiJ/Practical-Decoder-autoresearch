# Final-run execution ownership — 2026-11-16 Pacific

This RSI chat owns boundary cleanup and final-run execution. The handoff from
chat01a1bd0d-48a8-77d1-9dd0-456365df5e22 relayed the human instruction:
“Go ahead and safely free enough storage for the final run once run11 finishes,
then let the RSI thread complete the study without deleting important models
or results.” Existing direct authorization and AGENTS.md remain applicable.

- Let slot10/run11 finish training, both endpoint publications, scoring and all
  fixed first outputs. Accept execution and verify actual worker/supervisor exit.
- Acquire `research_state/tokenizer-worker-lock/.lock` nonblocking before any
  research-payload cleanup. Preserve selected/fresh/scientific endpoints, original
  failures and their payloads, first outputs, source/config/log/metric/receipts.
- Remeasure the complete remaining requirement: retained slot10 endpoints, slot11
  outputs, restart/atomic overlap, metadata, failure allowance and10GiB reserve
  plus2GiB headroom. The reported1.08GB deficit is not a fixed deletion target.
- Retire only obsolete accepted engineering/pilot restart payloads with verified
  closed consumers. Use exact path/size/hash manifests, process/open-file checks,
  durable intent/deletion journals and retained-evidence hash verification.
  Continue across eligible classes to the full requirement and modest churn margin.
- Use the independent external-cache lock only for newly eligible cache entries.
  Do not repeat exhausted scans or bypass a held research lock.
- After recovery, reconcile capacity and complete memory/storage/time admission.
  Launch frozen slot11 depth16/context128 bundle3 without a new permission request.
- Complete frozen statistical decisions, common128/RAW analyses, fixed generation
  ratings, completion audit and concise results. Commit/push substantive evidence
  without force. No partial panel, changed roles/gates/schedules or test opening.

If audited candidates cannot cover the full requirement, record the exact residual
and one concrete remedy. Do not delete important results or weaken admission.
