# Research execution and blocker recovery

These instructions apply to work in this repository, including scheduled
continuations. They record the user's request to stop repeatedly stalling on
the same issue. Follow the latest human authorization over older local plans.

## Resolve blockers instead of repeating them

- Identify the exact blocked action and the evidence for the blocker. Do not
  turn a training-capacity constraint into a ban on source integration,
  small numerical tests, analysis, or other independent useful work.
- On the first blocker, inspect available recovery options and execute the
  authorized, bounded option. Do not stop at listing options or promising work.
- If the same blocker is encountered again without a relevant state change,
  stop repeating the same check. In that continuation, either take a different
  authorized recovery action, complete a concrete independent work item, or
  give one actionable escalation explaining exactly what cannot be done and
  what external change is needed. Record the recovery tried and the next action.
- An unchanged heartbeat is not research progress. Do not create commits,
  metadata scaffolding, or repeated notifications merely to record unchanged
  state. Once independent work is exhausted, record a precise wake condition;
  check it only at an appropriate interval or after new evidence/user steering.
- Never describe work as continuing when no worker or concrete work item is
  advancing. Report completion, failures, and actionable blockers accurately.

## Manage capacity before it stops experiments

- Forecast the complete retained-output high-water requirement before freezing
  a panel, including serialization overlap, metadata, and the existing reserve.
- When capacity is insufficient, promptly inventory large research artifacts
  and identify obsolete screens, completed pilots, duplicate caches, and old
  archives. Archival does not make an obsolete checkpoint permanently necessary.
- Apply current user authorization to concrete cleanup candidates; do not
  silently treat an earlier self-authored retain-all plan as overriding newer
  permission to retire old checkpoints. Record narrow policy exceptions.
- Use exact path/size/hash allowlists, the shared worker lock, process checks,
  durable deletion journals, and verification of retained evidence. Preserve
  selected models, fresh confirmations, failed artifacts, and active work unless
  the user specifically authorizes changing those protections. Never remove
  unrelated files or data merely because it is large.
- After cleanup, measure actual free space and immediately update stale blocker
  state and the recorded next action. Continue the enabled work; do not leave
  future heartbeats stuck on a resolved capacity condition.
- If authorized cleanup cannot meet the requirement, state the remaining byte
  deficit and a concrete storage option once. Continue independent work. Do not
  shrink confirmation requirements, relax guards, or launch a partial panel to
  hide the capacity problem.

## Keep engineering tied to research

- Prefer the smallest implementation needed for the next real numerical,
  checkpoint, inference, or quality test. Extend ledger/metadata infrastructure
  only for a demonstrated failure or an explicit requirement of that test.
- Preserve immutable historical evidence and frozen source snapshots. Put new
  implementations outside frozen namespaces; never reopen closed campaigns or
  retry closed failed fixtures just to manufacture progress.
- Before ending a substantive continuation, reconcile actual Git/process/log
  state with progress records, identify the next executable action, and commit
  and push authorized source/evidence changes without force.
- Approval-review timeouts are execution failures, not missing human consent.
  Try permitted independent work and report the exact blocked action rather
  than asking again for permission already granted.
