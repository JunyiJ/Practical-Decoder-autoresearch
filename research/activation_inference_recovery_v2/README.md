# Activation inference recovery v2 prerequisites

This version is not frozen and cannot dispatch inference. Activation001 and
inference recovery001 remain permanently closed to entries. Their original
runtime sources and failure evidence are unchanged. Preserve the valid recovery001
seed3611 timing pair; only seeds3612/3613 remain eligible for the new finite plan.

## Completed-slot verification under live monitoring

`inventory.py` checks the full output root under the existing metadata lock at
one bounded instant. It separately hashes every completed slot file and its
associated control records. A later audit compares this exact scoped proof while
other slots' allowlisted live monitors/logs may change. Every observation still
checks global path, alias, file-count, size, STOP/failure, temporary-file and disk
limits. Long original-source/checkpoint audits run outside the metadata lock.

The scoped-inventory prerequisite passed27 tests, including10 sequential real
bounded pipe subprocesses. Both prior3611/current3612 and prior3612/current3613
were covered. A handshake proves that the actual parent monitor changed during
child prior-slot verification. Whole-root equality fails during these valid
updates; scoped prior hashes remain equal. Same-size prior-result modification,
changed prior controls, extra prior files and unknown controls are rejected.
This reproduces the failure mechanism; it does not recover the exact differing
monitor field from the historical failure, which was not logged.

## Carried first pair

`carried_pair.py` is a read-only historical importer. It does not invoke the
failed v1 root's bind, budget or entry paths. It replays original training/source
and16 protected checkpoint proofs, checks the frozen v1 runtime and plan, then
independently reconstructs accepted3611 request, process, guards, model lineage,
worker outputs and timing arithmetic. It pins all27 files of the failed v1 root,
including the failed3612 entry, and requires both historical process groups idle.
The deterministic import explicitly forbids repeating3611 and grants no dispatch.

The carried-pair prerequisite passed13 disposable provenance tests and an actual
read-only import of the retained first pair. Test model/GPU/process components
are explicit doubles; the actual import uses original file hashes and timing
records without loading models or repeating inference. Guard finished6.246s,
peakRSS234,799,104bytes. Both campaign inventories and evidence stayed unchanged.

Evidence: `../tokenizer-evidence/activation-preparation-001/` files prefixed
`inference-recovery-v2-scoped-inventory-prerequisite-001` and
`inference-recovery-v2-carried-pair-prerequisite-001`. Each wrapper executed once;
do not overwrite or rerun it. Reviews pin the exact corresponding Python files.

## Remaining launch gates

Integrate a v2 controller, worker and final review with carried3611 credit and
only new3612/3613 slots. Bind both original training and carried-pair imports;
compare completed-slot proofs around long audits. Exercise the actual later-child
entry integration under real parent monitor/pipe updates, including failure and
tampering rejection. Test serial acceptance, partial/failure latches, aggregation
of one carried plus two new pairs, and final receipts. Preserve original model,
cache, numerical timing, statistical and resource rules without modifying frozen
source. Complete source applicability and capacity audits, freeze the exact new
finite root/plan and commit/push before any new inference.

The scientific conclusion is unchanged: GELU fails the predeclared quality gates
against SiLU; no adoption. One timing pair is accepted, so the three-pair timing
aggregate and formal terminal receipt remain outstanding. No training, checkpoint
regeneration, cleanup or holdout scoring occurred during these prerequisites.
