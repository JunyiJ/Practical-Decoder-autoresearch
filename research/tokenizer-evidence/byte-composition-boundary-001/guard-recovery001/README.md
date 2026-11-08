# Recovery of the slot14 output-inventory guard failure

The original quality001 controller exited1 after its slot14 worker exited91.
All1,200 step records and all five scheduled diagnostics exist. The terminal
worker guard records `guard_error`, `FileNotFoundError(2, 'No such file or
directory')`, at919.518729750067seconds. The guard uses `os._exit(91)`, so no
worker STOP was written. The original controller STOP remains closed failed.
This is an execution/guard failure, not a measured negative quality result.

`failure.json` preserves hashes of1,211 original source/step/diagnostic/failure
records plus the retained449,317,111-byte RAW checkpoint. The RAW hash is
`9b08f859de55955ae10177c4c9c16a2da33a36fffdb381df8f9e5e63296e814a`.
Original evidence and the failed worker/controller namespaces are immutable.

## Demonstrated mechanism and exact correction

The original guard first collects paths satisfying `is_file()` and later calls
`stat()` again to sum their sizes. The original checkpoint writer fsyncs a
temporary, creates a hard link at the final name, then unlinks the temporary.
A temporary can disappear between those two guard operations. The original
terminal record omitted the failing filename and traceback, so it does not
identify that path conclusively. `reproduce.py` demonstrates this exact race
with a new seven-byte filesystem fixture and the unchanged original Guard.

The new guard in `research/byte_input_quality_recovery_v1/guard.py` samples each
path once per inventory. A FileNotFoundError invalidates the entire sample and
causes a bounded full rescan, at most three total attempts. It never substitutes
zero or a stale lower size. Persistent disappearance and other filesystem errors
still fail. Regular temporary files and hard-link overlap remain charged;
symlinks and unsupported output types fail. Original RSS, disk reserve, MPS,
metadata/output and wall limits remain enforced. The original source is untouched.

The new filesystem fixture reproduced the old error and verified corrected
publication sampling, hard-link overlap, symlink rejection, persistent-error
failure after exactly three attempts, and unchanged permission-error propagation.
It completed in0.004546seconds without model/GPU execution. Its original
failure and passing correction are retained together in `reproduction.json`.

## Saved checkpoint integrity and recovery scope

`inspect_raw.py` used the shared nonblocking research lock and a120-second CPU
guard with2GiB RSS and8GiB disk reserve. It finished in1.745seconds, peak RSS
785,137,664bytes. The qualified RAW reader verified file size/hash before and
after loading. The audit matched the original plan's roles, identity, data and
recipe; all1,200 saved history rows matched their original step logs; all41
averaging snapshots were present; all88 parameter tensors and their averages
had the expected finite FP32 shapes, totaling28,492,416parameters. The saved
train/validation sampler hashes matched both other fresh1 arms. All scheduled
diagnostics retained their state-preservation flags. This is saved-state
integrity evidence, not retroactive acceptance of the failed scientific slot.

The separately frozen `../quality-recovery001/plan.json` binds the correction,
proofs and immutable original plan. Recovery slot14 restores the existing RAW
into a pristine qualified context, exports its existing H200 averages, and runs
the unchanged post-training validation/cache/saved-review work. It performs zero
training updates. The post-training suffix is byte-identical to the original
driver, SHA256`f2af9b13f453541df19e45d879968bcd06850a673860256ea2b8c490d1ed0624`;
AST inspection confirms the recovery driver has no `.step()` call. Original
step/diagnostic evidence is copied byte-for-byte into the new consumer namespace.
The original RAW stays in its original path; no duplicate RAW is created.

Only never-entered slots15–17 are trained afterward, using the original frozen
numerical driver, selected learning rates and reserved fresh2 roles. No seed is
replaced or redrawn for training, and no tuning, trajectory, comparison, horizon,
bootstrap or acceptance threshold changes. Restoration reconstructs the original
initializer identity solely to load saved weights; it is not another initializer
search. The original controller is not restarted or rewritten as successful.

Capacity at recovery freeze was16,831,553,536free bytes versus a15,589,246,165-byte
remaining entry requirement, preserving original reserve and extra headroom.
All original final pairs remain in the same forecast; extra small recovery
metadata fits the original512MiB allowance. Original failed elapsed time is
charged against slot14's7,200-second limit, leaving6,280.481seconds for its
postprocessing. Total campaign time includes time elapsed since the original
controller launch, including diagnosis, under the original130,800-second bound.

The original guard remains failed; its final observation was lost when the
inventory exception replaced it with null. Recovery guard receipts therefore
cover postprocessing and later runs, not a fabricated original terminal pass.
Independent final review must inspect this execution lineage explicitly before
any adoption. Original scientific quality gates remain unchanged.

A source-generation helper initially selected an earlier diagnostic `finally`
block while extracting the post-training suffix and raised `ValueError:
substring not found`. It wrote no driver. The corrected helper searches for the
closing `finally` only after the suffix start, then verifies exact suffix bytes
and parses every new source before freeze. The original tool error remains in
the conversation; no worker, numerical case or frozen file was altered by it.
