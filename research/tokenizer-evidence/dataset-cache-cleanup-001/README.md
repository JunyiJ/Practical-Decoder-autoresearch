# Authorized dataset cache cleanup, October31,2026

Removed exactly134 files/symlinks and83 directories from the manifest-listed
Hugging Face processed/downloaded dataset caches. No model cache, corpus, research
checkpoint, archive or unrelated program was removed. No cache users were found
with process/open-file inspection; held cache locks and the shared worker lock.

Removed allocated bytes:10,008,887,296 (9.32GiB). Observed free-space increase:
9,952,747,520B. Free after verified completion:19,183,828,992B (17.87GiB),
above the current15,170,221,264B full-panel requirement by4,013,607,728B.
Recheck exact capacity before dispatch; no reduction of the8GiB reserve.

All research weights/archives and portable immutable evidence were byte-hashed
before and after, and model cache file identities stayed unchanged. Original
remote datasets are unaffected; removed caches require download/rebuild if reused.

The checkpoint review classifies437 remaining loose checkpoint files.
No additional irreversible deletion was justified: selected/fresh/failure evidence
is protected, and numerical fixtures require retained dependency evidence or a
separate verified archival operation. Storage is no longer the current blocker.
