"""Prospective inference-only reader; does not authorize dispatch or retry.

Keep frozen src unchanged. Call only OUTSIDE MetadataWriter._serialized(): this
reader opens its own metadata flock and is deliberately not reentrant. All
mutating producers must use the original bounded writer and the same root.
The shared worker/campaign FDs alone do not serialize inherited processes.
"""
import time

from src.research.tokenizer_wide_hybrid_metadata import snapshot
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter

CENSUS_SECONDS = 2.0


def synchronized_snapshot(writer):
    """Serialize, scan twice, and reject instability without a census retry.

    Producer-side overlap accounting continues to bound temporary bytes while
    the reader waits for publication. Leftover temporary files remain counted;
    no missing-path exception is swallowed. This is an observation, not proof
    of immutable content provenance, which needs separate hash verification.
    The 2s allowance includes lock wait and both scans; the enclosing worker
    watchdog must still bound a blocked filesystem call. No timing region may
    contain this function. Callers check STOP, disk and panel limits separately.
    """
    if not isinstance(writer, MetadataWriter):
        raise ValueError('Original bounded metadata writer required')
    started = time.monotonic()
    with writer._serialized():
        first = snapshot(writer.root)
        second = snapshot(writer.root)
        if first != second:
            raise RuntimeError('Metadata inventory changed while serialized; no retry')
        if time.monotonic() - started >= CENSUS_SECONDS:
            raise TimeoutError('Synchronized census time limit; no retry')
    # Lock ownership/identity was checked by the original writer. Returning the
    # original schema retains false dispatch/transient-certification flags.
    return second
