"""Future-only paired-score diagnostics; never imports or reopens a campaign."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path

POLICY = 'readout_projection_native_distribution_validation_v1'
ATOL = 3e-5
MAX_BYTES = 512*1024


def publish_then_require(path, cpu_report, mps_report, provenance):
    """Persist both finite scorer reports before enforcing the unchanged gate.

    The caller must already have validated physical targets, checkpoint identity,
    runtime, source binding and resource admission. This is not a scorer or an
    acceptance substitute. No frozen worker imports this module.
    """
    path = Path(path)
    if path.suffix != '.json' or path.is_symlink():
        raise ValueError('New ordinary JSON destination required')
    for report in (cpu_report, mps_report):
        if (type(report) is not dict or type(report.get('ce')) is not float
                or not math.isfinite(report['ce']) or report['ce'] < 0
                or type(report.get('target')) is not dict):
            raise ValueError('Already validated finite scorer reports required')
    if type(provenance) is not dict or not provenance:
        raise ValueError('Explicit caller provenance required')
    cpu, mps = copy.deepcopy(cpu_report), copy.deepcopy(mps_report)
    targets_equal = cpu['target'] == mps['target']
    coverage = all(t.get('policy') == POLICY and t.get('targets') == 34278
                   and t.get('target_bytes') == 111536 and t.get('context') == 128
                   and t.get('first_target_offset') == 1 and t.get('tail_retained') is True
                   and t.get('holdout_scored') is False
                   for t in (cpu['target'], mps['target']))
    difference = abs(cpu['ce'] - mps['ce'])
    result = dict(kind='projection_validation_pair_record_v1', cpu=cpu, mps=mps,
                  provenance=copy.deepcopy(provenance), cpu_mps_ce_difference=difference,
                  absolute_ce_tolerance=ATOL, ce_pass=difference <= ATOL,
                  target_equal=targets_equal, canonical_coverage_pass=coverage,
                  accepted=difference <= ATOL and targets_equal and coverage,
                  physical_scorer_or_checkpoint_acceptance=False)
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    if len(raw) > MAX_BYTES:
        raise ValueError('Paired report metadata cap exceeded')
    # Exclusive publication; caller chooses a new admitted directory. No overwrite,
    # checkpoint mutation or fallback if the evidence cannot be made durable.
    with path.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    if not result['accepted']:
        raise ValueError('CPU/MPS comparison rejected; paired reports saved SHA256=' + hashlib.sha256(raw).hexdigest())
    return result
