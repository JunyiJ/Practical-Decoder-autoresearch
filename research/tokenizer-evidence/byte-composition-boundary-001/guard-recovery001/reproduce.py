"""New tiny filesystem fixture for the demonstrated output-inventory failure."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

from research.byte_input_quality_v1 import entry
from research.byte_input_quality_recovery_v1.guard import retained_bytes

out = Path(__file__).resolve().parent
started = time.perf_counter()
cases = {}
with tempfile.TemporaryDirectory(prefix='byte-guard-race-') as directory:
    root = Path(directory)
    tmp, final = root/'raw.pt.tmp', root/'raw.pt'
    tmp.write_bytes(b'payload'); os.link(tmp, final)
    entry.WEIGHTS = root
    guard = entry.Guard(root/'unused.json', root)
    guard.started = time.monotonic()
    original_is_file = Path.is_file
    def disappear_after_inventory(path):
        result = original_is_file(path)
        if path == tmp and result:
            tmp.unlink()
        return result
    try:
        with patch.object(Path, 'is_file', disappear_after_inventory):
            guard._sample()
    except FileNotFoundError as exc:
        cases['original_race'] = dict(type=type(exc).__name__, filename=exc.filename, reason=repr(exc))
    else:
        raise AssertionError('Original guard must reproduce the disappearing temporary failure')
    assert final.read_bytes() == b'payload' and not tmp.exists()
    os.link(final, tmp)
    assert retained_bytes([root]) == 14  # Preserve original overlap charging.
    original_lstat = os.lstat
    disappeared = []
    def disappear_before_stat(path, *args, **kwargs):
        if Path(path) == tmp and not disappeared:
            disappeared.append(True); tmp.unlink()
        return original_lstat(path, *args, **kwargs)
    with patch.object(os, 'lstat', disappear_before_stat):
        assert retained_bytes([root]) == 7
    assert disappeared and final.read_bytes() == b'payload'
    cases['corrected_race'] = 'whole inventory retried; final payload counted exactly'
    cases['hard_link_overlap'] = 'both paths charged before publication unlink'
    bad = root/'unsafe'; bad.symlink_to(final)
    try:
        retained_bytes([root])
    except ValueError:
        cases['symlink'] = 'rejected'
    else:
        raise AssertionError('Symlink must fail')
    bad.unlink()
    calls = []
    def always_missing(path, *args, **kwargs):
        calls.append(str(path)); raise FileNotFoundError(2, 'fixture churn', str(path))
    try:
        with patch.object(os, 'lstat', always_missing):
            retained_bytes([root])
    except FileNotFoundError:
        assert len(calls) == 3
        cases['persistent_missing'] = 'failed after exactly three bounded attempts'
    else:
        raise AssertionError('Persistent missing paths must fail')
    try:
        with patch.object(os, 'lstat', side_effect=PermissionError('fixture denied')):
            retained_bytes([root])
    except PermissionError:
        cases['permission_error'] = 'propagated unchanged'
    else:
        raise AssertionError('Other filesystem errors must fail')

sources = [Path(__file__), Path('research/byte_input_quality_recovery_v1/guard.py'),
           Path('research/byte_input_quality_v1/entry.py'), Path('research/byte_input_training_v2.py')]
result = dict(accepted=True, cases=cases, elapsed_seconds=time.perf_counter()-started,
              source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              no_model_execution=True, no_training_replay=True,
              limits_unchanged=True, scope='Filesystem inventory only; does not accept the saved model or recover quality001.')
with (out/'reproduction.json').open('x') as f:
    json.dump(result, f, indent=2, sort_keys=True); f.write('\n')
print(json.dumps(result, indent=2))
