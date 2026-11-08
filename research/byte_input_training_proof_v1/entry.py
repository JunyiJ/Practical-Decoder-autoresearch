"""Stdlib outer guard and once-only original numerical entry."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'research/tokenizer-evidence/byte-composition-boundary-001'
OUT = BASE/'training-math001'
ARCHIVES = ROOT/'research_runs/byte-input-training-math001'
ADMISSION = BASE/'training-admission001'
TORCH = None
CAP = 7516192768


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(1048576): h.update(data)
    return h.hexdigest()


def pins(records):
    for pin in records:
        path = Path(pin['path'])
        assert path.stat().st_size == pin['size'] and sha(path) == pin['sha256'], str(path)


def publish(name, record):
    raw = (json.dumps(record, sort_keys=True, allow_nan=False)+'\n').encode()
    assert sum(p.stat().st_size for p in OUT.iterdir())+len(raw) <= CAP
    with (OUT/name).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


class Guard(CpuPrerequisiteGuard):
    def _sample(self):
        sample, reason = super()._sample()
        paths = list(OUT.iterdir()) + list(ARCHIVES.iterdir())
        if any(p.is_symlink() or not p.is_file() for p in paths): reason = 'unsafe_output'
        sample['output_bytes'] = sum(p.lstat().st_size for p in paths)
        if sample['output_bytes'] > CAP: reason = 'output_limit'
        if TORCH is not None:
            sample['mps_driver_bytes'] = TORCH.mps.driver_allocated_memory()
            if sample['mps_driver_bytes'] >= 6442450944: reason = 'mps_limit'
        return sample, reason


def main(expected):
    global TORCH
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert {p.name for p in OUT.iterdir()} == {'plan.json'}, 'Original once-only entry'
        assert not list(ARCHIVES.iterdir()), 'Exclusive archive destination'
        assert sha(OUT/'plan.json') == expected
        plan = json.loads((OUT/'plan.json').read_text())
        receipt = json.loads((ADMISSION/'review.json').read_text())
        assert receipt['accepted'] and receipt['numerical_plan_sha256'] == expected
        assert json.loads((ADMISSION/'guard.json').read_text())['status'] == 'finished'
        assert json.loads((ADMISSION/'STOP').read_text())['status'] == 'complete_original_admission'
        stat = os.statvfs(ROOT)
        assert stat.f_bavail*stat.f_frsize >= CAP+8589934592+2147483648
        publish('attempt.json', dict(pid=os.getpid(), plan_sha256=expected, no_replay=True))
        try:
            with Guard(OUT/'guard.json', ROOT, timeout_seconds=3600,
                       rss_limit_bytes=4294967296, reserve_bytes=8589934592, poll_seconds=.25):
                pins(plan['sources']); pins(plan['assets'])
                for key,value in plan['environment'].items(): assert os.environ.get(key) == value, key
                assert sys.flags.optimize == 0, 'Proof assertions must be enabled'
                import torch
                TORCH = torch
                from .driver import run
                run(plan, ARCHIVES, publish)
                pins(plan['sources']); pins(plan['assets'])
            publish('STOP', dict(status='complete_original_training_context', no_replay=True))
        except BaseException as exc:
            if not (OUT/'STOP').exists():
                publish('STOP', dict(status='closed_failed_original_training_context',
                                    error=repr(exc), no_replay=True))
            raise


if __name__ == '__main__': main(sys.argv[1])
