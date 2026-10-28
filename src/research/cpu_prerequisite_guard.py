"""Opt-in CPU fixture guard with a durable diagnostic outside pytest capture.

This does not dispatch tests, authorize retries or change training watchdogs.
The journal path must be new. Polling is not an instantaneous memory guarantee.
"""
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import threading
import time


def observation(root):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(peak_rss_bytes=rss if sys.platform == 'darwin' else rss*1024,
                free_bytes=shutil.disk_usage(root).free)


def violation(sample, elapsed, rss_limit_bytes, reserve_bytes, timeout_seconds):
    if sample['peak_rss_bytes'] >= rss_limit_bytes:
        return 'peak_rss_limit'
    if sample['free_bytes'] < reserve_bytes:
        return 'disk_reserve'
    if elapsed >= timeout_seconds:
        return 'wall_time_limit'
    return None


class CpuPrerequisiteGuard:
    def __init__(self, journal, root, *, rss_limit_bytes=4*1024**3,
                 reserve_bytes=8*1024**3, timeout_seconds=600, poll_seconds=1):
        for value in (rss_limit_bytes, reserve_bytes, timeout_seconds, poll_seconds):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError('CPU guard limits must be positive finite numbers')
        if poll_seconds > timeout_seconds:
            raise ValueError('CPU guard polling must fit within wall-time limit')
        self.journal, self.root = Path(journal), Path(root)
        self.limits = dict(rss_limit_bytes=rss_limit_bytes, reserve_bytes=reserve_bytes,
                           timeout_seconds=timeout_seconds)
        self.poll_seconds = poll_seconds
        self.stop = threading.Event()
        self.started = None
        self.stream = None
        self.thread = None

    def _write(self, status, sample=None, reason=None):
        record = dict(status=status, pid=os.getpid(), limits=self.limits,
                      elapsed_seconds=time.monotonic()-self.started,
                      observation=sample, reason=reason)
        # An independently opened file descriptor survives pytest fd capture.
        # Flush and fsync before hard exit; no stdout/stderr dependency.
        self.stream.seek(0)
        self.stream.write((json.dumps(record, sort_keys=True)+'\n').encode())
        self.stream.truncate()
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def _sample(self):
        sample = observation(self.root)
        reason = violation(sample, time.monotonic()-self.started, **self.limits)
        return sample, reason

    def _watch(self):
        try:
            while not self.stop.is_set():
                sample, reason = self._sample()
                self._write('failed' if reason else 'watching', sample, reason)
                if reason:
                    os._exit(90)
                self.stop.wait(self.poll_seconds)
        except BaseException as exc:
            try:
                self._write('guard_error', reason=repr(exc))
            finally:
                os._exit(91)

    def __enter__(self):
        if self.started is not None:
            raise RuntimeError('CPU guard instances cannot be reused')
        self.stream = self.journal.open('xb')
        self.started = time.monotonic()
        self._write('starting')
        self.thread = threading.Thread(target=self._watch, daemon=True, name='cpu-prerequisite-guard')
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join()
        try:
            sample, reason = self._sample()
            self._write('failed' if reason else 'finished' if exc is None else 'fixture_exception',
                        sample, reason if reason else repr(exc) if exc is not None else None)
            if reason:
                os._exit(90)
        finally:
            self.stream.close()
        return False
