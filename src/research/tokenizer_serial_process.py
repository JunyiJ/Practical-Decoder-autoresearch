"""Bounded fresh-only subprocess primitive, not a training dispatcher.

Caller supplies a frozen argv/request, audited prerequisites, acquired global
and run locks, and full remaining disk requirement. The training worker must
independently run its existing 4GiB RSS/6GiB MPS-driver watchdog. This parent
checks elapsed time, output and reserve; it does not measure MPS memory.
"""
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

RESERVE = 8*1024**3


def snapshot(output):
    paths = list(output.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('Worker output may not contain symlinks')
    return dict(free_bytes=shutil.disk_usage(output).free,
                output_bytes=sum(p.stat().st_size for p in paths if p.is_file()),
                log_bytes=(output/'worker.log').stat().st_size if (output/'worker.log').exists() else 0)


def _group_exists(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _stop(process):
    """Terminate only the session created for this child, then reap its leader."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    # A dead leader can still leave descendants; kill the owned group as well.
    if _group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=2)


def run_process(argv, cwd, output, lock_fds, *, required_free_bytes,
                output_limit_bytes, log_limit_bytes=4*1024**2,
                timeout_seconds=7200, poll_seconds=.2):
    """One fresh process; exceptions are retained and never retried here.

    required_free_bytes includes all remaining arms plus the 8GiB reserve at
    dispatch. During execution the reserve and this job's output limits apply;
    the controller must recheck all remaining allocations before the next arm.
    Guard observations are sampled, not instantaneous peak guarantees.
    """
    if (not isinstance(argv, (tuple, list)) or not argv
            or any(not isinstance(a, str) or not a for a in argv)):
        raise ValueError('Explicit nonempty argv required; no shell command')
    for value in (required_free_bytes, output_limit_bytes, log_limit_bytes):
        if type(value) is not int or value <= 0:
            raise ValueError('Positive integer byte limits required')
    if required_free_bytes < RESERVE or log_limit_bytes > output_limit_bytes:
        raise ValueError('Reserve/output policy violated')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
           for v in (timeout_seconds, poll_seconds)) or poll_seconds > timeout_seconds:
        raise ValueError('Invalid bounded polling/time limits')
    if len(lock_fds) != 2 or len(set(lock_fds)) != 2:
        raise ValueError('Distinct global and run lock descriptors required')
    for fd in lock_fds:
        if type(fd) is not int or fd <= 2:
            raise ValueError('Invalid inherited lock descriptor')
        os.fstat(fd)
    output = Path(output).absolute(); cwd = Path(cwd).resolve()
    output.mkdir(exist_ok=False)  # terminal even for previous success/failure
    journal = output/'process.json'
    state = dict(status='preflight', argv=list(argv), pid=None, returncode=None,
                 required_free_bytes=required_free_bytes, reserve_bytes=RESERVE,
                 timeout_seconds=timeout_seconds, output_limit_bytes=output_limit_bytes,
                 log_limit_bytes=log_limit_bytes, observation=None, error=None)
    started = time.monotonic(); process = None
    def save():
        state['elapsed_seconds'] = time.monotonic()-started
        temporary = journal.with_suffix('.tmp')
        with temporary.open('w') as stream:
            json.dump(state, stream, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(journal)
    def check(initial=False):
        sample = snapshot(output); state['observation'] = sample
        if sample['free_bytes'] < (required_free_bytes if initial else RESERVE):
            raise RuntimeError('Insufficient disk for remaining allocations' if initial else 'Disk reserve crossed')
        if sample['log_bytes'] > log_limit_bytes or sample['output_bytes'] > output_limit_bytes:
            raise RuntimeError('Worker output limit crossed')
        if time.monotonic()-started >= timeout_seconds:
            raise TimeoutError('Worker wall time limit crossed')
    save()
    try:
        check(initial=True)
        with (output/'worker.log').open('xb', buffering=0) as log:
            process = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=tuple(lock_fds), close_fds=True, shell=False)
            state.update(status='running', pid=process.pid); save()
            while process.poll() is None:
                check(); save(); time.sleep(poll_seconds)
            state['returncode'] = process.wait()
            # Check again even for an immediately exiting child.
            check()
            if state['returncode'] != 0:
                raise RuntimeError('Worker exited with code '+str(state['returncode']))
            if _group_exists(process.pid):
                raise RuntimeError('Worker left live descendants')
        state['status'] = 'completed'; save()
        # The terminal journal also counts toward output allocation.
        check()
        return state
    except BaseException as exc:
        if process is not None:
            try:
                _stop(process)
            except BaseException as cleanup_error:
                # Preserve both the original failure and uncertain cleanup.
                # A controller must halt and inspect processes in this case.
                state['cleanup_error'] = repr(cleanup_error)
            state['returncode'] = process.returncode
        state.update(status='failed', error=repr(exc)); save()
        raise
