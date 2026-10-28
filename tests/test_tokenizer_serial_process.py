"""Tiny serial child processes only; no torch, training, model or corpus."""
import contextlib
import fcntl
import json
import os
import sys
from pathlib import Path

import pytest

from src.research import tokenizer_serial_process as p


@contextlib.contextmanager
def locks(tmp_path):
    with (tmp_path/'global.lock').open('a+b') as a, (tmp_path/'run.lock').open('a+b') as b:
        for s in (a, b): fcntl.flock(s.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield a.fileno(), b.fileno()


def run(tmp_path, fds, code, **kwargs):
    options = dict(required_free_bytes=p.RESERVE+1024**2, output_limit_bytes=65536,
                   log_limit_bytes=4096, timeout_seconds=5, poll_seconds=.02)
    options.update(kwargs)
    return p.run_process([sys.executable, '-c', code], tmp_path, tmp_path/'job', fds, **options)


def state(tmp_path):
    return json.loads((tmp_path/'job/process.json').read_text())


def test_success_inherits_both_locks_and_refuses_second_attempt(tmp_path):
    with locks(tmp_path) as fds:
        code = ('import os,fcntl\n'
                f'fds={fds!r}\n'
                'for fd in fds: os.fstat(fd)\n'
                "for name in ('global.lock','run.lock'):\n"
                " with open(name,'a+b') as stream:\n"
                '  try: fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)\n'
                '  except BlockingIOError: pass\n'
                "  else: raise RuntimeError('missing inherited lock')\n"
                "print('fixture success')\n")
        result = run(tmp_path, fds, code)
        assert result['status'] == 'completed' and result['returncode'] == 0
        assert not p._group_exists(result['pid'])
        before = (tmp_path/'job/process.json').read_bytes()
        with pytest.raises(FileExistsError): run(tmp_path, fds, "raise RuntimeError('must not execute')")
        assert (tmp_path/'job/process.json').read_bytes() == before
    assert (tmp_path/'job/worker.log').read_text() == 'fixture success\n'


def test_full_remaining_preflight_blocks_spawn(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('spawn must not happen')
    monkeypatch.setattr(p.subprocess, 'Popen', forbidden)
    with locks(tmp_path) as fds:
        with pytest.raises(RuntimeError, match='remaining allocations'):
            run(tmp_path, fds, 'pass', required_free_bytes=2**80)
    assert state(tmp_path)['status'] == 'failed' and state(tmp_path)['pid'] is None


def test_nonzero_exit_is_terminal_and_retained(tmp_path):
    with locks(tmp_path) as fds:
        with pytest.raises(RuntimeError, match='code 86'):
            run(tmp_path, fds, "import sys; print('synthetic guard exit'); sys.exit(86)")
        assert state(tmp_path)['returncode'] == 86 and state(tmp_path)['status'] == 'failed'
        with pytest.raises(FileExistsError): run(tmp_path, fds, 'pass')
    assert 'synthetic guard exit' in (tmp_path/'job/worker.log').read_text()


@pytest.mark.parametrize('ignore_term', [False, True])
def test_timeout_reaps_owned_child(tmp_path, ignore_term):
    code = ('import signal,time; ' + ('signal.signal(signal.SIGTERM,signal.SIG_IGN); ' if ignore_term else '')
            + 'time.sleep(30)')
    with locks(tmp_path) as fds:
        with pytest.raises(TimeoutError): run(tmp_path, fds, code, timeout_seconds=.15)
    s = state(tmp_path)
    assert s['status'] == 'failed' and s['returncode'] < 0
    assert not p._group_exists(s['pid'])


def test_fast_exit_still_checks_output(tmp_path):
    with locks(tmp_path) as fds:
        with pytest.raises(RuntimeError, match='output limit'):
            run(tmp_path, fds, "print('x'*8192)", poll_seconds=.2)
    assert state(tmp_path)['status'] == 'failed'
    assert (tmp_path/'job/worker.log').stat().st_size > 4096


@pytest.mark.parametrize('cause', ['disk', 'observer', 'interrupt'])
def test_parent_guard_or_interruption_cleans_child(tmp_path, monkeypatch, cause):
    original = p.snapshot; calls = 0
    def sample(output):
        nonlocal calls
        calls += 1
        if calls == 1: return original(output)
        if cause == 'disk': return dict(free_bytes=p.RESERVE-1, output_bytes=0, log_bytes=0)
        if cause == 'observer': raise OSError('synthetic observer failure')
        raise KeyboardInterrupt('synthetic parent interrupt')
    monkeypatch.setattr(p, 'snapshot', sample)
    expected = dict(disk=RuntimeError, observer=OSError, interrupt=KeyboardInterrupt)[cause]
    with locks(tmp_path) as fds:
        with pytest.raises(expected): run(tmp_path, fds, 'import time; time.sleep(30)')
    s = state(tmp_path)
    assert s['status'] == 'failed' and not p._group_exists(s['pid'])


def test_output_files_count_and_symlinks_rejected(tmp_path):
    root = tmp_path/'output'; root.mkdir()
    (root/'blob').write_bytes(b'12345')
    assert p.snapshot(root)['output_bytes'] == 5
    (root/'alias').symlink_to(root/'blob')
    with pytest.raises(ValueError, match='symlinks'): p.snapshot(root)


@pytest.mark.parametrize('overrides', [dict(required_free_bytes=1), dict(timeout_seconds=0),
    dict(poll_seconds=10), dict(output_limit_bytes=True), dict(log_limit_bytes=100000)])
def test_invalid_limits_never_create_job(tmp_path, overrides):
    with locks(tmp_path) as fds:
        with pytest.raises(ValueError): run(tmp_path, fds, 'pass', **overrides)
    assert not (tmp_path/'job').exists()


def test_invalid_lock_descriptors_rejected(tmp_path):
    with locks(tmp_path) as fds:
        for bad in ((fds[0], fds[0]), (1, 2), (fds[0],)):
            with pytest.raises(ValueError): run(tmp_path, bad, 'pass')
    assert not (tmp_path/'job').exists()


def test_cleanup_diagnostic_cannot_hide_original_failure(tmp_path, monkeypatch):
    original = p._stop
    def cleanup(process):
        original(process)
        raise OSError('synthetic cleanup observer failure')
    monkeypatch.setattr(p, '_stop', cleanup)
    with locks(tmp_path) as fds:
        with pytest.raises(RuntimeError, match='code 3'):
            run(tmp_path, fds, 'import sys; sys.exit(3)')
    s = state(tmp_path)
    assert s['status'] == 'failed' and 'code 3' in s['error']
    assert 'synthetic cleanup observer failure' in s['cleanup_error']
    assert not p._group_exists(s['pid'])


def test_watched_trial_includes_sibling_checkpoint_output(tmp_path):
    with locks(tmp_path) as fds:
        with pytest.raises(RuntimeError, match='output limit'):
            run(tmp_path, fds, "from pathlib import Path; Path('checkpoint.bin').write_bytes(b'x'*70000)",
                watch_root=tmp_path)
    assert state(tmp_path)['status'] == 'failed'
    assert (tmp_path/'checkpoint.bin').stat().st_size == 70000
