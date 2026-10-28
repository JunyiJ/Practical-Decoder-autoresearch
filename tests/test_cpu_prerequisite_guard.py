"""Tiny subprocess fixtures only; never runs a research model or failed fixture."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard, violation


def test_boundaries_and_priority():
    assert violation({'peak_rss_bytes': 100, 'free_bytes': 100}, 1, 100, 100, 2) == 'peak_rss_limit'
    assert violation({'peak_rss_bytes': 99, 'free_bytes': 99}, 1, 100, 100, 2) == 'disk_reserve'
    assert violation({'peak_rss_bytes': 99, 'free_bytes': 100}, 2, 100, 100, 2) == 'wall_time_limit'
    assert violation({'peak_rss_bytes': 99, 'free_bytes': 100}, 1, 100, 100, 2) is None


@pytest.mark.parametrize('limit', [0, -1, float('nan'), float('inf'), True])
def test_invalid_limits_fail_before_file_creation(tmp_path, limit):
    path = tmp_path/'guard.json'
    with pytest.raises(ValueError):
        CpuPrerequisiteGuard(path, tmp_path, rss_limit_bytes=limit)
    assert not path.exists()


def test_success_exception_and_exclusive_journal(tmp_path):
    path = tmp_path/'success.json'
    with CpuPrerequisiteGuard(path, tmp_path, reserve_bytes=1):
        pass
    record = json.loads(path.read_text())
    assert record['status'] == 'finished' and record['observation']['peak_rss_bytes'] > 0
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        with CpuPrerequisiteGuard(path, tmp_path, reserve_bytes=1):
            pass
    assert path.read_bytes() == original
    error_path = tmp_path/'exception.json'
    with pytest.raises(RuntimeError, match='fixture failure'):
        with CpuPrerequisiteGuard(error_path, tmp_path, reserve_bytes=1):
            raise RuntimeError('fixture failure')
    assert json.loads(error_path.read_text())['status'] == 'fixture_exception'


@pytest.mark.parametrize('kind', ['rss', 'disk', 'time', 'observer'])
def test_hard_exit_preserves_diagnostic_with_captured_stdio(tmp_path, kind):
    journal = tmp_path/f'{kind}.json'
    code = '''
import os, sys, time
import src.research.cpu_prerequisite_guard as guard
# Simulate pytest replacing stdout/stderr with disposable capture descriptors.
capture = os.open(os.devnull, os.O_WRONLY)
os.dup2(capture, 1); os.dup2(capture, 2)
kind = sys.argv[2]
options = dict(reserve_bytes=1, timeout_seconds=10, poll_seconds=.01)
if kind == 'rss': options['rss_limit_bytes'] = 1
if kind == 'disk': options['reserve_bytes'] = 10**18
if kind == 'time': options['timeout_seconds'] = .05
if kind == 'observer':
    def broken(root): raise RuntimeError('observer failed')
    guard.observation = broken
with guard.CpuPrerequisiteGuard(sys.argv[1], sys.argv[3], **options):
    time.sleep(2)
'''
    result = subprocess.run([sys.executable, '-c', code, str(journal), kind, str(tmp_path)],
                            cwd=Path(__file__).resolve().parents[1], timeout=15,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == (91 if kind == 'observer' else 90)
    record = json.loads(journal.read_text())
    assert record['status'] == ('guard_error' if kind == 'observer' else 'failed')
    reason = dict(rss='peak_rss_limit', disk='disk_reserve', time='wall_time_limit')
    if kind != 'observer':
        assert record['reason'] == reason[kind]
        assert record['observation'] is not None
    else:
        assert 'observer failed' in record['reason']
