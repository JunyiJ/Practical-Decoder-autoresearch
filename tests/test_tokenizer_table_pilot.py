"""Finite pilot launch invariants without allocating training checkpoints."""
import json
from types import SimpleNamespace
import pytest
from src.research import tokenizer_table_pilot as p


def test_geometry_and_complete_allocation(tmp_path):
    assert p.required_free([]) == 18691843584
    for k in p.KINDS:
        c = p.config(tmp_path/k, k)
        assert c['model']['dim'] == 256 and c['model']['weight_tying_kind'] == k
        assert c['training']['seed'] == 3291 and c['training']['max_iters'] == 20
        assert c['training']['weight_averaging']['start_step'] == 10
        assert c['training']['weight_averaging']['interval'] == 5
    assert p.required_free(list(p.KINDS)) == p.RESERVE+4*1024**2+5*sum(p.ALLOWANCE.values())
    with pytest.raises(ValueError): p.required_free(['scaled_tied'])
    with pytest.raises(ValueError): p.required_free(list(p.KINDS)+['scaled_tied'])


def setup(monkeypatch):
    monkeypatch.setattr(p.subprocess, 'check_output', lambda args, **kw: '' if 'status' in args else 'source')
    monkeypatch.setattr(p.shutil, 'disk_usage', lambda path: SimpleNamespace(free=40*1024**3))


def test_freeze_requests_and_refuse_reentry(tmp_path, monkeypatch):
    setup(monkeypatch); root = tmp_path/'pilot'; p.prepare(root); plan, state = p.verify(root)
    state['status'] = 'failed'; p.atomic_json(root/'state.json', state)
    with pytest.raises(ValueError, match='freshly prepared'): p.run(root)
    req = root/p.KINDS[0]/'request.json'; data = json.loads(req.read_text())
    data['config']['training']['seed'] += 1; p.atomic_json(req, data)
    with pytest.raises(ValueError, match='request changed'): p.verify(root)


def test_full_panel_guard_precedes_creation(tmp_path, monkeypatch):
    setup(monkeypatch)
    monkeypatch.setattr(p.shutil, 'disk_usage', lambda path: SimpleNamespace(free=p.required_free([])-1))
    root = tmp_path/'pilot'
    with pytest.raises(RuntimeError, match='does not fit'): p.prepare(root)
    assert not root.exists()


def test_worker_failure_stops_serial_panel(tmp_path, monkeypatch):
    setup(monkeypatch); monkeypatch.setattr(p, 'ROOT', tmp_path)
    (tmp_path/'research_state/tokenizer-worker-lock').mkdir(parents=True)
    root = tmp_path/'pilot'; p.prepare(root)
    calls = []
    class Failed:
        pid = 123; returncode = 86
        def __init__(self, args, **kw):
            calls.append(args); assert len(kw['pass_fds']) == 2 and kw['start_new_session']
        def poll(self): return 86
    monkeypatch.setattr(p.subprocess, 'Popen', Failed)
    monkeypatch.setattr(p.signal, 'signal', lambda *args: None)
    with pytest.raises(RuntimeError, match='worker failed'): p.run(root)
    assert len(calls) == 1
    assert json.loads((root/'state.json').read_text())['status'] == 'failed'
    with pytest.raises(ValueError, match='freshly prepared'): p.run(root)
    assert len(calls) == 1


def test_timeout_terminates_worker_and_preserves_failure(tmp_path, monkeypatch):
    setup(monkeypatch); monkeypatch.setattr(p, 'ROOT', tmp_path)
    (tmp_path/'research_state/tokenizer-worker-lock').mkdir(parents=True)
    root = tmp_path/'pilot'; p.prepare(root)
    kills = []; elapsed = iter([0, 601])
    class Hung:
        pid = 123; returncode = None
        def __init__(self, *args, **kw): pass
        def poll(self): return self.returncode
        def wait(self, timeout): self.returncode = -15; return -15
    monkeypatch.setattr(p.subprocess, 'Popen', Hung)
    monkeypatch.setattr(p.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(p.time, 'monotonic', lambda: next(elapsed))
    monkeypatch.setattr(p.os, 'killpg', lambda pid, sig: kills.append((pid, sig)))
    with pytest.raises(RuntimeError, match='timeout'): p.run(root)
    assert kills == [(123, p.signal.SIGTERM)]
    assert json.loads((root/p.KINDS[0]/'guard-failure.json').read_text())['reason'] == 'timeout'
    assert not (root/p.KINDS[1]/'worker.log').exists()
