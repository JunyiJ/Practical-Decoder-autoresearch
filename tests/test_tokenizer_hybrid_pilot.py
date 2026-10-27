"""Pilot controller prerequisites without starting training or accessing test loss."""
import importlib
import json
from types import SimpleNamespace

import pytest

pilot = importlib.import_module('research.tokenizer-evidence.hybrid-preparation-001.run_pilot')


def test_pilot_exact_geometry_and_budget(tmp_path):
    cfg = pilot.config(tmp_path)
    assert cfg['model']['attention_layout'] == ['kda', 'kda', 'kda', 'gated_mla']
    assert cfg['model']['dim'] == 64 and cfg['model']['rope'] == 0
    assert cfg['training']['seed'] == 2991 and cfg['training']['max_iters'] == 20
    assert cfg['training']['learning_rate'] == .00125
    assert cfg['training']['lr_schedule']['start_step'] == 10
    assert cfg['training']['weight_averaging']['interval'] == 5


def test_pilot_serial_guard_and_no_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, 'ROOT', tmp_path)
    (tmp_path/'research_state/tokenizer-worker-lock').mkdir(parents=True)
    monkeypatch.setattr(pilot.subprocess, 'check_output', lambda args, **kw: '' if 'status' in args else 'source-commit')
    monkeypatch.setattr(pilot.shutil, 'disk_usage', lambda path: SimpleNamespace(free=10*1024**3))
    monkeypatch.setattr(pilot.signal, 'signal', lambda *args: None)
    root = tmp_path/'pilot'
    monkeypatch.setattr(pilot.sys, 'argv', ['pilot', 'prepare', '--output', str(root)])
    pilot.main()
    plan = pilot.verify(root)
    assert plan['job_seconds'] == 600 and plan['max_output_bytes'] == 256*1024**2
    calls = []
    class Process:
        pid = 123
        def __init__(self, args, **kwargs):
            calls.append(args)
            assert len(kwargs['pass_fds']) == 2 and kwargs['start_new_session']
            (root/'result.json').write_text('{}')
        def wait(self, timeout):
            assert timeout == 600
            return 0
    monkeypatch.setattr(pilot.subprocess, 'Popen', Process)
    monkeypatch.setattr(pilot.sys, 'argv', ['pilot', 'run', '--output', str(root)])
    pilot.main()
    assert len(calls) == 1 and json.loads((root/'state.json').read_text())['status'] == 'completed'
    with pytest.raises(AssertionError):
        pilot.main()
    assert len(calls) == 1
    # Changing a frozen request field cannot quietly resume or launch a worker.
    plan['config']['training']['learning_rate'] *= 2
    (root/'plan.json').write_text(json.dumps(plan))
    with pytest.raises(AssertionError):
        pilot.verify(root)


def test_pilot_requires_full_output_headroom(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot.subprocess, 'check_output', lambda *args, **kw: '')
    monkeypatch.setattr(pilot.shutil, 'disk_usage', lambda path: SimpleNamespace(free=8*1024**3))
    root = tmp_path/'insufficient'
    monkeypatch.setattr(pilot.sys, 'argv', ['pilot', 'prepare', '--output', str(root)])
    with pytest.raises(AssertionError):
        pilot.main()
    assert not root.exists()
