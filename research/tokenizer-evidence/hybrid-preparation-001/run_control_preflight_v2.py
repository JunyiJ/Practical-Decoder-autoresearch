"""Single guarded invocation of the frozen two-control MPS feasibility check."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from src.research.runner import atomic_json, session_lock, source_fingerprint
from src.research.protocol import file_digest
from src.research.tokenizer_study import ROOT, runtime_identity

out = ROOT/'research/tokenizer-evidence/hybrid-preparation-001'
plan = json.loads((out/'control-plan-v2.json').read_text())
assert plan['source'] == source_fingerprint() and plan['runtime'] == runtime_identity()
for name, value in plan['files'].items():
    assert file_digest(ROOT/name) == value
run = ROOT/'research_runs/tokenizer-hybrid-control-preflight-002'
run.mkdir(exist_ok=False)
assert shutil.disk_usage(ROOT).free > 8*1024**3+4*1024**2
with session_lock(run) as fd, session_lock(ROOT/'research_state/tokenizer-worker-lock') as gfd:
    if (run/'STOP').exists():
        raise RuntimeError('Stopped')
    with (run/'worker.log').open('x') as log:
        proc = subprocess.Popen([sys.executable, '-m', 'research.tokenizer-evidence.hybrid-preparation-001.control_preflight_v2'],
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(fd,gfd))
        def stop(signum, frame):
            (run/'STOP').write_text('Stop requested\n'); os.killpg(proc.pid,signal.SIGTERM)
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:
            code=proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            code=124
    assert plan['source']==source_fingerprint()
    atomic_json(run/'state.json',dict(status='passed' if code==0 else 'failed',exit_code=code))
    shutil.copyfile(run/'worker.log',out/'control-preflight-v2.log')
    shutil.copyfile(run/'state.json',out/'control-state-v2.json')
    if code:raise RuntimeError(f'Control feasibility failed: {code}')
