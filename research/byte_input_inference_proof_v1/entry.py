"""Once-only bounded read-only inference consumer entry; no training replay."""
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'research/tokenizer-evidence/byte-composition-boundary-001'
OUT=BASE/'inference-math001'
ARCHIVES=ROOT/'research_runs/byte-input-inference-math001'
CAP=2*1024**3
TORCH=None

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while x:=f.read(1048576):h.update(x)
    return h.hexdigest()

def pins(rows):
    for r in rows:
        p=Path(r['path']);assert p.is_file() and not p.is_symlink() and p.stat().st_size==r['size'] and sha(p)==r['sha256'],str(p)

def publish(name,value):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    assert len(raw)<32*1024**2
    with (OUT/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())

class Guard(CpuPrerequisiteGuard):
    def _sample(self):
        sample,reason=super()._sample()
        paths=list(OUT.iterdir())+list(ARCHIVES.iterdir())
        if any(p.is_symlink() or not p.is_file() for p in paths):reason='unsafe_output'
        sample['output_bytes']=sum(p.lstat().st_size for p in paths)
        if sample['output_bytes']>CAP:reason='output_limit'
        if TORCH is not None:
            sample['mps_driver_bytes']=TORCH.mps.driver_allocated_memory()
            if sample['mps_driver_bytes']>=6*1024**3:reason='mps_limit'
        return sample,reason

def main(expected):
    global TORCH
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert {p.name for p in OUT.iterdir()}=={'plan.json'} and not list(ARCHIVES.iterdir()),'Original once-only entry'
        assert sha(OUT/'plan.json')==expected
        plan=json.loads((OUT/'plan.json').read_text())
        assert plan['forecast']['entry_bytes']==CAP+10*1024**3
        stat=os.statvfs(ROOT);assert stat.f_bavail*stat.f_frsize>=plan['forecast']['entry_bytes']
        publish('attempt.json',dict(pid=os.getpid(),plan_sha256=expected,no_replay=True))
        try:
            with Guard(OUT/'guard.json',ROOT,timeout_seconds=1800,rss_limit_bytes=4*1024**3,
                       reserve_bytes=8*1024**3,poll_seconds=.25):
                pins(plan['sources']);pins(plan['assets']);pins(plan['parents']);pins(plan['prerequisites'])
                assert sys.flags.optimize==0
                for key,value in plan['environment'].items():assert os.environ.get(key)==value,key
                for relative in plan['new_python_sources']:
                    tree=ast.parse((ROOT/relative).read_text())
                    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                        and n.func.attr in ('step','backward','restore_pristine') for n in ast.walk(tree)),relative
                accepted=json.loads((BASE/'training-review001/review.json').read_text())
                assert accepted['accepted'] and len(accepted['cases'])==6
                assert all(p in accepted['archives'] for p in plan['parents']) and len(plan['parents'])==6
                assert plan['required']==dict(cases=6,updates=0,full_validation_targets=205668,
                    invalid_transactions=78,exact_weights=True,exact_logits=True,atol=4e-6,rtol=4e-5)
                publish('admission.json',dict(accepted=True,training_replayed=False,
                    role_policy='Existing parent identities reconstructed solely for loading. No new training roles, training updates, seed search or role census.',
                    plan_sha256=expected,source_files=len(plan['sources'])))
                import torch
                TORCH=torch
                from .driver import run
                run(plan,ARCHIVES,publish)
                pins(plan['sources']);pins(plan['assets']);pins(plan['parents'])
            publish('STOP',dict(status='complete_original_inference_consumers',no_replay=True))
        except BaseException as exc:
            if not (OUT/'STOP').exists():publish('STOP',dict(status='closed_failed_original_inference_consumers',error=repr(exc),no_replay=True))
            raise

if __name__=='__main__':main(sys.argv[1])
