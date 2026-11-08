"""Pre-execution evidence-completeness correction, no role census replay."""
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'research/tokenizer-evidence/byte-composition-boundary-001'
OUT = BASE/'model-admission002'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1048576):h.update(block)
    return h.hexdigest()


def save(name,value):
    data=(json.dumps(value,sort_keys=True)+'\n').encode()
    assert sum(p.stat().st_size for p in OUT.iterdir())+len(data)<2097152
    with (OUT/name).open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())


def main(expected):
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not list(OUT.iterdir())
        path=BASE/'model-math002/plan.json';assert sha(path)==expected
        plan=json.loads(path.read_text())
        save('attempt.json',dict(pid=os.getpid(),numerical_plan_sha256=expected))
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=120,
                                     rss_limit_bytes=4294967296,reserve_bytes=8589934592,poll_seconds=.25):
                predecessor=BASE/'model-math001'
                assert {p.name for p in predecessor.iterdir()}=={'plan.json','WITHDRAWN_BEFORE_EXECUTION.json'}
                old=json.loads((predecessor/'plan.json').read_text())
                old_review=json.loads((BASE/'model-admission001/review.json').read_text())
                assert old_review['accepted'] and old_review['numerical_plan_sha256']==sha(predecessor/'plan.json')
                assert json.loads((BASE/'model-admission001/guard.json').read_text())['status']=='finished'
                assert json.loads((BASE/'model-admission001/STOP').read_text())['status']=='complete_original_admission'
                roles=json.loads((BASE/'model-admission001/roles.json').read_text())
                assert roles['status']=='reserved_exclusive_unused' and not roles['matches']
                assert roles['roles']==old['roles']==plan['roles']
                assert plan['tolerances']==old['tolerances'] and plan['required_evidence']==old['required_evidence']
                for pin in plan['sources']+plan['assets']+plan['prior_admission_pins']:
                    p=Path(pin['path']);assert p.stat().st_size==pin['size'] and sha(p)==pin['sha256']
                folder=ROOT/'research/byte_input_proof_v1'
                old_driver=(folder/'driver.py').read_text()
                extra="    for n,p in actual_parameters.items():\n        compare(out, prefix+'-initial-'+n, p, ref_parameters[n], 0., 0.)\n"
                assert (folder/'driver002.py').read_text()==old_driver.replace('    assert all(exact_init.values())',extra+'    assert all(exact_init.values())')
                old_entry=(folder/'entry.py').read_text()
                assert (folder/'entry002.py').read_text()==old_entry.replace("model-math001","model-math002").replace("model-admission001","model-admission002").replace('from .driver import run','from .driver002 import run')
                for name in ('driver002.py','entry002.py','admission_delta.py'):
                    compile(ast.parse((folder/name).read_text()),name,'exec')
                extra_bytes=9*(2*28492416+4*16351859)
                assert plan['forecast']['raw_coordinate_upper_bytes']==old['forecast']['raw_coordinate_upper_bytes']+extra_bytes
                assert plan['forecast']['raw_coordinate_upper_bytes']+plan['forecast']['codebook_framing_metadata_allowance']<17179869184
                fs=os.statvfs(ROOT);free=fs.f_bavail*fs.f_frsize
                assert free>=plan['forecast']['entry_bytes']
                save('review.json',dict(accepted=True,numerical_plan_sha256=expected,
                    original_never_executed=True,roles_reused_from_unused_reservation_only=True,
                    correction='Save all exact initializer coordinates before assertion; scientific cases/tolerances unchanged',
                    observed_free_bytes=free))
            save('STOP',dict(status='complete_original_admission',no_replay=True))
        except BaseException as exc:
            save('STOP',dict(status='closed_failed_admission_delta',error=repr(exc),no_replay=True));raise


if __name__=='__main__':main(sys.argv[1])
